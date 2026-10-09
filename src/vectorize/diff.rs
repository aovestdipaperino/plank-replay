//! Recovers a steering vector from the weights of a model and an edited copy.
//!
//! Abliteration replaces each matrix that writes into the residual stream,
//! `W`, with `W' = W - λ d dᵀ W` for a unit direction `d`. The difference
//! `Δ = W' - W = -λ d (dᵀ W)` is rank one, and `d` is its only left singular
//! vector. ds4's runtime steering, `y - s d dᵀ y` on a layer's `attn_out` or
//! `ffn_out`, is the same edit applied to the matrix's output, so `d` at
//! scale `λ` reproduces it exactly.
//!
//! Per layer and component this accumulates three `width x width` Gram
//! matrices over every changed residual writer (each routed expert counts as
//! one more block of columns): `G = Δ Δᵀ`, `B = W Wᵀ` and `X = Δ Wᵀ`. Then:
//!
//! - `d` is the top eigenvector of `G`;
//! - `explained = μ₁ / tr G` is the share of the change along `d` (1 for a
//!   clean rank-one edit), and `second = √(μ₂ / μ₁)` the next singular value
//!   relative to the first;
//! - `λ = -dᵀ X d / dᵀ B d`, a least-squares fit of `dᵀ Δ = -λ dᵀ W`. Rounding
//!   noise from re-quantizing is uncorrelated with `W`, so it does not bias
//!   `λ`, and its sign separates removal (positive) from amplification.
//!
//! Only byte-different tensors are dequantized, so a pair that shares most of
//! its payload (an edit patched into a copy) costs little more than reading
//! both files once.

use std::collections::HashMap;
use std::fs::File;
use std::os::unix::fs::FileExt as _;
use std::path::{Path, PathBuf};

use candle_core::quantized::{GgmlDType, ggml_file::qtensor_from_ggml};
use candle_core::{DType, Device, Tensor};

use super::gguf::{self, TensorInfo};
use super::{Component, Direction, Profile, VectorizeError};

/// Floats of one dequantized batch kept in memory at once (about 1 GiB).
const BATCH_FLOATS: u64 = 1 << 28;

/// Bytes compared per read when looking for changed tensors.
const COMPARE_CHUNK: usize = 16 << 20;

/// Squarings of the Gram matrix before reading its top eigenvector: the
/// second eigenvalue's share shrinks to its 64th power.
const SQUARINGS: usize = 6;

/// The residual writers of each component, by tensor-name suffix.
const WRITERS: &[(Component, &str)] = &[
    (Component::AttnOut, "attn_output_b.weight"),
    (Component::AttnOut, "attn_output.weight"),
    (Component::FfnOut, "ffn_down.weight"),
    (Component::FfnOut, "ffn_down_exps.weight"),
    (Component::FfnOut, "ffn_down_shexp.weight"),
];

fn failed(e: impl std::fmt::Display) -> VectorizeError {
    VectorizeError::msg(e.to_string())
}

/// What the difference says about one layer's component.
#[derive(Debug, Clone)]
pub struct LayerFit {
    /// The layer.
    pub layer: usize,
    /// The activation the changed matrices write.
    pub component: Component,
    /// The residual writers whose bytes differ.
    pub changed: Vec<String>,
    /// The unit direction, `width` floats; empty when nothing changed.
    pub direction: Vec<f32>,
    /// The fitted strength: positive removes the direction, negative
    /// amplifies it. 0 when nothing changed.
    pub lambda: f64,
    /// Share of the change's energy along the direction, `0..=1`.
    pub explained: f64,
    /// Second singular value of the change relative to the first.
    pub second: f64,
}

impl LayerFit {
    /// Whether any residual writer of this layer's component changed.
    #[must_use]
    pub fn is_changed(&self) -> bool {
        !self.changed.is_empty()
    }
}

/// The outcome of [`WeightDiff::run`].
#[derive(Debug, Clone)]
pub struct DiffReport {
    /// The shape the fits were made for.
    pub profile: Profile,
    /// One fit per layer and component, layer-major, attention first.
    pub fits: Vec<LayerFit>,
    /// Changed tensors that are not residual writers of a normal layer
    /// (embeddings, the output head, MTP layers, other matrices): steering
    /// cannot express these.
    pub other_changes: Vec<String>,
    /// Tensors present in only one of the files.
    pub unmatched: Vec<String>,
}

impl DiffReport {
    /// The fits of one component, in layer order.
    pub fn component(&self, component: Component) -> impl Iterator<Item = &LayerFit> {
        self.fits.iter().filter(move |f| f.component == component)
    }

    /// Components with at least one changed layer.
    #[must_use]
    pub fn changed_components(&self) -> Vec<Component> {
        [Component::AttnOut, Component::FfnOut]
            .into_iter()
            .filter(|&c| self.component(c).any(LayerFit::is_changed))
            .collect()
    }

    /// The steering vector for `component`: each changed layer's direction
    /// scaled by `√|λ|`, so ds4 reproduces the edit at scale `sign`
    /// (`1` for removal, `-1` for amplification) and every unchanged layer is
    /// a zero row. Returns the vector and `sign`, taken from the majority of
    /// the changed layers' strengths.
    #[must_use]
    pub fn direction(&self, component: Component) -> (Direction, f32) {
        let fits: Vec<&LayerFit> = self.component(component).collect();
        let positive = fits.iter().filter(|f| f.lambda > 0.0).count();
        let negative = fits.iter().filter(|f| f.lambda < 0.0).count();
        let sign = if negative > positive { -1.0 } else { 1.0 };
        let w = self.profile.width;
        let mut data = vec![0.0f32; self.profile.layers * w];
        for fit in fits.iter().filter(|f| f.is_changed()) {
            // A layer whose sign disagrees with the majority cannot be
            // expressed under one scale; it is left at zero.
            if fit.lambda * f64::from(sign) <= 0.0 {
                continue;
            }
            #[allow(clippy::cast_possible_truncation, reason = "the file format is f32")]
            let k = fit.lambda.abs().sqrt() as f32;
            for (o, &x) in data[fit.layer * w..(fit.layer + 1) * w]
                .iter_mut()
                .zip(&fit.direction)
            {
                *o = k * x;
            }
        }
        (Direction::from_values(self.profile, data), sign)
    }
}

/// Progress of a [`WeightDiff::run`].
#[derive(Debug, Clone, Copy)]
pub enum Step<'a> {
    /// Comparing tensor bytes: `done` of `total` tensors.
    Comparing {
        /// Tensors compared so far.
        done: usize,
        /// Tensors to compare.
        total: usize,
    },
    /// One layer's component has been fitted.
    Fitted(&'a LayerFit),
}

/// A base model and an edited copy of it, ready to compare.
#[derive(Debug)]
pub struct WeightDiff {
    base: PathBuf,
    edited: PathBuf,
    profile: Profile,
    base_tensors: Vec<TensorInfo>,
    edited_tensors: HashMap<String, TensorInfo>,
    device: Device,
}

impl WeightDiff {
    /// Reads both tensor tables.
    ///
    /// # Errors
    /// Fails when either file is not a usable GGUF.
    pub fn open(
        base: impl AsRef<Path>,
        edited: impl AsRef<Path>,
        profile: Profile,
    ) -> Result<Self, VectorizeError> {
        let base = base.as_ref().to_path_buf();
        let edited = edited.as_ref().to_path_buf();
        let base_tensors = gguf::tensors(&base)?;
        let edited_tensors = gguf::tensors(&edited)?
            .into_iter()
            .map(|t| (t.name.clone(), t))
            .collect();
        Ok(Self {
            base,
            edited,
            profile,
            base_tensors,
            edited_tensors,
            device: pick_device(),
        })
    }

    /// Computes on `device` instead of the default (Metal when available).
    #[must_use]
    pub fn on(mut self, device: Device) -> Self {
        self.device = device;
        self
    }

    /// Whether the fits run on the GPU.
    #[must_use]
    pub fn uses_gpu(&self) -> bool {
        !self.device.is_cpu()
    }

    /// Compares every tensor and fits each changed layer component.
    ///
    /// # Errors
    /// Fails on a read error, when a changed residual writer differs in shape
    /// or type between the files, or when its type cannot be dequantized.
    pub fn run(&self, on_step: &mut dyn FnMut(Step<'_>)) -> Result<DiffReport, VectorizeError> {
        let base_file = File::open(&self.base).map_err(|e| VectorizeError::io(&self.base, e))?;
        let edited_file =
            File::open(&self.edited).map_err(|e| VectorizeError::io(&self.edited, e))?;

        let mut unmatched: Vec<String> = self
            .edited_tensors
            .keys()
            .filter(|n| !self.base_tensors.iter().any(|t| &t.name == *n))
            .map(|n| format!("{n} (only in the edited file)"))
            .collect();
        let mut changed: Vec<(&TensorInfo, &TensorInfo)> = Vec::new();
        let total = self.base_tensors.len();
        for (done, b) in self.base_tensors.iter().enumerate() {
            on_step(Step::Comparing { done, total });
            let Some(e) = self.edited_tensors.get(&b.name) else {
                unmatched.push(format!("{} (only in the base file)", b.name));
                continue;
            };
            if !same_bytes(&base_file, b, &edited_file, e)
                .map_err(|err| VectorizeError::io(&self.edited, err))?
            {
                changed.push((b, e));
            }
        }
        on_step(Step::Comparing { done: total, total });
        unmatched.sort();

        let mut by_slot: HashMap<(usize, Component), Vec<(&TensorInfo, &TensorInfo)>> =
            HashMap::new();
        let mut other_changes = Vec::new();
        for (b, e) in changed {
            match self.writer_slot(&b.name) {
                Some(slot) => by_slot.entry(slot).or_default().push((b, e)),
                None => other_changes.push(b.name.clone()),
            }
        }
        other_changes.sort();

        let mut fits = Vec::with_capacity(self.profile.layers * 2);
        for layer in 0..self.profile.layers {
            for component in [Component::AttnOut, Component::FfnOut] {
                let pairs = by_slot.remove(&(layer, component)).unwrap_or_default();
                let fit = if pairs.is_empty() {
                    LayerFit {
                        layer,
                        component,
                        changed: Vec::new(),
                        direction: Vec::new(),
                        lambda: 0.0,
                        explained: 0.0,
                        second: 0.0,
                    }
                } else {
                    self.fit(layer, component, &pairs, &base_file, &edited_file)?
                };
                on_step(Step::Fitted(&fit));
                fits.push(fit);
            }
        }
        Ok(DiffReport {
            profile: self.profile,
            fits,
            other_changes,
            unmatched,
        })
    }

    /// The layer and component `name` writes, if it is a residual writer of
    /// a normal layer.
    fn writer_slot(&self, name: &str) -> Option<(usize, Component)> {
        let rest = name.strip_prefix("blk.")?;
        let (layer, suffix) = rest.split_once('.')?;
        let layer: usize = layer.parse().ok()?;
        if layer >= self.profile.layers {
            return None;
        }
        WRITERS
            .iter()
            .find(|(_, s)| *s == suffix)
            .map(|&(c, _)| (layer, c))
    }

    fn fit(
        &self,
        layer: usize,
        component: Component,
        pairs: &[(&TensorInfo, &TensorInfo)],
        base_file: &File,
        edited_file: &File,
    ) -> Result<LayerFit, VectorizeError> {
        let width = self.profile.width;
        let zeros = || Tensor::zeros((width, width), DType::F32, &self.device).map_err(failed);
        let (mut g_delta, mut g_base, mut g_cross) = (zeros()?, zeros()?, zeros()?);
        for (base, edited) in pairs {
            check_pair(base, edited, width)?;
            let (cols, rows) = (base.dims[0], base.dims[1]);
            let experts = base.dims.get(2).copied().unwrap_or(1);
            let floats = cols * rows;
            let slice = base.bytes / experts;
            let batch = (BATCH_FLOATS / floats).clamp(1, experts);
            let mut first = 0;
            while first < experts {
                let n = batch.min(experts - first);
                let range = first * slice..(first + n) * slice;
                let wt = self.load(base_file, base, range.clone(), n, rows, cols)?;
                let et = self.load(edited_file, edited, range, n, rows, cols)?;
                let delta = (&et - &wt).map_err(failed)?;
                let wt_t = wt.t().and_then(|t| t.contiguous()).map_err(failed)?;
                let gram = |a: &Tensor, bt: &Tensor| -> Result<Tensor, VectorizeError> {
                    a.matmul(bt).and_then(|m| m.sum(0)).map_err(failed)
                };
                let delta_t = delta.t().and_then(|t| t.contiguous()).map_err(failed)?;
                g_delta = (&g_delta + gram(&delta, &delta_t)?).map_err(failed)?;
                g_base = (&g_base + gram(&wt, &wt_t)?).map_err(failed)?;
                g_cross = (&g_cross + gram(&delta, &wt_t)?).map_err(failed)?;
                first += n;
            }
        }
        let (direction, mu1) = top_eigen(&g_delta)?;
        let trace = to_f64(&g_delta)?.iter().step_by(width + 1).sum::<f64>();
        let deflated = deflate(&g_delta, &direction, mu1, &self.device)?;
        let (_, mu2) = top_eigen(&deflated)?;
        let quad = |m: &Tensor| -> Result<f64, VectorizeError> {
            let v = to_f64(m)?;
            Ok(quadratic(&v, &direction))
        };
        let (along_x, along_b) = (quad(&g_cross)?, quad(&g_base)?);
        let lambda = if along_b > 0.0 {
            -along_x / along_b
        } else {
            0.0
        };
        #[allow(clippy::cast_possible_truncation, reason = "the file format is f32")]
        let direction: Vec<f32> = direction.iter().map(|&v| v as f32).collect();
        Ok(LayerFit {
            layer,
            component,
            changed: pairs.iter().map(|(b, _)| b.name.clone()).collect(),
            direction,
            lambda,
            explained: if trace > 0.0 { mu1 / trace } else { 0.0 },
            second: if mu1 > 0.0 {
                (mu2.max(0.0) / mu1).sqrt()
            } else {
                0.0
            },
        })
    }

    /// Dequantizes `n` consecutive `rows x cols` slices of `info` (experts,
    /// or the one matrix) from `range` of its payload into an `(n, rows,
    /// cols)` tensor on the compute device.
    fn load(
        &self,
        file: &File,
        info: &TensorInfo,
        range: std::ops::Range<u64>,
        n: u64,
        rows: u64,
        cols: u64,
    ) -> Result<Tensor, VectorizeError> {
        let dtype = ggml_dtype(info.ty).ok_or_else(|| {
            VectorizeError::msg(format!(
                "{} is {}, which cannot be dequantized here",
                info.name,
                gguf::type_name(info.ty)
            ))
        })?;
        let len = usize::try_from(range.end - range.start).map_err(failed)?;
        let mut raw = vec![0u8; len];
        file.read_exact_at(&mut raw, info.offset + range.start)
            .map_err(|e| VectorizeError::msg(format!("{}: {e}", info.name)))?;
        let (n, rows, cols) = (to_usize(n)?, to_usize(rows)?, to_usize(cols)?);
        let slice = len / n;
        // Each slice dequantizes on its own thread; candle's CPU path is
        // single-threaded and is the slow half of a fit.
        let parts: Vec<Result<Tensor, VectorizeError>> = std::thread::scope(|s| {
            let handles: Vec<_> = raw
                .chunks(slice)
                .map(|bytes| {
                    s.spawn(move || {
                        qtensor_from_ggml(dtype, bytes, vec![rows, cols], &Device::Cpu)
                            .and_then(|q| q.dequantize(&Device::Cpu))
                            .map_err(failed)
                    })
                })
                .collect();
            handles
                .into_iter()
                .map(|h| {
                    h.join().unwrap_or_else(|_| {
                        Err(VectorizeError::msg("a dequantize thread panicked"))
                    })
                })
                .collect()
        });
        let parts = parts.into_iter().collect::<Result<Vec<_>, _>>()?;
        Tensor::stack(&parts, 0)
            .and_then(|t| t.to_device(&self.device))
            .map_err(failed)
    }
}

/// Metal when the machine has it, else the CPU.
fn pick_device() -> Device {
    #[cfg(target_os = "macos")]
    if let Ok(d) = Device::new_metal(0) {
        return d;
    }
    Device::Cpu
}

fn to_usize(n: u64) -> Result<usize, VectorizeError> {
    usize::try_from(n).map_err(failed)
}

/// Whether the two tensors hold the same bytes; a different size is a
/// difference.
fn same_bytes(
    base: &File,
    base_info: &TensorInfo,
    edited: &File,
    edited_info: &TensorInfo,
) -> std::io::Result<bool> {
    let (ai, bi) = (base_info, edited_info);
    if ai.bytes != bi.bytes || ai.dims != bi.dims || ai.ty != bi.ty {
        return Ok(false);
    }
    let mut x = vec![0u8; COMPARE_CHUNK];
    let mut y = vec![0u8; COMPARE_CHUNK];
    let mut at = 0u64;
    while at < ai.bytes {
        let n = usize::try_from((ai.bytes - at).min(COMPARE_CHUNK as u64)).unwrap_or(COMPARE_CHUNK);
        base.read_exact_at(&mut x[..n], ai.offset + at)?;
        edited.read_exact_at(&mut y[..n], bi.offset + at)?;
        if x[..n] != y[..n] {
            return Ok(false);
        }
        at += n as u64;
    }
    Ok(true)
}

/// Checks that a changed writer can be compared: same shape and type in both
/// files, `width` output rows, and a type the dequantizer reads.
fn check_pair(base: &TensorInfo, edited: &TensorInfo, width: usize) -> Result<(), VectorizeError> {
    if base.dims != edited.dims || base.ty != edited.ty {
        return Err(VectorizeError::msg(format!(
            "{}: {:?} {} in the base but {:?} {} in the edited file; the two must share \
             format and quantization",
            base.name,
            base.dims,
            gguf::type_name(base.ty),
            edited.dims,
            gguf::type_name(edited.ty)
        )));
    }
    if base.dims.len() < 2 || base.dims.len() > 3 || base.dims[1] != width as u64 {
        return Err(VectorizeError::msg(format!(
            "{}: shape {:?} does not write {width}-wide rows",
            base.name, base.dims
        )));
    }
    if ggml_dtype(base.ty).is_none() {
        return Err(VectorizeError::msg(format!(
            "{} is {}, which cannot be dequantized here",
            base.name,
            gguf::type_name(base.ty)
        )));
    }
    Ok(())
}

/// The candle type for a GGML type id, for the types it dequantizes.
fn ggml_dtype(ty: u32) -> Option<GgmlDType> {
    Some(match ty {
        0 => GgmlDType::F32,
        1 => GgmlDType::F16,
        30 => GgmlDType::BF16,
        2 => GgmlDType::Q4_0,
        3 => GgmlDType::Q4_1,
        6 => GgmlDType::Q5_0,
        7 => GgmlDType::Q5_1,
        8 => GgmlDType::Q8_0,
        10 => GgmlDType::Q2K,
        11 => GgmlDType::Q3K,
        12 => GgmlDType::Q4K,
        13 => GgmlDType::Q5K,
        14 => GgmlDType::Q6K,
        _ => return None,
    })
}

fn to_f64(m: &Tensor) -> Result<Vec<f64>, VectorizeError> {
    m.to_device(&Device::Cpu)
        .and_then(|t| t.flatten_all())
        .and_then(|t| t.to_dtype(DType::F64))
        .and_then(|t| t.to_vec1::<f64>())
        .map_err(failed)
}

/// `vᵀ M v` for a square row-major `m`.
fn quadratic(m: &[f64], v: &[f64]) -> f64 {
    let n = v.len();
    m.chunks_exact(n)
        .zip(v)
        .map(|(row, &vi)| vi * row.iter().zip(v).map(|(a, b)| a * b).sum::<f64>())
        .sum()
}

/// The top eigenvector (unit, its largest-magnitude entry positive) and
/// eigenvalue of a symmetric positive semi-definite `g`.
///
/// `g` is squared [`SQUARINGS`] times, normalized each time, so the top
/// eigenvalue dominates by the 64th power of its gap; a few plain power steps
/// on `g` itself then polish the vector. A zero matrix gives a zero vector.
fn top_eigen(g: &Tensor) -> Result<(Vec<f64>, f64), VectorizeError> {
    let n = g.dim(0).map_err(failed)?;
    let mut p = g.clone();
    for _ in 0..SQUARINGS {
        let norm = p
            .sqr()
            .and_then(|t| t.sum_all())
            .and_then(|t| t.to_scalar::<f32>())
            .map_err(failed)?
            .sqrt();
        if !(norm.is_finite() && norm > 0.0) {
            break;
        }
        let q = p.affine(1.0 / f64::from(norm), 0.0).map_err(failed)?;
        p = q.matmul(&q).map_err(failed)?;
    }
    let p = to_f64(&p)?;
    let g = to_f64(g)?;
    // Start from the column of `p` with the most mass: `p` is close to
    // `v vᵀ`, so every column is a multiple of `v`.
    let col = (0..n)
        .max_by(|&a, &b| p[a * n + a].total_cmp(&p[b * n + b]))
        .unwrap_or(0);
    let mut v: Vec<f64> = (0..n).map(|r| p[r * n + col]).collect();
    if !normalize(&mut v) {
        return Ok((vec![0.0; n], 0.0));
    }
    for _ in 0..4 {
        let mut next = matvec(&g, &v);
        if !normalize(&mut next) {
            break;
        }
        v = next;
    }
    let mu = quadratic(&g, &v);
    let big = v
        .iter()
        .copied()
        .max_by(|a, b| a.abs().total_cmp(&b.abs()))
        .unwrap_or(0.0);
    if big < 0.0 {
        for x in &mut v {
            *x = -*x;
        }
    }
    Ok((v, mu))
}

/// `g - μ v vᵀ`, on `device`.
fn deflate(g: &Tensor, v: &[f64], mu: f64, device: &Device) -> Result<Tensor, VectorizeError> {
    let n = v.len();
    #[allow(clippy::cast_possible_truncation, reason = "the Gram matrices are f32")]
    let col: Vec<f32> = v.iter().map(|&x| x as f32).collect();
    let col = Tensor::from_vec(col, (n, 1), device).map_err(failed)?;
    let outer = col
        .matmul(&col.t().map_err(failed)?)
        .and_then(|o| o.affine(mu, 0.0))
        .map_err(failed)?;
    (g - outer).map_err(failed)
}

fn matvec(m: &[f64], v: &[f64]) -> Vec<f64> {
    m.chunks_exact(v.len())
        .map(|row| row.iter().zip(v).map(|(a, b)| a * b).sum())
        .collect()
}

/// Scales `v` to unit length; `false` (and `v` untouched) when it is zero.
fn normalize(v: &mut [f64]) -> bool {
    let n2: f64 = v.iter().map(|x| x * x).sum();
    if !(n2 > 0.0 && n2.is_finite()) {
        return false;
    }
    let inv = 1.0 / n2.sqrt();
    for x in v.iter_mut() {
        *x *= inv;
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::quantized::QTensor;

    const WIDTH: usize = 16;

    fn profile() -> Profile {
        Profile {
            name: "test",
            layers: 3,
            width: WIDTH,
            residual_dump: "ffn_out",
            residual_branches: 1,
        }
    }

    /// A deterministic pseudo-random matrix, `rows x cols`, row-major.
    fn matrix(rows: usize, cols: usize, seed: u64) -> Vec<f32> {
        let mut state = seed;
        (0..rows * cols)
            .map(|_| {
                state = state
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                #[allow(clippy::cast_precision_loss, reason = "test data")]
                let x = (state >> 40) as f32 / (1u64 << 24) as f32;
                x - 0.5
            })
            .collect()
    }

    fn unit(seed: u64) -> Vec<f32> {
        let mut d = matrix(WIDTH, 1, seed);
        let n = d.iter().map(|x| x * x).sum::<f32>().sqrt();
        for x in &mut d {
            *x /= n;
        }
        d
    }

    /// `W - λ d dᵀ W` for a row-major `rows x cols` `w` with `rows = d.len()`.
    fn ablate(w: &[f32], d: &[f32], lambda: f32) -> Vec<f32> {
        let cols = w.len() / d.len();
        let mut out = w.to_vec();
        for c in 0..cols {
            let dot: f32 = (0..d.len()).map(|r| d[r] * w[r * cols + c]).sum();
            for r in 0..d.len() {
                out[r * cols + c] -= lambda * d[r] * dot;
            }
        }
        out
    }

    /// A tensor to write: name, GGML dims, type id, payload.
    type Entry = (String, Vec<u64>, u32, Vec<u8>);

    fn f32_entry(name: &str, dims: &[u64], data: &[f32]) -> Entry {
        let bytes = data.iter().flat_map(|x| x.to_le_bytes()).collect();
        (name.to_string(), dims.to_vec(), 0, bytes)
    }

    fn q8_entry(name: &str, rows: usize, cols: usize, data: &[f32]) -> Entry {
        let t = Tensor::from_vec(data.to_vec(), (rows, cols), &Device::Cpu).unwrap();
        let q = QTensor::quantize(&t, GgmlDType::Q8_0).unwrap();
        let bytes = q.data().unwrap().into_owned();
        (name.to_string(), vec![cols as u64, rows as u64], 8, bytes)
    }

    fn push_str(b: &mut Vec<u8>, s: &str) {
        b.extend_from_slice(&(s.len() as u64).to_le_bytes());
        b.extend_from_slice(s.as_bytes());
    }

    /// Writes a GGUF v3 file holding `entries`, 32-byte aligned.
    fn write(path: &Path, entries: &[Entry]) {
        let mut b = b"GGUF".to_vec();
        b.extend_from_slice(&3u32.to_le_bytes());
        b.extend_from_slice(&(entries.len() as u64).to_le_bytes());
        b.extend_from_slice(&1u64.to_le_bytes());
        push_str(&mut b, "general.architecture");
        b.extend_from_slice(&8u32.to_le_bytes());
        push_str(&mut b, "test");
        let mut offset = 0u64;
        for (name, dims, ty, data) in entries {
            push_str(&mut b, name);
            b.extend_from_slice(&u32::try_from(dims.len()).unwrap().to_le_bytes());
            for d in dims {
                b.extend_from_slice(&d.to_le_bytes());
            }
            b.extend_from_slice(&ty.to_le_bytes());
            b.extend_from_slice(&offset.to_le_bytes());
            offset += (data.len() as u64).div_ceil(32) * 32;
        }
        while !b.len().is_multiple_of(32) {
            b.push(0);
        }
        for (_, _, _, data) in entries {
            b.extend_from_slice(data);
            while !b.len().is_multiple_of(32) {
                b.push(0);
            }
        }
        std::fs::write(path, b).unwrap();
    }

    fn dir(name: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("pt-diff-{name}-{}", std::process::id()));
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    fn run(base: &[Entry], edited: &[Entry], name: &str) -> DiffReport {
        let d = dir(name);
        let (b, e) = (d.join("base.gguf"), d.join("edited.gguf"));
        write(&b, base);
        write(&e, edited);
        let report = WeightDiff::open(&b, &e, profile())
            .unwrap()
            .on(Device::Cpu)
            .run(&mut |_| {})
            .unwrap();
        let _ = std::fs::remove_dir_all(d);
        report
    }

    fn cosine(a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b).map(|(x, y)| x * y).sum()
    }

    #[test]
    fn an_exact_rank_one_edit_is_recovered_with_its_strength() {
        let cols = 24;
        let experts = 4;
        let d = unit(11);
        let attn = matrix(WIDTH, 32, 1);
        let exps = matrix(WIDTH * experts, cols, 2);
        let edited_exps: Vec<f32> = exps
            .chunks(WIDTH * cols)
            .flat_map(|e| ablate(e, &d, 0.8))
            .collect();
        let base = [
            f32_entry("blk.0.attn_output.weight", &[32, WIDTH as u64], &attn),
            f32_entry(
                "blk.1.ffn_down_exps.weight",
                &[cols as u64, WIDTH as u64, experts as u64],
                &exps,
            ),
            f32_entry(
                "token_embd.weight",
                &[WIDTH as u64, 4],
                &matrix(4, WIDTH, 3),
            ),
        ];
        let mut edited = base.clone();
        edited[1] = f32_entry(
            "blk.1.ffn_down_exps.weight",
            &[cols as u64, WIDTH as u64, experts as u64],
            &edited_exps,
        );
        edited[2] = f32_entry(
            "token_embd.weight",
            &[WIDTH as u64, 4],
            &matrix(4, WIDTH, 4),
        );
        let report = run(&base, &edited, "exact");

        assert_eq!(report.changed_components(), [Component::FfnOut]);
        assert_eq!(report.other_changes, ["token_embd.weight"]);
        let fit = report
            .component(Component::FfnOut)
            .find(|f| f.layer == 1)
            .unwrap();
        assert!(cosine(&fit.direction, &d).abs() > 0.9999, "{fit:?}");
        assert!((fit.lambda - 0.8).abs() < 1e-3, "{}", fit.lambda);
        assert!(fit.explained > 0.9999 && fit.second < 1e-2, "{fit:?}");

        let (vector, sign) = report.direction(Component::FfnOut);
        assert!((sign - 1.0).abs() < f32::EPSILON);
        let row = vector.layer(1);
        let norm = row.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 0.8f32.sqrt()).abs() < 1e-3, "{norm}");
        assert!(vector.layer(0).iter().all(|&x| x == 0.0));
        assert!(vector.layer(2).iter().all(|&x| x == 0.0));
    }

    #[test]
    fn an_amplifying_edit_gets_a_negative_scale() {
        let d = unit(21);
        let w = matrix(WIDTH, 64, 5);
        let base = [f32_entry("blk.2.ffn_down.weight", &[64, WIDTH as u64], &w)];
        let edited = [f32_entry(
            "blk.2.ffn_down.weight",
            &[64, WIDTH as u64],
            &ablate(&w, &d, -0.5),
        )];
        let report = run(&base, &edited, "amplify");
        let fit = report
            .component(Component::FfnOut)
            .find(|f| f.layer == 2)
            .unwrap();
        assert!((fit.lambda + 0.5).abs() < 1e-3, "{}", fit.lambda);
        assert!((report.direction(Component::FfnOut).1 + 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn requantized_q8_weights_still_give_the_direction() {
        let (rows, cols) = (WIDTH, 512);
        let d = unit(31);
        let w = matrix(rows, cols, 6);
        let base = [q8_entry("blk.0.attn_output_b.weight", rows, cols, &w)];
        let edited = [q8_entry(
            "blk.0.attn_output_b.weight",
            rows,
            cols,
            &ablate(&w, &d, 1.0),
        )];
        let report = run(&base, &edited, "q8");
        let fit = report
            .component(Component::AttnOut)
            .find(|f| f.layer == 0)
            .unwrap();
        assert!(cosine(&fit.direction, &d).abs() > 0.999, "{fit:?}");
        assert!((fit.lambda - 1.0).abs() < 0.02, "{}", fit.lambda);
        assert!(fit.explained > 0.95, "{fit:?}");
    }

    #[test]
    fn identical_files_change_nothing() {
        let base = [f32_entry(
            "blk.0.ffn_down.weight",
            &[8, WIDTH as u64],
            &matrix(WIDTH, 8, 7),
        )];
        let report = run(&base, &base, "same");
        assert!(report.changed_components().is_empty());
        assert!(report.other_changes.is_empty() && report.unmatched.is_empty());
    }

    #[test]
    fn a_mismatched_format_is_refused() {
        let w = matrix(WIDTH, 32, 8);
        let base = [f32_entry("blk.0.ffn_down.weight", &[32, WIDTH as u64], &w)];
        let edited = [q8_entry("blk.0.ffn_down.weight", WIDTH, 32, &w)];
        let d = dir("mismatch");
        let (b, e) = (d.join("base.gguf"), d.join("edited.gguf"));
        write(&b, &base);
        write(&e, &edited);
        let err = WeightDiff::open(&b, &e, profile())
            .unwrap()
            .on(Device::Cpu)
            .run(&mut |_| {})
            .unwrap_err();
        let _ = std::fs::remove_dir_all(d);
        assert!(err.to_string().contains("share format"), "{err}");
    }
}
