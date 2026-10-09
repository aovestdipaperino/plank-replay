//! Turns per-layer activation captures into a normalized direction.

use std::io::Write;
use std::path::Path;

use super::{Profile, VectorizeError};

/// Running sums over target and control captures.
///
/// The direction is `target - control`, optionally made orthogonal to the
/// control mean. With ds4's projection edit, a positive scale then strips
/// the target's component from activations that sit near the control.
///
/// This is heretic's computation (`modifiers/abliteration.py`) with its
/// `bad` prompts as the target and its `good` prompts as the control: each
/// side is averaged over its own captures in `f64`, the difference is
/// normalized, and with `orthogonalize` (heretic's `orthogonalize_direction`,
/// on by default there) the part along the normalized control mean is
/// removed and the result normalized again.
#[derive(Debug, Clone)]
pub struct Accumulator {
    profile: Profile,
    targets: usize,
    controls: usize,
    pairs: usize,
    target_sum: Vec<f64>,
    control_sum: Vec<f64>,
    pair_sum: Vec<f64>,
}

impl Accumulator {
    /// Starts an empty accumulator for `profile`'s shape.
    #[must_use]
    pub fn new(profile: Profile) -> Self {
        let n = profile.layers * profile.width;
        Self {
            profile,
            targets: 0,
            controls: 0,
            pairs: 0,
            target_sum: vec![0.0; n],
            control_sum: vec![0.0; n],
            pair_sum: vec![0.0; n],
        }
    }

    /// Number of pairs added so far.
    #[must_use]
    pub fn pairs(&self) -> usize {
        self.pairs
    }

    /// Adds one target capture, `layers * width` floats, layer-major.
    ///
    /// # Panics
    /// Panics if the capture does not match the profile's shape, which would
    /// be a bug in the caller.
    pub fn add_target(&mut self, target: &[f32]) {
        assert_eq!(
            target.len(),
            self.target_sum.len(),
            "target capture has the wrong shape"
        );
        for (sum, &x) in self.target_sum.iter_mut().zip(target) {
            *sum += f64::from(x);
        }
        self.targets += 1;
    }

    /// Adds one control capture, `layers * width` floats, layer-major.
    ///
    /// # Panics
    /// Panics if the capture does not match the profile's shape.
    pub fn add_control(&mut self, control: &[f32]) {
        assert_eq!(
            control.len(),
            self.control_sum.len(),
            "control capture has the wrong shape"
        );
        for (sum, &x) in self.control_sum.iter_mut().zip(control) {
            *sum += f64::from(x);
        }
        self.controls += 1;
    }

    /// Adds one target and one control capture as a pair, which is also what
    /// `pair_normalize` averages over.
    ///
    /// # Panics
    /// Panics if either capture does not match the profile's shape.
    pub fn add_pair(&mut self, target: &[f32], control: &[f32]) {
        self.add_target(target);
        self.add_control(control);
        let w = self.profile.width;
        for layer in 0..self.profile.layers {
            let span = layer * w..(layer + 1) * w;
            let mut diff: Vec<f64> = target[span.clone()]
                .iter()
                .zip(&control[span.clone()])
                .map(|(&t, &f)| f64::from(t) - f64::from(f))
                .collect();
            normalize(&mut diff);
            for (sum, d) in self.pair_sum[span].iter_mut().zip(diff) {
                *sum += d;
            }
        }
        self.pairs += 1;
    }

    /// Computes the direction from the pairs added so far.
    ///
    /// With `pair_normalize`, each pair's difference is normalized before
    /// averaging, so no single pair dominates; otherwise the direction is the
    /// difference of the two means. With `orthogonalize`, the component
    /// parallel to the control mean is removed, so the vector does not simply
    /// encode the baseline activation.
    ///
    /// # Panics
    /// Panics if either side has no capture, or `pair_normalize` is asked
    /// for without pairs.
    #[must_use]
    pub fn finish(&self, orthogonalize: bool, pair_normalize: bool) -> Direction {
        assert!(self.targets > 0 && self.controls > 0, "nothing to average");
        assert!(!pair_normalize || self.pairs > 0, "no pairs to normalize");
        #[allow(clippy::cast_precision_loss, reason = "capture counts are small")]
        let (nt, nc, n) = (self.targets as f64, self.controls as f64, self.pairs as f64);
        let w = self.profile.width;
        let mut data = Vec::with_capacity(self.profile.layers * w);
        for layer in 0..self.profile.layers {
            let span = layer * w..(layer + 1) * w;
            let mut dir: Vec<f64> = if pair_normalize {
                self.pair_sum[span.clone()].iter().map(|x| x / n).collect()
            } else {
                self.target_sum[span.clone()]
                    .iter()
                    .zip(&self.control_sum[span.clone()])
                    .map(|(t, f)| t / nt - f / nc)
                    .collect()
            };
            normalize(&mut dir);
            if orthogonalize {
                let mut base: Vec<f64> = self.control_sum[span].iter().map(|x| x / nc).collect();
                normalize(&mut base);
                let p = dot(&dir, &base);
                for (d, b) in dir.iter_mut().zip(&base) {
                    *d -= p * b;
                }
                normalize(&mut dir);
            }
            #[allow(clippy::cast_possible_truncation, reason = "the file format is f32")]
            data.extend(dir.iter().map(|&x| x as f32));
        }
        Direction {
            profile: self.profile,
            data,
        }
    }
}

/// A finished steering vector: one direction per layer, unit length when
/// built from captures, scaled by the fitted strength when recovered from a
/// weight difference.
#[derive(Debug, Clone)]
pub struct Direction {
    profile: Profile,
    data: Vec<f32>,
}

impl Direction {
    /// A vector from raw layer-major values, as [`super::diff`] builds them.
    ///
    /// # Panics
    /// Panics if `data` is not `layers * width` floats.
    #[must_use]
    pub fn from_values(profile: Profile, data: Vec<f32>) -> Self {
        assert_eq!(
            data.len(),
            profile.layers * profile.width,
            "values do not match the profile's shape"
        );
        Self { profile, data }
    }

    /// The shape the vector was built for.
    #[must_use]
    pub fn profile(&self) -> Profile {
        self.profile
    }

    /// The flat layer-major values.
    #[must_use]
    pub fn as_slice(&self) -> &[f32] {
        &self.data
    }

    /// The direction for one layer.
    ///
    /// # Panics
    /// Panics if `layer` is out of range.
    #[must_use]
    pub fn layer(&self, layer: usize) -> &[f32] {
        let w = self.profile.width;
        &self.data[layer * w..(layer + 1) * w]
    }

    /// The raw little-endian `f32` matrix ds4 loads.
    #[must_use]
    pub fn to_le_bytes(&self) -> Vec<u8> {
        self.data.iter().flat_map(|x| x.to_le_bytes()).collect()
    }

    /// Writes the raw little-endian `f32` matrix ds4 loads.
    ///
    /// # Errors
    /// Fails when the file cannot be written.
    pub fn write_f32(&self, path: impl AsRef<Path>) -> Result<(), VectorizeError> {
        let path = path.as_ref();
        let bytes = self.to_le_bytes();
        std::fs::File::create(path)
            .and_then(|mut f| f.write_all(&bytes))
            .map_err(|e| VectorizeError::io(path, e))
    }
}

/// Heretic's symmetric winsorization, applied to one capture before it is
/// averaged: per layer, the magnitudes of the `width` components are clamped
/// to their `quantile`-quantile, computed as `torch.quantile` does (linear
/// interpolation between the two nearest ranks). It tames the "massive
/// activations" some models carry in a few components.
///
/// # Panics
/// Panics if `quantile` is outside `0..=1` or the capture is not a whole
/// number of layers.
pub fn winsorize(capture: &mut [f32], width: usize, quantile: f64) {
    assert!((0.0..=1.0).contains(&quantile), "quantile out of range");
    assert!(
        width > 0 && capture.len().is_multiple_of(width),
        "ragged capture"
    );
    let mut sorted = vec![0.0f32; width];
    for layer in capture.chunks_exact_mut(width) {
        for (s, x) in sorted.iter_mut().zip(layer.iter()) {
            *s = x.abs();
        }
        sorted.sort_unstable_by(f32::total_cmp);
        #[allow(clippy::cast_precision_loss, reason = "widths are small")]
        let pos = quantile * (width - 1) as f64;
        #[allow(
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            reason = "pos lies in 0..width"
        )]
        let (lo, hi) = (pos.floor() as usize, pos.ceil() as usize);
        let frac = pos - pos.floor();
        #[allow(clippy::cast_possible_truncation, reason = "the capture is f32")]
        let threshold =
            (f64::from(sorted[lo]) + (f64::from(sorted[hi]) - f64::from(sorted[lo])) * frac) as f32;
        for x in layer.iter_mut() {
            *x = x.clamp(-threshold, threshold);
        }
    }
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// Scales `v` to unit length; a zero vector is left alone.
fn normalize(v: &mut [f64]) {
    let n2 = dot(v, v);
    if n2 > 0.0 {
        let inv = 1.0 / n2.sqrt();
        for x in v {
            *x *= inv;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const TINY: Profile = Profile {
        name: "tiny",
        layers: 2,
        width: 3,
        residual_dump: "ffn_out",
        residual_branches: 1,
    };

    fn norm(v: &[f32]) -> f32 {
        v.iter().map(|x| x * x).sum::<f32>().sqrt()
    }

    #[test]
    fn the_direction_points_from_the_from_mean_to_the_to_mean() {
        let mut acc = Accumulator::new(TINY);
        acc.add_pair(
            &[2.0, 0.0, 0.0, 0.0, 0.0, 5.0],
            &[0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        );
        let d = acc.finish(false, false);
        assert_eq!(d.layer(0), &[1.0, 0.0, 0.0]);
        assert_eq!(d.layer(1), &[0.0, 0.0, 1.0]);
    }

    #[test]
    fn orthogonalizing_removes_the_from_mean_component() {
        let mut acc = Accumulator::new(TINY);
        acc.add_pair(
            &[1.0, 1.0, 0.0, 1.0, 1.0, 0.0],
            &[1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
        );
        let d = acc.finish(true, false);
        // target - control = (0,1,0), already orthogonal to control = (1,0,0).
        assert_eq!(d.layer(0), &[0.0, 1.0, 0.0]);

        let mut acc = Accumulator::new(TINY);
        acc.add_pair(
            &[2.0, 1.0, 0.0, 2.0, 1.0, 0.0],
            &[1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
        );
        let plain = acc.finish(false, false);
        let ortho = acc.finish(true, false);
        assert!(plain.layer(0)[0] > 0.5);
        assert!(ortho.layer(0)[0].abs() < 1e-6);
        assert!((norm(ortho.layer(0)) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn pair_normalizing_weights_every_pair_equally() {
        let mut acc = Accumulator::new(TINY);
        // A huge pair along x, a small one along y.
        acc.add_pair(&[100.0, 0.0, 0.0, 1.0, 0.0, 0.0], &[0.0; 6]);
        acc.add_pair(&[0.0, 1.0, 0.0, 1.0, 0.0, 0.0], &[0.0; 6]);
        let d = acc.finish(false, true);
        let l0 = d.layer(0);
        assert!((l0[0] - l0[1]).abs() < 1e-6, "{l0:?}");
        let means = acc.finish(false, false);
        assert!(means.layer(0)[0] > 0.99);
    }

    #[test]
    fn each_side_is_averaged_over_its_own_count() {
        let mut acc = Accumulator::new(TINY);
        // Targets average to (2,0,0); one control at (0,0,0).
        acc.add_target(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0]);
        acc.add_target(&[3.0, 0.0, 0.0, 0.0, 0.0, 1.0]);
        acc.add_control(&[0.0, 1.0, 0.0, 0.0, 0.0, 0.0]);
        let d = acc.finish(false, false);
        // (2,-1,0) normalized.
        let n = 5f32.sqrt();
        assert!((d.layer(0)[0] - 2.0 / n).abs() < 1e-6);
        assert!((d.layer(0)[1] + 1.0 / n).abs() < 1e-6);
    }

    #[test]
    #[allow(
        clippy::float_cmp,
        reason = "the clamped values are exact by construction"
    )]
    fn winsorizing_clamps_each_layer_to_its_quantile() {
        // Layer 0 magnitudes 1,2,3,4,100: the 0.75 quantile sits at rank 3
        // (4.0); layer 1 at 0.5 between ranks interpolates 2.5 of 1..4.
        let mut c = [1.0, -2.0, 3.0, 4.0, -100.0];
        winsorize(&mut c, 5, 0.75);
        assert_eq!(c, [1.0, -2.0, 3.0, 4.0, -4.0]);
        let mut c = [1.0, 2.0, 3.0, -4.0];
        winsorize(&mut c, 4, 0.5);
        assert_eq!(c, [1.0, 2.0, 2.5, -2.5]);
        // Quantile 1 clamps nothing.
        let mut c = [1.0, -9.0, 3.0];
        winsorize(&mut c, 3, 1.0);
        assert_eq!(c, [1.0, -9.0, 3.0]);
    }

    #[test]
    fn the_file_is_flat_little_endian_f32() {
        let mut acc = Accumulator::new(TINY);
        acc.add_pair(&[1.0, 0.0, 0.0, 0.0, 1.0, 0.0], &[0.0; 6]);
        let path = std::env::temp_dir().join(format!(
            "plank-tools-test-dir-{}-{:?}.f32",
            std::process::id(),
            std::thread::current().id()
        ));
        acc.finish(false, false).write_f32(&path).unwrap();
        let bytes = std::fs::read(&path).unwrap();
        std::fs::remove_file(&path).unwrap();
        assert_eq!(bytes.len(), 6 * 4);
        assert_eq!(&bytes[0..4], &1.0f32.to_le_bytes());
        assert_eq!(&bytes[16..20], &1.0f32.to_le_bytes());
    }
}
