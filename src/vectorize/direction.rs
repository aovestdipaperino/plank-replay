//! Turns per-layer activation captures into a normalized direction.

use std::io::Write;
use std::path::Path;

use super::{Profile, VectorizeError};

/// Running sums over paired target/control captures.
///
/// The direction is `target - control`, optionally made orthogonal to the
/// control mean. With ds4's projection edit, a positive scale then strips
/// the target's component from activations that sit near the control.
#[derive(Debug, Clone)]
pub struct Accumulator {
    profile: Profile,
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

    /// Adds one pair of captures, each `layers * width` floats, layer-major.
    ///
    /// # Panics
    /// Panics if either capture does not match the profile's shape, which
    /// would be a bug in the caller.
    pub fn add_pair(&mut self, target: &[f32], control: &[f32]) {
        let n = self.profile.layers * self.profile.width;
        assert_eq!(target.len(), n, "target capture has the wrong shape");
        assert_eq!(control.len(), n, "control capture has the wrong shape");
        for (sum, &x) in self.target_sum.iter_mut().zip(target) {
            *sum += f64::from(x);
        }
        for (sum, &x) in self.control_sum.iter_mut().zip(control) {
            *sum += f64::from(x);
        }
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
    /// Panics if no pair has been added.
    #[must_use]
    pub fn finish(&self, orthogonalize: bool, pair_normalize: bool) -> Direction {
        assert!(self.pairs > 0, "no pairs to average");
        #[allow(clippy::cast_precision_loss, reason = "pair counts are small")]
        let n = self.pairs as f64;
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
                    .map(|(t, f)| t / n - f / n)
                    .collect()
            };
            normalize(&mut dir);
            if orthogonalize {
                let mut base: Vec<f64> = self.control_sum[span].iter().map(|x| x / n).collect();
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

/// A finished steering vector: one unit direction per layer.
#[derive(Debug, Clone)]
pub struct Direction {
    profile: Profile,
    data: Vec<f32>,
}

impl Direction {
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
