//! `pt vectorize --diff`: recover a steering vector from a model and an
//! edited copy of it, without running either.

use std::io::IsTerminal;
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::Instant;

use plank_tools::vectorize::diff::{DiffReport, LayerFit, Step, WeightDiff};
use plank_tools::vectorize::{Component, Direction, Model, Profile, VectorStore, VectorizeError};

use super::CliError;

/// What a `--diff` run needs from the parsed arguments.
#[derive(Debug)]
pub struct DiffArgs<'a> {
    /// The base model as given on the command line.
    pub model_arg: &'a str,
    /// The resolved base model.
    pub model: &'a Model,
    /// Its steering shape.
    pub profile: Profile,
    /// The edited copy.
    pub edited: &'a Path,
    /// `--component`, when given as `ffn_out` or `attn_out`.
    pub component: Option<Component>,
    /// `-o FILE`.
    pub out: Option<&'a Path>,
    /// `-n NAME`.
    pub name: Option<&'a str>,
    /// The vector store.
    pub store: &'a VectorStore,
    /// `-f`.
    pub force: bool,
    /// `-q`.
    pub quiet: bool,
}

#[allow(clippy::needless_pass_by_value, reason = "used as a map_err adapter")]
fn failed(e: VectorizeError) -> CliError {
    CliError::Failed(e.to_string())
}

/// Runs the comparison, prints the per-layer fit, and saves the vector.
pub fn run(args: &DiffArgs<'_>) -> Result<ExitCode, CliError> {
    let say = |line: &str| {
        if !args.quiet {
            eprintln!("{line}");
        }
    };
    let started = Instant::now();
    let diff = WeightDiff::open(args.model.path(), args.edited, args.profile).map_err(failed)?;
    say(&format!(
        "base   : {}\nedited : {}\nshape  : {} layers x {}  compute: {}",
        args.model.path().display(),
        args.edited.display(),
        args.profile.layers,
        args.profile.width,
        if diff.uses_gpu() { "Metal" } else { "CPU" },
    ));
    let live = !args.quiet && std::io::stderr().is_terminal();
    let report = diff
        .run(&mut |step| match step {
            Step::Comparing { done, total } if live => {
                eprint!("\r\x1b[Kcomparing tensors {done}/{total}");
            }
            Step::Fitted(fit) if fit.is_changed() && !args.quiet => {
                if live {
                    eprint!("\r\x1b[K");
                }
                eprintln!("{}", fit_line(fit));
            }
            _ => {}
        })
        .map_err(failed)?;
    if live {
        eprint!("\r\x1b[K");
    }
    for note in notes(&report) {
        say(&note);
    }

    let changed = report.changed_components();
    let component = choose_component(args.component, &changed)?;
    let (direction, sign) = report.direction(component);
    let took = super::vectorize::human(started.elapsed().as_secs_f64());
    let layers = report
        .component(component)
        .filter(|f| f.is_changed())
        .count();
    let shape = format!(
        "{} x {} f32, {layers} edited layer(s), {took}",
        args.profile.layers, args.profile.width
    );

    save(args, &report, component, &direction, sign, &shape)?;
    let scale = scale_flag(component);
    #[allow(clippy::cast_possible_truncation, reason = "the sign is ±1")]
    let s = sign as i32;
    let effect = if s > 0 {
        "removes the direction, as the edit did"
    } else {
        "amplifies the direction, as the edit did"
    };
    match args.name {
        Some(name) => println!(
            "\nthe edit is reproduced at scale {s} ({effect}):\n  plank --dir-steering {name} --dir-steering-{scale} {s}\n  \
             or in engines.local.json: \"steering\": {{ \"direction\": \"{name}\", \"{scale}\": {s} }}"
        ),
        None => println!(
            "\nthe edit is reproduced at {scale} scale {s} ({effect}); plank loads directions by \
             name: rerun with -n NAME to store this one in vectors.json"
        ),
    }
    Ok(ExitCode::SUCCESS)
}

/// Writes the vector to `-o` (with its sidecar) and/or the store under `-n`.
///
/// A failed store write with no `-o` rescues the vector to `./<name>.f32`.
fn save(
    args: &DiffArgs<'_>,
    report: &DiffReport,
    component: Component,
    direction: &Direction,
    sign: f32,
    shape: &str,
) -> Result<(), CliError> {
    if let Some(out) = args.out {
        if let Some(parent) = out.parent().filter(|p| !p.as_os_str().is_empty()) {
            std::fs::create_dir_all(parent).map_err(|e| format!("{}: {e}", parent.display()))?;
        }
        direction.write_f32(out).map_err(failed)?;
        let meta = super::vectorize::meta_path(out);
        write_meta(args, report, component, sign, &meta)?;
        println!(
            "wrote {} ({shape})\nwrote {}",
            out.display(),
            meta.display()
        );
    }
    if let Some(name) = args.name {
        let key = args.model.key();
        match args.store.put(&key, name, direction, args.force) {
            Ok(replaced) => println!(
                "{} vector `{name}` for `{key}` in {} ({shape})",
                if replaced { "replaced" } else { "stored" },
                args.store.path().display()
            ),
            Err(e) if args.out.is_none() => {
                let rescue = PathBuf::from(format!("{name}.f32"));
                let kept = direction.write_f32(&rescue).is_ok();
                return Err(CliError::Failed(if kept {
                    format!("{e}\nthe vector was saved to {} instead", rescue.display())
                } else {
                    e.to_string()
                }));
            }
            Err(e) => return Err(failed(e)),
        }
    }
    Ok(())
}

/// `--component` when given, else the one component the edit touched.
fn choose_component(
    asked: Option<Component>,
    changed: &[Component],
) -> Result<Component, CliError> {
    match (asked, changed) {
        (Some(c), _) if changed.contains(&c) => Ok(c),
        (_, []) => Err(CliError::Failed(
            "no residual writer differs between the two files; there is no vector to recover"
                .into(),
        )),
        (Some(c), _) => Err(CliError::Failed(format!(
            "the edit does not touch {c}; it changed {}",
            names(changed)
        ))),
        (None, [one]) => Ok(*one),
        (None, _) => Err(CliError::Usage(format!(
            "the edit changed both {}; ds4 takes one vector per run, so pick one with \
             `--component` (and run again with another name for the other)",
            names(changed)
        ))),
    }
}

fn names(components: &[Component]) -> String {
    components
        .iter()
        .map(|c| c.as_str())
        .collect::<Vec<_>>()
        .join(" and ")
}

/// The ds4 scale a component's vector is applied with.
fn scale_flag(component: Component) -> &'static str {
    match component {
        Component::AttnOut => "attn",
        Component::FfnOut | Component::Residual => "ffn",
    }
}

/// `layer 12 attn_out  λ +3.405  along d 90.2%  σ₂/σ₁ 0.057  (1 tensor)`
fn fit_line(fit: &LayerFit) -> String {
    format!(
        "layer {:>2} {:<8}  λ {:+.3}  along d {:>5.1}%  σ₂/σ₁ {:.3}  ({} tensor{})",
        fit.layer,
        fit.component.as_str(),
        fit.lambda,
        fit.explained * 100.0,
        fit.second,
        fit.changed.len(),
        if fit.changed.len() == 1 { "" } else { "s" },
    )
}

/// Warnings about what the vector cannot carry.
fn notes(report: &DiffReport) -> Vec<String> {
    let mut out = Vec::new();
    if !report.unmatched.is_empty() {
        out.push(format!(
            "note: {} tensor(s) exist in only one file, e.g. {}; are these the same model?",
            report.unmatched.len(),
            report.unmatched[0]
        ));
    }
    if !report.other_changes.is_empty() {
        let shown: Vec<&str> = report
            .other_changes
            .iter()
            .take(5)
            .map(String::as_str)
            .collect();
        out.push(format!(
            "note: {} other changed tensor(s) are not residual writers of a steerable layer, \
             so the vector cannot reproduce them: {}{}",
            report.other_changes.len(),
            shown.join(", "),
            if report.other_changes.len() > shown.len() {
                ", ..."
            } else {
                ""
            }
        ));
    }
    let weak: Vec<usize> = report
        .fits
        .iter()
        .filter(|f| f.is_changed() && f.explained < 0.5)
        .map(|f| f.layer)
        .collect();
    if !weak.is_empty() {
        out.push(format!(
            "note: in layer(s) {weak:?} less than half of the change lies along one direction: \
             the edit there is not a rank-one ablation (fine-tuning?), or quantization noise \
             dominates it, and the vector only approximates it"
        ));
    }
    for c in report.changed_components() {
        let (pos, neg) =
            report
                .component(c)
                .filter(|f| f.is_changed())
                .fold((0, 0), |(p, n), f| {
                    if f.lambda > 0.0 {
                        (p + 1, n)
                    } else {
                        (p, n + 1)
                    }
                });
        if pos > 0 && neg > 0 {
            out.push(format!(
                "note: {c} removes the direction in {pos} layer(s) and amplifies it in {neg}; \
                 one scale cannot do both, so the minority layers are left at zero"
            ));
        }
    }
    out
}

/// Writes the provenance sidecar, with the per-layer fit.
fn write_meta(
    args: &DiffArgs<'_>,
    report: &DiffReport,
    component: Component,
    sign: f32,
    path: &Path,
) -> Result<(), CliError> {
    let layers: Vec<serde_json::Value> = report
        .component(component)
        .filter(|f| f.is_changed())
        .map(|f| {
            serde_json::json!({
                "layer": f.layer,
                "lambda": f.lambda,
                "explained": f.explained,
                "second_over_first": f.second,
                "tensors": f.changed,
            })
        })
        .collect();
    let meta = serde_json::json!({
        "format": "ds4-directional-steering-v1",
        "shape": [args.profile.layers, args.profile.width],
        "profile": args.profile.name,
        "component": component.as_str(),
        "method": "weight diff: top left singular vector of (edited - base) per layer, \
                   scaled by sqrt(|lambda|) with lambda = -d'(dW)W'd / d'WW'd",
        "model": args.model_arg,
        "model_path": args.model.path(),
        "edited_path": args.edited,
        "scale": sign,
        "layers": layers,
        "other_changes": report.other_changes,
        "note": "rows are not unit length: each carries its layer's fitted strength, so \
                 the edit is reproduced at the given scale and unchanged layers are zero",
    });
    let text = serde_json::to_string_pretty(&meta).map_err(|e| e.to_string())? + "\n";
    std::fs::write(path, text).map_err(|e| format!("{}: {e}", path.display()))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_one_changed_component_is_chosen() {
        assert_eq!(
            choose_component(None, &[Component::AttnOut]).unwrap(),
            Component::AttnOut
        );
    }

    #[test]
    fn two_changed_components_need_a_choice() {
        let both = [Component::AttnOut, Component::FfnOut];
        assert!(matches!(
            choose_component(None, &both),
            Err(CliError::Usage(_))
        ));
        assert_eq!(
            choose_component(Some(Component::FfnOut), &both).unwrap(),
            Component::FfnOut
        );
    }

    #[test]
    fn an_untouched_component_is_refused() {
        assert!(choose_component(Some(Component::FfnOut), &[Component::AttnOut]).is_err());
        assert!(choose_component(None, &[]).is_err());
    }
}
