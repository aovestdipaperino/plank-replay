//! `pt vectorize`: build a directional-steering vector for a plank model.

use std::io::IsTerminal;
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::{Duration, Instant};

use plank_tools::vectorize::{
    Accumulator, Capture, Component, Direction, Fetch, HubCache, Model, PROFILES, Profile,
    PromptSource, Tick, VectorStore,
};

use super::CliError;

/// Usage text for `pt vectorize --help`.
pub const USAGE: &str = "\
pt vectorize - build a steering vector from two prompt sets

USAGE:
    pt vectorize <model> --from SRC --to SRC (-n NAME | -o FILE) [OPTIONS]
    pt vectorize <model> --diff EDITED.gguf (-n NAME | -o FILE) [OPTIONS]

Loads the model once with the ds4 engine linked into pt, runs every prompt
through it with activation dumps on, and writes one unit direction per layer
separating the two sets. The computation is heretic's (github.com/p-e-w/heretic):
the residual stream after each layer at the last prompt token, averaged per
set, `from - to` normalized and made orthogonal to the `to` mean. With
`--from` as heretic's bad prompts and `--to` as its good ones, the vector is
heretic's refusal direction. At runtime a positive scale pushes the model
towards `to` (it strips the `from` component); a negative scale pushes it
towards `from`.

With --diff, nothing is run: the vector is read out of the weights instead.
EDITED must be <model> with an abliteration-style edit (W' = W - λ d dᵀ W on
the matrices that write the residual stream), in the same format and
quantization. Per layer, the edit's direction d is the top singular vector of
the weight difference and λ its fitted strength; each row is d scaled by
√|λ|, so the vector reproduces the edit at scale 1 (-1 for an amplifying
edit). Only byte-different tensors are dequantized; the fit runs on Metal.

The vector is stored under NAME in ~/.plank/models/vectors.json (created if
missing), keyed by the plank engine name, and/or written to FILE with its
metadata next to it as FILE.json.

ARGS:
    <model>             plank engine name from its catalog (engines.json, plus
                        ~/.plank/engines.local.json), e.g. ds4vision; a .gguf
                        path is mapped back to the engine it belongs to

PROMPT SOURCES (SRC):
    path/to/file        Local .txt (one prompt per line, # comments), .jsonl or .json
    owner/name          Hugging Face dataset, `train` split (or the first one);
    owner:name          the same, written with a colon
    owner/name:path     a .txt/.jsonl/.json file in the dataset repo, or a split
                        such as `test` (a data file name like
                        data/train-00000-of-00001.parquet selects that split).
                        HF_API_KEY (or HF_TOKEN) is sent when set.
                        Rows are cached in ~/.cache/plank-tools/hf (or
                        $XDG_CACHE_HOME/plank-tools/hf) and reused; an
                        interrupted or rate-limited fetch resumes from there.
                        Both sides are sized first and neither is fetched
                        beyond the smaller one (or --limit).

OPTIONS:
        --diff EDITED   Recover the vector from the weights of EDITED, an
                        edited copy of <model> (replaces --from/--to); with
                        --component ffn_out or attn_out to pick a component
                        when the edit touched both
        --from SRC      Baseline prompts; repeat to combine sources
        --to SRC        Target prompts; repeat to combine sources
    -n, --name NAME     Store the vector as NAME for this model in
                        ~/.plank/models/vectors.json (base64 f32)
    -o, --out FILE      Also (or instead) write the raw little-endian f32 file
    -l, --limit N       Use at most N prompt pairs
        --column NAME   Field holding the prompt in JSON or dataset rows
        --metal DIR     Metal kernel sources (default: $DS4_METAL_DIR, the
                        plank/refs/ds4/metal pt was built against, or
                        ../share/plank/metal beside the pt binary)
        --profile NAME  Override the shape detected from the model
                        (deepseek-v4-flash, glm-5.3-flash, qwen3.8-flash-next)
        --component C   residual (default; heretic's residual stream after each
                        layer), ffn_out (ds4 build_direction.py's dump) or
                        attn_out
        --winsorize Q   Clamp each capture's per-layer magnitudes to their
                        Q-quantile before averaging (heretic's
                        winsorization_quantile; off by default)
        --ctx N         Context size per capture (default: 512)
        --system TEXT   System prompt (default: \"You are a helpful assistant.\")
        --think         Capture after <think> instead of a direct answer
        --pair-normalize
                        Average normalized per-pair differences
        --no-orthogonalize
                        Keep the component parallel to the `to` mean
        --refresh       Download Hugging Face sources again, replacing the cache
        --no-cache      Neither read nor write the dataset cache
    -f, --force         Replace an existing model/NAME pair or output file
    -q, --quiet         Do not report progress
    -h, --help          Show this help
";

/// Parsed `pt vectorize` arguments.
#[derive(Debug)]
#[allow(
    clippy::struct_excessive_bools,
    reason = "each bool is an independent CLI flag"
)]
struct Args {
    model: String,
    diff: Option<PathBuf>,
    component_given: bool,
    from: Vec<PromptSource>,
    to: Vec<PromptSource>,
    out: Option<PathBuf>,
    name: Option<String>,
    limit: Option<usize>,
    column: Option<String>,
    metal: Option<PathBuf>,
    profile: Option<Profile>,
    component: Component,
    ctx: u32,
    system: String,
    think: bool,
    winsorize: Option<f64>,
    pair_normalize: bool,
    orthogonalize: bool,
    force: bool,
    quiet: bool,
    refresh: bool,
    no_cache: bool,
}

/// Runs `pt vectorize`.
pub fn run(args: &[String]) -> Result<ExitCode, CliError> {
    let args = parse_args(args)?;
    vectorize(&args)
}

fn parse_args(args: &[String]) -> Result<Args, CliError> {
    let usage = |m: String| CliError::Usage(m);
    let mut model = None;
    let mut diff = None;
    let mut component_given = false;
    let (mut from, mut to) = (Vec::new(), Vec::new());
    let mut out = None;
    let mut name = None;
    let mut limit = None;
    let mut column = None;
    let mut metal = None;
    let mut profile = None;
    let mut component = Component::Residual;
    let mut winsorize = None;
    let mut ctx = 512;
    let mut system = "You are a helpful assistant.".to_string();
    let (mut think, mut pair_normalize, mut orthogonalize) = (false, false, true);
    let (mut force, mut quiet) = (false, false);
    let (mut refresh, mut no_cache) = (false, false);

    let mut it = args.iter();
    while let Some(arg) = it.next() {
        let mut value = |flag: &str| {
            it.next()
                .cloned()
                .ok_or_else(|| usage(format!("`{flag}` needs a value")))
        };
        match arg.as_str() {
            "-h" | "--help" => return Err(CliError::Help),
            "--diff" => diff = Some(PathBuf::from(value("--diff")?)),
            "--from" => from.push(PromptSource::parse(&value("--from")?)),
            "--to" => to.push(PromptSource::parse(&value("--to")?)),
            "-o" | "--out" => out = Some(PathBuf::from(value("--out")?)),
            "-n" | "--name" => {
                let v = value("--name")?;
                if v.trim().is_empty() {
                    return Err(usage("`--name` cannot be empty".into()));
                }
                name = Some(v);
            }
            "-l" | "--limit" => limit = Some(positive("--limit", &value("--limit")?)?),
            "--column" => column = Some(value("--column")?),
            "--metal" => metal = Some(PathBuf::from(value("--metal")?)),
            "--profile" => profile = Some(named_profile(&value("--profile")?)?),
            "--component" => {
                component = named_component(&value("--component")?)?;
                component_given = true;
            }
            "--ctx" => ctx = positive("--ctx", &value("--ctx")?)?,
            "--system" => system = value("--system")?,
            "--think" => think = true,
            "--winsorize" => winsorize = Some(quantile(&value("--winsorize")?)?),
            "--pair-normalize" => pair_normalize = true,
            "--no-orthogonalize" => orthogonalize = false,
            "-f" | "--force" => force = true,
            "-q" | "--quiet" => quiet = true,
            "--refresh" => refresh = true,
            "--no-cache" => no_cache = true,
            other if other.starts_with('-') => {
                return Err(usage(format!("unknown option `{other}`")));
            }
            other if model.is_none() => model = Some(other.to_string()),
            other => return Err(usage(format!("unexpected argument `{other}`"))),
        }
    }

    let model = model.ok_or_else(|| usage("a model name is required".into()))?;
    check_sources(
        diff.is_some(),
        &from,
        &to,
        component_given.then_some(component),
    )?;
    if out.is_none() && name.is_none() {
        return Err(usage(
            "say where the vector goes: `-n NAME`, `-o FILE`, or both".into(),
        ));
    }
    Ok(Args {
        model,
        diff,
        component_given,
        from,
        to,
        out,
        name,
        limit,
        column,
        metal,
        profile,
        component,
        ctx,
        system,
        think,
        winsorize,
        pair_normalize,
        orthogonalize,
        force,
        quiet,
        refresh,
        no_cache,
    })
}

/// Checks the prompt sources against the mode: `--diff` takes none, a
/// capture run needs both sides.
fn check_sources(
    diff: bool,
    from: &[PromptSource],
    to: &[PromptSource],
    component: Option<Component>,
) -> Result<(), CliError> {
    let usage = |m: &str| Err(CliError::Usage(m.to_string()));
    if diff {
        if !from.is_empty() || !to.is_empty() {
            return usage("`--diff` reads the vector from weights; drop `--from` and `--to`");
        }
        if component == Some(Component::Residual) {
            return usage("`--diff` recovers ffn_out or attn_out directions, not residual");
        }
    } else if from.is_empty() || to.is_empty() {
        return usage("both `--from` and `--to` are required");
    }
    Ok(())
}

/// Refuses to overwrite an existing `-o` file or its sidecar without `-f`.
fn check_outputs(args: &Args) -> Result<(), CliError> {
    if let (Some(out), false) = (&args.out, args.force) {
        for path in [out.clone(), meta_path(out)] {
            if path.exists() {
                return Err(format!(
                    "{} already exists; pass --force to overwrite it",
                    path.display()
                )
                .into());
            }
        }
    }
    Ok(())
}

/// Hands a `--diff` run to [`super::vecdiff`].
fn run_diff(
    args: &Args,
    model: &Model,
    profile: Profile,
    edited: &Path,
    store: &VectorStore,
) -> Result<ExitCode, CliError> {
    super::vecdiff::run(&super::vecdiff::DiffArgs {
        model_arg: &args.model,
        model,
        profile,
        edited,
        component: args.component_given.then_some(args.component),
        out: args.out.as_deref(),
        name: args.name.as_deref(),
        store,
        force: args.force,
        quiet: args.quiet,
    })
}

/// Parses a `--component` name.
fn named_component(v: &str) -> Result<Component, CliError> {
    Component::parse(v).ok_or_else(|| {
        CliError::Usage(format!(
            "`--component` is residual, ffn_out or attn_out, not `{v}`"
        ))
    })
}

/// Parses a `--winsorize` quantile, strictly between 0 and 1.
fn quantile(v: &str) -> Result<f64, CliError> {
    v.parse()
        .ok()
        .filter(|q: &f64| *q > 0.0 && *q < 1.0)
        .ok_or_else(|| {
            CliError::Usage(format!(
                "`--winsorize` needs a quantile between 0 and 1, not `{v}`"
            ))
        })
}

/// Looks up a `--profile` name, listing the known ones on a miss.
fn named_profile(v: &str) -> Result<Profile, CliError> {
    Profile::named(v).ok_or_else(|| {
        let names: Vec<&str> = PROFILES.iter().map(|(p, _)| p.name).collect();
        CliError::Usage(format!(
            "unknown profile `{v}`; known: {}",
            names.join(", ")
        ))
    })
}

/// Parses a number above zero for `flag`.
fn positive<T: std::str::FromStr + Default + PartialOrd>(
    flag: &str,
    v: &str,
) -> Result<T, CliError> {
    v.parse()
        .ok()
        .filter(|n| *n > T::default())
        .ok_or_else(|| CliError::Usage(format!("`{flag}` needs a positive number, not `{v}`")))
}

fn vectorize(args: &Args) -> Result<ExitCode, CliError> {
    let failed = |e: plank_tools::vectorize::VectorizeError| CliError::Failed(e.to_string());
    check_outputs(args)?;

    // Everything that can fail fast is checked before the model loads, since
    // that alone takes minutes on the large models.
    let model = Model::resolve(&args.model).map_err(failed)?;
    let store = VectorStore::new(VectorStore::default_path());
    let model_key = model.key();
    if let (Some(name), false) = (&args.name, args.force)
        && store.contains(&model_key, name).map_err(failed)?
    {
        return Err(format!(
            "{}: `{model_key}` already has a vector named `{name}`; pass --force to replace it",
            store.path().display()
        )
        .into());
    }
    let profile = match args.profile {
        Some(p) => p,
        None => model.profile().map_err(failed)?,
    };
    if let Some(edited) = &args.diff {
        return run_diff(args, &model, profile, edited, &store);
    }
    let capture = Capture::new(model.path(), profile)
        .ctx(args.ctx)
        .system(args.system.clone())
        .think(args.think)
        .component(args.component);
    let mut capture = match &args.metal {
        Some(dir) => capture.metal_dir(dir).map_err(failed)?,
        None => capture,
    };
    let say = |line: String| {
        if !args.quiet {
            eprintln!("{line}");
        }
    };
    say(format!(
        "model  : {} ({}, {})\nkey    : {}\nshape  : {} layers x {}  component: {}\nmetal  : {}",
        model.name(),
        model.path().display(),
        model.architecture().unwrap_or("no architecture key"),
        model.engine().map_or_else(
            || format!("{} (no plank engine uses this file)", model.key()),
            |e| format!("{e} (plank engine)"),
        ),
        profile.layers,
        profile.width,
        args.component,
        capture.metal_source_dir().display(),
    ));
    capture.preflight().map_err(failed)?;
    let column = args.column.as_deref();
    let cache = if args.no_cache {
        HubCache::disabled()
    } else {
        HubCache::default().refresh(args.refresh)
    };
    let (to, from) = load_prompts(args, column, &cache)?;
    let pairs = to.len().min(from.len());
    say(format!(
        "pairs  : {pairs} ({} to, {} from), two captures each",
        to.len(),
        from.len()
    ));
    if let Some(parent) = args
        .out
        .as_deref()
        .and_then(Path::parent)
        .filter(|p| !p.as_os_str().is_empty())
    {
        std::fs::create_dir_all(parent).map_err(|e| format!("{}: {e}", parent.display()))?;
    }

    let started = Instant::now();
    say("loading the model...".to_string());
    capture.load().map_err(failed)?;
    say(format!(
        "loaded in {}",
        human(started.elapsed().as_secs_f64())
    ));
    let acc = capture_pairs(
        &mut capture,
        profile,
        &to,
        &from,
        args.winsorize,
        args.quiet,
    )
    .map_err(failed)?;

    let direction = acc.finish(args.orthogonalize, args.pair_normalize);
    let took = human(started.elapsed().as_secs_f64());
    save(args, &direction, &model, &store, pairs, &took)?;
    Ok(ExitCode::SUCCESS)
}

/// Writes the vector wherever it was asked for and says how to use it.
///
/// A store write is the last step, after captures that can take hours, so if
/// it fails and no `-o` file was asked for, the vector is rescued to
/// `./<name>.f32` rather than lost.
fn save(
    args: &Args,
    direction: &Direction,
    model: &Model,
    store: &VectorStore,
    pairs: usize,
    took: &str,
) -> Result<(), CliError> {
    let failed = |e: plank_tools::vectorize::VectorizeError| CliError::Failed(e.to_string());
    let profile = direction.profile();
    let shape = format!(
        "{} x {} f32, {pairs} pair(s), {took}",
        profile.layers, profile.width
    );
    if let Some(out) = &args.out {
        direction.write_f32(out).map_err(failed)?;
        let meta = meta_path(out);
        write_meta(args, model, profile, pairs, &meta)?;
        println!(
            "wrote {} ({shape})\nwrote {}",
            out.display(),
            meta.display()
        );
    }
    if let Some(name) = &args.name {
        let model_key = model.key();
        match store.put(&model_key, name, direction, args.force) {
            Ok(replaced) => println!(
                "{} vector `{name}` for `{model_key}` in {} ({shape})",
                if replaced { "replaced" } else { "stored" },
                store.path().display()
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
    // A residual direction is applied where heretic ablates against it, to
    // each layer's FFN output.
    let scale = match args.component {
        Component::Residual | Component::FfnOut => "ffn",
        Component::AttnOut => "attn",
    };
    if let Some(name) = &args.name {
        println!(
            "\nuse it in plank (positive = towards `to`):\n  plank --dir-steering {name} --dir-steering-{scale} 1\n  \
             or in engines.local.json: \"steering\": {{ \"direction\": \"{name}\", \"{scale}\": 1 }}"
        );
    } else {
        // plank loads directions by name from vectors.json only.
        println!(
            "\nplank loads directions by name: rerun with -n NAME to store this one in vectors.json"
        );
    }
    Ok(())
}

/// Writes the provenance sidecar next to the vector.
fn write_meta(
    args: &Args,
    model: &Model,
    profile: Profile,
    pairs: usize,
    path: &Path,
) -> Result<(), CliError> {
    let meta = serde_json::json!({
        "format": "ds4-directional-steering-v1",
        "shape": [profile.layers, profile.width],
        "profile": profile.name,
        "component": args.component.as_str(),
        "dump": args.component.dump(profile).0,
        "hc_branches_averaged": args.component.dump(profile).1,
        "winsorization_quantile": args.winsorize,
        "method": "heretic: normalize(mean(from) - mean(to)), projected off normalize(mean(to))",
        "thinking": args.think,
        "pair_normalize": args.pair_normalize,
        "orthogonalize_control_mean": args.orthogonalize,
        "control": "to",
        "pairs": pairs,
        "to": args.to.iter().map(ToString::to_string).collect::<Vec<_>>(),
        "from": args.from.iter().map(ToString::to_string).collect::<Vec<_>>(),
        "model": args.model,
        "model_path": model.path(),
        "system": args.system,
        "ctx": args.ctx,
        "direction": "from - to, orthogonal to the `to` mean",
        "note": "positive scale moves towards `to`, negative towards `from`",
    });
    let text = serde_json::to_string_pretty(&meta).map_err(|e| e.to_string())? + "\n";
    std::fs::write(path, text).map_err(|e| format!("{}: {e}", path.display()))?;
    Ok(())
}

/// Captures every `to`/`from` pair, reporting progress on stderr.
fn capture_pairs(
    capture: &mut Capture,
    profile: Profile,
    to: &[String],
    from: &[String],
    winsorize: Option<f64>,
    quiet: bool,
) -> Result<Accumulator, plank_tools::vectorize::VectorizeError> {
    let mut progress = Progress::new(to.len().min(from.len()), quiet);
    let mut acc = Accumulator::new(profile);
    let clamp = |rows: &mut Vec<f32>| {
        if let Some(q) = winsorize {
            plank_tools::vectorize::winsorize(rows, profile.width, q);
        }
    };
    for (t, f) in to.iter().zip(from) {
        let mut t_rows = progress.capture(capture, t, "to")?;
        let mut f_rows = progress.capture(capture, f, "from")?;
        clamp(&mut t_rows);
        clamp(&mut f_rows);
        // `from` is the target and `to` the control: the vector is
        // `from - to`, orthogonal to the `to` mean, so a positive scale strips
        // the `from` component and moves the model towards `to`.
        acc.add_pair(&f_rows, &t_rows);
    }
    progress.finish();
    Ok(acc)
}

/// The progress display for a run: one live line on a terminal, one line
/// per finished capture otherwise.
#[derive(Debug)]
struct Progress {
    pairs: usize,
    done: usize,
    spent: Duration,
    quiet: bool,
    live: bool,
    cols: usize,
}

impl Progress {
    fn new(pairs: usize, quiet: bool) -> Self {
        Self {
            pairs,
            done: 0,
            spent: Duration::ZERO,
            quiet,
            live: std::io::stderr().is_terminal(),
            cols: terminal_cols(),
        }
    }

    /// Runs one capture, redrawing the live line on every tick.
    fn capture(
        &mut self,
        capture: &mut Capture,
        prompt: &str,
        side: &str,
    ) -> Result<Vec<f32>, plank_tools::vectorize::VectorizeError> {
        let started = Instant::now();
        let result = capture.activations_with(prompt, &mut |tick: Tick<'_>| {
            if !self.quiet && self.live {
                let line = self.line(side, tick.elapsed, tick.ds4_says);
                eprint!("\r\x1b[K{line}");
            }
        });
        if result.is_err() {
            self.clear();
            return result;
        }
        let took = started.elapsed();
        let pair = self.done / 2 + 1;
        self.done += 1;
        self.spent += took;
        if !self.quiet && !self.live {
            let total = self.pairs * 2;
            #[allow(clippy::cast_precision_loss, reason = "capture counts are small")]
            let (fraction, left) = (
                self.done as f64 / total as f64,
                self.spent.as_secs_f64() / self.done as f64 * (total - self.done) as f64,
            );
            eprintln!(
                "pair {pair}/{} {side:<4} {} {:>3.0}%  took {}  ETA ~{}",
                self.pairs,
                bar(fraction, 20),
                fraction * 100.0,
                human(took.as_secs_f64()),
                human(left),
            );
        }
        result
    }

    fn clear(&self) {
        if !self.quiet && self.live {
            eprint!("\r\x1b[K");
        }
    }

    fn finish(&self) {
        self.clear();
    }

    /// `pair 3/20 to    ▕████░░░░▏  12%  0m 41s  ETA ~9m 30s  ds4: …`
    fn line(&self, side: &str, elapsed: Duration, ds4_says: &str) -> String {
        let total = self.pairs * 2;
        #[allow(clippy::cast_precision_loss, reason = "capture counts are small")]
        let (fraction, eta) = if self.done == 0 {
            (0.0, None)
        } else {
            let avg = self.spent.as_secs_f64() / self.done as f64;
            let current = (elapsed.as_secs_f64() / avg).min(0.95);
            let left = avg * (total - self.done) as f64 - elapsed.as_secs_f64();
            (
                (self.done as f64 + current) / total as f64,
                Some(left.max(0.0)),
            )
        };
        let pair = self.done / 2 + 1;
        let head = format!(
            "pair {pair}/{} {side:<4} {} {:>3.0}%  {}  {}",
            self.pairs,
            bar(fraction, 20),
            fraction * 100.0,
            human(elapsed.as_secs_f64()),
            eta.map_or_else(
                || "ETA after the first prompt".to_string(),
                |s| format!("ETA ~{}", human(s))
            ),
        );
        // ds4 prefixes most of its own messages with `ds4: ` already.
        let says = ds4_says.strip_prefix("ds4: ").unwrap_or(ds4_says);
        let line = if says.is_empty() {
            head
        } else {
            format!("{head}  ds4: {says}")
        };
        fit(&line, self.cols.saturating_sub(1))
    }
}

/// Sizes both sides, then loads each up to the smaller size (or `--limit`).
///
/// Prompts past the shorter set are never paired, so sizing first (one row
/// per Hugging Face split) keeps a large dataset from being downloaded only
/// to be cut.
fn load_prompts(
    args: &Args,
    column: Option<&str>,
    cache: &HubCache,
) -> Result<(Vec<String>, Vec<String>), CliError> {
    let failed = |e: plank_tools::vectorize::VectorizeError| CliError::Failed(e.to_string());
    let to_size = side_size(&args.to, column, cache, args.quiet).map_err(failed)?;
    let from_size = side_size(&args.from, column, cache, args.quiet).map_err(failed)?;
    let want = [args.limit, to_size, from_size].into_iter().flatten().min();
    if !args.quiet {
        let shown = |n: Option<usize>| n.map_or_else(|| "?".to_string(), |n| n.to_string());
        eprintln!(
            "sizes  : to {}, from {}{} -> {} prompt(s) each",
            shown(to_size),
            shown(from_size),
            args.limit
                .map_or_else(String::new, |l| format!(", --limit {l}")),
            shown(want),
        );
    }
    let to = load_all("to", &args.to, column, want, cache, args.quiet).map_err(failed)?;
    let from = load_all("from", &args.from, column, want, cache, args.quiet).map_err(failed)?;
    Ok((to, from))
}

/// The total size of one side's sources, or `None` if any is unknown.
///
/// Rate-limit waits are reported, so a slow sizing step is never silent.
fn side_size(
    sources: &[PromptSource],
    column: Option<&str>,
    cache: &HubCache,
    quiet: bool,
) -> Result<Option<usize>, plank_tools::vectorize::VectorizeError> {
    let mut total = Some(0usize);
    for source in sources {
        let size = source.size(column, cache, &mut |fetch| {
            if let Fetch::Waiting { wait, reason, .. } = fetch
                && !quiet
            {
                eprintln!(
                    "sizing {source}: {reason}, retrying in {}",
                    human(wait.as_secs_f64())
                );
            }
        })?;
        total = total.zip(size).map(|(a, b)| a + b);
    }
    Ok(total)
}

/// Loads and concatenates every source, keeping at most `limit` prompts.
///
/// On a terminal the line for each source is redrawn as pages arrive or a
/// rate limit is waited out; otherwise only waits and the result are logged.
fn load_all(
    side: &str,
    sources: &[PromptSource],
    column: Option<&str>,
    limit: Option<usize>,
    cache: &HubCache,
    quiet: bool,
) -> Result<Vec<String>, plank_tools::vectorize::VectorizeError> {
    let live = !quiet && std::io::stderr().is_terminal();
    let cols = terminal_cols();
    let mut all = Vec::new();
    for source in sources {
        let left = limit.map(|l| l - all.len());
        if left == Some(0) {
            break;
        }
        let what = if matches!(source, PromptSource::Hub { .. }) {
            "fetching"
        } else {
            "reading"
        };
        let label = format!("{side:<4} {what} {source}");
        if live {
            eprint!("{}", fit(&format!("{label} …"), cols - 1));
        }
        let started = Instant::now();
        let (mut cached, mut fetched) = (false, false);
        let mut report = |fetch: Fetch| {
            let status = match fetch {
                Fetch::Cached { rows } => {
                    cached = true;
                    format!("{rows} row(s) in the cache")
                }
                Fetch::Rows { have, total } => {
                    fetched = true;
                    total.map_or_else(|| format!("{have} rows"), |t| format!("{have}/{t} rows"))
                }
                Fetch::Waiting {
                    wait,
                    attempt,
                    attempts,
                    reason,
                } => {
                    let message = format!(
                        "{reason}, retrying in {} (retry {attempt}/{})",
                        human(wait.as_secs_f64()),
                        attempts - 1
                    );
                    if !live && !quiet {
                        eprintln!("{label}: {message}");
                    }
                    message
                }
                _ => return,
            };
            if live {
                eprint!("\r\x1b[K{}", fit(&format!("{label} … {status}"), cols - 1));
            }
        };
        let result = source.load_with(column, left, cache, &mut report);
        if live {
            eprint!("\r\x1b[K");
        }
        let prompts = result.inspect_err(|_| {
            if !quiet {
                eprintln!("{label} … failed");
            }
        })?;
        if !quiet {
            let origin = if cached && !fetched {
                ", from the cache"
            } else {
                ""
            };
            eprintln!(
                "{label} … {} prompt(s) in {}{origin}",
                prompts.len(),
                human(started.elapsed().as_secs_f64())
            );
        }
        all.extend(prompts);
    }
    Ok(all)
}

/// A `width`-cell bar filled to `fraction`.
fn bar(fraction: f64, width: usize) -> String {
    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        clippy::cast_precision_loss,
        reason = "a small, clamped cell count"
    )]
    let filled = ((fraction.clamp(0.0, 1.0) * width as f64).round() as usize).min(width);
    format!("▕{}{}▏", "█".repeat(filled), "░".repeat(width - filled))
}

/// Cuts `line` to at most `cols` characters, marking the cut with `…`.
fn fit(line: &str, cols: usize) -> String {
    if line.chars().count() <= cols {
        return line.to_string();
    }
    let mut out: String = line.chars().take(cols.saturating_sub(1)).collect();
    out.push('…');
    out
}

/// The terminal width from `COLUMNS` or `stty size`, else 100.
fn terminal_cols() -> usize {
    let from_env = std::env::var("COLUMNS").ok().and_then(|c| c.parse().ok());
    from_env
        .or_else(|| {
            let tty = std::fs::File::open("/dev/tty").ok()?;
            let out = std::process::Command::new("stty")
                .arg("size")
                .stdin(tty)
                .stderr(std::process::Stdio::null())
                .output()
                .ok()?;
            String::from_utf8_lossy(&out.stdout)
                .split_whitespace()
                .nth(1)?
                .parse()
                .ok()
        })
        .filter(|&c: &usize| c >= 40)
        .unwrap_or(100)
}

/// `FILE.json` beside the vector, or `FILE.meta.json` if FILE is a `.json`.
pub(super) fn meta_path(out: &Path) -> PathBuf {
    let candidate = out.with_extension("json");
    if candidate == out {
        out.with_extension("meta.json")
    } else {
        candidate
    }
}

/// Seconds as `1h 02m`, `3m 05s` or `12s`.
pub(super) fn human(secs: f64) -> String {
    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "durations are small and non-negative"
    )]
    let s = secs.max(0.0).round() as u64;
    match (s / 3600, s / 60 % 60, s % 60) {
        (0, 0, s) => format!("{s}s"),
        (0, m, s) => format!("{m}m {s:02}s"),
        (h, m, _) => format!("{h}h {m:02}m"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn strings(args: &[&str]) -> Vec<String> {
        args.iter().map(ToString::to_string).collect()
    }

    #[test]
    fn sources_repeat_and_defaults_apply() {
        let a = parse_args(&strings(&[
            "qwen",
            "--to",
            "a.txt",
            "--to",
            "o/n:train",
            "--from",
            "b.txt",
            "-o",
            "v.f32",
        ]))
        .unwrap();
        assert_eq!(a.model, "qwen");
        assert_eq!(a.to.len(), 2);
        assert!(matches!(a.to[1], PromptSource::Hub { .. }));
        assert_eq!((a.ctx, a.component), (512, Component::Residual));
        assert!(a.orthogonalize && !a.pair_normalize);
    }

    #[test]
    fn missing_pieces_are_usage_errors() {
        for args in [
            vec!["--to", "a", "--from", "b", "-o", "v"],
            vec!["m", "--to", "a", "-o", "v"],
            vec!["m", "--to", "a", "--from", "b"],
            vec![
                "m",
                "--to",
                "a",
                "--from",
                "b",
                "-o",
                "v",
                "--profile",
                "llama",
            ],
        ] {
            assert!(
                matches!(parse_args(&strings(&args)), Err(CliError::Usage(_))),
                "{args:?}"
            );
        }
    }

    #[test]
    fn a_name_alone_is_enough_and_limit_moved_to_l() {
        let a = parse_args(&strings(&[
            "m", "--to", "a", "--from", "b", "-n", "succinct", "-l", "20",
        ]))
        .unwrap();
        assert_eq!(a.name.as_deref(), Some("succinct"));
        assert_eq!((a.out, a.limit), (None, Some(20)));

        let both = parse_args(&strings(&[
            "m", "--to", "a", "--from", "b", "--name", "x", "-o", "x.f32",
        ]))
        .unwrap();
        assert!(both.name.is_some() && both.out.is_some());

        for bad in [
            vec!["-n", " "],
            vec!["-n", "x", "-l", "0"],
            vec!["-n", "x", "-n"],
        ] {
            let mut args = vec!["m", "--to", "a", "--from", "b"];
            args.extend(bad);
            assert!(
                matches!(parse_args(&strings(&args)), Err(CliError::Usage(_))),
                "{args:?}"
            );
        }
    }

    #[test]
    fn metadata_sits_beside_the_vector() {
        assert_eq!(meta_path(Path::new("out/v.f32")), Path::new("out/v.json"));
        assert_eq!(meta_path(Path::new("v")), Path::new("v.json"));
        assert_eq!(meta_path(Path::new("v.json")), Path::new("v.meta.json"));
    }

    #[test]
    fn the_bar_fills_proportionally_and_clamps() {
        assert_eq!(bar(0.0, 4), "▕░░░░▏");
        assert_eq!(bar(0.5, 4), "▕██░░▏");
        assert_eq!(bar(2.0, 4), "▕████▏");
    }

    #[test]
    fn long_lines_are_cut_with_an_ellipsis() {
        assert_eq!(fit("short", 10), "short");
        assert_eq!(fit("ds4: loading █ tensors", 8), "ds4: lo…");
    }

    #[test]
    fn the_live_line_waits_for_a_first_timing_then_estimates() {
        let mut p = Progress {
            pairs: 2,
            done: 0,
            spent: Duration::ZERO,
            quiet: false,
            live: true,
            cols: 200,
        };
        let first = p.line("to", Duration::from_secs(3), "ds4: loading");
        assert!(first.starts_with("pair 1/2 to "), "{first}");
        assert!(first.contains("ETA after the first prompt"), "{first}");
        assert!(first.ends_with("ds4: loading"), "{first}");

        // One 10s capture done, 2s into the second: 3 left at 10s, minus 2s.
        p.done = 1;
        p.spent = Duration::from_secs(10);
        let next = p.line("from", Duration::from_secs(2), "");
        assert!(next.starts_with("pair 1/2 from "), "{next}");
        assert!(next.contains(" 30%"), "{next}");
        assert!(next.contains("ETA ~28s"), "{next}");
    }

    #[test]
    fn durations_read_naturally() {
        assert_eq!(human(7.4), "7s");
        assert_eq!(human(185.0), "3m 05s");
        assert_eq!(human(3720.0), "1h 02m");
    }
}
