//! Headless control of LM Studio via its `lms` CLI — no need to open the app
//! (port of lmstudioControl.ts).
//!
//! The CLI talks to LM Studio's local core directly, so it works even when the
//! HTTP server requires an API key (listing/loading don't need the token; only
//! inference over /v1 does). It is used to: enumerate installed models, start
//! the server headless, estimate a model's memory cost, and SAFE-LOAD big
//! models (>~20B) with capped context + partial GPU offload so they don't
//! OOM-crash the machine. Inference itself still goes through the
//! OpenAI-compatible /v1 server ([`crate::LmStudioProvider`]).
//!
//! Every child runs with stdin null (it can never steal the keyboard), is
//! killed when its timeout expires, and — for the long operations — when the
//! caller's cancel token fires.

use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::sync::{LazyLock, Mutex};
use std::time::{Duration, Instant};

use qo_core::{CancellationToken, ProviderError};
use regex::Regex;
use serde_json::Value;

use crate::http;

/// An installed LLM as `lms ls --json` reports it.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct LmsModel {
    /// Model key usable as the /v1 "model" id and `lms load` path.
    pub key: String,
    /// Full model path ("publisher/Repo-GGUF/file.gguf") — what `lms load --exact` wants.
    pub path: Option<String>,
    pub display_name: String,
    /// Billions of parameters (best-effort).
    pub params_b: f64,
    pub params_string: String,
    pub arch: String,
    pub size_gb: f64,
    pub max_context: u64,
    pub tool_use: bool,
    pub vision: bool,
}

/// Safe-load policy (config `[lmstudio]`).
#[derive(Clone, Debug, PartialEq)]
pub struct SafeLoadOpts {
    /// Params threshold to treat a model as "big".
    pub big_params_b: f64,
    /// On-disk size threshold (GB) to treat a model as "big".
    pub big_size_gb: f64,
    /// Capped context for big models.
    pub big_context: u64,
    /// Auto-unload idle big models after this many seconds.
    pub ttl_seconds: u64,
    /// Context window to load SMALL models with (the harness's configured
    /// `model.context_tokens`). LM Studio's default is 4096, which is smaller
    /// than the agent's system prompt + tool schemas — every tool turn then
    /// fails with "n_keep >= n_ctx". 0 = leave LM Studio's default.
    pub context_tokens: u64,
    /// Reload even if the model is already loaded (e.g. after a context overflow).
    pub force: bool,
}

impl Default for SafeLoadOpts {
    /// The `[lmstudio]` config defaults.
    fn default() -> Self {
        Self { big_params_b: 20.0, big_size_gb: 14.0, big_context: 8192, ttl_seconds: 1800, context_tokens: 0, force: false }
    }
}

/// A model currently loaded (`lms ps --json`).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct LoadedModel {
    pub identifier: String,
    pub model_key: String,
    pub context_length: u64,
}

/// GPU memory snapshot (GB) from nvidia-smi.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GpuMem {
    pub total_gb: f64,
    pub free_gb: f64,
}

/// Outcome of [`LmStudioControl::safe_load`].
#[derive(Clone, Debug, Default, PartialEq)]
pub struct SafeLoadResult {
    pub ok: bool,
    /// The CLI's combined output (or "already loaded").
    pub detail: String,
    pub big: bool,
    pub reloaded: bool,
    pub skipped: bool,
}

/// Result of one `lms` run (execa `reject: false` semantics: a spawn failure
/// or timeout is reported through `exit_code: None`, never as an error).
#[derive(Clone, Debug, Default)]
struct RunOutput {
    exit_code: Option<i32>,
    stdout: String,
    /// stdout followed by stderr.
    all: String,
}

impl RunOutput {
    fn ok(&self) -> bool {
        self.exit_code == Some(0)
    }
}

/// Resolve the lms CLI path: configured → default install → PATH → "lms".
pub fn lms_bin(configured: Option<&str>) -> String {
    if let Some(c) = configured.filter(|c| !c.is_empty() && Path::new(c).exists()) {
        return c.to_string();
    }
    if let Some(home) = dirs::home_dir() {
        for name in ["lms.exe", "lms"] {
            let p = home.join(".lmstudio").join("bin").join(name);
            if p.exists() {
                return p.to_string_lossy().into_owned();
            }
        }
    }
    match which::which("lms") {
        Ok(p) => p.to_string_lossy().into_owned(),
        Err(_) => "lms".to_string(), // hope it's on PATH
    }
}

pub struct LmStudioControl {
    bin: String,
    gpu_mem_cache: Mutex<Option<(Instant, Option<GpuMem>)>>,
}

impl LmStudioControl {
    /// `bin_path`: config `lmstudio.cli_path` (empty/None = auto-detect).
    pub fn new(bin_path: Option<&str>) -> Self {
        Self { bin: lms_bin(bin_path), gpu_mem_cache: Mutex::new(None) }
    }

    /// The resolved CLI path.
    pub fn bin(&self) -> &str {
        &self.bin
    }

    /// Is the lms CLI actually present/usable?
    pub fn installed(&self) -> bool {
        self.bin != "lms" || which::which("lms").is_ok()
    }

    async fn run(&self, args: &[&str], timeout_ms: u64, cancel: Option<&CancellationToken>) -> Result<RunOutput, ProviderError> {
        run_cmd(Path::new(&self.bin), args, timeout_ms, cancel).await
    }

    /// Best-effort GPU memory probe (nvidia-smi), cached for 10 s. Used to
    /// decide whether a model that is "small" by parameter count still won't
    /// fit next to whatever else is using the card (e.g. a 6.5 GB 9B model on
    /// a 6 GB RTX 3050 with 2.8 GB already taken by other apps).
    pub async fn gpu_mem(&self) -> Option<GpuMem> {
        if let Some((at, mem)) = *self.gpu_mem_cache.lock().unwrap_or_else(|e| e.into_inner()) {
            if at.elapsed() < Duration::from_secs(10) {
                return mem;
            }
        }
        let out = run_cmd(
            Path::new("nvidia-smi"),
            &["--query-gpu=memory.total,memory.used", "--format=csv,noheader,nounits"],
            5_000,
            None,
        )
        .await
        .unwrap_or_default();
        // No nvidia-smi → fall back to the configured size thresholds.
        let mem = if out.ok() { parse_gpu_mem(&out.stdout) } else { None };
        *self.gpu_mem_cache.lock().unwrap_or_else(|e| e.into_inner()) = Some((Instant::now(), mem));
        mem
    }

    /// Is this model "big" for THIS machine: over the configured thresholds, or
    /// larger than the GPU memory currently free (so it would spill to CPU anyway)?
    pub fn is_big(&self, model: &LmsModel, opts: &SafeLoadOpts, gpu: Option<GpuMem>) -> bool {
        is_big(model, opts, gpu)
    }

    /// Context length a model should be loaded with under these options (0 = default).
    pub fn context_for(&self, model: &LmsModel, opts: &SafeLoadOpts, gpu: Option<GpuMem>) -> u64 {
        context_for(model, opts, gpu)
    }

    /// All installed LLMs (embeddings excluded), parsed from `lms ls --json`.
    pub async fn list(&self) -> Vec<LmsModel> {
        match self.run(&["ls", "--json"], 30_000, None).await {
            Ok(r) => parse_ls_json(&r.stdout),
            Err(_) => Vec::new(),
        }
    }

    /// Identifiers of currently loaded models (`lms ps`).
    pub async fn loaded(&self) -> Vec<String> {
        self.loaded_detail()
            .await
            .into_iter()
            .map(|m| if m.identifier.is_empty() { m.model_key } else { m.identifier })
            .filter(|s| !s.is_empty())
            .collect()
    }

    /// Loaded models with the context length each was loaded with (`lms ps --json`).
    pub async fn loaded_detail(&self) -> Vec<LoadedModel> {
        match self.run(&["ps", "--json"], 20_000, None).await {
            Ok(r) => parse_ps_json(&r.stdout),
            Err(_) => Vec::new(),
        }
    }

    /// Ensure the local server is up. Fast path: an HTTP probe (≈100 ms) — if
    /// it already answers (200 or 401 "needs key"), skip the slow
    /// `lms server start` (which spawns the CLI and can take ~10 s). Only
    /// starts it when truly down. `base_url` is `[lmstudio].base_url` (empty =
    /// `http://localhost:<port>`).
    ///
    /// Bounded (1.5 s probe + 30 s CLI); dropping the future (e.g. aborting the
    /// startup task) kills the `lms` child.
    pub async fn ensure_server(&self, port: u16, bind: &str, base_url: &str) -> bool {
        let root = server_root(port, Some(base_url));
        let client = http::build_client();
        let req = client.get(format!("{root}/v1/models"));
        if let Ok(res) = http::send("lmstudio", req, None, Some(1_500)).await {
            if matches!(res.status().as_u16(), 200 | 401) {
                return true; // already running
            }
        }
        // Not up — start it.
        let port = port.to_string();
        match self.run(&["server", "start", "--port", &port, "--bind", bind], 30_000, None).await {
            Ok(r) => r.ok(),
            Err(_) => false,
        }
    }

    /// Estimated memory (GB) to load a model, without loading it.
    pub async fn estimate_gb(&self, key: &str, cancel: Option<&CancellationToken>) -> anyhow::Result<Option<f64>> {
        let r = self.run(&["load", key, "--estimate-only", "--yes"], 60_000, cancel).await?;
        Ok(parse_estimate(&r.all))
    }

    /// Load a model with crash-safe parameters. Big models (≥ thresholds) get
    /// a capped context and partial GPU offload (the rest runs on CPU/RAM) plus
    /// a TTL, so they run slowly instead of exhausting VRAM/RAM and crashing
    /// the PC. Smaller models load with LM Studio's automatic settings (and the
    /// harness's context window when configured).
    ///
    /// Errors only with [`ProviderError::Interrupted`] (the load child is killed).
    pub async fn safe_load(
        &self,
        model: &LmsModel,
        opts: &SafeLoadOpts,
        cancel: Option<&CancellationToken>,
    ) -> anyhow::Result<SafeLoadResult> {
        let gpu = self.gpu_mem().await;
        let big = is_big(model, opts, gpu);
        let ctx = context_for(model, opts, gpu);

        // Already loaded? `lms load` refuses a duplicate identifier, so decide
        // here: keep it when its context is big enough, otherwise unload and reload.
        let mut reloaded = false;
        let loaded = self.loaded_detail().await;
        if let Some(current) = loaded.iter().find(|m| m.identifier == model.key || m.model_key == model.key) {
            let enough = ctx == 0 || current.context_length >= ctx;
            if enough && !opts.force {
                return Ok(SafeLoadResult { ok: true, detail: "already loaded".into(), big, reloaded, skipped: true });
            }
            let id = if current.identifier.is_empty() { model.key.as_str() } else { current.identifier.as_str() };
            self.unload(id).await;
            reloaded = true;
        }

        let extra = load_args(model, opts, gpu);
        let extra: Vec<&str> = extra.iter().map(String::as_str).collect();
        // Prefer an exact match on the full path: the short key "qwen3.5-0.8b"
        // would otherwise fuzzy-match "qwen3.5-0.8b-toolcalling" too ("2 models
        // match…"). Fall back to the key if an older CLI rejects the path form.
        let mut r = RunOutput { exit_code: Some(1), ..Default::default() };
        if let Some(path) = model.path.as_deref() {
            let mut args = vec!["load", path, "--exact"];
            args.extend(&extra);
            r = self.run(&args, 600_000, cancel).await?;
        }
        if !r.ok() {
            let mut args = vec!["load", model.key.as_str()];
            args.extend(&extra);
            r = self.run(&args, 600_000, cancel).await?;
        }
        Ok(SafeLoadResult { ok: r.ok(), detail: r.all.trim().to_string(), big, reloaded, skipped: false })
    }

    pub async fn unload(&self, key: &str) {
        let _ = self.run(&["unload", key], 20_000, None).await;
    }
}

/// Run `bin args…` with stdin null, a timeout, and an optional cancel token.
/// The child is killed when the timeout expires or the token fires (the
/// future owning it is dropped with `kill_on_drop`).
async fn run_cmd(
    bin: &Path,
    args: &[&str],
    timeout_ms: u64,
    cancel: Option<&CancellationToken>,
) -> Result<RunOutput, ProviderError> {
    if cancel.is_some_and(|c| c.is_cancelled()) {
        return Err(ProviderError::Interrupted);
    }
    let child = tokio::process::Command::new(PathBuf::from(bin))
        .args(args)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .kill_on_drop(true)
        .spawn();
    let Ok(child) = child else {
        return Ok(RunOutput::default()); // not installed
    };
    tokio::select! {
        biased;
        _ = http::cancelled(cancel) => Err(ProviderError::Interrupted),
        _ = tokio::time::sleep(Duration::from_millis(timeout_ms)) => Ok(RunOutput::default()),
        out = child.wait_with_output() => Ok(match out {
            Ok(o) => {
                let stdout = String::from_utf8_lossy(&o.stdout).into_owned();
                let stderr = String::from_utf8_lossy(&o.stderr);
                let all = if stderr.is_empty() { stdout.clone() } else { format!("{stdout}{stderr}") };
                RunOutput { exit_code: o.status.code(), stdout, all }
            }
            Err(_) => RunOutput::default(),
        }),
    }
}

/// `http://localhost:1234/v1/` → `http://localhost:1234` (or the port default).
fn server_root(port: u16, base_url: Option<&str>) -> String {
    let base = base_url.filter(|b| !b.is_empty()).map(str::to_string).unwrap_or_else(|| format!("http://localhost:{port}"));
    let b = base.trim_end_matches('/');
    b.strip_suffix("/v1").unwrap_or(b).to_string()
}

/// Over the thresholds, or larger than (90 % of) the GPU memory currently free.
pub fn is_big(model: &LmsModel, opts: &SafeLoadOpts, gpu: Option<GpuMem>) -> bool {
    if model.params_b >= opts.big_params_b || model.size_gb >= opts.big_size_gb {
        return true;
    }
    gpu.is_some_and(|g| model.size_gb > g.free_gb * 0.9)
}

/// Capped context for big models, the configured window for small ones,
/// never above the model's own maximum (0 = LM Studio's default).
pub fn context_for(model: &LmsModel, opts: &SafeLoadOpts, gpu: Option<GpuMem>) -> u64 {
    let want = if is_big(model, opts, gpu) { opts.big_context } else { opts.context_tokens };
    if want == 0 {
        return 0;
    }
    if model.max_context > 0 {
        want.min(model.max_context)
    } else {
        want
    }
}

/// The `lms load` flags after the model path/key.
pub fn load_args(model: &LmsModel, opts: &SafeLoadOpts, gpu: Option<GpuMem>) -> Vec<String> {
    let big = is_big(model, opts, gpu);
    let ctx = context_for(model, opts, gpu);
    let mut extra = vec!["--yes".to_string(), "--identifier".to_string(), model.key.clone()];
    if ctx > 0 {
        extra.push("-c".into());
        extra.push(ctx.to_string());
    }
    if big {
        extra.push("--ttl".into());
        extra.push(opts.ttl_seconds.to_string());
        // Partial GPU offload so the overflow lands on system RAM rather than
        // OOM-ing the GPU: sized from the memory actually free when known
        // (leave ~25% of it for the KV cache), else the old fixed rule.
        match gpu {
            Some(g) if model.size_gb > g.free_gb * 0.9 => {
                let ratio = ((g.free_gb * 0.75) / model.size_gb).clamp(0.1, 0.9);
                extra.push("--gpu".into());
                extra.push(format!("{ratio:.2}"));
            }
            _ if model.size_gb >= 18.0 => {
                extra.push("--gpu".into());
                extra.push("0.5".into());
            }
            _ => {}
        }
    }
    extra
}

static PARAMS_MULT: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?i)([\d.]+)\s*x\s*([\d.]+)\s*B").expect("valid regex"));
static PARAMS: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"(?i)([\d.]+)\s*B").expect("valid regex"));
static ESTIMATE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?i)Estimated Total Memory:\s*([\d.]+)\s*GB").expect("valid regex"));

/// JS `parseFloat` on a `[\d.]+` match ("1.2.3" → 1.2).
fn parse_float(s: &str) -> f64 {
    let mut end = 0;
    let mut dot = false;
    for (i, c) in s.char_indices() {
        if c == '.' {
            if dot {
                break;
            }
            dot = true;
        }
        end = i + c.len_utf8();
    }
    s[..end].parse().unwrap_or(0.0)
}

/// "35B-A3B" → 35, "7.5B" → 7.5, "64x2.6B" → 64×2.6, "20B" → 20.
pub fn parse_params(s: &str) -> f64 {
    if s.is_empty() {
        return 0.0;
    }
    if let Some(c) = PARAMS_MULT.captures(s) {
        return parse_float(&c[1]) * parse_float(&c[2]);
    }
    PARAMS.captures(s).map(|c| parse_float(&c[1])).unwrap_or(0.0)
}

/// `lms ls --json` stdout → installed LLMs.
pub fn parse_ls_json(stdout: &str) -> Vec<LmsModel> {
    let text = if stdout.trim().is_empty() { "[]" } else { stdout };
    let Ok(Value::Array(arr)) = serde_json::from_str::<Value>(text) else {
        return Vec::new();
    };
    let s = |m: &Value, k: &str| m.get(k).and_then(Value::as_str).filter(|s| !s.is_empty()).map(str::to_string);
    let truthy = |m: &Value, k: &str| m.get(k).is_some_and(crate::wire::truthy);
    arr.iter()
        .filter(|m| m.get("type").and_then(Value::as_str) == Some("llm"))
        .filter_map(|m| {
            let key = s(m, "modelKey")?;
            let params_string = s(m, "paramsString").unwrap_or_default();
            Some(LmsModel {
                path: s(m, "path").or_else(|| s(m, "indexedModelIdentifier")),
                display_name: s(m, "displayName").unwrap_or_else(|| key.clone()),
                params_b: parse_params(&params_string),
                params_string,
                arch: s(m, "architecture").unwrap_or_default(),
                size_gb: m.get("sizeBytes").and_then(Value::as_f64).unwrap_or(0.0) / 1e9,
                max_context: m.get("maxContextLength").and_then(Value::as_u64).unwrap_or(0),
                tool_use: truthy(m, "trainedForToolUse"),
                vision: truthy(m, "vision"),
                key,
            })
        })
        .collect()
}

/// `lms ps --json` stdout → loaded models.
pub fn parse_ps_json(stdout: &str) -> Vec<LoadedModel> {
    let text = if stdout.trim().is_empty() { "[]" } else { stdout };
    let Ok(Value::Array(arr)) = serde_json::from_str::<Value>(text) else {
        return Vec::new();
    };
    let s = |m: &Value, k: &str| match m.get(k) {
        Some(Value::String(s)) if !s.is_empty() => Some(s.clone()),
        Some(Value::Number(n)) => Some(n.to_string()),
        _ => None,
    };
    arr.iter()
        .map(|m| LoadedModel {
            identifier: s(m, "identifier").or_else(|| s(m, "modelKey")).unwrap_or_default(),
            model_key: s(m, "modelKey").or_else(|| s(m, "identifier")).unwrap_or_default(),
            // `Number(m.contextLength) || 0`: numbers or numeric strings.
            context_length: match m.get("contextLength") {
                Some(Value::Number(n)) => n.as_f64().filter(|f| *f > 0.0).map_or(0, |f| f as u64),
                Some(Value::String(s)) => s.trim().parse::<f64>().ok().filter(|f| *f > 0.0).map_or(0, |f| f as u64),
                _ => 0,
            },
        })
        .collect()
}

/// nvidia-smi `total, used` (MiB) → GB snapshot.
pub fn parse_gpu_mem(stdout: &str) -> Option<GpuMem> {
    let line = stdout.trim().lines().next()?;
    let mut parts = line.split(',').map(|x| x.trim().parse::<f64>().ok());
    let total = parts.next()??;
    let used = parts.next()??;
    (total > 0.0 && used >= 0.0).then(|| GpuMem { total_gb: total / 1024.0, free_gb: (total - used).max(0.0) / 1024.0 })
}

/// "Estimated Total Memory: 6.42 GB" → 6.42.
pub fn parse_estimate(all: &str) -> Option<f64> {
    ESTIMATE.captures(all).map(|c| parse_float(&c[1]))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn model(params_b: f64, size_gb: f64, max_context: u64) -> LmsModel {
        LmsModel { key: "m".into(), params_b, size_gb, max_context, ..Default::default() }
    }

    #[test]
    fn params_parsing() {
        assert_eq!(parse_params("35B-A3B"), 35.0);
        assert_eq!(parse_params("7.5B"), 7.5);
        assert!((parse_params("64x2.6B") - 166.4).abs() < 1e-9);
        assert_eq!(parse_params("20B"), 20.0);
        assert_eq!(parse_params(""), 0.0);
        assert_eq!(parse_params("unknown"), 0.0);
    }

    #[test]
    fn ls_and_ps_parsing() {
        let ls = r#"[
          {"type":"llm","modelKey":"qwen3.5-9b","path":"lmstudio-community/Qwen3.5-9B-GGUF/q4.gguf","displayName":"Qwen3.5 9B",
           "paramsString":"9B","architecture":"qwen35","sizeBytes":6500000000,"maxContextLength":262144,"trainedForToolUse":true},
          {"type":"embedding","modelKey":"nomic-embed"},
          {"type":"llm"}
        ]"#;
        let m = parse_ls_json(ls);
        assert_eq!(m.len(), 1);
        assert_eq!(m[0].key, "qwen3.5-9b");
        assert_eq!(m[0].params_b, 9.0);
        assert!((m[0].size_gb - 6.5).abs() < 1e-9);
        assert!(m[0].tool_use && !m[0].vision);
        assert_eq!(m[0].path.as_deref(), Some("lmstudio-community/Qwen3.5-9B-GGUF/q4.gguf"));
        assert!(parse_ls_json("garbage").is_empty());
        assert!(parse_ls_json("").is_empty());

        let ps = r#"[{"identifier":"qwen3.5-9b","modelKey":"qwen3.5-9b","contextLength":16384},{"modelKey":"x","contextLength":"4096"}]"#;
        let p = parse_ps_json(ps);
        assert_eq!(p[0].context_length, 16384);
        assert_eq!(p[1].identifier, "x");
        assert_eq!(p[1].context_length, 4096);
    }

    #[test]
    fn gpu_and_estimate_parsing() {
        let g = parse_gpu_mem("6144, 2867\n").expect("gpu");
        assert!((g.total_gb - 6.0).abs() < 1e-9);
        assert!((g.free_gb - 3.2001953125).abs() < 1e-9);
        assert!(parse_gpu_mem("").is_none());
        assert!(parse_gpu_mem("N/A, N/A").is_none());
        assert_eq!(parse_estimate("...\nEstimated Total Memory: 6.42 GB\n"), Some(6.42));
        assert_eq!(parse_estimate("nope"), None);
    }

    #[test]
    fn safe_load_policy() {
        let opts = SafeLoadOpts { context_tokens: 16384, ..Default::default() };
        // Small model, no GPU info: configured context, capped at the model max.
        let small = model(9.0, 6.5, 8192);
        assert!(!is_big(&small, &opts, None));
        assert_eq!(context_for(&small, &opts, None), 8192);
        assert_eq!(load_args(&small, &opts, None), ["--yes", "--identifier", "m", "-c", "8192"]);
        // Small by params but doesn't fit the free VRAM → big, ratio offload.
        let gpu = Some(GpuMem { total_gb: 6.0, free_gb: 3.2 });
        assert!(is_big(&small, &opts, gpu));
        assert_eq!(
            load_args(&small, &opts, gpu),
            ["--yes", "--identifier", "m", "-c", "8192", "--ttl", "1800", "--gpu", "0.37"]
        );
        // Big by params, no GPU info, ≥ 18 GB → fixed half offload.
        let big = model(32.0, 19.0, 0);
        assert_eq!(context_for(&big, &opts, None), 8192);
        assert_eq!(
            load_args(&big, &opts, None),
            ["--yes", "--identifier", "m", "-c", "8192", "--ttl", "1800", "--gpu", "0.5"]
        );
        // No configured context for small models → LM Studio default (no -c).
        let opts0 = SafeLoadOpts::default();
        assert_eq!(context_for(&small, &opts0, None), 0);
    }

    #[test]
    fn server_root_normalisation() {
        assert_eq!(server_root(1234, None), "http://localhost:1234");
        assert_eq!(server_root(1234, Some("http://127.0.0.1:4321/v1/")), "http://127.0.0.1:4321");
    }

    #[tokio::test]
    async fn missing_binary_is_not_an_error() {
        let ctl = LmStudioControl { bin: "definitely-not-a-real-lms-binary-xyz".into(), gpu_mem_cache: Mutex::new(None) };
        assert!(ctl.list().await.is_empty());
        assert!(ctl.loaded().await.is_empty());
        assert_eq!(ctl.estimate_gb("x", None).await.ok().flatten(), None);
    }
}
