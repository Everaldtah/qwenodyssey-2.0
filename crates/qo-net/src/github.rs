//! GitHub tools: full read/write access to the user's GitHub account via the
//! REST API. The auth token is resolved automatically so the agent inherits the
//! user's GitHub login with zero config:
//!
//!   1. explicit config token  (`github.token`)
//!   2. an environment variable (`github.token_env`, then `GITHUB_TOKEN`, then `GH_TOKEN`)
//!   3. the GitHub CLI's own token via `gh auth token` (already logged in on this box)
//!
//! With these the model can browse/read repos & files, create repos,
//! upload/commit files, open PRs, and review PRs (diff + submit reviews) — i.e.
//! read, write and review code on the user's account.

use std::process::Stdio;
use std::sync::{Arc, LazyLock, Mutex};
use std::time::Duration;

use async_trait::async_trait;
use base64::engine::{general_purpose, DecodePaddingMode, GeneralPurpose, GeneralPurposeConfig};
use base64::Engine as _;
use qo_core::tool::SharedTool;
use qo_core::{Tool, ToolContext, ToolResult};
use regex::Regex;
use serde_json::{json, Map, Value};

use crate::util::{cancellable, js_num, num_arg, num_or, str_arg, take_chars, truthy, value_to_string, INTERRUPTED};

/// Options for [`create_github_tools`] (mirrors the TS `GithubConfig` / `[github]` config).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GithubConfig {
    /// Personal access token. Leave blank to fall back to env / gh CLI.
    pub token: String,
    /// Env var to read the token from when `token` is blank.
    pub token_env: String,
    /// Default owner for owner/repo args (blank = the authenticated user).
    pub default_owner: String,
    /// REST API base (override for GitHub Enterprise).
    pub api_base: String,
}

impl Default for GithubConfig {
    fn default() -> Self {
        Self {
            token: String::new(),
            token_env: "GITHUB_TOKEN".into(),
            default_owner: String::new(),
            api_base: "https://api.github.com".into(),
        }
    }
}

const UA: &str = "qwenodyssey-github-tool";
const ACCEPT_JSON: &str = "application/vnd.github+json";
const ACCEPT_DIFF: &str = "application/vnd.github.v3.diff";
/// Generous per-request deadline (the TS had none): a hung API call must not
/// hold the turn forever. PR diffs can be large, hence the headroom.
const REQUEST_TIMEOUT: Duration = Duration::from_secs(120);
/// `gh auth token` answers instantly when logged in; never wait long for it.
const GH_CLI_TIMEOUT: Duration = Duration::from_secs(15);

type EnvLookup = Arc<dyn Fn(&str) -> Option<String> + Send + Sync>;

/// A raw API response: status, parsed JSON (`Null` for empty / non-JSON bodies
/// such as a raw diff) and the body text.
#[derive(Clone, Debug)]
pub struct GhResponse {
    pub ok: bool,
    pub status: u16,
    pub json: Value,
    pub text: String,
}

/// Authenticated REST client shared by the GitHub tools. Memoises the token
/// fetched from `gh` (so we don't shell out on every call) and the login used
/// for owner defaulting.
pub struct GithubClient {
    cfg: GithubConfig,
    http: reqwest::Client,
    env: EnvLookup,
    /// Program used for the CLI fallback (`gh`); overridable for tests.
    gh_program: String,
    cached_token: Mutex<Option<String>>,
    login: Mutex<Option<String>>,
}

impl GithubClient {
    pub fn new(cfg: GithubConfig) -> Self {
        Self::with_env(cfg, Arc::new(|k: &str| std::env::var(k).ok()), "gh")
    }

    /// Construct with an injected environment lookup and `gh` program (tests).
    pub(crate) fn with_env(cfg: GithubConfig, env: EnvLookup, gh_program: &str) -> Self {
        let http = reqwest::Client::builder()
            .connect_timeout(Duration::from_secs(15))
            .timeout(REQUEST_TIMEOUT)
            .build()
            .unwrap_or_default();
        Self {
            cfg,
            http,
            env,
            gh_program: gh_program.to_string(),
            cached_token: Mutex::new(None),
            login: Mutex::new(None),
        }
    }

    pub fn config(&self) -> &GithubConfig {
        &self.cfg
    }

    /// Resolve the token: config → env (`token_env`, `GITHUB_TOKEN`, `GH_TOKEN`)
    /// → memoised `gh auth token`.
    pub async fn resolve_token(&self) -> anyhow::Result<String> {
        if !self.cfg.token.is_empty() {
            return Ok(self.cfg.token.clone());
        }
        let env_name = if self.cfg.token_env.is_empty() { "GITHUB_TOKEN" } else { self.cfg.token_env.as_str() };
        for name in [env_name, "GITHUB_TOKEN", "GH_TOKEN"] {
            if let Some(v) = (self.env)(name).filter(|v| !v.is_empty()) {
                return Ok(v);
            }
        }
        if let Some(t) = self.cached_token.lock().ok().and_then(|g| g.clone()) {
            return Ok(t);
        }
        // Fall back to the GitHub CLI's stored login.
        if let Some(tok) = self.gh_cli_token().await {
            if let Ok(mut g) = self.cached_token.lock() {
                *g = Some(tok.clone());
            }
            return Ok(tok);
        }
        anyhow::bail!(
            "No GitHub token. Set github.token in config, export {}, or run `gh auth login`.",
            if self.cfg.token_env.is_empty() { "GITHUB_TOKEN" } else { self.cfg.token_env.as_str() }
        )
    }

    /// `gh auth token`, with stdin closed (it must never read the keyboard) and
    /// a short deadline. `None` when gh is missing / not logged in.
    async fn gh_cli_token(&self) -> Option<String> {
        let child = tokio::process::Command::new(&self.gh_program)
            .args(["auth", "token"])
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .kill_on_drop(true)
            .spawn()
            .ok()?;
        let out = tokio::time::timeout(GH_CLI_TIMEOUT, child.wait_with_output()).await.ok()?.ok()?;
        if !out.status.success() {
            return None;
        }
        let tok = String::from_utf8_lossy(&out.stdout).trim().to_string();
        (!tok.is_empty()).then_some(tok)
    }

    /// One REST call. `path` is appended to the API base unless it is already an
    /// absolute URL.
    pub async fn api(&self, method: reqwest::Method, path: &str, body: Option<&Value>, accept: &str) -> anyhow::Result<GhResponse> {
        let token = self.resolve_token().await?;
        let base = self.cfg.api_base.trim_end_matches('/');
        let base = if base.is_empty() { "https://api.github.com" } else { base };
        let url = if HTTP_URL.is_match(path) { path.to_string() } else { format!("{base}{path}") };
        let mut req = self
            .http
            .request(method, &url)
            .header(reqwest::header::AUTHORIZATION, format!("Bearer {token}"))
            .header(reqwest::header::ACCEPT, accept)
            .header("X-GitHub-Api-Version", "2022-11-28")
            .header(reqwest::header::USER_AGENT, UA);
        if let Some(b) = body {
            req = req.header(reqwest::header::CONTENT_TYPE, "application/json").body(b.to_string());
        }
        let res = req.send().await.map_err(|e| anyhow::anyhow!(describe(&e)))?;
        let status = res.status();
        let text = res.text().await.map_err(|e| anyhow::anyhow!(describe(&e)))?;
        // Non-JSON (e.g. a raw diff) leaves json null.
        let json = if text.is_empty() { Value::Null } else { serde_json::from_str(&text).unwrap_or(Value::Null) };
        Ok(GhResponse { ok: status.is_success(), status: status.as_u16(), json, text })
    }

    async fn get(&self, path: &str) -> anyhow::Result<GhResponse> {
        self.api(reqwest::Method::GET, path, None, ACCEPT_JSON).await
    }

    /// The authenticated login (cached) — used to default the repo owner.
    pub async fn whoami_login(&self) -> anyhow::Result<String> {
        if let Some(l) = self.login.lock().ok().and_then(|g| g.clone()) {
            return Ok(l);
        }
        let r = self.get("/user").await?;
        if !r.ok {
            let msg = r.json.get("message").filter(|m| truthy(Some(m))).map(value_to_string);
            anyhow::bail!(msg.unwrap_or_else(|| format!("HTTP {}", r.status)));
        }
        let login = r.json.get("login").map(value_to_string).unwrap_or_default();
        self.remember_login(&login);
        Ok(login)
    }

    fn remember_login(&self, login: &str) {
        if let Ok(mut g) = self.login.lock() {
            *g = Some(login.to_string());
        }
    }

    /// Split an "owner/repo" or bare "repo" arg, defaulting owner to the
    /// configured default or the authenticated user.
    pub async fn split_repo(&self, repo_arg: &str) -> anyhow::Result<(String, String)> {
        let raw = repo_arg.trim();
        if raw.contains('/') {
            let mut parts = raw.split('/');
            let owner = parts.next().unwrap_or("").to_string();
            let repo = parts.next().unwrap_or("").to_string();
            return Ok((owner, repo));
        }
        let owner = if self.cfg.default_owner.is_empty() { self.whoami_login().await? } else { self.cfg.default_owner.clone() };
        Ok((owner, raw.to_string()))
    }
}

static HTTP_URL: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"(?i)^https?://").expect("static regex"));

fn describe(e: &reqwest::Error) -> String {
    if e.is_timeout() {
        return format!("request timed out after {}ms", REQUEST_TIMEOUT.as_millis());
    }
    let mut msg = e.to_string();
    let mut src = std::error::Error::source(e);
    while let Some(s) = src {
        msg = format!("{msg}: {s}");
        src = s.source();
    }
    msg
}

fn fail(label: &str, r: &GhResponse) -> ToolResult {
    let msg = match r.json.get("message").filter(|m| truthy(Some(m))) {
        Some(m) => value_to_string(m),
        None if !r.text.is_empty() => r.text.clone(),
        None => format!("HTTP {}", r.status),
    };
    ToolResult::err(format!("{label} failed (HTTP {}): {msg}", r.status))
}

/// `String(v ?? "")`-style field read for output lines.
fn field(v: &Value, ptr: &str) -> String {
    v.pointer(ptr).map(value_to_string).unwrap_or_default()
}

/// GitHub's base64 payloads are wrapped at 60 columns; Node's decoder skips the
/// newlines and tolerates missing padding, so this one does too.
fn decode_base64(s: &str) -> String {
    const LENIENT: GeneralPurpose = GeneralPurpose::new(
        &base64::alphabet::STANDARD,
        GeneralPurposeConfig::new().with_decode_padding_mode(DecodePaddingMode::Indifferent),
    );
    let compact: String = s.chars().filter(|c| !c.is_ascii_whitespace()).collect();
    match LENIENT.decode(compact.as_bytes()) {
        Ok(bytes) => String::from_utf8_lossy(&bytes).into_owned(),
        Err(_) => String::new(),
    }
}

/// Strip leading slashes (`replace(/^\/+/, "")`).
fn repo_path(args: &Value) -> String {
    str_arg(args, &["path"]).trim_start_matches('/').to_string()
}

/* ── tools ── */

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Kind {
    Whoami,
    ListRepos,
    GetFile,
    PutFile,
    CreateRepo,
    ListPrs,
    GetPr,
    CreatePr,
    ReviewPr,
    CreateIssue,
}

const ALL: [Kind; 10] = [
    Kind::Whoami,
    Kind::ListRepos,
    Kind::GetFile,
    Kind::PutFile,
    Kind::CreateRepo,
    Kind::ListPrs,
    Kind::GetPr,
    Kind::CreatePr,
    Kind::ReviewPr,
    Kind::CreateIssue,
];

/// Build the ten `github_*` tools sharing one authenticated client.
pub fn create_github_tools(cfg: GithubConfig) -> Vec<SharedTool> {
    tools_for(Arc::new(GithubClient::new(cfg)))
}

/// Build the tools around an existing client (e.g. one the chat layer also
/// uses for its startup `github_whoami` probe).
pub fn tools_for(client: Arc<GithubClient>) -> Vec<SharedTool> {
    ALL.iter().map(|&kind| Arc::new(GithubTool { kind, client: client.clone() }) as SharedTool).collect()
}

struct GithubTool {
    kind: Kind,
    client: Arc<GithubClient>,
}

#[async_trait]
impl Tool for GithubTool {
    fn name(&self) -> &str {
        match self.kind {
            Kind::Whoami => "github_whoami",
            Kind::ListRepos => "github_list_repos",
            Kind::GetFile => "github_get_file",
            Kind::PutFile => "github_put_file",
            Kind::CreateRepo => "github_create_repo",
            Kind::ListPrs => "github_list_prs",
            Kind::GetPr => "github_get_pr",
            Kind::CreatePr => "github_create_pr",
            Kind::ReviewPr => "github_review_pr",
            Kind::CreateIssue => "github_create_issue",
        }
    }

    fn description(&self) -> &str {
        match self.kind {
            Kind::Whoami => {
                "Verify the GitHub login the agent is acting as and show its scopes. Call this first if \
                 you're unsure whether GitHub access is configured."
            }
            Kind::ListRepos => {
                "List repositories on the authenticated GitHub account (most recently pushed first). \
                 Use to discover what repos exist before reading or writing."
            }
            Kind::GetFile => {
                "Read a file (or list a directory) from a GitHub repo. Returns decoded file contents so you \
                 can review code that isn't checked out locally."
            }
            Kind::PutFile => {
                "Create or update (upload/commit) a file in a GitHub repo on a branch. Writes code directly to \
                 GitHub — fetches the existing file's sha automatically when updating. Use to push new or changed files."
            }
            Kind::CreateRepo => {
                "Create a new repository on the authenticated account. Optionally auto-initialise it with a README \
                 so you can immediately commit files to it."
            }
            Kind::ListPrs => "List pull requests in a repo (open by default). Use before reviewing.",
            Kind::GetPr => {
                "Fetch a pull request's details AND its full diff so you can review the code. Returns title, \
                 description, changed-file stats and the unified diff."
            }
            Kind::CreatePr => "Open a pull request from a head branch into a base branch.",
            Kind::ReviewPr => {
                "Submit a review on a pull request after reading it with github_get_pr. Choose an event: \
                 APPROVE, REQUEST_CHANGES, or COMMENT, with a body explaining the review."
            }
            Kind::CreateIssue => "Open an issue in a repo (e.g. to file a bug or track work found while reviewing).",
        }
    }

    fn mutating(&self) -> bool {
        matches!(self.kind, Kind::PutFile | Kind::CreateRepo | Kind::CreatePr | Kind::ReviewPr | Kind::CreateIssue)
    }

    async fn run(&self, args: Value, ctx: &ToolContext) -> ToolResult {
        match cancellable(ctx, self.exec(&args, ctx)).await {
            None => ToolResult::err(INTERRUPTED),
            Some(Ok(r)) => r,
            // What the TS registry reported for a tool that threw (no token,
            // owner lookup failed, network error …).
            Some(Err(e)) => {
                let name = self.name();
                ctx.log(json!({"tool": name, "error": format!("{e:#}")}));
                ToolResult::err(format!("Tool \"{name}\" error: {e:#}"))
            }
        }
    }
}

impl GithubTool {
    async fn exec(&self, args: &Value, ctx: &ToolContext) -> anyhow::Result<ToolResult> {
        let c = &self.client;
        use reqwest::Method;
        match self.kind {
            Kind::Whoami => {
                let r = c.get("/user").await?;
                if !r.ok {
                    return Ok(fail("github_whoami", &r));
                }
                let login = field(&r.json, "/login");
                c.remember_login(&login);
                ctx.log(json!({"tool": "github_whoami", "login": login}));
                let name = r.json.get("name").filter(|v| truthy(Some(v))).map(value_to_string);
                let output = format!(
                    "Authenticated as {login} ({}) — {} public repos.",
                    name.as_deref().unwrap_or("no name"),
                    field(&r.json, "/public_repos")
                );
                Ok(ToolResult::ok(output).with_data(r.json))
            }

            Kind::ListRepos => {
                let per = num_or(args, "limit", 30.0).clamp(1.0, 100.0);
                let r = c
                    .get(&format!(
                        "/user/repos?per_page={}&sort=pushed&affiliation=owner,collaborator,organization_member",
                        js_num(per)
                    ))
                    .await?;
                if !r.ok {
                    return Ok(fail("github_list_repos", &r));
                }
                let lines: Vec<String> = r
                    .json
                    .as_array()
                    .map(|a| a.as_slice())
                    .unwrap_or_default()
                    .iter()
                    .map(|x| {
                        let private = if truthy(x.get("private")) { " (private)" } else { "" };
                        let desc = x.get("description").filter(|d| truthy(Some(d))).map(value_to_string);
                        format!("{}{private} — {}", field(x, "/full_name"), desc.as_deref().unwrap_or("no description"))
                    })
                    .collect();
                ctx.log(json!({"tool": "github_list_repos", "count": lines.len()}));
                let out = if lines.is_empty() { "(no repositories)".to_string() } else { lines.join("\n") };
                Ok(ToolResult::ok(out).with_data(r.json))
            }

            Kind::GetFile => {
                let (owner, repo) = c.split_repo(&str_arg(args, &["repo"])).await?;
                let path = repo_path(args);
                let ref_q = match str_arg(args, &["ref"]) {
                    r if r.is_empty() => String::new(),
                    r => format!("?ref={}", urlencoding::encode(&r)),
                };
                let r = c.get(&format!("/repos/{owner}/{repo}/contents/{path}{ref_q}")).await?;
                if !r.ok {
                    return Ok(fail("github_get_file", &r));
                }
                ctx.log(json!({"tool": "github_get_file", "repo": format!("{owner}/{repo}"), "path": path}));
                if let Some(entries) = r.json.as_array() {
                    let listing = entries
                        .iter()
                        .map(|e| {
                            let icon = if e.get("type").and_then(Value::as_str) == Some("dir") { "📁" } else { "  " };
                            format!("{icon} {}", field(e, "/path"))
                        })
                        .collect::<Vec<_>>()
                        .join("\n");
                    let shown = if path.is_empty() { "/" } else { path.as_str() };
                    return Ok(ToolResult::ok(format!("Directory {shown}:\n{listing}")).with_data(r.json.clone()));
                }
                let raw = field(&r.json, "/content");
                let content = if r.json.get("encoding").and_then(Value::as_str) == Some("base64") { decode_base64(&raw) } else { raw };
                let sha = field(&r.json, "/sha");
                let file_path = field(&r.json, "/path");
                Ok(ToolResult::ok(format!("{owner}/{repo}/{file_path} (sha {}):\n\n{content}", take_chars(&sha, 7)))
                    .with_data(json!({"sha": sha, "content": content, "path": file_path})))
            }

            Kind::PutFile => {
                let (owner, repo) = c.split_repo(&str_arg(args, &["repo"])).await?;
                let path = repo_path(args);
                let message = match str_arg(args, &["message"]) {
                    m if m.is_empty() => format!("Update {path}"),
                    m => m,
                };
                let branch = str_arg(args, &["branch"]);
                let content = args.get("content").filter(|v| !v.is_null());
                let Some(content) = content.filter(|_| !path.is_empty()) else {
                    return Ok(ToolResult::err("github_put_file needs `path` and `content`."));
                };

                // Look up the existing sha (required by the API to update; absent = create).
                let ref_q = if branch.is_empty() { String::new() } else { format!("?ref={}", urlencoding::encode(&branch)) };
                let existing = c.get(&format!("/repos/{owner}/{repo}/contents/{path}{ref_q}")).await?;
                let sha = if existing.ok && !existing.json.is_array() {
                    existing.json.get("sha").filter(|s| truthy(Some(s))).map(value_to_string)
                } else {
                    None
                };

                let mut body = Map::new();
                body.insert("message".into(), json!(message));
                body.insert(
                    "content".into(),
                    json!(general_purpose::STANDARD.encode(value_to_string(content).as_bytes())),
                );
                if let Some(s) = &sha {
                    body.insert("sha".into(), json!(s));
                }
                if !branch.is_empty() {
                    body.insert("branch".into(), json!(branch));
                }
                let r = c.api(Method::PUT, &format!("/repos/{owner}/{repo}/contents/{path}"), Some(&Value::Object(body)), ACCEPT_JSON).await?;
                if !r.ok {
                    return Ok(fail("github_put_file", &r));
                }
                ctx.log(json!({"tool": "github_put_file", "repo": format!("{owner}/{repo}"), "path": path, "updated": sha.is_some()}));
                let commit = field(&r.json, "/commit/sha");
                Ok(ToolResult::ok(format!(
                    "{} {owner}/{repo}/{path} → commit {}",
                    if sha.is_some() { "Updated" } else { "Created" },
                    take_chars(&commit, 7)
                ))
                .with_data(r.json))
            }

            Kind::CreateRepo => {
                let name = str_arg(args, &["name"]).trim().to_string();
                if name.is_empty() {
                    return Ok(ToolResult::err("github_create_repo needs a `name`."));
                }
                let mut body = Map::new();
                body.insert("name".into(), json!(name));
                let desc = str_arg(args, &["description"]);
                if !desc.is_empty() {
                    body.insert("description".into(), json!(desc));
                }
                // Private by default for safety; auto-init by default so it's immediately writable.
                body.insert("private".into(), json!(args.get("private") != Some(&Value::Bool(false))));
                body.insert("auto_init".into(), json!(args.get("auto_init") != Some(&Value::Bool(false))));
                let r = c.api(Method::POST, "/user/repos", Some(&Value::Object(body)), ACCEPT_JSON).await?;
                if !r.ok {
                    return Ok(fail("github_create_repo", &r));
                }
                let full = field(&r.json, "/full_name");
                ctx.log(json!({"tool": "github_create_repo", "repo": full}));
                Ok(ToolResult::ok(format!("Created {full} → {}", field(&r.json, "/html_url"))).with_data(r.json))
            }

            Kind::ListPrs => {
                let (owner, repo) = c.split_repo(&str_arg(args, &["repo"])).await?;
                let state = args.get("state").map(value_to_string).unwrap_or_default();
                let state = if ["open", "closed", "all"].contains(&state.as_str()) { state } else { "open".into() };
                let r = c.get(&format!("/repos/{owner}/{repo}/pulls?state={state}&per_page=30")).await?;
                if !r.ok {
                    return Ok(fail("github_list_prs", &r));
                }
                let lines: Vec<String> = r
                    .json
                    .as_array()
                    .map(|a| a.as_slice())
                    .unwrap_or_default()
                    .iter()
                    .map(|p| {
                        format!(
                            "#{} [{}] {} — {} ({} → {})",
                            field(p, "/number"),
                            field(p, "/state"),
                            field(p, "/title"),
                            field(p, "/user/login"),
                            field(p, "/head/ref"),
                            field(p, "/base/ref")
                        )
                    })
                    .collect();
                ctx.log(json!({"tool": "github_list_prs", "repo": format!("{owner}/{repo}"), "count": lines.len()}));
                let out = if lines.is_empty() { "(no pull requests)".to_string() } else { lines.join("\n") };
                Ok(ToolResult::ok(out).with_data(r.json))
            }

            Kind::GetPr => {
                let (owner, repo) = c.split_repo(&str_arg(args, &["repo"])).await?;
                let Some(num) = num_arg(args, "number").filter(|n| *n != 0.0) else {
                    return Ok(ToolResult::err("github_get_pr needs a PR `number`."));
                };
                let num = js_num(num);
                let meta = c.get(&format!("/repos/{owner}/{repo}/pulls/{num}")).await?;
                if !meta.ok {
                    return Ok(fail("github_get_pr", &meta));
                }
                let diff = c.api(Method::GET, &format!("/repos/{owner}/{repo}/pulls/{num}"), None, ACCEPT_DIFF).await?;
                ctx.log(json!({"tool": "github_get_pr", "repo": format!("{owner}/{repo}"), "number": meta.json.get("number").cloned().unwrap_or(Value::Null)}));
                let p = &meta.json;
                let body = p.get("body").filter(|b| truthy(Some(b))).map(value_to_string);
                let header = format!(
                    "#{} {} — {} ({} → {})\nstate: {}  +{} -{} across {} files\n\n{}\n\n--- DIFF ---\n",
                    field(p, "/number"),
                    field(p, "/title"),
                    field(p, "/user/login"),
                    field(p, "/head/ref"),
                    field(p, "/base/ref"),
                    field(p, "/state"),
                    field(p, "/additions"),
                    field(p, "/deletions"),
                    field(p, "/changed_files"),
                    body.as_deref().unwrap_or("(no description)")
                );
                let diff_text = if diff.ok { diff.text } else { "(diff unavailable)".into() };
                Ok(ToolResult::ok(header + &diff_text).with_data(meta.json))
            }

            Kind::CreatePr => {
                let (owner, repo) = c.split_repo(&str_arg(args, &["repo"])).await?;
                let mut body = Map::new();
                body.insert("title".into(), json!(str_arg(args, &["title"])));
                body.insert("head".into(), json!(str_arg(args, &["head"])));
                let base = str_arg(args, &["base"]);
                body.insert("base".into(), json!(if base.is_empty() { "main".to_string() } else { base }));
                let pr_body = str_arg(args, &["body"]);
                if !pr_body.is_empty() {
                    body.insert("body".into(), json!(pr_body));
                }
                body.insert("draft".into(), json!(args.get("draft") == Some(&Value::Bool(true))));
                let r = c.api(Method::POST, &format!("/repos/{owner}/{repo}/pulls"), Some(&Value::Object(body)), ACCEPT_JSON).await?;
                if !r.ok {
                    return Ok(fail("github_create_pr", &r));
                }
                let number = field(&r.json, "/number");
                ctx.log(json!({"tool": "github_create_pr", "repo": format!("{owner}/{repo}"), "number": r.json.get("number").cloned().unwrap_or(Value::Null)}));
                Ok(ToolResult::ok(format!("Opened PR #{number} → {}", field(&r.json, "/html_url"))).with_data(r.json))
            }

            Kind::ReviewPr => {
                let (owner, repo) = c.split_repo(&str_arg(args, &["repo"])).await?;
                let num = num_arg(args, "number").filter(|n| *n != 0.0);
                let event = match str_arg(args, &["event"]) {
                    e if e.is_empty() => "COMMENT".to_string(),
                    e => e.to_uppercase(),
                };
                let Some(num) = num else {
                    return Ok(ToolResult::err("github_review_pr needs a PR `number`."));
                };
                if !["APPROVE", "REQUEST_CHANGES", "COMMENT"].contains(&event.as_str()) {
                    return Ok(ToolResult::err("event must be APPROVE, REQUEST_CHANGES, or COMMENT."));
                }
                let num = js_num(num);
                let body = json!({"body": str_arg(args, &["body"]), "event": event});
                let r = c.api(Method::POST, &format!("/repos/{owner}/{repo}/pulls/{num}/reviews"), Some(&body), ACCEPT_JSON).await?;
                if !r.ok {
                    return Ok(fail("github_review_pr", &r));
                }
                ctx.log(json!({"tool": "github_review_pr", "repo": format!("{owner}/{repo}"), "number": num, "event": event}));
                Ok(ToolResult::ok(format!("Submitted {event} review on PR #{num}.")).with_data(r.json))
            }

            Kind::CreateIssue => {
                let (owner, repo) = c.split_repo(&str_arg(args, &["repo"])).await?;
                let mut body = Map::new();
                body.insert("title".into(), json!(str_arg(args, &["title"])));
                let issue_body = str_arg(args, &["body"]);
                if !issue_body.is_empty() {
                    body.insert("body".into(), json!(issue_body));
                }
                if let Some(labels) = args.get("labels").filter(|l| l.is_array()) {
                    body.insert("labels".into(), labels.clone());
                }
                let r = c.api(Method::POST, &format!("/repos/{owner}/{repo}/issues"), Some(&Value::Object(body)), ACCEPT_JSON).await?;
                if !r.ok {
                    return Ok(fail("github_create_issue", &r));
                }
                let number = field(&r.json, "/number");
                ctx.log(json!({"tool": "github_create_issue", "repo": format!("{owner}/{repo}"), "number": r.json.get("number").cloned().unwrap_or(Value::Null)}));
                Ok(ToolResult::ok(format!("Opened issue #{number} → {}", field(&r.json, "/html_url"))).with_data(r.json))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testutil::{serve, Reply};
    use std::collections::HashMap;

    fn no_env() -> EnvLookup {
        Arc::new(|_: &str| None)
    }

    fn env_of(pairs: &[(&str, &str)]) -> EnvLookup {
        let m: HashMap<String, String> = pairs.iter().map(|(k, v)| (k.to_string(), v.to_string())).collect();
        Arc::new(move |k: &str| m.get(k).cloned())
    }

    #[tokio::test]
    async fn token_resolution_order() {
        let cfg = GithubConfig { token: "cfg-tok".into(), ..Default::default() };
        let c = GithubClient::with_env(cfg, env_of(&[("GITHUB_TOKEN", "env-tok")]), "qo_no_such_gh_xyz");
        assert_eq!(c.resolve_token().await.unwrap(), "cfg-tok");

        let cfg = GithubConfig { token_env: "MY_TOK".into(), ..Default::default() };
        let c = GithubClient::with_env(cfg.clone(), env_of(&[("MY_TOK", "mine"), ("GITHUB_TOKEN", "gh")]), "qo_no_such_gh_xyz");
        assert_eq!(c.resolve_token().await.unwrap(), "mine");
        let c = GithubClient::with_env(cfg.clone(), env_of(&[("GH_TOKEN", "ghtok"), ("MY_TOK", "")]), "qo_no_such_gh_xyz");
        assert_eq!(c.resolve_token().await.unwrap(), "ghtok");

        // No config, no env, no gh binary → the TS error text.
        let c = GithubClient::with_env(cfg, no_env(), "qo_no_such_gh_xyz");
        let e = c.resolve_token().await.unwrap_err().to_string();
        assert!(e.starts_with("No GitHub token. Set github.token in config, export MY_TOK"), "{e}");
    }

    #[tokio::test]
    async fn tools_against_mock_api() {
        let server = serve(Arc::new(|req| {
            let path = req.path.as_str();
            match (req.method.as_str(), path) {
                ("GET", "/user") => Reply::json(200, json!({"login": "octo", "name": null, "public_repos": 3})),
                ("GET", "/repos/octo/demo/contents/src/a.txt") => Reply::json(
                    200,
                    json!({"path": "src/a.txt", "sha": "abcdef1234", "encoding": "base64", "content": "aGVs\nbG8="}),
                ),
                ("GET", "/repos/octo/demo/contents/") => Reply::json(
                    200,
                    json!([{"type": "dir", "path": "src"}, {"type": "file", "path": "README.md"}]),
                ),
                ("GET", "/repos/octo/demo/contents/new.txt?ref=dev") => Reply::json(404, json!({"message": "Not Found"})),
                ("PUT", "/repos/octo/demo/contents/new.txt") => Reply::json(201, json!({"commit": {"sha": "1234567890"}})),
                ("GET", "/repos/octo/demo/pulls/7") => {
                    if req.header("accept").is_some_and(|a| a.contains("diff")) {
                        Reply::text(200, "text/plain", "diff --git a/x b/x\n+hi\n")
                    } else {
                        Reply::json(
                            200,
                            json!({"number": 7, "title": "T", "user": {"login": "u"}, "head": {"ref": "h"},
                                   "base": {"ref": "main"}, "state": "open", "additions": 1, "deletions": 0,
                                   "changed_files": 1, "body": null}),
                        )
                    }
                }
                ("POST", "/repos/other/x/issues") => Reply::json(422, json!({"message": "Validation Failed"})),
                _ => Reply::text(500, "text/plain", "unexpected"),
            }
        }))
        .await;

        let cfg = GithubConfig { token: "t0k".into(), api_base: format!("{}/", server.base), ..Default::default() };
        let client = Arc::new(GithubClient::with_env(cfg, no_env(), "qo_no_such_gh_xyz"));
        let tools = tools_for(client);
        let tool = |n: &str| tools.iter().find(|t| t.name() == n).cloned().unwrap();
        let ctx = ToolContext::new(".");

        let r = tool("github_whoami").run(json!({}), &ctx).await;
        assert_eq!(r.output, "Authenticated as octo (no name) — 3 public repos.");

        // Bare repo name → owner defaults to the login.
        let r = tool("github_get_file").run(json!({"repo": "demo", "path": "/src/a.txt"}), &ctx).await;
        assert!(r.ok, "{}", r.output);
        assert_eq!(r.output, "octo/demo/src/a.txt (sha abcdef1):\n\nhello");

        let r = tool("github_get_file").run(json!({"repo": "octo/demo", "path": ""}), &ctx).await;
        assert_eq!(r.output, "Directory /:\n📁 src\n   README.md");

        let r = tool("github_put_file")
            .run(json!({"repo": "octo/demo", "path": "new.txt", "content": "x", "branch": "dev"}), &ctx)
            .await;
        assert_eq!(r.output, "Created octo/demo/new.txt → commit 1234567");
        let r = tool("github_put_file").run(json!({"repo": "octo/demo", "path": "new.txt"}), &ctx).await;
        assert_eq!(r.output, "github_put_file needs `path` and `content`.");

        let r = tool("github_get_pr").run(json!({"repo": "demo", "number": "7"}), &ctx).await;
        assert!(r.output.starts_with("#7 T — u (h → main)\nstate: open  +1 -0 across 1 files\n\n(no description)\n\n--- DIFF ---\ndiff --git"), "{}", r.output);

        let r = tool("github_create_issue").run(json!({"repo": "other/x", "title": "bug"}), &ctx).await;
        assert!(!r.ok);
        assert_eq!(r.output, "github_create_issue failed (HTTP 422): Validation Failed");

        let r = tool("github_review_pr").run(json!({"repo": "octo/demo", "number": 7, "event": "nope"}), &ctx).await;
        assert_eq!(r.output, "event must be APPROVE, REQUEST_CHANGES, or COMMENT.");

        // The PUT carried base64 content, the looked-up branch and the bearer token.
        let reqs = server.requests();
        let put = reqs.iter().find(|r| r.method == "PUT").unwrap();
        let body: Value = serde_json::from_str(&put.body).unwrap();
        assert_eq!(body, json!({"message": "Update new.txt", "content": "eA==", "branch": "dev"}));
        assert_eq!(put.header("authorization"), Some("Bearer t0k"));
        assert_eq!(put.header("x-github-api-version"), Some("2022-11-28"));
        // /user was fetched once for whoami, then served from the login cache.
        assert_eq!(reqs.iter().filter(|r| r.path == "/user").count(), 1);
    }

    #[tokio::test]
    async fn missing_token_is_reported_like_a_thrown_tool_error() {
        let cfg = GithubConfig::default();
        let tools = tools_for(Arc::new(GithubClient::with_env(cfg, no_env(), "qo_no_such_gh_xyz")));
        let r = tools[0].run(json!({}), &ToolContext::new(".")).await;
        assert!(!r.ok);
        assert!(r.output.starts_with("Tool \"github_whoami\" error: No GitHub token."), "{}", r.output);
    }

    #[test]
    fn names_and_mutating_flags() {
        let tools = create_github_tools(GithubConfig::default());
        let names: Vec<&str> = tools.iter().map(|t| t.name()).collect();
        assert_eq!(names.len(), 10);
        let specs: Vec<String> = crate::specs::github_tool_specs().into_iter().map(|s| s.name).collect();
        assert_eq!(names, specs);
        let mutating: Vec<&str> = tools.iter().filter(|t| t.mutating()).map(|t| t.name()).collect();
        assert_eq!(
            mutating,
            ["github_put_file", "github_create_repo", "github_create_pr", "github_review_pr", "github_create_issue"]
        );
        assert_eq!(decode_base64("aGVs\nbG8"), "hello");
    }
}
