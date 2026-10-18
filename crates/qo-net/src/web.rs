//! Internet access tools: `web_search` (find pages), `web_fetch` (read a page),
//! `web_research` (search + read + condense in one call) and `deep_research`
//! (multi-round, citation-indexed research in one call).
//!
//! Default search provider is keyless DuckDuckGo HTML; Tavily/Brave/SearXNG are
//! supported when an api_key / url is configured. Whatever the primary is, an
//! empty or failed search falls back through the keyless DuckDuckGo HTML and
//! DuckDuckGo Lite endpoints, so a markup change or soft rate-limit is never a
//! dead end.
//!
//! Every request has a deadline (`fetch_timeout_ms`, covering connect, headers
//! *and* body), page downloads are capped at [`MAX_PAGE_BYTES`], and page
//! fetches pass an SSRF guard — on the first URL and on every redirect hop — so
//! a tool call can't be used to probe the machine's own services or the LAN.
//! All futures here are drop-safe: the chat layer cancels a tool by dropping it.

use std::sync::{Arc, LazyLock, Mutex};
use std::time::Duration;

use async_trait::async_trait;
use indexmap::IndexMap;
use qo_core::{Tool, ToolContext, ToolResult};
use regex::Regex;
use serde_json::{json, Value};

use crate::util::{cancellable, char_len, collapse_ws, num_or, str_arg, take_chars, INTERRUPTED};

/// Browser-like UA: several search/doc sites serve bots an empty shell.
const UA: &str =
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36";

/// Hard cap on how much of one page/response body is downloaded. The model only
/// ever sees `fetch_chars` of it; this just stops a multi-GB file (or an endless
/// stream) from eating memory and time.
pub const MAX_PAGE_BYTES: usize = 5 * 1024 * 1024;

/// In-session readable-page cache size (bounded FIFO).
const PAGE_CACHE_MAX: usize = 64;

/// Which engine `web_search` asks first.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum SearchProvider {
    /// Keyless DuckDuckGo HTML (the default, and the universal fallback).
    #[default]
    DuckDuckGo,
    /// Tavily search API (needs `api_key`).
    Tavily,
    /// Brave Search API (needs `api_key`).
    Brave,
    /// A SearXNG instance (needs `searxng_url`).
    Searxng,
}

impl SearchProvider {
    /// Parse the config value (`web.provider`); unknown values mean DuckDuckGo.
    pub fn parse(s: &str) -> Self {
        match s.trim().to_ascii_lowercase().as_str() {
            "tavily" => SearchProvider::Tavily,
            "brave" => SearchProvider::Brave,
            "searxng" => SearchProvider::Searxng,
            _ => SearchProvider::DuckDuckGo,
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            SearchProvider::DuckDuckGo => "duckduckgo",
            SearchProvider::Tavily => "tavily",
            SearchProvider::Brave => "brave",
            SearchProvider::Searxng => "searxng",
        }
    }
}

/// Search API endpoints. Only overridden by tests (pointing at a local mock).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WebEndpoints {
    pub ddg_html: String,
    pub ddg_lite: String,
    pub tavily: String,
    pub brave: String,
}

impl Default for WebEndpoints {
    fn default() -> Self {
        Self {
            ddg_html: "https://html.duckduckgo.com/html/".into(),
            ddg_lite: "https://lite.duckduckgo.com/lite/".into(),
            tavily: "https://api.tavily.com/search".into(),
            brave: "https://api.search.brave.com/res/v1/web/search".into(),
        }
    }
}

/// Options for [`create_web_tools`] (mirrors the TS `WebConfig` / `[web]` config).
#[derive(Clone, Debug)]
pub struct WebConfig {
    pub provider: SearchProvider,
    pub api_key: String,
    pub searxng_url: String,
    pub max_results: usize,
    /// Max characters of page text handed back by `web_fetch`.
    pub fetch_chars: usize,
    /// Abort an individual page/search fetch after this many ms (0 = no timeout).
    pub fetch_timeout_ms: u64,
    pub endpoints: WebEndpoints,
    /// Disable the SSRF guard. Only for tests against a loopback mock server or a
    /// trusted intranet — never set this from model-controlled input.
    pub allow_private_hosts: bool,
}

impl Default for WebConfig {
    fn default() -> Self {
        Self {
            provider: SearchProvider::DuckDuckGo,
            api_key: String::new(),
            searxng_url: String::new(),
            max_results: 5,
            fetch_chars: 8000,
            fetch_timeout_ms: 15_000,
            endpoints: WebEndpoints::default(),
            allow_private_hosts: false,
        }
    }
}

/// One search hit.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct SearchResult {
    pub title: String,
    pub url: String,
    pub snippet: String,
}

/// Shared HTTP state behind the four web tools: client, config and the
/// in-session page cache (a `web_research` followed by a `web_fetch` of the same
/// URL — very common — shouldn't re-download the page).
pub struct WebClient {
    cfg: WebConfig,
    http: reqwest::Client,
    cache: Mutex<IndexMap<String, String>>,
}

impl WebClient {
    pub fn new(cfg: WebConfig) -> Self {
        let allow_private = cfg.allow_private_hosts;
        // Re-check every redirect hop: a public URL must not be able to bounce
        // the fetch onto localhost / the LAN.
        let policy = reqwest::redirect::Policy::custom(move |attempt| {
            if attempt.previous().len() >= 10 {
                return attempt.error("too many redirects");
            }
            if !allow_private {
                if let Err(e) = assert_public_url(attempt.url().as_str()) {
                    return attempt.error(e);
                }
            }
            attempt.follow()
        });
        let http = reqwest::Client::builder()
            .redirect(policy)
            .connect_timeout(Duration::from_secs(15))
            .build()
            // Never fall back to a client that follows redirects unguarded.
            .or_else(|_| reqwest::Client::builder().redirect(reqwest::redirect::Policy::none()).build())
            .unwrap_or_default();
        Self { cfg, http, cache: Mutex::new(IndexMap::new()) }
    }

    pub fn config(&self) -> &WebConfig {
        &self.cfg
    }

    fn timeout_ms(&self) -> u64 {
        self.cfg.fetch_timeout_ms
    }

    /// Send with the configured deadline (covering the body read too) and map a
    /// timeout to the TS wording.
    async fn send(&self, req: reqwest::RequestBuilder) -> anyhow::Result<(reqwest::StatusCode, String, String)> {
        let ms = self.timeout_ms();
        let fut = async {
            let res = req.send().await?;
            let status = res.status();
            let ctype = res
                .headers()
                .get(reqwest::header::CONTENT_TYPE)
                .and_then(|v| v.to_str().ok())
                .unwrap_or("")
                .to_string();
            let body = read_capped(res, MAX_PAGE_BYTES).await?;
            Ok::<_, reqwest::Error>((status, ctype, body))
        };
        if ms == 0 {
            return fut.await.map_err(|e| anyhow::anyhow!(describe_reqwest(&e, ms)));
        }
        match tokio::time::timeout(Duration::from_millis(ms), fut).await {
            Err(_) => anyhow::bail!("fetch timed out after {ms}ms"),
            Ok(r) => r.map_err(|e| anyhow::anyhow!(describe_reqwest(&e, ms))),
        }
    }

    /// Resilient search: try the configured provider first, then fall back
    /// through a ladder of keyless engines. An empty/failed result just advances
    /// to the next engine; each step is isolated so one bad provider never aborts
    /// the whole search. Errors only when every step failed.
    pub async fn search(&self, query: &str) -> anyhow::Result<Vec<SearchResult>> {
        #[derive(Clone, Copy)]
        enum Step {
            Tavily,
            Brave,
            Searxng,
            Ddg,
            DdgLite,
        }
        let mut ladder = Vec::with_capacity(3);
        match self.cfg.provider {
            SearchProvider::Tavily => ladder.push(Step::Tavily),
            SearchProvider::Brave => ladder.push(Step::Brave),
            SearchProvider::Searxng => ladder.push(Step::Searxng),
            // duckduckgo is the default primary as well as the universal fallback below.
            SearchProvider::DuckDuckGo => {}
        }
        // Keyless fallbacks, always appended (and the primary for the default config).
        ladder.push(Step::Ddg);
        ladder.push(Step::DdgLite);

        let mut last_err: Option<anyhow::Error> = None;
        for step in ladder {
            let r = match step {
                Step::Tavily => self.tavily(query).await,
                Step::Brave => self.brave(query).await,
                Step::Searxng => self.searxng(query).await,
                Step::Ddg => self.duckduckgo(query).await,
                Step::DdgLite => self.ddg_lite(query).await,
            };
            match r {
                Ok(results) if !results.is_empty() => return Ok(results),
                Ok(_) => {}
                Err(e) => last_err = Some(e),
            }
        }
        match last_err {
            Some(e) => Err(e),
            None => Ok(Vec::new()),
        }
    }

    async fn duckduckgo(&self, query: &str) -> anyhow::Result<Vec<SearchResult>> {
        let req = self
            .http
            .post(&self.cfg.endpoints.ddg_html)
            .header(reqwest::header::USER_AGENT, UA)
            .form(&[("q", query)]);
        let (_, _, html) = self.send(req).await?;
        Ok(parse_ddg_html(&html, self.cfg.max_results))
    }

    async fn ddg_lite(&self, query: &str) -> anyhow::Result<Vec<SearchResult>> {
        let req = self
            .http
            .post(&self.cfg.endpoints.ddg_lite)
            .header(reqwest::header::USER_AGENT, UA)
            .form(&[("q", query)]);
        let (_, _, html) = self.send(req).await?;
        Ok(parse_ddg_lite(&html, self.cfg.max_results))
    }

    async fn tavily(&self, query: &str) -> anyhow::Result<Vec<SearchResult>> {
        if self.cfg.api_key.is_empty() {
            anyhow::bail!("Tavily requires web.api_key");
        }
        let req = self.http.post(&self.cfg.endpoints.tavily).json(&json!({
            "api_key": self.cfg.api_key,
            "query": query,
            "max_results": self.cfg.max_results,
            "include_answer": false,
        }));
        let (status, _, body) = self.send(req).await?;
        if !status.is_success() {
            anyhow::bail!("Tavily HTTP {}", status.as_u16());
        }
        let v: Value = serde_json::from_str(&body)?;
        Ok(json_results(v.get("results"), "content", self.cfg.max_results))
    }

    async fn brave(&self, query: &str) -> anyhow::Result<Vec<SearchResult>> {
        if self.cfg.api_key.is_empty() {
            anyhow::bail!("Brave requires web.api_key");
        }
        let url = format!(
            "{}?q={}&count={}",
            self.cfg.endpoints.brave,
            urlencoding::encode(query),
            self.cfg.max_results
        );
        let req = self
            .http
            .get(url)
            .header(reqwest::header::ACCEPT, "application/json")
            .header("X-Subscription-Token", &self.cfg.api_key);
        let (status, _, body) = self.send(req).await?;
        if !status.is_success() {
            anyhow::bail!("Brave HTTP {}", status.as_u16());
        }
        let v: Value = serde_json::from_str(&body)?;
        let results = v.get("web").and_then(|w| w.get("results"));
        Ok(json_results(results, "description", self.cfg.max_results))
    }

    async fn searxng(&self, query: &str) -> anyhow::Result<Vec<SearchResult>> {
        if self.cfg.searxng_url.is_empty() {
            anyhow::bail!("SearXNG requires web.searxng_url");
        }
        let base = self.cfg.searxng_url.trim_end_matches('/');
        let url = format!("{base}/search?q={}&format=json", urlencoding::encode(query));
        let req = self.http.get(url).header(reqwest::header::USER_AGENT, UA);
        let (status, _, body) = self.send(req).await?;
        if !status.is_success() {
            anyhow::bail!("SearXNG HTTP {}", status.as_u16());
        }
        let v: Value = serde_json::from_str(&body)?;
        Ok(json_results(v.get("results"), "content", self.cfg.max_results))
    }

    /// Fetch a URL and return its readable text (HTML → text), SSRF-guarded and
    /// cached for the session.
    pub async fn fetch_readable(&self, url: &str) -> anyhow::Result<String> {
        if !self.cfg.allow_private_hosts {
            assert_public_url(url).map_err(anyhow::Error::msg)?;
        }
        if let Some(hit) = self.cache.lock().ok().and_then(|c| c.get(url).cloned()) {
            return Ok(hit);
        }
        let req = self.http.get(url).header(reqwest::header::USER_AGENT, UA);
        let (status, ctype, raw) = self.send(req).await?;
        if !status.is_success() {
            anyhow::bail!("HTTP {}", status.as_u16());
        }
        let text = if ctype.to_ascii_lowercase().contains("html") { html_to_text(&raw) } else { raw };
        if let Ok(mut cache) = self.cache.lock() {
            if cache.len() >= PAGE_CACHE_MAX && !cache.contains_key(url) {
                cache.shift_remove_index(0);
            }
            cache.insert(url.to_string(), text.clone());
        }
        Ok(text)
    }
}

/// Read a response body up to `cap` bytes (the rest is dropped) as lossy UTF-8
/// — `Response.text()` in the TS always decoded as UTF-8 too.
async fn read_capped(mut res: reqwest::Response, cap: usize) -> Result<String, reqwest::Error> {
    let mut buf: Vec<u8> = Vec::new();
    while let Some(chunk) = res.chunk().await? {
        let room = cap.saturating_sub(buf.len());
        if chunk.len() >= room {
            buf.extend_from_slice(&chunk[..room]);
            break;
        }
        buf.extend_from_slice(&chunk);
    }
    Ok(String::from_utf8_lossy(&buf).into_owned())
}

fn describe_reqwest(e: &reqwest::Error, ms: u64) -> String {
    if e.is_timeout() {
        return format!("fetch timed out after {ms}ms");
    }
    // Surface the innermost cause (e.g. "Refused: … (SSRF guard)" from a redirect).
    let mut msg = e.to_string();
    let mut src = std::error::Error::source(e);
    while let Some(s) = src {
        msg = format!("{msg}: {s}");
        src = s.source();
    }
    msg
}

fn json_results(arr: Option<&Value>, snippet_key: &str, max: usize) -> Vec<SearchResult> {
    let Some(items) = arr.and_then(|a| a.as_array()) else {
        return Vec::new();
    };
    let s = |v: &Value, k: &str| v.get(k).and_then(|x| x.as_str()).unwrap_or("").to_string();
    items
        .iter()
        .take(max)
        .map(|r| SearchResult { title: s(r, "title"), url: s(r, "url"), snippet: s(r, snippet_key) })
        .collect()
}

/* ── DuckDuckGo scraping ── */

static DDG_LINK: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r#"<a[^>]+class="result__a"[^>]+href="([^"]+)"[^>]*>([\s\S]*?)</a>"#).expect("static regex")
});
static DDG_SNIP: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r#"class="result__snippet"[^>]*>([\s\S]*?)</a>"#).expect("static regex"));
static LITE_LINK: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r#"<a[^>]+class="result-link"[^>]+href="([^"]+)"[^>]*>([\s\S]*?)</a>"#).expect("static regex")
});
static LITE_SNIP: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r#"class="result-snippet"[^>]*>([\s\S]*?)</td>"#).expect("static regex"));
static UDDG: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"[?&]uddg=([^&]+)").expect("static regex"));

/// Two passes over the DDG HTML page: titles+links, then snippets, zipped by order.
pub fn parse_ddg_html(html: &str, max: usize) -> Vec<SearchResult> {
    zip_results(html, &DDG_LINK, &DDG_SNIP, max)
}

/// DuckDuckGo Lite — a far simpler, more stable HTML endpoint than the main one.
/// Results live in a table: result link rows carry class "result-link"; the
/// snippet is the following "result-snippet" cell.
pub fn parse_ddg_lite(html: &str, max: usize) -> Vec<SearchResult> {
    zip_results(html, &LITE_LINK, &LITE_SNIP, max)
}

fn zip_results(html: &str, link_re: &Regex, snip_re: &Regex, max: usize) -> Vec<SearchResult> {
    let mut links: Vec<(String, String)> = Vec::new();
    for c in link_re.captures_iter(html) {
        let url = decode_ddg_url(&c[1]);
        let title = html_to_text(&c[2]).trim().to_string();
        if !url.is_empty() && !title.is_empty() {
            links.push((title, url));
        }
    }
    let snippets: Vec<String> = snip_re.captures_iter(html).map(|c| html_to_text(&c[1]).trim().to_string()).collect();
    links
        .into_iter()
        .take(max)
        .enumerate()
        .map(|(i, (title, url))| SearchResult { title, url, snippet: snippets.get(i).cloned().unwrap_or_default() })
        .collect()
}

/// DDG wraps results as `//duckduckgo.com/l/?uddg=<encoded>&...`.
fn decode_ddg_url(href: &str) -> String {
    if let Some(c) = UDDG.captures(href) {
        return match urlencoding::decode(&c[1]) {
            Ok(s) => s.into_owned(),
            Err(_) => c[1].to_string(),
        };
    }
    if href.starts_with("//") {
        format!("https:{href}")
    } else {
        href.to_string()
    }
}

/* ── SSRF guard ── */

/// Refuse loopback / link-local / private-network hosts so a tool call (or a
/// redirect) can't be used to probe the machine's own services or the local
/// network. Only http(s) public hosts are allowed. Returns the refusal text.
pub fn assert_public_url(url: &str) -> Result<(), String> {
    let u = url::Url::parse(url).map_err(|_| format!("Invalid URL: {url}"))?;
    if u.scheme() != "http" && u.scheme() != "https" {
        return Err(format!("Refused: only http(s) URLs are allowed (got {}:).", u.scheme()));
    }
    let host_str = u.host_str().unwrap_or("").to_ascii_lowercase();
    let blocked = match u.host() {
        None => true,
        Some(url::Host::Domain(d)) => {
            let host = d.to_ascii_lowercase();
            host == "localhost"
                || host == "0.0.0.0"
                || host == "::1"
                || host.ends_with(".localhost")
                || host.ends_with(".internal")
                || host.ends_with(".local")
        }
        Some(url::Host::Ipv4(ip)) => ipv4_blocked(ip),
        Some(url::Host::Ipv6(ip)) => {
            let seg = ip.segments();
            ip.is_loopback()
                || ip.is_unspecified()
                || (seg[0] & 0xffc0) == 0xfe80 // link-local
                || (seg[0] & 0xfe00) == 0xfc00 // unique local
                || ip.to_ipv4_mapped().is_some_and(ipv4_blocked)
        }
    };
    if blocked {
        return Err(format!("Refused: \"{host_str}\" is a private/loopback address (SSRF guard)."));
    }
    Ok(())
}

fn ipv4_blocked(ip: std::net::Ipv4Addr) -> bool {
    let [a, b, _, _] = ip.octets();
    a == 127 // loopback
        || a == 10 // private
        || (a == 172 && (16..=31).contains(&b)) // private
        || (a == 192 && b == 168) // private
        || (a == 169 && b == 254) // link-local
        || a == 0
}

/* ── extractive condensing ── */

/// Extractive summariser: pick the sentences most relevant to `query` (with a
/// mild bias toward the top of the page) up to `max_chars`, preserving reading
/// order. No model call — keeps large pages from flooding the context. With no
/// query terms it falls back to the leading sentences (the article's lede).
pub fn condense(text: &str, query: &str, max_chars: usize) -> String {
    let collapsed = collapse_ws(text);
    let sentences: Vec<&str> = split_sentences(&collapsed)
        .into_iter()
        .map(str::trim)
        .filter(|s| {
            let n = char_len(s);
            (30..=500).contains(&n)
        })
        .collect();
    if sentences.is_empty() {
        return take_chars(text, max_chars).to_string();
    }

    let lower_q = query.to_lowercase();
    let terms: Vec<&str> = TERM_RE.find_iter(&lower_q).map(|m| m.as_str()).filter(|w| w.len() > 2).collect();

    struct Scored<'a> {
        s: &'a str,
        i: usize,
        len: usize,
        score: f64,
    }
    let mut scored: Vec<Scored> = sentences
        .iter()
        .enumerate()
        .map(|(i, s)| {
            let sl = s.to_lowercase();
            let mut score = 0.0;
            for t in &terms {
                if sl.contains(t) {
                    score += 2.0;
                }
            }
            score += (3.0 - i as f64 * 0.03).max(0.0); // earlier sentences are usually more informative
            Scored { s, i, len: char_len(s), score }
        })
        .collect();

    // Stable sort, highest score first (ties keep reading order, like the TS).
    scored.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
    let mut picked: Vec<&Scored> = Vec::new();
    let mut len = 0usize;
    for x in &scored {
        if len + x.len + 1 > max_chars {
            continue;
        }
        picked.push(x);
        len += x.len + 1;
    }
    if picked.is_empty() {
        return take_chars(text, max_chars).to_string();
    }
    picked.sort_by_key(|x| x.i);
    picked.iter().map(|x| x.s).collect::<Vec<_>>().join(" ")
}

static TERM_RE: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"[a-z0-9]+").expect("static regex"));

/// `split(/(?<=[.!?])\s+(?=[A-Z0-9"'])/)` on whitespace-collapsed text (the regex
/// crate has no look-around, so this walks the string).
fn split_sentences(s: &str) -> Vec<&str> {
    let chars: Vec<(usize, char)> = s.char_indices().collect();
    let mut out = Vec::new();
    let mut start = 0usize;
    for k in 1..chars.len().saturating_sub(1) {
        let (idx, c) = chars[k];
        if c == ' '
            && matches!(chars[k - 1].1, '.' | '!' | '?')
            && matches!(chars[k + 1].1, 'A'..='Z' | '0'..='9' | '"' | '\'')
        {
            out.push(&s[start..idx]);
            start = idx + 1;
        }
    }
    out.push(&s[start..]);
    out
}

/* ── html → text ── */

static SCRIPT_RE: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"(?i)<script[\s\S]*?</script>").expect("static regex"));
static STYLE_RE: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"(?i)<style[\s\S]*?</style>").expect("static regex"));
static NOSCRIPT_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?i)<noscript[\s\S]*?</noscript>").expect("static regex"));
static BLOCK_END_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?i)</(p|div|li|h[1-6]|tr|br|section|article)>").expect("static regex"));
static TAG_RE: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"<[^>]+>").expect("static regex"));
static ENTITY_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?i)&(#x?[0-9a-f]+|[A-Za-z0-9_]+);").expect("static regex"));
static HSPACE_RE: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"[ \t]+").expect("static regex"));
static MANY_NL_RE: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"\n{3,}").expect("static regex"));

/// Strip scripts/styles/tags, turn block ends into newlines, decode entities.
pub fn html_to_text(html: &str) -> String {
    let s = SCRIPT_RE.replace_all(html, " ");
    let s = STYLE_RE.replace_all(&s, " ");
    let s = NOSCRIPT_RE.replace_all(&s, " ");
    let s = BLOCK_END_RE.replace_all(&s, "\n");
    let s = TAG_RE.replace_all(&s, " ");
    let s = decode_entities(&s);
    let s = HSPACE_RE.replace_all(&s, " ");
    let s = MANY_NL_RE.replace_all(&s, "\n\n");
    s.trim().to_string()
}

fn decode_entities(s: &str) -> String {
    ENTITY_RE
        .replace_all(s, |c: &regex::Captures| {
            let whole = &c[0];
            let code = &c[1];
            if let Some(num) = code.strip_prefix('#') {
                let n = if let Some(hex) = num.strip_prefix(['x', 'X']) {
                    u32::from_str_radix(hex, 16).ok()
                } else {
                    // parseInt(…, 10) reads the leading digits only.
                    let digits: String = num.chars().take_while(char::is_ascii_digit).collect();
                    digits.parse::<u32>().ok()
                };
                return n.and_then(char::from_u32).map(String::from).unwrap_or_else(|| whole.to_string());
            }
            match code.to_ascii_lowercase().as_str() {
                "amp" => "&".into(),
                "lt" => "<".into(),
                "gt" => ">".into(),
                "quot" => "\"".into(),
                "apos" => "'".into(),
                "nbsp" => " ".into(),
                _ => whole.to_string(),
            }
        })
        .into_owned()
}

/* ── tools ── */

/// Build the four web tools (`web_search`, `web_fetch`, `web_research`,
/// `deep_research`) sharing one client + page cache.
pub fn create_web_tools(cfg: WebConfig) -> Vec<qo_core::tool::SharedTool> {
    let client = Arc::new(WebClient::new(cfg));
    vec![
        Arc::new(WebSearchTool { client: client.clone() }),
        Arc::new(WebFetchTool { client: client.clone() }),
        Arc::new(WebResearchTool { client: client.clone() }),
        Arc::new(DeepResearchTool { client }),
    ]
}

struct WebSearchTool {
    client: Arc<WebClient>,
}

#[async_trait]
impl Tool for WebSearchTool {
    fn name(&self) -> &str {
        "web_search"
    }
    fn description(&self) -> &str {
        "Search the internet for current information you don't already know (docs, APIs, \
         library usage, errors, recent facts). Returns titles, URLs and snippets. Follow up \
         with web_fetch to read a promising result in full."
    }
    fn mutating(&self) -> bool {
        false
    }
    async fn run(&self, args: Value, ctx: &ToolContext) -> ToolResult {
        // Esc drops the in-flight search/fetch immediately.
        cancellable(ctx, self.exec(args, ctx)).await.unwrap_or_else(|| ToolResult::err(INTERRUPTED))
    }
}

impl WebSearchTool {
    async fn exec(&self, args: Value, ctx: &ToolContext) -> ToolResult {
        let query = str_arg(&args, &["query", "q"]).trim().to_string();
        if query.is_empty() {
            return ToolResult::err("No query given");
        }
        match self.client.search(&query).await {
            Ok(results) => {
                ctx.log(json!({"tool": "web_search", "query": query, "results": results.len()}));
                if results.is_empty() {
                    return ToolResult::ok(format!("No results for \"{query}\"."));
                }
                let text = results
                    .iter()
                    .enumerate()
                    .map(|(i, r)| format!("{}. {}\n   {}\n   {}", i + 1, r.title, r.url, r.snippet))
                    .collect::<Vec<_>>()
                    .join("\n\n");
                ToolResult::ok(text)
            }
            Err(e) => ToolResult::err(format!("web_search failed: {e}")),
        }
    }
}

struct WebFetchTool {
    client: Arc<WebClient>,
}

static HTTP_URL_RE: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"(?i)^https?://").expect("static regex"));

#[async_trait]
impl Tool for WebFetchTool {
    fn name(&self) -> &str {
        "web_fetch"
    }
    fn description(&self) -> &str {
        "Fetch a URL and return its readable text (HTML stripped). Pass `query` to get back only \
         the most relevant sentences (condensed, token-cheap) instead of the whole page. Use after \
         web_search, or directly when you know the URL (e.g. official docs)."
    }
    fn mutating(&self) -> bool {
        false
    }
    async fn run(&self, args: Value, ctx: &ToolContext) -> ToolResult {
        // Esc drops the in-flight search/fetch immediately.
        cancellable(ctx, self.exec(args, ctx)).await.unwrap_or_else(|| ToolResult::err(INTERRUPTED))
    }
}

impl WebFetchTool {
    async fn exec(&self, args: Value, ctx: &ToolContext) -> ToolResult {
        let url = str_arg(&args, &["url"]).trim().to_string();
        if !HTTP_URL_RE.is_match(&url) {
            return ToolResult::err("Provide an http(s) URL.");
        }
        let query = str_arg(&args, &["query"]).trim().to_string();
        let max = self.client.cfg.fetch_chars;
        match self.client.fetch_readable(&url).await {
            Ok(text) => {
                let out = if !query.is_empty() {
                    condense(&text, &query, max)
                } else {
                    let mut o = take_chars(&text, max).to_string();
                    if char_len(&text) > max {
                        o.push_str("\n…(truncated)");
                    }
                    o
                };
                ctx.log(json!({"tool": "web_fetch", "url": url, "bytes": char_len(&text), "condensed": !query.is_empty()}));
                ToolResult::ok(format!("Source: {url}\n\n{out}"))
            }
            Err(e) => ToolResult::err(format!("web_fetch failed: {e}")),
        }
    }
}

/// One-shot research: search + read the top pages + condense each to a compact,
/// query-focused digest. This is how the model answers current-info questions
/// (news, latest, prices…) without pulling whole pages into context — the
/// condensing is extractive (done in code), so it costs ZERO model tokens.
struct WebResearchTool {
    client: Arc<WebClient>,
}

#[async_trait]
impl Tool for WebResearchTool {
    fn name(&self) -> &str {
        "web_research"
    }
    fn description(&self) -> &str {
        "Answer a question from the live web in one call: searches, reads the top results, and returns \
         a COMPACT query-focused digest (key sentences + source URLs) rather than raw pages. Use this \
         for anything current/online (today's news, latest releases, prices, facts) — it summarises \
         large pages cheaply so it won't blow up the context."
    }
    fn mutating(&self) -> bool {
        false
    }
    async fn run(&self, args: Value, ctx: &ToolContext) -> ToolResult {
        // Esc drops the in-flight search/fetch immediately.
        cancellable(ctx, self.exec(args, ctx)).await.unwrap_or_else(|| ToolResult::err(INTERRUPTED))
    }
}

impl WebResearchTool {
    async fn exec(&self, args: Value, ctx: &ToolContext) -> ToolResult {
        let query = str_arg(&args, &["query"]).trim().to_string();
        if query.is_empty() {
            return ToolResult::err("No query given");
        }
        let pages = num_or(&args, "pages", 3.0).clamp(1.0, 5.0).floor() as usize;
        let results = match self.client.search(&query).await {
            Ok(r) => r,
            Err(e) => return ToolResult::err(format!("web_research failed: {e}")),
        };
        if results.is_empty() {
            return ToolResult::ok(format!("No web results for \"{query}\"."));
        }
        let chosen: Vec<&SearchResult> = results.iter().take(pages).collect();
        let budget_per = (3500 / chosen.len()).max(500);
        let mut parts = Vec::with_capacity(chosen.len());
        for r in chosen {
            let mut digest = r.snippet.clone();
            // A page that fails to load falls back to the search snippet.
            if let Ok(text) = self.client.fetch_readable(&r.url).await {
                let c = condense(&text, &query, budget_per);
                if char_len(&c) > char_len(&digest) {
                    digest = c;
                }
            }
            parts.push(format!("### {}\n{}\n{}", r.title, r.url, digest));
        }
        ctx.log(json!({"tool": "web_research", "query": query, "sources": parts.len()}));
        ToolResult::ok(format!(
            "Digest for \"{query}\" ({} sources, condensed):\n\n{}",
            parts.len(),
            parts.join("\n\n")
        ))
    }
}

/// Deep, multi-round research in ONE tool call. Runs several related searches,
/// reads many pages, de-dupes by URL and caps per domain, then returns a single
/// citation-indexed digest ([1], [2]…) with a Sources list. This lets a small
/// model do genuine multi-hop research without spending its outer tool-step
/// budget — the looping happens in here, extractively (no model tokens).
struct DeepResearchTool {
    client: Arc<WebClient>,
}

fn domain_of(u: &str) -> String {
    match url::Url::parse(u).ok().and_then(|p| p.host_str().map(str::to_string)) {
        Some(h) => h.strip_prefix("www.").map(str::to_string).unwrap_or(h),
        None => u.to_string(),
    }
}

#[async_trait]
impl Tool for DeepResearchTool {
    fn name(&self) -> &str {
        "deep_research"
    }
    fn description(&self) -> &str {
        "Thoroughly research a topic from the live web in ONE call: runs several related searches, reads \
         many sources, de-duplicates them, and returns a single citation-indexed report (key findings with \
         [n] markers + a Sources list). Use for hard/multi-faceted questions where one search isn't enough; \
         for a quick lookup use web_research instead."
    }
    fn mutating(&self) -> bool {
        false
    }
    async fn run(&self, args: Value, ctx: &ToolContext) -> ToolResult {
        // Esc drops the in-flight search/fetch immediately.
        cancellable(ctx, self.exec(args, ctx)).await.unwrap_or_else(|| ToolResult::err(INTERRUPTED))
    }
}

impl DeepResearchTool {
    async fn exec(&self, args: Value, ctx: &ToolContext) -> ToolResult {
        let query = str_arg(&args, &["query"]).trim().to_string();
        if query.is_empty() {
            return ToolResult::err("No query given");
        }
        let depth = num_or(&args, "depth", 2.0).clamp(1.0, 3.0);

        // Build a handful of related queries to widen coverage beyond one search.
        let modifiers = ["", "latest", "explained in detail", "pros and cons", "how it works"];
        let n_queries = ((depth + 1.0).floor() as usize).min(modifiers.len());
        let queries: Vec<String> = modifiers[..n_queries]
            .iter()
            .map(|m| if m.is_empty() { query.clone() } else { format!("{query} {m}") })
            .collect();

        // Collect results across rounds, de-duping by URL and capping per domain
        // so one site doesn't dominate.
        let mut by_url: IndexMap<String, SearchResult> = IndexMap::new();
        let mut per_domain: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
        for q in &queries {
            // One round failing is fine — keep going.
            let results = self.client.search(q).await.unwrap_or_default();
            for r in results {
                if r.url.is_empty() || by_url.contains_key(&r.url) {
                    continue;
                }
                let d = domain_of(&r.url);
                let n = per_domain.get(&d).copied().unwrap_or(0);
                if n >= 2 {
                    continue; // at most 2 pages from any single domain
                }
                per_domain.insert(d, n + 1);
                by_url.insert(r.url.clone(), r);
            }
            if by_url.len() >= 8 {
                break; // enough material gathered
            }
        }

        let chosen: Vec<SearchResult> = by_url.into_values().take(8).collect();
        if chosen.is_empty() {
            return ToolResult::ok(format!("No web results for \"{query}\"."));
        }

        let budget_per = (4000 / chosen.len()).max(400);
        let mut findings = Vec::with_capacity(chosen.len());
        let mut sources = Vec::with_capacity(chosen.len());
        for (idx, r) in chosen.iter().enumerate() {
            let n = idx + 1;
            let mut digest = r.snippet.clone();
            // A failed page keeps the search snippet.
            if let Ok(text) = self.client.fetch_readable(&r.url).await {
                let c = condense(&text, &query, budget_per);
                if char_len(&c) > char_len(&digest) {
                    digest = c;
                }
            }
            findings.push(format!("[{n}] {}\n{digest}", r.title));
            sources.push(format!("[{n}] {}", r.url));
        }
        ctx.log(json!({"tool": "deep_research", "query": query, "rounds": queries.len(), "sources": chosen.len()}));
        ToolResult::ok(format!(
            "Research digest for \"{query}\" ({} sources across {} searches):\n\n{}\n\n── Sources ──\n{}",
            chosen.len(),
            queries.len(),
            findings.join("\n\n"),
            sources.join("\n")
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ssrf_guard_cases() {
        assert!(assert_public_url("https://example.com/x").is_ok());
        assert!(assert_public_url("http://8.8.8.8/").is_ok());
        for bad in [
            "http://localhost:8080/admin",
            "http://127.0.0.1/",
            "http://10.0.0.5/",
            "http://172.16.0.1/",
            "http://192.168.1.1/",
            "http://169.254.1.1/",
            "http://0.0.0.0/",
            "http://[::1]/",
            "http://[fe80::1]/",
            "http://printer.local/",
            "http://metadata.google.internal/",
            "http://2130706433/", // 127.0.0.1 in decimal form
        ] {
            let e = assert_public_url(bad).unwrap_err();
            assert!(e.contains("SSRF guard"), "{bad}: {e}");
        }
        assert!(assert_public_url("file:///etc/passwd").unwrap_err().contains("only http(s)"));
        assert!(assert_public_url("not a url").unwrap_err().starts_with("Invalid URL"));
    }

    #[test]
    fn html_to_text_strips_and_decodes() {
        let html = "<html><head><style>p{}</style><script>var x=1;</script></head>\
                    <body><h1>Title &amp; more</h1><p>Hello&nbsp;<b>world</b> &#39;q&#x27; &lt;3</p>\
                    <noscript>enable js</noscript></body></html>";
        let t = html_to_text(html);
        assert!(t.contains("Title & more"), "{t}");
        assert!(t.contains("Hello world 'q' <3"), "{t}");
        assert!(!t.contains("var x") && !t.contains("enable js") && !t.contains("p{}"));
        assert_eq!(decode_entities("&#12345678901;&bogus;"), "&#12345678901;&bogus;");
    }

    #[test]
    fn ddg_parsers() {
        let html = r#"<div><a rel="nofollow" class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Fa&amp;rut=x">Example <b>A</b></a>
            <a class="result__snippet" href="x">Snippet <b>one</b></a></div>
            <div><a rel="nofollow" class="result__a" href="https://b.example.org/">B</a>
            <a class="result__snippet" href="y">Snippet two</a></div>"#;
        let r = parse_ddg_html(html, 5);
        assert_eq!(r.len(), 2);
        assert_eq!(r[0].url, "https://example.com/a");
        assert_eq!(r[0].title, "Example A");
        assert_eq!(r[0].snippet, "Snippet one");
        assert_eq!(r[1].url, "https://b.example.org/");
        assert_eq!(parse_ddg_html(html, 1).len(), 1);

        let lite = r#"<tr><td><a rel="nofollow" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Flite.example%2F" class='x'>no</a></td></tr>
            <tr><td><a rel="nofollow" class="result-link" href="https://lite.example/">Lite hit</a></td></tr>
            <tr><td class="result-snippet">Lite snippet</td></tr>"#;
        let r = parse_ddg_lite(lite, 5);
        assert_eq!(r.len(), 1);
        assert_eq!(r[0].title, "Lite hit");
        assert_eq!(r[0].snippet, "Lite snippet");
    }

    #[test]
    fn sentence_split() {
        let parts = split_sentences("One thing. Two things! three? Four \"five\". 6 six");
        assert_eq!(parts, vec!["One thing.", "Two things! three?", "Four \"five\".", "6 six"]);
    }

    #[test]
    fn provider_parse() {
        assert_eq!(SearchProvider::parse("SearXNG"), SearchProvider::Searxng);
        assert_eq!(SearchProvider::parse("whatever"), SearchProvider::DuckDuckGo);
        assert_eq!(SearchProvider::Brave.as_str(), "brave");
    }

    /* ── ported from tests/webTools.test.ts ── */

    use crate::testutil::{serve, MockServer, Recorded, Reply};
    use qo_core::CancellationToken;

    fn test_cfg() -> WebConfig {
        WebConfig { max_results: 5, fetch_chars: 4000, fetch_timeout_ms: 5000, ..WebConfig::default() }
    }

    fn tool_named(tools: &[qo_core::tool::SharedTool], name: &str) -> qo_core::tool::SharedTool {
        tools.iter().find(|t| t.name() == name).cloned().expect("tool")
    }

    #[test]
    fn exposes_the_four_web_tools() {
        let tools = create_web_tools(test_cfg());
        let mut names: Vec<&str> = tools.iter().map(|t| t.name()).collect();
        names.sort_unstable();
        assert_eq!(names, ["deep_research", "web_fetch", "web_research", "web_search"]);
        assert!(tools.iter().all(|t| !t.mutating()));
    }

    #[tokio::test]
    async fn web_fetch_refuses_loopback_private_and_non_http() {
        let tools = create_web_tools(test_cfg());
        let fetch = tool_named(&tools, "web_fetch");
        let ctx = ToolContext::new(".");
        let r = fetch.run(json!({"url": "http://localhost:8080/admin"}), &ctx).await;
        assert!(!r.ok);
        assert!(Regex::new(r"(?i)private|loopback|SSRF").unwrap().is_match(&r.output), "{}", r.output);
        for u in ["http://127.0.0.1/", "http://10.0.0.5/", "http://192.168.1.1/", "http://169.254.1.1/"] {
            assert!(!fetch.run(json!({"url": u}), &ctx).await.ok, "{u}");
        }
        let r = fetch.run(json!({"url": "file:///etc/passwd"}), &ctx).await;
        assert_eq!((r.ok, r.output.as_str()), (false, "Provide an http(s) URL."));
    }

    #[test]
    fn condense_prefers_sentences_with_query_terms() {
        let text = "The weather today is mild and pleasant across the region. \
                    The new graphics card delivers a massive performance uplift for gaming. \
                    Local traffic was lighter than usual this morning.";
        assert!(condense(text, "graphics card performance", 200).contains("graphics card"));
    }

    #[test]
    fn condense_falls_back_to_leading_sentences() {
        let text = "First sentence is reasonably long and informative here. Second sentence follows along too.";
        assert!(condense(text, "", 200).contains("First sentence"));
        // Nothing sentence-shaped → plain prefix.
        assert_eq!(condense("short", "q", 3), "sho");
    }

    /* ── headless search / fetch against a loopback mock ── */

    fn ddg_page(host: &str, n: usize) -> String {
        (1..=n)
            .map(|i| {
                format!(
                    r##"<div class="result"><a rel="nofollow" class="result__a" href="//duckduckgo.com/l/?uddg={}&amp;rut=1">Result {i}</a>
                       <a class="result__snippet" href="#">Snippet number {i}</a></div>"##,
                    urlencoding::encode(&format!("http://{host}/page{i}"))
                )
            })
            .collect()
    }

    const ARTICLE: &str = "<html><head><title>t</title><script>evil()</script></head><body>\
        <h1>Rust async runtimes</h1>\
        <p>Tokio is the most widely used async runtime for the Rust language today.</p>\
        <p>Unrelated paragraph about gardening tips and the best soil for tomatoes.</p>\
        <p>Cancellation in Tokio works by dropping the future that is being polled.</p></body></html>";

    /// DDG HTML → `/ddg`, Lite → `/lite`, Tavily → `/tavily`; pages → `/page<n>`.
    async fn search_mock(ddg_results: usize, lite_results: usize) -> MockServer {
        serve(Arc::new(move |req: &Recorded| {
            let host = req.header("host").unwrap_or("").to_string();
            match req.path.as_str() {
                "/ddg" => Reply::html(&ddg_page(&host, ddg_results)),
                "/lite" => {
                    let rows: String = (1..=lite_results)
                        .map(|i| format!(r#"<tr><td><a rel="nofollow" class="result-link" href="http://{host}/page{i}">Lite {i}</a></td></tr><tr><td class="result-snippet">Lite snippet {i}</td></tr>"#))
                        .collect();
                    Reply::html(&rows)
                }
                "/tavily" => Reply::json(500, json!({"error": "down"})),
                p if p.starts_with("/page") => Reply::html(ARTICLE),
                "/slow" => Reply::html(ARTICLE).delayed(3000),
                "/big" => Reply::text(200, "text/plain", &"x".repeat(MAX_PAGE_BYTES + 100_000)),
                "/moved" => Reply::text(302, "text/plain", "").header("Location", "/page1"),
                _ => Reply::text(404, "text/plain", "nope"),
            }
        }))
        .await
    }

    fn mock_cfg(server: &MockServer) -> WebConfig {
        WebConfig {
            endpoints: WebEndpoints {
                ddg_html: format!("{}/ddg", server.base),
                ddg_lite: format!("{}/lite", server.base),
                tavily: format!("{}/tavily", server.base),
                brave: format!("{}/brave", server.base),
            },
            allow_private_hosts: true,
            ..test_cfg()
        }
    }

    #[tokio::test]
    async fn search_formats_results_and_falls_back_through_the_ladder() {
        let server = search_mock(3, 2).await;
        let tools = create_web_tools(mock_cfg(&server));
        let ctx = ToolContext::new(".");
        let r = tool_named(&tools, "web_search").run(json!({"q": "rust"}), &ctx).await;
        assert!(r.ok, "{}", r.output);
        assert!(r.output.starts_with("1. Result 1\n   http://127.0.0.1:"), "{}", r.output);
        assert!(r.output.contains("/page1\n   Snippet number 1\n\n2. Result 2"));
        let form = server.requests().into_iter().find(|q| q.path == "/ddg").unwrap();
        assert_eq!(form.body, "q=rust");
        assert!(form.header("user-agent").unwrap_or("").starts_with("Mozilla/5.0"));

        // DDG HTML empty → DDG Lite answers.
        let server = search_mock(0, 2).await;
        let client = WebClient::new(mock_cfg(&server));
        let hits = client.search("rust").await.unwrap();
        assert_eq!(hits.iter().map(|h| h.title.as_str()).collect::<Vec<_>>(), ["Lite 1", "Lite 2"]);
        assert_eq!(hits[1].snippet, "Lite snippet 2");

        // A failing keyed primary (Tavily HTTP 500) falls back to the keyless engines.
        let server = search_mock(1, 0).await;
        let client = WebClient::new(WebConfig { provider: SearchProvider::Tavily, api_key: "k".into(), ..mock_cfg(&server) });
        assert_eq!(client.search("rust").await.unwrap()[0].title, "Result 1");

        // Every engine empty → no results, not an error.
        let server = search_mock(0, 0).await;
        let r = tool_named(&create_web_tools(mock_cfg(&server)), "web_search").run(json!({"query": "zzz"}), &ctx).await;
        assert_eq!((r.ok, r.output.as_str()), (true, "No results for \"zzz\"."));
    }

    #[tokio::test]
    async fn fetch_extracts_text_condenses_truncates_and_caches() {
        let server = search_mock(0, 0).await;
        let tools = create_web_tools(WebConfig { fetch_chars: 120, ..mock_cfg(&server) });
        let fetch = tool_named(&tools, "web_fetch");
        let ctx = ToolContext::new(".");
        let url = format!("{}/page1", server.base);

        let r = fetch.run(json!({"url": url}), &ctx).await;
        assert!(r.ok, "{}", r.output);
        assert!(r.output.starts_with(&format!("Source: {url}\n\nt Rust async runtimes\n Tokio is")), "{:?}", r.output);
        assert!(r.output.ends_with("\n…(truncated)"));
        assert!(!r.output.contains("evil()"));

        let r = fetch.run(json!({"url": url, "query": "tokio cancellation"}), &ctx).await;
        assert!(r.output.contains("Cancellation in Tokio"), "{}", r.output);
        assert!(!r.output.contains("gardening"));
        // Second and third fetch of the same URL came from the page cache.
        assert_eq!(server.requests().iter().filter(|q| q.path == "/page1").count(), 1);

        // Redirects are followed (and re-checked by the SSRF policy when it is on).
        let r = fetch.run(json!({"url": format!("{}/moved", server.base)}), &ctx).await;
        assert!(r.output.contains("Rust async runtimes"), "{}", r.output);

        let r = fetch.run(json!({"url": format!("{}/missing", server.base)}), &ctx).await;
        assert_eq!((r.ok, r.output.as_str()), (false, "web_fetch failed: HTTP 404"));

        // Bodies are capped at MAX_PAGE_BYTES.
        let client = WebClient::new(mock_cfg(&server));
        let big = client.fetch_readable(&format!("{}/big", server.base)).await.unwrap();
        assert_eq!(big.len(), MAX_PAGE_BYTES);
    }

    #[tokio::test]
    async fn fetch_deadline_and_esc_interrupt() {
        let server = search_mock(0, 0).await;
        let slow = format!("{}/slow", server.base);
        let tools = create_web_tools(WebConfig { fetch_timeout_ms: 300, ..mock_cfg(&server) });
        let r = tool_named(&tools, "web_fetch").run(json!({"url": slow}), &ToolContext::new(".")).await;
        assert_eq!((r.ok, r.output.as_str()), (false, "web_fetch failed: fetch timed out after 300ms"));

        let tools = create_web_tools(mock_cfg(&server));
        let token = CancellationToken::new();
        let mut ctx = ToolContext::new(".");
        ctx.cancel = Some(token.clone());
        let t2 = token.clone();
        tokio::spawn(async move {
            tokio::time::sleep(Duration::from_millis(150)).await;
            t2.cancel();
        });
        let start = std::time::Instant::now();
        let r = tool_named(&tools, "web_fetch").run(json!({"url": slow}), &ctx).await;
        assert_eq!((r.ok, r.output.as_str()), (false, "interrupted"));
        assert!(start.elapsed() < Duration::from_millis(2000));
        // An already-cancelled turn doesn't even start the request.
        let r = tool_named(&tools, "web_search").run(json!({"query": "x"}), &ctx).await;
        assert_eq!(r.output, "interrupted");
    }

    #[tokio::test]
    async fn research_tools_digest_pages_with_citations() {
        let server = search_mock(4, 0).await;
        let tools = create_web_tools(mock_cfg(&server));
        let ctx = ToolContext::new(".");

        let r = tool_named(&tools, "web_research").run(json!({"query": "tokio runtime", "pages": 2}), &ctx).await;
        assert!(r.output.starts_with("Digest for \"tokio runtime\" (2 sources, condensed):\n\n### Result 1\nhttp://"), "{}", r.output);
        assert!(r.output.contains("Tokio is the most widely used async runtime"));

        // deep_research de-dupes by URL and keeps at most 2 pages per domain
        // (every mock page lives on 127.0.0.1).
        let r = tool_named(&tools, "deep_research").run(json!({"query": "tokio", "depth": 1}), &ctx).await;
        assert!(r.output.starts_with("Research digest for \"tokio\" (2 sources across 2 searches):\n\n[1] Result 1\n"), "{}", r.output);
        assert!(r.output.contains("\n\n── Sources ──\n[1] http://127.0.0.1:"));
        assert!(r.output.contains("\n[2] http://127.0.0.1:"));
    }
}
