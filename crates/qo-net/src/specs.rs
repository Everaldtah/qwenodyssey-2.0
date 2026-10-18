//! JSON-schema tool specs advertised to the model for this crate's tools
//! (`WEB_TOOL_SPECS`, `GITHUB_TOOL_SPECS`, `KNOWLEDGE_TOOL_SPECS` in the TS
//! `cli/chatTools.ts`). Names match the registered tool names exactly so calls
//! dispatch directly. MCP specs come from the servers ([`crate::mcp::McpSetup::specs`]).

use qo_core::ToolSpec;
use serde_json::{json, Value};

fn str_prop(description: &str) -> Value {
    json!({"type": "string", "description": description})
}

fn spec(name: &str, description: &str, parameters: Value) -> ToolSpec {
    ToolSpec { name: name.into(), description: description.into(), parameters }
}

const REPO_ARG: &str = "Repo as \"owner/name\" or just \"name\".";

/// Internet tools — included when `web.enabled`.
pub fn web_tool_specs() -> Vec<ToolSpec> {
    vec![
        spec(
            "web_search",
            "Search the internet for information you don't already know or that may be current \
             (library docs, APIs, error messages, recent facts). Returns titles, URLs and snippets.",
            json!({"type": "object", "properties": {"query": str_prop("What to search for.")}, "required": ["query"]}),
        ),
        spec(
            "web_fetch",
            "Fetch a URL and return its readable text (HTML stripped). Pass `query` to get only the most \
             relevant sentences (condensed, token-cheap) instead of the whole page.",
            json!({
                "type": "object",
                "properties": {
                    "url": str_prop("The http(s) URL to fetch."),
                    "query": str_prop("Optional: condense the page to sentences relevant to this, to save tokens."),
                },
                "required": ["url"],
            }),
        ),
        spec(
            "web_research",
            "Answer a question from the LIVE web in one call: searches, reads the top results, and returns a \
             compact query-focused digest (key sentences + sources) instead of raw pages. Use for anything \
             current/online — today's news, latest releases, prices, recent facts. Summarises large pages \
             cheaply so it won't overflow the context.",
            json!({
                "type": "object",
                "properties": {
                    "query": str_prop("What to research, e.g. \"today's top news headlines\"."),
                    "pages": {"type": "integer", "description": "How many top results to read & condense (default 3, max 5)."},
                },
                "required": ["query"],
            }),
        ),
        spec(
            "deep_research",
            "Thoroughly research a topic from the LIVE web in ONE call: runs several related searches, reads \
             many sources, de-duplicates them, and returns a single citation-indexed report (key findings with \
             [n] markers + a Sources list). Use for hard/multi-faceted questions where one search isn't enough; \
             for a quick lookup use web_research instead.",
            json!({
                "type": "object",
                "properties": {
                    "query": str_prop("The topic/question to research thoroughly."),
                    "depth": {"type": "integer", "description": "How many related searches to run (1-3, default 2). More = broader."},
                },
                "required": ["query"],
            }),
        ),
    ]
}

/// GitHub tools — included when `github.enabled` and a login/token is available.
pub fn github_tool_specs() -> Vec<ToolSpec> {
    vec![
        spec(
            "github_whoami",
            "Show which GitHub account the agent is acting as (and its scopes). Use to confirm GitHub access.",
            json!({"type": "object", "properties": {}}),
        ),
        spec(
            "github_list_repos",
            "List repositories on the user's GitHub account (most recently pushed first). Use to discover repos.",
            json!({
                "type": "object",
                "properties": {"limit": {"type": "integer", "description": "Max repos to list (default 30, max 100)."}},
            }),
        ),
        spec(
            "github_get_file",
            "Read a file (or list a directory) from a GitHub repo to review code that isn't checked out locally.",
            json!({
                "type": "object",
                "properties": {
                    "repo": str_prop("Repo as \"owner/name\" or just \"name\" (defaults owner to your account)."),
                    "path": str_prop("Path within the repo. Empty/'/' lists the root directory."),
                    "ref": str_prop("Optional branch, tag, or commit SHA."),
                },
                "required": ["repo", "path"],
            }),
        ),
        spec(
            "github_put_file",
            "Create or update (upload/commit) a file in a GitHub repo. Writes code straight to GitHub; the \
             existing file's sha is fetched automatically when updating.",
            json!({
                "type": "object",
                "properties": {
                    "repo": str_prop(REPO_ARG),
                    "path": str_prop("Path within the repo to write."),
                    "content": str_prop("Full new file contents (UTF-8)."),
                    "message": str_prop("Commit message."),
                    "branch": str_prop("Optional branch to commit to (default the repo's default branch)."),
                },
                "required": ["repo", "path", "content"],
            }),
        ),
        spec(
            "github_create_repo",
            "Create a new repository on the user's account (private + auto-initialised by default).",
            json!({
                "type": "object",
                "properties": {
                    "name": str_prop("Repository name."),
                    "description": str_prop("Optional description."),
                    "private": {"type": "boolean", "description": "Private repo (default true)."},
                    "auto_init": {"type": "boolean", "description": "Initialise with a README so it's immediately writable (default true)."},
                },
                "required": ["name"],
            }),
        ),
        spec(
            "github_list_prs",
            "List pull requests in a repo (open by default).",
            json!({
                "type": "object",
                "properties": {
                    "repo": str_prop(REPO_ARG),
                    "state": {"type": "string", "enum": ["open", "closed", "all"], "description": "Default 'open'."},
                },
                "required": ["repo"],
            }),
        ),
        spec(
            "github_get_pr",
            "Fetch a pull request's details AND full unified diff so you can review the code change.",
            json!({
                "type": "object",
                "properties": {
                    "repo": str_prop(REPO_ARG),
                    "number": {"type": "integer", "description": "PR number."},
                },
                "required": ["repo", "number"],
            }),
        ),
        spec(
            "github_create_pr",
            "Open a pull request from a head branch into a base branch.",
            json!({
                "type": "object",
                "properties": {
                    "repo": str_prop(REPO_ARG),
                    "title": str_prop("PR title."),
                    "head": str_prop("Head branch (the branch with your changes)."),
                    "base": str_prop("Base branch to merge into (default 'main')."),
                    "body": str_prop("Optional PR description."),
                    "draft": {"type": "boolean", "description": "Open as a draft (default false)."},
                },
                "required": ["repo", "title", "head"],
            }),
        ),
        spec(
            "github_review_pr",
            "Submit a review on a PR after reading it with github_get_pr. Pick an event and explain it.",
            json!({
                "type": "object",
                "properties": {
                    "repo": str_prop(REPO_ARG),
                    "number": {"type": "integer", "description": "PR number."},
                    "event": {"type": "string", "enum": ["APPROVE", "REQUEST_CHANGES", "COMMENT"], "description": "Review verdict."},
                    "body": str_prop("Review comment / summary."),
                },
                "required": ["repo", "number", "event", "body"],
            }),
        ),
        spec(
            "github_create_issue",
            "Open an issue in a repo (e.g. file a bug found while reviewing).",
            json!({
                "type": "object",
                "properties": {
                    "repo": str_prop(REPO_ARG),
                    "title": str_prop("Issue title."),
                    "body": str_prop("Optional issue body."),
                    "labels": {"type": "array", "items": {"type": "string"}, "description": "Optional labels."},
                },
                "required": ["repo", "title"],
            }),
        ),
    ]
}

/// Long-term memory tools — included when `knowledge.enabled`.
pub fn knowledge_tool_specs() -> Vec<ToolSpec> {
    vec![
        spec(
            "knowledge_search",
            "Search your long-term memory for what you've already learned. Check here BEFORE the web.",
            json!({
                "type": "object",
                "properties": {"query": str_prop("Topic, concept, codebase, or API to recall.")},
                "required": ["query"],
            }),
        ),
        spec(
            "knowledge_read",
            "Read a full note from memory by its slug.",
            json!({
                "type": "object",
                "properties": {"slug": str_prop("The note slug, e.g. from knowledge_search results.")},
                "required": ["slug"],
            }),
        ),
        spec(
            "knowledge_save",
            "Save or update a note in long-term memory so you remember it permanently. Use whenever \
             you learn something durable (how code works, an API, a fix, a looked-up fact). Reusing a \
             title updates that note. Cite source URLs when available.",
            json!({
                "type": "object",
                "properties": {
                    "title": str_prop("Short, specific note title (also the update key)."),
                    "content": str_prop("Distilled, durable knowledge in markdown. Use [[wikilinks]] to relate notes."),
                    "tags": {"type": "array", "items": {"type": "string"}, "description": "Optional tags."},
                    "sources": {"type": "array", "items": {"type": "string"}, "description": "Optional source URLs."},
                },
                "required": ["title", "content"],
            }),
        ),
        spec(
            "knowledge_list",
            "List the titles of everything in your knowledge vault.",
            json!({"type": "object", "properties": {}}),
        ),
        spec(
            "record_lesson",
            "Record a durable LESSON learned from a mistake or a user correction so you don't repeat it. \
             Lessons are auto-recalled in future turns. Use after a tool/command failed and you found the \
             right approach, or when the user corrects you.",
            json!({
                "type": "object",
                "properties": {
                    "lesson": str_prop("The reusable rule/takeaway, 1-2 sentences (name the correct tool/command/approach)."),
                    "title": str_prop("Optional short title."),
                },
                "required": ["lesson"],
            }),
        ),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn specs_are_well_formed() {
        for s in web_tool_specs().iter().chain(&github_tool_specs()).chain(&knowledge_tool_specs()) {
            assert_eq!(s.parameters["type"], "object", "{}", s.name);
            assert!(s.parameters["properties"].is_object(), "{}", s.name);
            for r in s.parameters.get("required").and_then(Value::as_array).into_iter().flatten() {
                let key = r.as_str().unwrap();
                assert!(s.parameters["properties"].get(key).is_some(), "{} requires unknown {key}", s.name);
            }
        }
        let web: Vec<String> = web_tool_specs().into_iter().map(|s| s.name).collect();
        assert_eq!(web, ["web_search", "web_fetch", "web_research", "deep_research"]);
    }
}
