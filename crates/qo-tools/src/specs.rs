//! JSON-schema tool specs advertised to the model during chat (ported from
//! src/cli/chatTools.ts — the local-tool parts; web/knowledge/GitHub specs live
//! in qo-net and the swarm spec in qo-swarm), plus [`compact_tool_specs`].

use std::collections::BTreeSet;

use qo_core::ToolSpec;
use serde_json::{json, Map, Value};

fn str_prop(description: &str) -> Value {
    json!({ "type": "string", "description": description })
}

fn spec(name: &str, description: &str, parameters: Value) -> ToolSpec {
    ToolSpec { name: name.to_string(), description: description.to_string(), parameters }
}

/// The subset of registry tools exposed in interactive chat. Names MUST match
/// the registry tool names so calls dispatch directly. Includes `update_plan`
/// (the plan tool is always registered in chat).
pub fn chat_tool_specs() -> Vec<ToolSpec> {
    vec![
        spec(
            "run_shell",
            "Execute ONE shell command in the project directory and return its combined stdout/stderr. \
             Use this for any real action: checking the OS, network/wifi status, running tests, git, \
             listing processes, installing packages, etc. On Windows the shell is PowerShell. Each call \
             is a FRESH shell (cwd/env do NOT persist between calls — chain with ';' or use shell_session). \
             It blocks until the command finishes or times out (default ~4 min). For a command you expect \
             to be slow, pass a larger `timeout_ms`. For a genuinely long-running or interactive job (cloning \
             a large repo, install, build, a server, a REPL), prefer shell_session — it runs persistently and \
             you poll it with shell_session_read instead of blocking. Be efficient: fetch/clone ONCE; never \
             loop downloading files one-by-one when a single clone, tarball, or API call would do.",
            json!({
                "type": "object",
                "properties": {
                    "command": str_prop("The exact command line to run."),
                    "timeout_ms": { "type": "integer", "description": "Max wall-clock time before abort (default ~240000, max 600000). Raise it for slow commands." }
                },
                "required": ["command"]
            }),
        ),
        spec(
            "shell_help",
            "Look up the EXACT verified Windows PowerShell command for a task BEFORE running it when \
             you're unsure (login/event logs, processes, services, network, users, disk, files, hardware, \
             performance, firewall, scheduled tasks, installed software). Returns ready-to-run commands; \
             then call run_shell with one. Saves you from guessing cmdlet names/flags.",
            json!({
                "type": "object",
                "properties": { "query": str_prop("Task or topic, e.g. \"login logs last week\", \"top memory processes\", \"wifi\".") }
            }),
        ),
        spec(
            "read_file",
            "Read a file's contents. The path may be relative to the project dir OR an absolute path \
             like \"C:\\Projects\\Overstory\\README.md\". For large files, pass `offset` (1-based start \
             line) and `limit` (line count) to read one page at a time; the result tells you the next offset.",
            json!({
                "type": "object",
                "properties": {
                    "path": str_prop("File path — relative to the project, or absolute."),
                    "offset": { "type": "integer", "description": "1-based line to start at (optional; for paging large files)." },
                    "limit": { "type": "integer", "description": "Max lines to return (optional; default pages ~800)." }
                },
                "required": ["path"]
            }),
        ),
        spec(
            "write_file",
            "Create a file, or replace an existing file's ENTIRE contents. To change part of an existing \
             file use apply_edit instead. Replacing a file you have not read this session is refused \
             unless overwrite=true.",
            json!({
                "type": "object",
                "properties": {
                    "path": str_prop("Path to the file, relative to the project root."),
                    "content": str_prop("Full new contents of the file."),
                    "overwrite": { "type": "boolean", "description": "Set true to replace an existing file you have not read." }
                },
                "required": ["path", "content"]
            }),
        ),
        spec(
            "list_files",
            "List files under a directory (recursively) matching an optional glob. Use `path` to target \
             a specific directory, including an absolute one outside the project.",
            json!({
                "type": "object",
                "properties": {
                    "path": str_prop("Directory to list — relative or absolute (e.g. \"C:\\Projects\\Overstory\"). Optional; defaults to the project dir."),
                    "pattern": str_prop("Glob like \"src/**/*.ts\". Optional.")
                }
            }),
        ),
        spec(
            "tree",
            "Show a compact, depth-limited directory tree. Use `path` to point at any directory — \
             including an absolute path the user names, e.g. \"C:\\Projects\\Overstory\".",
            json!({
                "type": "object",
                "properties": {
                    "path": str_prop("Directory to show — relative or absolute. Optional; defaults to the project dir."),
                    "depth": { "type": "integer", "description": "Max depth (default 2)." }
                }
            }),
        ),
        spec(
            "grep",
            "Search file contents for a regex pattern. Use `path` to search a specific (or absolute) directory.",
            json!({
                "type": "object",
                "properties": {
                    "pattern": str_prop("Regular expression to search for."),
                    "path": str_prop("Directory to search — relative or absolute. Optional; defaults to the project dir."),
                    "glob": str_prop("Optional glob to limit files, e.g. \"**/*.ts\".")
                },
                "required": ["pattern"]
            }),
        ),
        spec("git_status", "Show the git working-tree status.", json!({ "type": "object", "properties": {} })),
        spec(
            "git_diff",
            "Show the current git diff, optionally for one path.",
            json!({ "type": "object", "properties": { "path": str_prop("Optional path to scope the diff.") } }),
        ),
        plan_tool_spec(),
    ]
}

/// The `update_plan` spec (part of [`chat_tool_specs`]).
pub fn plan_tool_spec() -> ToolSpec {
    spec(
        "update_plan",
        "Lay out and track a step-by-step plan for a MULTI-STEP task. Call it once at the start to \
         list the steps, then again to update statuses as you progress. Always pass the FULL plan \
         (the list is replaced each time). Keeps you on track over long tool chains.",
        json!({
            "type": "object",
            "properties": {
                "plan": {
                    "type": "array",
                    "description": "The full ordered list of steps.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "step": str_prop("Short description of the step."),
                            "status": { "type": "string", "enum": ["pending", "in_progress", "done"], "description": "Step status." }
                        },
                        "required": ["step", "status"]
                    }
                }
            },
            "required": ["plan"]
        }),
    )
}

/// Code-navigation + reliable-edit tools. These are the grounding layer that
/// lets small models behave like they have an IDE: locate definitions, see a
/// file's outline, find references, read one symbol, and apply precise
/// SEARCH/REPLACE edits instead of error-prone whole-file rewrites.
pub fn code_nav_tool_specs() -> Vec<ToolSpec> {
    vec![
        spec(
            "find_symbol",
            "Find where a function/class/interface/type/const is DEFINED (file:line + signature). \
             Call this before editing so you change the right place.",
            json!({
                "type": "object",
                "properties": {
                    "name": str_prop("Exact symbol name to locate."),
                    "kind": str_prop("Optional filter: function|class|method|interface|type|const|enum|struct|trait.")
                },
                "required": ["name"]
            }),
        ),
        spec(
            "search_symbols",
            "Fuzzy-search symbol names across the project when you don't know the exact name.",
            json!({
                "type": "object",
                "properties": { "query": str_prop("Substring to search for."), "limit": { "type": "number", "description": "Max results." } },
                "required": ["query"]
            }),
        ),
        spec(
            "outline_file",
            "List every definition in a file (functions/classes/methods/types) with line numbers — a \
             cheap structural map. Read this before reading a whole large file.",
            json!({
                "type": "object",
                "properties": { "path": str_prop("File path relative to the project root.") },
                "required": ["path"]
            }),
        ),
        spec(
            "find_references",
            "Find every place a name is USED across the codebase (definitions flagged). Use before \
             renaming or changing a signature to understand the blast radius.",
            json!({
                "type": "object",
                "properties": { "name": str_prop("Symbol name to find uses of."), "limit": { "type": "number", "description": "Max results." } },
                "required": ["name"]
            }),
        ),
        spec(
            "read_symbol",
            "Read the body of ONE function/class/method by name instead of the whole file — keeps \
             context small. Optionally disambiguate with 'file'.",
            json!({
                "type": "object",
                "properties": {
                    "name": str_prop("Symbol name to read."),
                    "file": str_prop("Optional file path to disambiguate."),
                    "max_lines": { "type": "number", "description": "Max lines to return (default 200)." }
                },
                "required": ["name"]
            }),
        ),
        spec(
            "apply_edit",
            "Apply a PRECISE edit using a SEARCH/REPLACE block — the reliable way to edit existing files \
             (preferred over write_file for changes). Copy the 'search' text VERBATIM from the file. \
             Leave 'search' empty to create a new file or append. Whitespace-tolerant with auto-repair.",
            json!({
                "type": "object",
                "properties": {
                    "path": str_prop("File path relative to the project root."),
                    "search": str_prop("Exact text to find (verbatim from the file). Empty = create/append."),
                    "replace": str_prop("Replacement text.")
                },
                "required": ["path", "replace"]
            }),
        ),
    ]
}

/// Names of [`chat_tool_specs`] + [`code_nav_tool_specs`].
pub fn chat_tool_names() -> BTreeSet<String> {
    chat_tool_specs().into_iter().chain(code_nav_tool_specs()).map(|s| s.name).collect()
}

/// First 1–2 sentences of `text`, at most `cap` chars (sentence cut when one
/// falls after char 40, else a hard cut with "…").
fn first_sentences(text: &str, cap: usize) -> String {
    let t = text.split_whitespace().collect::<Vec<_>>().join(" ");
    let chars: Vec<char> = t.chars().collect();
    if chars.len() <= cap {
        return t;
    }
    let slice: String = chars[..cap].iter().collect();
    // Cut at a sentence boundary inside the cap when one exists.
    match slice.rfind(". ") {
        Some(dot_byte) if slice[..dot_byte].chars().count() > 40 => slice[..dot_byte + 1].to_string(),
        _ => format!("{}…", slice.trim_end()),
    }
}

/// Compact variants of tool specs for SMALL LOCAL MODELS. The full descriptions
/// are written for frontier models and run to whole paragraphs; with 35+ tools
/// they cost thousands of context tokens on EVERY request — a huge share of a 7B
/// model's 8–16K window — and long specs measurably *hurt* small-model tool
/// selection (they attend to the prose instead of the schema). This trims each
/// description to its first 1–2 sentences (≤ `max_chars`) and shortens parameter
/// descriptions to 90 chars, preserving names/types/required exactly so dispatch
/// is unaffected. Usage guidance still reaches the model via the system prompt,
/// which compaction never touches. The originals are not modified.
pub fn compact_tool_specs(specs: &[ToolSpec], max_chars: usize) -> Vec<ToolSpec> {
    specs
        .iter()
        .map(|s| {
            let mut params = match &s.parameters {
                Value::Object(m) => m.clone(),
                _ => Map::new(),
            };
            let props: Map<String, Value> = params
                .get("properties")
                .and_then(Value::as_object)
                .map(|p| {
                    p.iter()
                        .map(|(k, v)| {
                            let v = match v.get("description").and_then(Value::as_str) {
                                Some(d) => {
                                    let mut v2 = v.clone();
                                    v2["description"] = Value::String(first_sentences(d, 90));
                                    v2
                                }
                                None => v.clone(),
                            };
                            (k.clone(), v)
                        })
                        .collect()
                })
                .unwrap_or_default();
            params.insert("properties".to_string(), Value::Object(props));
            ToolSpec { name: s.name.clone(), description: first_sentences(&s.description, max_chars), parameters: Value::Object(params) }
        })
        .collect()
}

/// [`compact_tool_specs`] with the TS default cap (180 chars).
pub fn compact_tool_specs_default(specs: &[ToolSpec]) -> Vec<ToolSpec> {
    compact_tool_specs(specs, 180)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn verbose() -> Vec<ToolSpec> {
        vec![spec(
            "run_shell",
            "Execute ONE shell command in the project directory and return its combined stdout/stderr. \
             Use this for any real action: checking the OS, network/wifi status, running tests, git, \
             listing processes, installing packages, etc. On Windows the shell is PowerShell. Each call \
             is a FRESH shell (cwd/env do NOT persist between calls). It blocks until the command finishes \
             or times out. For slow commands pass a larger timeout_ms. Never loop downloading files.",
            json!({
                "type": "object",
                "properties": {
                    "command": { "type": "string", "description": "The exact command line to run." },
                    "timeout_ms": {
                        "type": "integer",
                        "description": "Max wall-clock time before abort (default ~240000, max 600000). Raise it for slow commands like builds or installs."
                    }
                },
                "required": ["command"]
            }),
        )]
    }

    // smallModelOptims.test.ts: compactToolSpecs
    #[test]
    fn shortens_but_preserves_names_types_required() {
        let c = compact_tool_specs_default(&verbose());
        assert_eq!(c[0].name, "run_shell");
        assert!(c[0].description.chars().count() <= 181);
        assert!(c[0].description.starts_with("Execute ONE shell command"));
        assert_eq!(c[0].parameters["required"], json!(["command"]));
        assert_eq!(c[0].parameters["properties"]["command"]["type"], json!("string"));
        assert_eq!(c[0].parameters["properties"]["timeout_ms"]["type"], json!("integer"));
        assert!(c[0].parameters["properties"]["timeout_ms"]["description"].as_str().unwrap().chars().count() <= 91);
    }

    #[test]
    fn cuts_at_sentence_boundary() {
        let c = compact_tool_specs_default(&verbose());
        assert!(c[0].description.ends_with('.') || c[0].description.ends_with('…'));
    }

    #[test]
    fn does_not_mutate_originals() {
        let v = verbose();
        let before = v[0].description.clone();
        let _ = compact_tool_specs_default(&v);
        assert_eq!(v[0].description, before);
    }

    #[test]
    fn chat_specs_cover_the_ts_set() {
        let names = chat_tool_names();
        for n in ["run_shell", "shell_help", "read_file", "write_file", "list_files", "tree", "grep", "git_status", "git_diff", "update_plan"] {
            assert!(names.contains(n), "{n}");
        }
        for n in ["find_symbol", "search_symbols", "outline_file", "find_references", "read_symbol", "apply_edit"] {
            assert!(names.contains(n), "{n}");
        }
    }
}
