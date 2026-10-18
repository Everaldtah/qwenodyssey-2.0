//! Planner agent (port of `src/agents/planner.ts`): decompose a task into a
//! few concrete steps and name the files involved.

use serde_json::Value;
use tokio_util::sync::CancellationToken;

use super::{ask, display_value, string_list};
use crate::deps::{extract_json, load_prompt, render};
use crate::repo_scanner::summarize_repo;
use qo_core::{Plan, PlanStep, Provider, RepoInfo};

/// At most this many file names go into the planner prompt.
const MAX_PROMPT_FILES: usize = 60;

/// The single-step plan used when planning is skipped or unparseable.
pub fn single_step_plan(task: &str, assumptions: Vec<String>) -> Plan {
    Plan {
        goal: task.to_string(),
        steps: vec![PlanStep { index: 1, title: task.to_string(), detail: Some(task.to_string()), files: Vec::new() }],
        files: Vec::new(),
        assumptions,
    }
}

/// Turn the planner's JSON into a [`Plan`]; `None` when it has no `steps` array.
pub fn plan_from_json(task: &str, parsed: &Value) -> Option<Plan> {
    let steps = parsed.get("steps")?.as_array()?;
    let goal = parsed.get("goal").and_then(Value::as_str).filter(|g| !g.is_empty()).unwrap_or(task).to_string();
    Some(Plan {
        goal,
        files: string_list(parsed, "files"),
        assumptions: string_list(parsed, "assumptions"),
        steps: steps
            .iter()
            .enumerate()
            .map(|(i, s)| PlanStep {
                index: s.get("index").and_then(Value::as_u64).map_or(i as u32 + 1, |n| n as u32),
                title: s.get("title").and_then(display_value).unwrap_or_else(|| format!("step {}", i + 1)),
                detail: s.get("detail").and_then(display_value),
                files: string_list(s, "files"),
            })
            .collect(),
    })
}

pub async fn plan(
    provider: &dyn Provider,
    task: &str,
    repo: &RepoInfo,
    file_list: &[String],
    cancel: Option<&CancellationToken>,
) -> anyhow::Result<Plan> {
    let files = file_list.iter().take(MAX_PROMPT_FILES).cloned().collect::<Vec<_>>().join("\n");
    let summary = summarize_repo(repo);
    let prompt = render(&load_prompt("planner")?, &[("task", task), ("repo", &summary), ("files", &files)]);
    let text = ask(provider, prompt, true, 0.1, None, cancel).await?;
    let parsed = extract_json(&text);
    // Fallback: a single-step plan so the pipeline can still proceed.
    Ok(parsed.as_ref().and_then(|p| plan_from_json(task, p)).unwrap_or_else(|| {
        single_step_plan(task, vec!["planner output was unstructured; using a single-step plan".to_string()])
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::FakeProvider;
    use serde_json::json;

    #[test]
    fn parses_planner_json() {
        let p = plan_from_json(
            "t",
            &json!({"goal": "", "files": ["a.rs", 3], "steps": [{"title": "one", "detail": "d", "files": ["a.rs"]}, {"index": 7}]}),
        )
        .unwrap();
        assert_eq!(p.goal, "t");
        assert_eq!(p.files, vec!["a.rs"]);
        assert_eq!(p.steps[0].index, 1);
        assert_eq!(p.steps[0].detail.as_deref(), Some("d"));
        assert_eq!(p.steps[1].index, 7);
        assert_eq!(p.steps[1].title, "step 2");
        assert!(plan_from_json("t", &json!({"goal": "g"})).is_none());
    }

    #[tokio::test]
    async fn falls_back_to_single_step_on_garbage() {
        let fp = FakeProvider::new(vec!["I think we should just do it".into()]);
        let repo = RepoInfo::default();
        let p = plan(&fp, "do the thing", &repo, &[], None).await.unwrap();
        assert_eq!(p.steps.len(), 1);
        assert_eq!(p.steps[0].title, "do the thing");
        assert_eq!(p.assumptions, vec!["planner output was unstructured; using a single-step plan"]);
        let req = fp.requests();
        assert!(req[0].json);
        assert_eq!(req[0].temperature, Some(0.1));
    }

    #[tokio::test]
    async fn uses_model_plan() {
        let fp = FakeProvider::new(vec![r#"{"goal":"G","files":["x.rs"],"steps":[{"index":1,"title":"edit x"}]}"#.into()]);
        let files: Vec<String> = (0..100).map(|i| format!("f{i}.rs")).collect();
        let p = plan(&fp, "t", &RepoInfo::default(), &files, None).await.unwrap();
        assert_eq!(p.goal, "G");
        assert_eq!(p.files, vec!["x.rs"]);
        let prompt = &fp.requests()[0].messages[1].content;
        assert!(prompt.contains("f59.rs") && !prompt.contains("f60.rs"), "only the first 60 files are listed");
    }
}
