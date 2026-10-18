Plan the task below. Decompose it into the smallest set of concrete steps a coder can follow.

TASK:
{{task}}

PROJECT:
{{repo}}

RELEVANT FILES (names only):
{{files}}

Rules:
- 2 to 6 steps. Fewer is better.
- Identify only the files that must be inspected or changed.
- Do NOT plan to edit files unrelated to the task.
- List risky assumptions explicitly.

Output ONLY this JSON:
{
  "goal": "<one sentence>",
  "files": ["path/one", "path/two"],
  "assumptions": ["..."],
  "steps": [
    {"index": 1, "title": "<short>", "detail": "<what to do>", "files": ["path"]}
  ]
}
