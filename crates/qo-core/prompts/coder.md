Implement the plan by producing file edits. Keep changes minimal and correct.

TASK:
{{task}}

PLAN:
{{plan}}

CONTEXT (relevant file contents, repo info, errors):
{{context}}

Rules:
- Produce the smallest change that satisfies the task.
- Prefer mode "patch" with a valid unified diff. Use "create" for new files,
  "rewrite" only when a patch is impractical.
- Keep imports correct. Do not reference symbols that do not exist.
- For unified diffs, use real @@ hunks with correct context lines.
- Add or update tests when the task implies behavior changes.

Output ONLY this JSON:
{
  "edits": [
    {
      "path": "relative/path",
      "mode": "patch | create | rewrite | delete",
      "diff": "<unified diff if mode=patch>",
      "content": "<full file content if mode=create/rewrite>",
      "rationale": "<one line>"
    }
  ],
  "notes": "<assumptions or follow-ups, one or two lines>"
}
