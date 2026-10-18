A change was applied and the tests/build failed. Diagnose and fix.

TASK:
{{task}}

FAILING OUTPUT:
{{errors}}

CURRENT RELEVANT FILES:
{{context}}

Rules:
- Read the error carefully. Fix the actual cause, not the symptom.
- Make the smallest patch that makes it pass.
- Do not disable or fake tests to force a pass.
- If the error is environmental (missing dependency, wrong command), say so in notes.

Output ONLY this JSON (same shape as the coder):
{
  "edits": [
    {"path": "...", "mode": "patch|create|rewrite|delete", "diff": "...", "content": "...", "rationale": "..."}
  ],
  "notes": "<root cause + fix, one or two lines>"
}
