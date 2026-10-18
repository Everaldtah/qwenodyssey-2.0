Review the proposed change for correctness before it is applied.

TASK:
{{task}}

PROPOSED EDITS (diffs):
{{edits}}

Check for:
- Correctness: does it actually do the task?
- Missing imports, undefined symbols, type errors.
- Obvious bugs or off-by-one / null issues.
- Security issues (injection, secrets, unsafe shell/exec).
- Style mismatches with the project.

Be strict but do not invent problems. If it is good, approve it.

Output ONLY this JSON:
{
  "approve": true,
  "summary": "<one line>",
  "issues": [
    {"severity": "info|warning|error", "file": "path", "message": "<short>"}
  ]
}
