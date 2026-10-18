You are editing a real codebase. Make the smallest correct change that satisfies the task.

TASK:
{{task}}

CONTEXT (relevant files, repo info, errors — read carefully before editing):
{{context}}

## How to edit

Return your changes as one or more SEARCH/REPLACE blocks. This is the ONLY
accepted edit format. For each block:

1. Put the file path on its own line.
2. Open with `<<<<<<< SEARCH`.
3. Copy the EXACT lines to find — character-for-character from the file above,
   including indentation. Include just enough surrounding lines to be unique.
4. Add a line with `=======`.
5. Write the replacement lines.
6. Close with `>>>>>>> REPLACE`.

Example:

src/math.ts
<<<<<<< SEARCH
export function add(a: number, b: number) {
  return a + b;
}
=======
export function add(a: number, b: number) {
  // guard against NaN
  if (Number.isNaN(a) || Number.isNaN(b)) throw new Error("NaN");
  return a + b;
}
>>>>>>> REPLACE

## Rules (follow exactly — these are where small models usually fail)

- The SEARCH text MUST match the file verbatim. If you're unsure of the exact
  text, use the find_symbol / read_symbol / outline_file tools first.
- Keep each SEARCH block small and unique. Do NOT paste a whole file.
- To create a NEW file, use an empty SEARCH section and put the full file
  contents in REPLACE.
- Multiple edits to the same file: emit multiple blocks (they apply in order).
- Keep imports correct. Never reference a symbol that doesn't exist — verify
  with find_symbol if unsure.
- Add or update tests when behaviour changes.
- Output ONLY the SEARCH/REPLACE blocks (a one-line note before each is fine).
  No JSON, no unified diffs, no commentary about what you "would" do.
