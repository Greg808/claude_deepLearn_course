# Frontend changes: code quality tooling

Prettier formats the frontend (`frontend/`: HTML, CSS, JS). The backend is untouched: black is a Python formatter and this change is front-end only.

## Added
- `package.json` / `package-lock.json`: Prettier 3.10.1 as a pinned dev dependency, plus the npm scripts `format` and `format:check`.
- `.prettierrc.json`: 4-space indent, single quotes, print width 100. This matches the existing style, so the diff stays small.
- `scripts/frontend-quality.sh`: runs `npm ci` if needed, then checks formatting (exits non-zero on issues). With `--fix` it rewrites the files.
- `.gitignore`: `node_modules/`.

## Changed
- `frontend/index.html`, `script.js`, `style.css`: reformatted once with Prettier. Only whitespace and syntax style changed (indentation, self-closing void tags, trailing commas, line wrapping). Behaviour is the same, and all tests pass, including the e2e tests.

## Usage
```bash
npm install                        # once
scripts/frontend-quality.sh        # check
scripts/frontend-quality.sh --fix  # format
```
