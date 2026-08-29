# Project Memory

- User prefers Chinese for project discussion and concise, precise explanations. Avoid vague shorthand such as "patch" unless immediately defined as the union of currently placed tiles.
- Use the local Anaconda Python when running scripts: `D:\anaconda3\python.exe`.
- Commit prefixes should use `doc:` for documentation, not `docs:`. The user also asks for specific prefixes such as `feat:`, `fix:`, `chore:`, and `refactor:`; follow the requested prefix exactly.
- Do not change code when the user is only asking a question. For implementation requests, make the change and verify when practical.
- Keep temporary test outputs out of the project after verification. Formal explorer output belongs in `outputs/dfs_hat`; `outputs/` and `references/` are ignored.
- Python scripts live in `src/`. The main automatic tiling search script is `src/tiling_explorer.py`; hat-specific scripts live in `src/hat/`.
- `src/tiling_explorer.py` currently uses an explicit DFS frame stack with conflict-directed backjumping, not Python recursion. It supports bitmap and angle concavity scoring, length-pair restrictions, per-step `trace.csv`, optional per-state HDF5, and PNG exports.
- For current hat explorer runs, use reflection, output to `outputs/dfs_hat`, and restrict glued edge lengths with `--allowed-length-pairs "1:1,1:2,2:2,sqrt3:sqrt3"`. After explorer runs, update `outputs/dfs_hat/tiles_trace.png`.
- Output file numbering should not be zero-padded: use names like `step_100_tiles_29.png` and `state_100_tiles_29.h5`.
- The README intentionally has two main conceptual sections: one about hat and the `Tile(a,b)` family, and one about search algorithms. Keep it concise and avoid non-Markdown hard wrapping.
