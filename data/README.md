# data/

Intentionally empty in git. The challenge data is **not** committed.

- On Renku it lives at `/home/renku/work/input-data/Project 2 - BRPMM`.
- Locally, put a copy at `challenge_docs/input-data/Project 2 - BRPMM` or set `BRPMM_DATA_DIR` in `.env`.

Load it through `brpmm.data.load_calibration(...)` / `load_test(...)`. Don't hard-code paths.
Use this folder only for small derived intermediates (it is gitignored).
