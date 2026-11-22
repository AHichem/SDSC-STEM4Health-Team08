# data/

Gitignored except this file. The challenge data is **never** committed.

```
data/
├── raw/         identical copy of "Project 2 - BRPMM" (process A, process B, template)
└── processed/   derived tables, e.g. process_A_features_{calibration,test}.parquet
```

- Create `raw/` with `make data` (or `python -m brpmm.prepare`; `--link` symlinks instead of copying).
  The source is the Renku folder `/home/renku/work/input-data/Project 2 - BRPMM`, or a local copy in
  `challenge_docs/input-data/Project 2 - BRPMM`.
- `processed/` is written by `make eda-a`.
- Always load through `brpmm.data.load_calibration(...)` / `load_test(...)`. Don't hard-code paths.
  `BRPMM_DATA_DIR` in `.env` overrides the location.
