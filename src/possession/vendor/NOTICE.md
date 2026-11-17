# Vendored files

The files in this directory are copied unmodified from the official challenge repository
<https://github.com/metrica-sports/more-than-a-hack-2027> at commit
`f8f2629ff1e0985f81d7e59d8413678514ee1530`.

| File | Source path | sha256 |
|---|---|---|
| `games.py` | `games.py` | `3b2122794e9e0476e2b3a871d14880e5fa47c4feaeacf99caef054c2309076ce` |
| `possession_metrics.py` | `evaluation/possession_metrics.py` | `341bad48904b489d6e875397b27687f5d6a4d22a397f3dc8af3a00d48f16874d` |
| `run_evaluation.py` | `evaluation/run_evaluation.py` | `1c61564fe52499ba4af3006fb5eb4147b825e92bf516651add80de49d267b19f` |
| `validate_submission.py` | `evaluation/validate_submission.py` | `07ac29499acbd337e1dc1de1832d63f6e1fa1d981e22f6ad08212529e2bbee14` |
| `LICENSE` | `LICENSE` | `b1f93728571a232a0e158abdd869891f2efd51894134fb20678292c2468ff3ec` |

Copyright (c) 2026-2027 Metrica Sports, distributed under the MIT License. The full licence text,
including its scope note, is in `LICENSE` next to this file. As that note says, the MIT License
covers the repository's code and documentation only; the match data (video, tracking, Smart
Tagging, homography) is not covered and is not included here. The public ground-truth CSV is also
not copied into this project: `possession.evaluate` reads it from the official repository checkout
(`MTAH_REPO_DIR`).

`__init__.py` is ours: it loads each file with `importlib` and registers it in `sys.modules`
under its original name, so the files keep their top-level imports (`import possession_metrics`,
`from games import ...`) without being edited.
`validate_submission.py` runs as a script directly from here:

```bash
.venv/bin/python src/possession/vendor/validate_submission.py outputs/baseline_smooth50.csv
```

To refresh, copy the files again from the official repository and update the table above.
