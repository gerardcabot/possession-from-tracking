import argparse
from pathlib import Path

from possession.paths import resolve_gt
from possession.vendor import possession_metrics as pm
from possession.vendor import run_evaluation as official

DEFAULT_GT = "public"
SUMMARY_KEYS = ("macro_f1", "spell_f1_mean", "n_pred_spells", "mean_spell_duration_s")


def score(submission: Path, gt: str = DEFAULT_GT) -> dict:
    gt_path = official.find_gt(str(resolve_gt(gt)))
    gt_codes = pm.label_codes(pm.read_label_csv(gt_path))
    result, reason = official.score_submission(
        str(submission), gt_codes, pm.TAUS, pm.BOOTSTRAP_N)
    if result is None:
        raise ValueError(f"{submission}: {reason}")
    return {key: result[key] for key in SUMMARY_KEYS}


def main() -> None:
    parser = argparse.ArgumentParser(description="Score a submission with the official scorer")
    parser.add_argument("submission", type=Path)
    parser.add_argument("--gt", default=DEFAULT_GT,
                        help="'public' or 'private' (read from MTAH_REPO_DIR), or a path to a "
                             "frame,label CSV (default: public)")
    args = parser.parse_args()
    for key, value in score(args.submission, args.gt).items():
        print(f"{key:<24}{value}")


if __name__ == "__main__":
    main()
