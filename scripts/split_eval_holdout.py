"""Split the eval parquet into a selection slice + a holdout validation slice.

Why this exists:
  Phase 4 used a single eval.parquet for both best_checkpoint selection
  during training AND the post-hoc U-test gate. That's fine for "does
  the policy beat random," but if we then train 10-20 seeds (or a
  sweep) and pick the highest-eval-reward one as deployment candidate,
  we're selecting partly on noise that happens to align with this
  specific eval distribution.

  Standard fix: a tertiary slice that the agent never sees during any
  selection step. It's only used to validate the chosen winner before
  shipping.

Output layout:
  data/episodes/<SYM>/<feature_version>/
    train.parquet            (unchanged)
    eval.parquet             (unchanged, kept for backward-compat)
    eval_selection.parquet   (first N rows of eval -- selection set)
    eval_holdout.parquet     (last M rows of eval -- final validation)
    norm_stats.json          (unchanged, shared by all three eval files)
    manifest.json            (unchanged)

The norm_stats.json doesn't need to be re-computed because it's
TRAIN-set z-score stats applied at eval time -- same for both slices.

Example:
  python -m scripts.split_eval_holdout \\
      --eval-parquet data/episodes/BTCUSDT/0.1.0/eval.parquet \\
      --holdout-days 1.0
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pyarrow.parquet as pq

from data.preprocessors.momentum_features import TICKS_PER_SECOND

logger = logging.getLogger(__name__)

TICKS_PER_DAY = 24 * 60 * 60 * TICKS_PER_SECOND  # 100ms ticks => 864_000 / day


def split_eval(eval_parquet: Path, holdout_days: float) -> tuple[Path, Path]:
    if not eval_parquet.exists():
        raise FileNotFoundError(f"eval parquet not found: {eval_parquet}")
    holdout_rows = int(round(holdout_days * TICKS_PER_DAY))
    pf = pq.ParquetFile(str(eval_parquet))
    total = pf.metadata.num_rows
    if holdout_rows >= total:
        raise ValueError(
            f"requested {holdout_rows} holdout rows but eval parquet only has {total} rows"
        )
    selection_rows = total - holdout_rows
    logger.info(
        "splitting %s (%d rows = %.2fd) -> selection=%d rows (%.2fd), holdout=%d rows (%.2fd)",
        eval_parquet, total, total / TICKS_PER_DAY,
        selection_rows, selection_rows / TICKS_PER_DAY,
        holdout_rows, holdout_rows / TICKS_PER_DAY,
    )

    # Walk row groups and accumulate. PyArrow writes RGs of ~200k rows by
    # default (episode_builder used 200k batches) so granularity is fine
    # but we trim the boundary RG to land exactly on the split row.
    selection_path = eval_parquet.parent / "eval_selection.parquet"
    holdout_path = eval_parquet.parent / "eval_holdout.parquet"
    schema = pf.schema_arrow

    sel_writer = pq.ParquetWriter(str(selection_path), schema, compression="zstd")
    hold_writer = pq.ParquetWriter(str(holdout_path), schema, compression="zstd")
    try:
        rows_seen = 0
        for batch in pf.iter_batches(batch_size=200_000):
            n = batch.num_rows
            if rows_seen + n <= selection_rows:
                sel_writer.write_table(_batch_to_table(batch, schema))
            elif rows_seen >= selection_rows:
                hold_writer.write_table(_batch_to_table(batch, schema))
            else:
                # batch straddles the split -- slice it
                cut = selection_rows - rows_seen
                first = batch.slice(0, cut)
                rest = batch.slice(cut)
                sel_writer.write_table(_batch_to_table(first, schema))
                hold_writer.write_table(_batch_to_table(rest, schema))
            rows_seen += n
    finally:
        sel_writer.close()
        hold_writer.close()

    logger.info("wrote %s + %s", selection_path, holdout_path)
    return selection_path, holdout_path


def _batch_to_table(batch, schema):
    import pyarrow as pa

    return pa.Table.from_batches([batch], schema=schema)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    p = argparse.ArgumentParser(description="Split eval.parquet into selection + holdout slices")
    p.add_argument("--eval-parquet", required=True)
    p.add_argument("--holdout-days", type=float, default=1.0)
    args = p.parse_args()
    split_eval(Path(args.eval_parquet), args.holdout_days)


if __name__ == "__main__":
    main()
