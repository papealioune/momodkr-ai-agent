from pathlib import Path

EXPECTED_TOP_LEVEL_DIRS = [
    "configs",
    "data",
    "docs",
    "envs",
    "evaluation",
    "execution",
    "live",
    "scripts",
    "serving",
    "tests",
    "training",
]

EXPECTED_CONFIGS = [
    "configs/data/binance_vision_v1.yaml",
    "configs/env/momodkr_v1.yaml",
    "configs/training/v1_engine_cold.yaml",
    "configs/live/governor_v1.yaml",
]


def test_top_level_dirs_exist(repo_root: Path) -> None:
    for d in EXPECTED_TOP_LEVEL_DIRS:
        assert (repo_root / d).is_dir(), f"missing top-level directory: {d}"


def test_phase0_artifacts(repo_root: Path) -> None:
    for f in [
        "pyproject.toml",
        "README.md",
        "CLAUDE.md",
        ".gitignore",
        "docs/LESSONS_LEARNED_FROM_MOLEAPP.md",
    ]:
        assert (repo_root / f).is_file(), f"missing required Phase 0 artifact: {f}"


def test_seed_configs_present(repo_root: Path) -> None:
    for c in EXPECTED_CONFIGS:
        assert (repo_root / c).is_file(), f"missing seed config: {c}"
