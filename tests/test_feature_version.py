from serving.feature_version import (
    ALL_FEATURE_NAMES,
    FEATURE_SPEC_CHECKSUM,
    FEATURE_VERSION,
    MARKET_FEATURE_DIM,
    MARKET_FEATURE_NAMES,
    OBS_DIM,
    POSITION_FEATURE_DIM,
    POSITION_FEATURE_NAMES,
    feature_spec_checksum,
)


def test_dims_match_env_config() -> None:
    assert MARKET_FEATURE_DIM == 26
    assert POSITION_FEATURE_DIM == 4
    assert OBS_DIM == 30
    assert len(ALL_FEATURE_NAMES) == OBS_DIM


def test_canonical_order_market_first_then_position() -> None:
    assert ALL_FEATURE_NAMES[:MARKET_FEATURE_DIM] == MARKET_FEATURE_NAMES
    assert ALL_FEATURE_NAMES[MARKET_FEATURE_DIM:] == POSITION_FEATURE_NAMES


def test_position_features_have_pos_prefix() -> None:
    for name in POSITION_FEATURE_NAMES:
        assert name.startswith("pos_"), f"position feature should start with 'pos_': {name}"


def test_no_duplicate_feature_names() -> None:
    assert len(set(ALL_FEATURE_NAMES)) == len(ALL_FEATURE_NAMES)


def test_checksum_is_deterministic() -> None:
    assert feature_spec_checksum() == FEATURE_SPEC_CHECKSUM
    assert len(FEATURE_SPEC_CHECKSUM) == 16
    assert all(c in "0123456789abcdef" for c in FEATURE_SPEC_CHECKSUM)


def test_version_is_semver_like() -> None:
    parts = FEATURE_VERSION.split(".")
    assert len(parts) == 3
    assert all(p.isdigit() for p in parts)
