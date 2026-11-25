from app.permissions import ALL_REQUIRED, affected_features, missing_permissions


def test_all_key_covers_everything():
    assert missing_permissions(["all"]) == []


def test_missing_listed_in_required_order():
    granted = [p for p in ALL_REQUIRED if p not in ("memoryAsset.create", "face.read")]
    assert missing_permissions(granted) == ["face.read", "memoryAsset.create"]


def test_affected_features_names_the_groups():
    assert affected_features(["memoryAsset.create"]) == ["memories strip"]
    assert affected_features(["asset.upload", "tag.asset"]) == ["upload results", "tags"]


def test_required_set_matches_plan():
    # docs/plan.md section 2 lists exactly these 16.
    assert set(ALL_REQUIRED) == {
        "asset.read", "asset.view", "asset.download", "asset.upload", "asset.update",
        "album.read", "album.create", "albumAsset.create", "person.read", "face.read",
        "tag.read", "tag.create", "tag.asset", "stack.create", "memory.create", "memoryAsset.create",
    }


def test_empty_password_env_means_generate():
    from app.config import Config

    cfg = Config(immich_url="http://h:2283", immich_api_key="k", photos_password="", _env_file=None)
    assert cfg.photos_password is None
