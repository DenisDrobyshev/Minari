from typing import Iterable, List, Optional

import pytest

from minari.storage import hosting


class FakeCloudStorage:
    """Minimal in-memory CloudStorage, to test listing without any remote."""

    def __init__(self, dataset_ids: List[str]):
        self.dataset_ids = dataset_ids

    def list_datasets(self, prefix: Optional[str] = None) -> Iterable[str]:
        for dataset_id in self.dataset_ids:
            if prefix is None or dataset_id.startswith(prefix):
                yield dataset_id

    def get_dataset_metadata(self, dataset_id: str) -> dict:
        return {"dataset_id": dataset_id}


@pytest.fixture
def patch_cloud_storage(monkeypatch):
    def patch(dataset_ids: List[str]):
        monkeypatch.setattr(
            hosting, "get_cloud_storage", lambda **_: FakeCloudStorage(dataset_ids)
        )

    return patch


@pytest.mark.parametrize(
    "dataset_ids",
    [
        ["D4RL/door/human-v0", "D4RL/door/human-v1", "D4RL/door/human-v2"],
        ["D4RL/door/human-v2", "D4RL/door/human-v1", "D4RL/door/human-v0"],
        ["D4RL/door/human-v1", "D4RL/door/human-v0", "D4RL/door/human-v2"],
        ["D4RL/door/human-v2", "D4RL/door/human-v0", "D4RL/door/human-v1"],
    ],
)
def test_list_remote_datasets_latest_version_ignores_listing_order(
    patch_cloud_storage, dataset_ids: List[str]
):
    """Only the highest version is returned, whatever order the remote lists them in.

    The remote gives no ordering guarantee: GCS lists blobs lexicographically
    (so `-v10` comes before `-v2`) and the Hugging Face API lists repos in
    arbitrary order.
    """
    patch_cloud_storage(dataset_ids)

    remote_datasets = hosting.list_remote_datasets(latest_version=True)

    assert list(remote_datasets.keys()) == ["D4RL/door/human-v2"]
    assert remote_datasets["D4RL/door/human-v2"]["dataset_id"] == "D4RL/door/human-v2"


def test_list_remote_datasets_latest_version_per_dataset(patch_cloud_storage):
    """Versions are tracked per (namespace, dataset name), not globally."""
    patch_cloud_storage(
        [
            "D4RL/door/human-v10",  # sorts before -v2 lexicographically
            "D4RL/door/human-v2",
            "D4RL/door/expert-v0",
            "atari/pong/expert-v1",
            "atari/pong/expert-v0",
        ]
    )

    remote_datasets = hosting.list_remote_datasets(latest_version=True)

    assert set(remote_datasets.keys()) == {
        "D4RL/door/human-v10",
        "D4RL/door/expert-v0",
        "atari/pong/expert-v1",
    }


def test_list_remote_datasets_all_versions(patch_cloud_storage):
    dataset_ids = ["D4RL/door/human-v1", "D4RL/door/human-v0"]
    patch_cloud_storage(dataset_ids)

    remote_datasets = hosting.list_remote_datasets()

    assert set(remote_datasets.keys()) == set(dataset_ids)
