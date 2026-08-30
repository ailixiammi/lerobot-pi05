from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest import mock

import prepare_lerobot_v21_local as converter


class PathGuardTest(unittest.TestCase):
    def make_source(self, base: Path) -> Path:
        source = base / "source"
        (source / "meta").mkdir(parents=True)
        (source / "meta" / "info.json").write_text("{}")
        (source / "data").mkdir()
        return source

    def test_rejects_same_source_and_staging(self) -> None:
        with TemporaryDirectory() as directory:
            source = self.make_source(Path(directory))
            with self.assertRaisesRegex(ValueError, "must differ"):
                converter.validate_paths(source, source)

    def test_refuses_to_overwrite_staging(self) -> None:
        with TemporaryDirectory() as directory:
            base = Path(directory)
            source = self.make_source(base)
            staging = base / "staging"
            staging.mkdir()
            with self.assertRaisesRegex(FileExistsError, "refusing to overwrite"):
                converter.validate_paths(source, staging)

    def test_uses_roundoff_tolerance_for_large_dataset_aggregation(self) -> None:
        dataset = object()
        reference_stats = {"episode_index": {"mean": [721.131897]}}
        with mock.patch.object(converter, "check_aggregate_stats") as checker:
            converter.validate_aggregate_stats(dataset, reference_stats)
        checker.assert_called_once_with(
            dataset,
            reference_stats,
            default_rtol_atol=(1e-5, 1e-4),
        )


if __name__ == "__main__":
    unittest.main()
