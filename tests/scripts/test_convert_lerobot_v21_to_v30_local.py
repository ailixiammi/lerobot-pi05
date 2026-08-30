from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import convert_lerobot_v21_to_v30_local as converter


class PathGuardTest(unittest.TestCase):
    def make_source(self, base: Path) -> Path:
        source = base / "source"
        (source / "meta").mkdir(parents=True)
        (source / "meta" / "info.json").write_text("{}")
        return source

    def test_rejects_same_source_and_output(self) -> None:
        with TemporaryDirectory() as directory:
            source = self.make_source(Path(directory))
            with self.assertRaisesRegex(ValueError, "must differ"):
                converter.validate_paths(source, source)

    def test_refuses_to_overwrite_output(self) -> None:
        with TemporaryDirectory() as directory:
            base = Path(directory)
            source = self.make_source(base)
            output = base / "output"
            output.mkdir()
            with self.assertRaisesRegex(FileExistsError, "refusing to overwrite"):
                converter.validate_paths(source, output)


if __name__ == "__main__":
    unittest.main()
