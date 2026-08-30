import unittest

import numpy as np
import pyarrow as pa

from scripts.augment_pi05_quantile_stats_local import compute_exact_quantiles
from scripts.augment_pi05_quantile_stats_local import missing_quantiles


class MissingQuantilesTest(unittest.TestCase):
    def test_reports_every_missing_required_quantile(self):
        missing = missing_quantiles({"actions": {"q01": [0.0]}}, ["actions", "state"])

        self.assertNotIn("q01", missing["actions"])
        self.assertEqual(missing["actions"], ["q10", "q50", "q90", "q99"])
        self.assertEqual(missing["state"], ["q01", "q10", "q50", "q90", "q99"])

    def test_accepts_complete_quantile_contract(self):
        complete = {key: [0.0] for key in ("q01", "q10", "q50", "q90", "q99")}

        self.assertEqual(missing_quantiles({"actions": complete}, ["actions"]), {})

    def test_computes_quantiles_directly_from_arrow_column(self):
        class DatasetStub:
            features = {"actions": {"dtype": "float32"}}
            hf_dataset = type(
                "HfDatasetStub",
                (),
                {"data": pa.table({"actions": [[0.0, 10.0], [2.0, 20.0]]})},
            )()
            meta = type("MetaStub", (), {"stats": {"actions": {"count": np.array([2])}}})()

        stats = compute_exact_quantiles(DatasetStub(), ["actions"])

        np.testing.assert_allclose(stats["actions"]["q50"], [1.0, 15.0])


if __name__ == "__main__":
    unittest.main()
