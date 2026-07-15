import unittest

from ml.training.create_realistic_labels import add_labels, extract_real_label, resolve_label


class CreateRealisticLabelsTest(unittest.TestCase):
    def test_extract_real_label_from_is_abandoned(self) -> None:
        self.assertEqual(extract_real_label({"isAbandoned": True}), 1)
        self.assertEqual(extract_real_label({"is_abandoned": False}), 0)

    def test_prefer_real_keeps_ground_truth(self) -> None:
        label, source, frustration, prob = resolve_label(
            {"isAbandoned": True, "timeOnPage": 10, "scrollDepth": 90},
            prefer_real=True,
            seed_value=0,
        )
        self.assertEqual(label, 1)
        self.assertEqual(source, "real")
        self.assertIsNone(frustration)
        self.assertIsNone(prob)

    def test_prefer_real_falls_back_to_synthetic(self) -> None:
        label, source, frustration, prob = resolve_label(
            {
                "timeOnPage": 150,
                "scrollDepth": 10,
                "clickMisses": 3,
                "hesitations": 2,
                "helpTriggered": True,
            },
            prefer_real=True,
            seed_value=1,
        )
        self.assertEqual(source, "synthetic")
        self.assertIn(label, (0, 1))
        self.assertIsNotNone(frustration)
        self.assertIsNotNone(prob)

    def test_add_labels_writes_label_source_fields(self) -> None:
        rows = [
            {"sessionId": "s1", "isAbandoned": False, "timeOnPage": 30, "scrollDepth": 80},
            {"sessionId": "s2", "timeOnPage": 180, "scrollDepth": 5, "clickMisses": 3, "hesitations": 2},
        ]
        labeled, stats = add_labels(rows, prefer_real=True)

        self.assertEqual(labeled[0]["label_source"], "real")
        self.assertEqual(labeled[0]["label"], 0)
        self.assertEqual(labeled[1]["label_source"], "synthetic")
        self.assertEqual(stats["real"], 1)
        self.assertEqual(stats["synthetic"], 1)


if __name__ == "__main__":
    unittest.main()
