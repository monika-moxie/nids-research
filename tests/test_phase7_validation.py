import unittest

from paper.run_repeated_seed_validation import mean_and_sample_std, summarize_metric_records


class PhaseSevenValidationTests(unittest.TestCase):
    def test_mean_and_sample_std(self):
        summary = mean_and_sample_std([0.7, 0.8, 0.9])
        self.assertEqual(summary["num_seeds"], 3)
        self.assertAlmostEqual(summary["mean"], 0.8)
        self.assertAlmostEqual(summary["sample_std"], 0.1)

    def test_one_seed_has_zero_sample_std(self):
        summary = mean_and_sample_std([0.8])
        self.assertEqual(summary["sample_std"], 0.0)

    def test_metric_summary_includes_all_reported_metrics(self):
        metrics = [
            {"accuracy": 0.8, "precision": 0.7, "recall": 0.9, "f1": 0.79, "roc_auc": 0.88},
            {"accuracy": 0.82, "precision": 0.72, "recall": 0.91, "f1": 0.80, "roc_auc": 0.89},
        ]
        summary = summarize_metric_records(metrics)
        self.assertEqual(summary["num_successful_runs"], 2)
        self.assertAlmostEqual(summary["metrics"]["f1"]["mean"], 0.795)

    def test_metric_summary_retains_failed_runs(self):
        records = [
            {"accuracy": 0.8, "precision": 0.7, "recall": 0.9, "f1": 0.79, "roc_auc": 0.88},
            {"error": "non-finite predictions"},
        ]
        summary = summarize_metric_records(records)
        self.assertEqual(summary["num_successful_runs"], 1)
        self.assertEqual(summary["num_failed_runs"], 1)
        self.assertEqual(summary["failure_messages"], ["non-finite predictions"])
