import unittest
from pathlib import Path

import pandas as pd
from fastapi.testclient import TestClient

from deployment_intelligence.app import app


PROJECT_ROOT = Path(__file__).resolve().parent.parent
TEST_CSV = PROJECT_ROOT / "data" / "raw" / "UNSW-NB15" / "UNSW_NB15_testing-set.csv"
MODEL_PATH = PROJECT_ROOT / "outputs" / "shared-baseline" / "baseline_mlp.pt"


@unittest.skipUnless(TEST_CSV.exists() and MODEL_PATH.exists(), "requires local NIDS data and model artifacts")
class DeploymentIntelligenceApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        frame = pd.read_csv(TEST_CSV, nrows=1)
        raw_row = frame.drop(columns=["id", "attack_cat", "label"]).iloc[0].to_dict()
        cls.valid_features = {
            key: value.item() if hasattr(value, "item") else value for key, value in raw_row.items()
        }

    def test_health_and_metadata(self):
        with TestClient(app) as client:
            health = client.get("/health")
            metadata = client.get("/metadata")

        self.assertEqual(health.status_code, 200)
        self.assertEqual(health.json()["status"], "ok")
        self.assertEqual(metadata.status_code, 200)
        self.assertEqual(len(metadata.json()["model_input_fields"]), 42)
        self.assertEqual(metadata.json()["preprocessed_feature_count"], 194)

    def test_dashboard_and_demo_flow_are_served(self):
        with TestClient(app) as client:
            dashboard = client.get("/")
            demo = client.get("/demo-flow")

        self.assertEqual(dashboard.status_code, 200)
        self.assertIn("NIDS Analyst Console", dashboard.text)
        self.assertEqual(demo.status_code, 200)
        self.assertEqual(set(demo.json()["features"]), set(self.valid_features))

    def test_predict_and_analyze_one_real_flow(self):
        with TestClient(app) as client:
            prediction = client.post("/predict", json={"features": self.valid_features})
            analysis = client.post("/analyze", json={"features": self.valid_features})

        self.assertEqual(prediction.status_code, 200)
        self.assertIn(prediction.json()["predicted_label"], {"normal", "attack"})
        self.assertGreaterEqual(prediction.json()["attack_probability"], 0.0)
        self.assertLessEqual(prediction.json()["attack_probability"], 1.0)
        self.assertEqual(analysis.status_code, 200)
        self.assertEqual(analysis.json()["certification"]["certificate_space"], "L2 radius in preprocessed feature space")

    def test_intelligence_returns_explanation_decision_and_draft_report(self):
        with TestClient(app) as client:
            response = client.post("/intelligence", json={"features": self.valid_features})

        self.assertEqual(response.status_code, 200)
        response_data = response.json()
        self.assertEqual(response_data["explanation"]["method"], "LIME")
        self.assertTrue(response_data["explanation"]["top_features"])
        self.assertIn(response_data["decision"]["action"], {"escalate", "suppress"})
        self.assertEqual(response_data["incident_report"]["status"], "draft")

    def test_unknown_or_missing_schema_is_rejected(self):
        invalid_features = dict(self.valid_features)
        invalid_features.pop("dur")
        invalid_features["not_a_training_feature"] = 1
        with TestClient(app) as client:
            response = client.post("/predict", json={"features": invalid_features})

        self.assertEqual(response.status_code, 422)
        self.assertIn("missing fields: dur", response.json()["detail"])
        self.assertIn("unknown fields: not_a_training_feature", response.json()["detail"])

    def test_batch_score_returns_only_flagged_rows_with_selectable_features(self):
        batch = pd.DataFrame([self.valid_features, self.valid_features])
        batch.insert(0, "id", [1, 2])
        batch["label"] = [0, 0]
        batch["attack_cat"] = ["Normal", "Normal"]
        csv_content = batch.to_csv(index=False).encode("utf-8")

        with TestClient(app) as client:
            response = client.post("/batch-score", files={"file": ("flows.csv", csv_content, "text/csv")})

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["total_rows"], 2)
        self.assertEqual(data["flagged_count"], len(data["alerts"]))
        for alert in data["alerts"]:
            self.assertEqual(set(alert["features"]), set(self.valid_features))
            self.assertGreaterEqual(alert["attack_probability"], 0.5)

    def test_batch_score_rejects_missing_required_columns(self):
        incomplete = pd.DataFrame([self.valid_features]).drop(columns=["dur"])
        csv_content = incomplete.to_csv(index=False).encode("utf-8")

        with TestClient(app) as client:
            response = client.post("/batch-score", files={"file": ("incomplete.csv", csv_content, "text/csv")})

        self.assertEqual(response.status_code, 422)
        self.assertIn("missing columns: dur", response.json()["detail"])
