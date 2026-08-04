import joblib
import pandas as pd

import config
from preprocessing import prepare_features


class HealthRiskPredictor:
    def __init__(self):
        missing = config.check_artifacts_exist()
        if missing:
            missing_str = "\n  - ".join(str(p) for p in missing)
            raise FileNotFoundError(
                "Missing model artifact(s):\n  - " + missing_str +
                "\n\nRun notebooks/06_XGBoost_Final.ipynb first - it saves all "
                "required files into the models/ folder."
            )

        self.model = joblib.load(config.MODEL_PATH)
        self.encoders = joblib.load(config.FEATURE_ENCODERS_PATH)
        self.target_encoder = joblib.load(config.TARGET_ENCODER_PATH)
        self.feature_names = joblib.load(config.FEATURE_NAMES_PATH)
        self.preprocessing_info = joblib.load(config.PREPROCESSING_INFO_PATH)

        self.class_labels = list(self.target_encoder.classes_)
        self.feature_cols = (
            self.preprocessing_info["numeric_columns"]
            + self.preprocessing_info["categorical_columns"]
        )

    def _predict_df(self, df: pd.DataFrame) -> pd.DataFrame:
        x = prepare_features(df, self.encoders, self.preprocessing_info, self.feature_names)
        proba = self.model.predict_proba(x)
        pred_idx = proba.argmax(axis=1)
        pred_label = self.target_encoder.inverse_transform(pred_idx)

        result = pd.DataFrame(proba, columns=[f"proba_{c}" for c in self.class_labels])
        result.insert(0, "predicted_health_condition", pred_label)
        return result

    def predict_single(self, record: dict) -> dict:
        """record: dict with the raw feature values for one person."""
        df = pd.DataFrame([{col: record.get(col) for col in self.feature_cols}])
        result = self._predict_df(df)
        row = result.iloc[0]
        probabilities = {c: float(row[f"proba_{c}"]) for c in self.class_labels}
        return {
            "predicted_health_condition": row["predicted_health_condition"],
            "probabilities": probabilities,
        }

    def predict_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """df: raw dataframe (e.g. loaded from an uploaded CSV / test.csv)."""
        result = self._predict_df(df)
        out = df.copy().reset_index(drop=True)
        out["predicted_health_condition"] = result["predicted_health_condition"]
        for c in self.class_labels:
            out[f"proba_{c}"] = result[f"proba_{c}"]
        return out


_predictor_instance = None


def get_predictor() -> HealthRiskPredictor:
    global _predictor_instance
    if _predictor_instance is None:
        _predictor_instance = HealthRiskPredictor()
    return _predictor_instance
