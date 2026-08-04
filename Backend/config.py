from pathlib import Path


BACKEND_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BACKEND_DIR.parent

DATASET_DIR = PROJECT_ROOT / "dataset"
MODELS_DIR = PROJECT_ROOT / "models"
SUBMISSIONS_DIR = PROJECT_ROOT / "submissions"


MODEL_PATH = MODELS_DIR / "xgboost_model.pkl"
FEATURE_ENCODERS_PATH = MODELS_DIR / "feature_encoders.pkl"
TARGET_ENCODER_PATH = MODELS_DIR / "target_encoder.pkl"
FEATURE_NAMES_PATH = MODELS_DIR / "feature_names.pkl"
PREPROCESSING_INFO_PATH = MODELS_DIR / "preprocessing_info.pkl"


TRAIN_CSV_PATH = DATASET_DIR / "train.csv"
TEST_CSV_PATH = DATASET_DIR / "test.csv"
SAMPLE_SUBMISSION_PATH = DATASET_DIR / "sample_submission.csv"
SUBMISSION_OUTPUT_PATH = SUBMISSIONS_DIR / "submission.csv"

TARGET_COLUMN = "health_condition"


def check_artifacts_exist() -> list:
    """Returns a list of any required model artifact paths that are missing.
    Used by predictor.py to fail with a clear error instead of a confusing
    FileNotFoundError deep in joblib."""
    required = [
        MODEL_PATH,
        FEATURE_ENCODERS_PATH,
        TARGET_ENCODER_PATH,
        FEATURE_NAMES_PATH,
        PREPROCESSING_INFO_PATH,
    ]
    return [p for p in required if not p.exists()]
