from pathlib import Path

PROJECT_FOLDER = Path(__file__).resolve().parents[2]

# important folders
DATA_FOLDER = PROJECT_FOLDER / "data"
RAW_DATA_FOLDER = DATA_FOLDER / "raw"
INTERIM_DATA_FOLDER = DATA_FOLDER / "interim"
PROCESSED_DATA_FOLDER = DATA_FOLDER / "processed"
MODELS_FOLDER = PROJECT_FOLDER / "models"
REPORTS_FOLDER = PROJECT_FOLDER / "reports"
REPORTS_FIGURES_FOLDER = REPORTS_FOLDER / "figures"

# data files
RAW_DATA_FILE = RAW_DATA_FOLDER / "ml_project1_data.csv"
INTERIM_DATA_FILE = INTERIM_DATA_FOLDER / "interim_data.parquet"
PROCESSED_DATA_FILE = PROCESSED_DATA_FOLDER / "processed_data.parquet"
PROCESSED_DATA_FILE_RFM = PROCESSED_DATA_FOLDER / "processed_data_rfm.parquet"

# model files
BEST_MODEL_FILE = MODELS_FOLDER / "best_model.joblib"
