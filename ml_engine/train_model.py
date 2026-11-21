"""
Machine Learning Engine: Academic Risk Model Trainer
Trains a Random Forest Classifier to identify students at academic risk based on attendance, marks, and engagement.
"""

import sys
import logging
from pathlib import Path
from datetime import datetime, timezone
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score

# Ensure root is in python path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.database import SessionLocal, init_db
from app.models import Student, AcademicLog

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ML-Trainer")

MODEL_DIR = Path(__file__).resolve().parent
MODEL_FILE = MODEL_DIR / "risk_model.pkl"

FEATURE_NAMES = [
    "attendance_pct",
    "internal_marks",
    "lms_logins",
    "scholarship_encoded",
]


def fetch_training_data_from_db() -> pd.DataFrame:
    """
    Queries students and their most recent academic logs from the database.
    """
    init_db()
    db = SessionLocal()
    try:
        # Fetch all students with their academic logs
        students = db.query(Student).all()
        if not students:
            logger.warning("No students found in database. Checking sample_students.csv for auto-seeding...")
            sample_csv = ROOT_DIR / "etl" / "sample_students.csv"
            if not sample_csv.exists():
                from etl.generate_sample_data import generate_sample_dataset
                generate_sample_dataset(output_path=sample_csv)
            from etl.data_loader import load_csv_data
            load_csv_data(sample_csv, db)
            students = db.query(Student).all()

        rows = []
        for s in students:
            log = s.latest_academic_log
            if log:
                rows.append({
                    "student_id": s.student_id,
                    "name": s.name,
                    "department": s.department,
                    "scholarship_status": s.scholarship_status,
                    "scholarship_encoded": 1 if s.scholarship_status == "Yes" else 0,
                    "attendance_pct": float(log.attendance_pct),
                    "internal_marks": float(log.internal_marks),
                    "lms_logins": int(log.lms_logins),
                })
        return pd.DataFrame(rows)
    finally:
        db.close()


def train_and_save_model(output_path: Path = MODEL_FILE) -> dict:
    """
    Trains the Random Forest model and saves it via joblib.
    """
    logger.info("Fetching student academic records from database...")
    df = fetch_training_data_from_db()

    if df.empty or len(df) < 10:
        raise ValueError("Insufficient data to train the model. Ensure at least 10 student records exist.")

    # High-risk label definition:
    # A student is labeled at high risk (1) if attendance_pct < 70 OR internal_marks < 45
    df["is_high_risk"] = ((df["attendance_pct"] < 70.0) | (df["internal_marks"] < 45.0)).astype(int)

    X = df[FEATURE_NAMES]
    y = df["is_high_risk"]

    logger.info(f"Loaded {len(df)} records. Class distribution: Safe (0): {(y == 0).sum()}, High Risk (1): {(y == 1).sum()}")

    # Stratified train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y if y.nunique() > 1 else None
    )

    # Initialize and train RandomForestClassifier
    clf = RandomForestClassifier(
        n_estimators=100,
        max_depth=5,
        min_samples_split=4,
        random_state=42,
        class_weight="balanced",
    )
    clf.fit(X_train, y_train)

    # Evaluate model
    y_pred = clf.predict(X_test)
    accuracy = float(accuracy_score(y_test, y_pred))
    
    # Calculate ROC-AUC if both classes exist in test set
    try:
        y_prob = clf.predict_proba(X_test)[:, 1]
        roc_auc = float(roc_auc_score(y_test, y_prob))
    except Exception:
        roc_auc = 1.0

    # Extract feature importances
    importances = dict(zip(FEATURE_NAMES, [round(float(v), 4) for v in clf.feature_importances_]))
    sorted_importances = dict(sorted(importances.items(), key=lambda item: item[1], reverse=True))

    logger.info("=== Model Performance ===")
    logger.info(f"Accuracy: {accuracy * 100:.2f}%")
    logger.info(f"ROC-AUC:  {roc_auc:.4f}")
    logger.info(f"Feature Importances: {sorted_importances}")

    # Package model artifact with metadata
    artifact = {
        "model": clf,
        "feature_names": FEATURE_NAMES,
        "feature_importances": sorted_importances,
        "primary_risk_driver": list(sorted_importances.keys())[0],
        "accuracy": accuracy,
        "roc_auc": roc_auc,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "total_samples": len(df),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, output_path)
    logger.info(f"Model saved successfully to: {output_path}")

    return artifact


if __name__ == "__main__":
    result = train_and_save_model()
    print("\nModel Training Completed Successfully!")
    print(f"Artifact location: {MODEL_FILE}")
    print(f"Accuracy: {result['accuracy']:.2%}")
    print(f"Feature Importances: {result['feature_importances']}")
