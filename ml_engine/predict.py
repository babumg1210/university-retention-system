"""
Machine Learning Engine: Academic Risk Predictor & Scoring Pipeline
Scores students, determines risk levels (High/Medium/Low), identifies top risk drivers,
and persists predictions to the database.
"""

import sys
import logging
from pathlib import Path
from datetime import datetime
import joblib
import pandas as pd
from sqlalchemy.orm import Session

# Ensure root directory is on python path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.database import SessionLocal, init_db
from app.models import Student, AcademicLog, RiskPrediction, utc_now
from ml_engine.train_model import MODEL_FILE, FEATURE_NAMES, train_and_save_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ML-Predictor")


def load_model_artifact(model_path: Path = MODEL_FILE):
    """
    Loads saved model artifact. If not found, automatically trains a new model.
    """
    if not model_path.exists():
        logger.warning(f"Model file not found at {model_path}. Auto-training initial model now...")
        train_and_save_model(model_path)
    return joblib.load(model_path)


def determine_top_risk_factor(attendance: float, marks: float, logins: int, risk_score: float) -> str:
    """
    Heuristically and analytically identifies the primary factor driving risk for an individual student.
    """
    attendance_flag = attendance < 70.0
    marks_flag = marks < 45.0
    logins_flag = logins < 15

    if attendance_flag and marks_flag:
        return f"Severe Compound Risk: Low Attendance ({attendance:.0f}%) & Marks ({marks:.0f}/100)"
    elif attendance_flag:
        return f"Low Attendance ({attendance:.0f}%, Below 70% Policy)"
    elif marks_flag:
        return f"Low Internal Marks ({marks:.0f}/100, Below 45 passing threshold)"
    elif logins_flag:
        return f"Low LMS Engagement ({logins} portal logins)"
    elif risk_score >= 0.40:
        # Borderline risk: find the weakest metric
        if attendance < 75.0:
            return f"Borderline Attendance ({attendance:.0f}%)"
        elif marks < 55.0:
            return f"Marginal Internal Marks ({marks:.0f}/100)"
        else:
            return f"Low LMS Activity ({logins} logins)"
    else:
        return "Stable Academic Performance (Good Standing)"


def run_batch_predictions(db: Session = None) -> dict:
    """
    Evaluates all students present in the database, generates risk scores,
    and updates or inserts risk predictions.
    """
    close_session_at_end = False
    if db is None:
        init_db()
        db = SessionLocal()
        close_session_at_end = True

    try:
        artifact = load_model_artifact()
        model = artifact["model"]

        students = db.query(Student).all()
        if not students:
            logger.warning("No students available to predict. Ingest student data first.")
            return {
                "status": "warning",
                "total_processed": 0,
                "high_risk": 0,
                "medium_risk": 0,
                "low_risk": 0,
                "message": "No students found in database to evaluate.",
            }

        records_to_predict = []
        student_obj_map = {}

        for s in students:
            log = s.latest_academic_log
            if log:
                records_to_predict.append({
                    "student_id": s.student_id,
                    "attendance_pct": float(log.attendance_pct),
                    "internal_marks": float(log.internal_marks),
                    "lms_logins": int(log.lms_logins),
                    "scholarship_encoded": 1 if s.scholarship_status == "Yes" else 0,
                })
                student_obj_map[s.student_id] = (s, log)

        if not records_to_predict:
            return {
                "status": "warning",
                "total_processed": 0,
                "high_risk": 0,
                "medium_risk": 0,
                "low_risk": 0,
                "message": "Students exist but have no academic logs recorded.",
            }

        pred_df = pd.DataFrame(records_to_predict)
        X = pred_df[FEATURE_NAMES]

        # Predict probability of being High Risk (Class 1)
        probabilities = model.predict_proba(X)[:, 1]

        high_count = 0
        med_count = 0
        low_count = 0

        now = utc_now()

        for idx, row in pred_df.iterrows():
            stu_id = row["student_id"]
            risk_score = float(probabilities[idx])
            attendance = float(row["attendance_pct"])
            marks = float(row["internal_marks"])
            logins = int(row["lms_logins"])

            # Classification thresholds
            if risk_score > 0.70 or attendance < 70.0 or marks < 45.0:
                risk_level = "High"
                high_count += 1
                # If deterministic rule triggered, ensure risk_score is visually high
                if risk_score < 0.70:
                    risk_score = max(risk_score, 0.75)
            elif risk_score >= 0.40:
                risk_level = "Medium"
                med_count += 1
            else:
                risk_level = "Low"
                low_count += 1

            top_factor = determine_top_risk_factor(attendance, marks, logins, risk_score)

            # Check if prediction record exists for student, upsert or create
            existing_pred = (
                db.query(RiskPrediction)
                .filter(RiskPrediction.student_id == stu_id)
                .order_by(RiskPrediction.predicted_at.desc())
                .first()
            )

            if existing_pred:
                existing_pred.risk_score = round(risk_score, 3)
                existing_pred.risk_level = risk_level
                existing_pred.top_risk_factor = top_factor
                existing_pred.predicted_at = now
            else:
                new_pred = RiskPrediction(
                    student_id=stu_id,
                    risk_score=round(risk_score, 3),
                    risk_level=risk_level,
                    top_risk_factor=top_factor,
                    predicted_at=now,
                )
                db.add(new_pred)

        db.commit()
        logger.info(
            f"Prediction batch complete. Total: {len(pred_df)} | High: {high_count} | Medium: {med_count} | Low: {low_count}"
        )

        return {
            "status": "success",
            "total_processed": len(pred_df),
            "high_risk": high_count,
            "medium_risk": med_count,
            "low_risk": low_count,
            "message": f"Successfully evaluated {len(pred_df)} students with ML Risk Engine.",
        }

    except Exception as e:
        db.rollback()
        logger.error(f"Error during risk prediction batch: {e}")
        raise e
    finally:
        if close_session_at_end:
            db.close()


if __name__ == "__main__":
    result = run_batch_predictions()
    print("\nPrediction Batch Summary:")
    print(f"Status: {result['status']}")
    print(f"Total Scored: {result.get('total_processed', 0)}")
    print(f"High Risk: {result.get('high_risk', 0)}")
    print(f"Medium Risk: {result.get('medium_risk', 0)}")
    print(f"Low Risk: {result.get('low_risk', 0)}")

