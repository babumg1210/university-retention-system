"""
Main Application Entrypoint: FastAPI & Jinja2 Web Application
Integrates PostgreSQL / SQLite Database, Scikit-Learn ML Risk Engine, and Bootstrap 5 Dashboard.
"""

import os
import io
import json
import logging
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Optional, List

from fastapi import FastAPI, Depends, Request, UploadFile, File, Form, HTTPException, status
from fastapi.responses import HTMLResponse, RedirectResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session
from sqlalchemy import func

from app.database import engine, get_db, init_db
from app.models import Student, AcademicLog, RiskPrediction, InterventionLog
from app.schemas import (
    StudentResponse,
    KPISummary,
    PredictionBatchResponse,
    UploadResponse,
)
from etl.data_loader import load_csv_data
from ml_engine.predict import run_batch_predictions
from ml_engine.train_model import train_and_save_model, MODEL_FILE
from ai_agents.advisor_agent import run_advisor_agent

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("RetentionApp")

# Base directories
BASE_DIR = Path(__file__).resolve().parent
TEMPLATES_DIR = BASE_DIR / "templates"
ROOT_DIR = BASE_DIR.parent
SAMPLE_CSV_PATH = ROOT_DIR / "etl" / "sample_students.csv"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager.
    Initializes database tables on startup.
    """
    logger.info("Initializing database tables...")
    init_db()
    # Check if sample CSV exists; if not, generate it so it's ready for sample download
    if not SAMPLE_CSV_PATH.exists():
        try:
            from etl.generate_sample_data import generate_sample_dataset
            generate_sample_dataset(output_path=SAMPLE_CSV_PATH)
        except Exception as e:
            logger.warning(f"Could not auto-generate sample dataset on startup: {e}")
    yield
    logger.info("Application shutdown complete.")


# Initialize FastAPI app
app = FastAPI(
    title="Intelligent Student Retention & Academic Risk Management System",
    description="Early-warning machine learning decision support system for university student retention.",
    version="1.0.0",
    lifespan=lifespan,
)

# Configure Jinja2 Templates
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------
def compute_kpi_metrics(students: List[Student]) -> dict:
    """
    Computes summary KPI indicators across the active student cohort.
    """
    total = len(students)
    if total == 0:
        return {
            "total_students": 0,
            "high_risk_count": 0,
            "medium_risk_count": 0,
            "low_risk_count": 0,
            "unassessed_count": 0,
            "avg_attendance": 0.0,
            "avg_internal_marks": 0.0,
            "high_risk_pct": 0.0,
        }

    high_risk = 0
    med_risk = 0
    low_risk = 0
    unassessed = 0

    attendance_values = []
    marks_values = []

    for s in students:
        pred = s.latest_risk_prediction
        if pred:
            if pred.risk_level == "High":
                high_risk += 1
            elif pred.risk_level == "Medium":
                med_risk += 1
            elif pred.risk_level == "Low":
                low_risk += 1
            else:
                unassessed += 1
        else:
            unassessed += 1

        log = s.latest_academic_log
        if log:
            attendance_values.append(log.attendance_pct)
            marks_values.append(log.internal_marks)

    avg_att = round(sum(attendance_values) / len(attendance_values), 1) if attendance_values else 0.0
    avg_marks = round(sum(marks_values) / len(marks_values), 1) if marks_values else 0.0
    high_pct = round((high_risk / total) * 100, 1)

    return {
        "total_students": total,
        "high_risk_count": high_risk,
        "medium_risk_count": med_risk,
        "low_risk_count": low_risk,
        "unassessed_count": unassessed,
        "avg_attendance": avg_att,
        "avg_internal_marks": avg_marks,
        "high_risk_pct": high_pct,
    }


def format_student_table_items(students: List[Student]) -> list:
    """
    Transforms SQLAlchemy Student models into flattened dictionaries for robust Jinja2 rendering.
    """
    items = []
    for s in students:
        log = s.latest_academic_log
        pred = s.latest_risk_prediction

        intervention = s.latest_intervention

        items.append({
            "student_id": s.student_id,
            "name": s.name,
            "department": s.department,
            "enrollment_year": s.enrollment_year,
            "scholarship_status": s.scholarship_status,
            "attendance_pct": float(log.attendance_pct) if log else None,
            "internal_marks": float(log.internal_marks) if log else None,
            "lms_logins": int(log.lms_logins) if log else None,
            "risk_score": float(pred.risk_score) if pred else None,
            "risk_level": pred.risk_level if pred else "Unassessed",
            "top_risk_factor": pred.top_risk_factor if pred else "Pending Prediction",
            "predicted_at": pred.predicted_at if pred else None,
            "has_intervention": bool(intervention),
            "intervention_id": intervention.id if intervention else None,
            "intervention_status": intervention.status if intervention else None,
        })
    return items


# -----------------------------------------------------------------------------
# Frontend Routes (HTML / Jinja2)
# -----------------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse, summary="Academic Retention Dashboard")
def dashboard_view(
    request: Request,
    message: Optional[str] = None,
    alert_type: Optional[str] = "success",
    db: Session = Depends(get_db),
):
    """
    Renders the primary executive retention dashboard with KPIs, visualizations,
    and a filterable student list with risk badges.
    """
    students = db.query(Student).all()
    kpi = compute_kpi_metrics(students)
    student_items = format_student_table_items(students)

    # Collect distinct departments for the dropdown filter
    departments = sorted(list({s.department for s in students}))

    return templates.TemplateResponse(
        request=request,
        name="dashboard.html",
        context={
            "students": student_items,
            "kpi": kpi,
            "departments": departments,
            "message": message,
            "alert_type": alert_type,
        },
    )


@app.get("/upload", response_class=HTMLResponse, summary="CSV Ingestion Page")
def upload_page_view(request: Request):
    """
    Renders the CSV upload and schema instructions interface.
    """
    return templates.TemplateResponse(request=request, name="upload.html", context={})


@app.post("/upload", summary="Handle CSV File Upload")
async def handle_csv_upload(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    """
    Ingests an uploaded institutional CSV file, parses and cleans records,
    and upserts into the database. Redirects back to dashboard.
    """
    if not file.filename.endswith(".csv"):
        return RedirectResponse(
            url="/?message=Invalid+file+type.+Please+upload+a+standard+.csv+file.&alert_type=danger",
            status_code=status.HTTP_303_SEE_OTHER,
        )

    try:
        contents = await file.read()
        buffer = io.BytesIO(contents)
        result = load_csv_data(buffer, db)

        msg = (
            f"Successfully+ingested+{result['total_records']}+records.+Updated+"
            f"{result['students_upserted']}+students+and+created+{result['logs_created']}+academic+logs."
        )
        return RedirectResponse(
            url=f"/?message={msg}&alert_type=success",
            status_code=status.HTTP_303_SEE_OTHER,
        )
    except Exception as e:
        logger.error(f"Upload failed: {e}")
        return RedirectResponse(
            url=f"/?message=Failed+to+process+file:+{str(e)[:100]}&alert_type=danger",
            status_code=status.HTTP_303_SEE_OTHER,
        )


@app.get("/download-sample", summary="Download Sample CSV Dataset")
def download_sample_csv():
    """
    Allows users to download the realistic sample CSV dataset directly from the web interface.
    """
    if not SAMPLE_CSV_PATH.exists():
        from etl.generate_sample_data import generate_sample_dataset
        generate_sample_dataset(output_path=SAMPLE_CSV_PATH)

    return FileResponse(
        path=str(SAMPLE_CSV_PATH),
        filename="sample_students.csv",
        media_type="text/csv",
    )


# -----------------------------------------------------------------------------
# REST API Endpoints (Swagger /docs Compatible)
# -----------------------------------------------------------------------------
@app.post(
    "/api/run-predictions",
    response_model=PredictionBatchResponse,
    tags=["Machine Learning"],
    summary="Trigger AI Risk Prediction Engine",
)
def api_run_predictions(db: Session = Depends(get_db)):
    """
    Runs the Scikit-Learn Random Forest scoring model against all students in the database.
    Updates the risk_predictions table with predicted risk scores, classifications,
    and primary risk drivers.
    """
    try:
        result = run_batch_predictions(db)
        return PredictionBatchResponse(
            status=result["status"],
            total_processed=result.get("total_processed", 0),
            high_risk=result.get("high_risk", 0),
            medium_risk=result.get("medium_risk", 0),
            low_risk=result.get("low_risk", 0),
            message=result.get("message", "Scoring completed."),
        )
    except Exception as e:
        logger.error(f"Error during API prediction call: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction Engine failed: {str(e)}",
        )


@app.post(
    "/api/train-model",
    tags=["Machine Learning"],
    summary="Retrain Random Forest Risk Model",
)
def api_train_model():
    """
    Retrains the Random Forest classifier using currently recorded database logs
    and serializes the new model to disk.
    """
    try:
        artifact = train_and_save_model()
        return {
            "status": "success",
            "message": "Model retrained and saved successfully.",
            "accuracy": artifact["accuracy"],
            "roc_auc": artifact["roc_auc"],
            "feature_importances": artifact["feature_importances"],
            "primary_risk_driver": artifact["primary_risk_driver"],
            "trained_at": artifact["trained_at"],
        }
    except Exception as e:
        logger.error(f"Retraining failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Retraining failed: {str(e)}",
        )


@app.get(
    "/api/kpi",
    response_model=KPISummary,
    tags=["Analytics"],
    summary="Get Cohort KPI Statistics",
)
def api_get_kpis(db: Session = Depends(get_db)):
    """
    Returns current high-level KPI retention metrics as JSON.
    """
    students = db.query(Student).all()
    kpi = compute_kpi_metrics(students)
    return KPISummary(**kpi)


@app.get(
    "/api/students",
    tags=["Analytics"],
    summary="List All Students With Academic Indicators",
)
def api_list_students(db: Session = Depends(get_db)):
    """
    Returns the complete list of students with their most recent academic metrics
    and risk scores.
    """
    students = db.query(Student).all()
    return format_student_table_items(students)


@app.get(
    "/api/seed-sample",
    tags=["Utility"],
    summary="Seed Database with 200 Sample Students",
)
def api_seed_sample(db: Session = Depends(get_db)):
    """
    Helper utility to generate and seed 200 sample students and auto-run predictions.
    """
    if not SAMPLE_CSV_PATH.exists():
        from etl.generate_sample_data import generate_sample_dataset
        generate_sample_dataset(output_path=SAMPLE_CSV_PATH)

    load_result = load_csv_data(SAMPLE_CSV_PATH, db)
    pred_result = run_batch_predictions(db)

    return RedirectResponse(
        url=f"/?message=Database+seeded+with+{load_result['total_records']}+students.+Predictions+executed+successfully!&alert_type=success",
        status_code=status.HTTP_303_SEE_OTHER,
    )


# -----------------------------------------------------------------------------
# AI Advisor Agent Endpoints (Phase 3)
# -----------------------------------------------------------------------------
@app.post(
    "/api/students/{student_id}/generate-plan",
    tags=["AI Advisor Agent"],
    summary="Generate AI Academic Recovery Plan via LangGraph Workflow",
)
def api_generate_recovery_plan(student_id: str, db: Session = Depends(get_db)):
    """
    Triggers the LangGraph AI Advisor Agent workflow:
    1. Fetches student performance and ML risk profile.
    2. Queries ChromaDB Handbook RAG for relevant institutional policies.
    3. Generates 3-step structured recovery plan and personalized email draft.
    4. Saves intervention record to the intervention_logs database table.
    """
    try:
        result = run_advisor_agent(student_id, db=db)
        if result.get("status") == "error":
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=result.get("message", f"Failed to generate recovery plan for {student_id}"),
            )
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error executing AI Advisor Agent for {student_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"AI Advisor Agent execution failed: {str(e)}",
        )


@app.get(
    "/api/students/{student_id}/latest-plan",
    tags=["AI Advisor Agent"],
    summary="Get Most Recent AI Recovery Plan For Student",
)
def api_get_latest_recovery_plan(student_id: str, db: Session = Depends(get_db)):
    """
    Fetches the latest recorded AI recovery plan and email draft from intervention_logs.
    """
    intervention = (
        db.query(InterventionLog)
        .filter(InterventionLog.student_id == student_id)
        .order_by(InterventionLog.created_at.desc())
        .first()
    )
    if not intervention:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No previous intervention plan found for student {student_id}",
        )

    try:
        parsed_payload = json.loads(intervention.ai_recovery_plan) if intervention.ai_recovery_plan else {}
    except Exception:
        parsed_payload = {"raw_plan": intervention.ai_recovery_plan}

    return {
        "status": "success",
        "student_id": student_id,
        "intervention_id": intervention.id,
        "intervention_status": intervention.status,
        "created_at": intervention.created_at.isoformat() if intervention.created_at else None,
        "advisor_notes": intervention.advisor_notes,
        "plan_data": parsed_payload,
    }
