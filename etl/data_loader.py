"""
ETL Pipeline: Academic Data Cleaner and Ingestion Loader
Reads, cleans, and upserts student profiles and academic logs into PostgreSQL / SQLite.
"""

import sys
import io
import logging
from pathlib import Path
from typing import Union
import pandas as pd
from sqlalchemy.orm import Session

# Add project root to path if running directly
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.database import SessionLocal, init_db
from app.models import Student, AcademicLog, utc_now

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("DataLoader")

REQUIRED_COLUMNS = [
    "student_id",
    "name",
    "department",
    "enrollment_year",
    "scholarship_status",
    "attendance_pct",
    "internal_marks",
    "lms_logins",
]


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans raw student dataframe: validates columns, coerces types, and trims strings.
    """
    # Normalize column names: lowercase and stripped
    df.columns = [str(col).strip().lower() for col in df.columns]

    # Verify required columns exist
    missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Uploaded CSV is missing mandatory columns: {', '.join(missing_cols)}")

    # Clean string columns
    df["student_id"] = df["student_id"].astype(str).str.strip()
    df["name"] = df["name"].astype(str).str.strip()
    df["department"] = df["department"].astype(str).str.strip()

    # Drop any entries missing critical identifiers
    df = df[df["student_id"] != ""].copy()
    df = df[df["name"] != ""].copy()

    # Coerce numeric types with boundaries
    df["enrollment_year"] = pd.to_numeric(df["enrollment_year"], errors="coerce").fillna(2023).astype(int)
    df["attendance_pct"] = pd.to_numeric(df["attendance_pct"], errors="coerce").fillna(0.0).clip(0.0, 100.0)
    df["internal_marks"] = pd.to_numeric(df["internal_marks"], errors="coerce").fillna(0.0).clip(0.0, 100.0)
    df["lms_logins"] = pd.to_numeric(df["lms_logins"], errors="coerce").fillna(0).clip(lower=0).astype(int)

    # Normalize scholarship_status
    df["scholarship_status"] = df["scholarship_status"].astype(str).str.strip().str.capitalize()
    df["scholarship_status"] = df["scholarship_status"].apply(lambda v: "Yes" if v in ["Yes", "Y", "True", "1"] else "No")

    return df


def load_csv_data(file_source: Union[str, Path, io.BytesIO, io.StringIO], db: Session) -> dict:
    """
    Ingests CSV file data into database using clean upsert semantics.
    Supports either file path string/Path or uploaded file bytes/stream.
    """
    if isinstance(file_source, (str, Path)):
        file_path = Path(file_source)
        if not file_path.exists():
            raise FileNotFoundError(f"Source file not found at: {file_path}")
        raw_df = pd.read_csv(file_path)
    else:
        # File-like object (e.g. from FastAPI UploadFile)
        raw_df = pd.read_csv(file_source)

    cleaned_df = clean_dataframe(raw_df)

    students_upserted = 0
    logs_created = 0

    try:
        for _, row in cleaned_df.iterrows():
            stu_id = str(row["student_id"])

            # 1. Upsert Student Record
            student = db.query(Student).filter(Student.student_id == stu_id).first()
            if student:
                # Update existing student metadata
                student.name = str(row["name"])
                student.department = str(row["department"])
                student.enrollment_year = int(row["enrollment_year"])
                student.scholarship_status = str(row["scholarship_status"])
            else:
                # Create new student record
                student = Student(
                    student_id=stu_id,
                    name=str(row["name"]),
                    department=str(row["department"]),
                    enrollment_year=int(row["enrollment_year"]),
                    scholarship_status=str(row["scholarship_status"]),
                )
                db.add(student)
            students_upserted += 1

            # 2. Add New Academic Log Record
            log_entry = AcademicLog(
                student_id=stu_id,
                attendance_pct=float(row["attendance_pct"]),
                internal_marks=float(row["internal_marks"]),
                lms_logins=int(row["lms_logins"]),
                recorded_date=utc_now(),
            )
            db.add(log_entry)
            logs_created += 1

        db.commit()
        logger.info(f"Successfully processed {len(cleaned_df)} records: {students_upserted} students, {logs_created} academic logs.")
        return {
            "status": "success",
            "total_records": len(cleaned_df),
            "students_upserted": students_upserted,
            "logs_created": logs_created,
        }

    except Exception as e:
        db.rollback()
        logger.error(f"ETL Load failed, transaction rolled back. Error: {e}")
        raise e


def run_standalone_loader(csv_file: str = None):
    """CLI runner for testing ETL pipeline."""
    init_db()
    if not csv_file:
        csv_file = ROOT_DIR / "etl" / "sample_students.csv"
        if not Path(csv_file).exists():
            print(f"Sample data not found. Generating now at {csv_file}...")
            from etl.generate_sample_data import generate_sample_dataset
            generate_sample_dataset(output_path=csv_file)

    db = SessionLocal()
    try:
        print(f"Starting data ingestion from: {csv_file}")
        result = load_csv_data(csv_file, db)
        print("Data Ingestion Complete!")
        print(f"  - Total records: {result['total_records']}")
        print(f"  - Students upserted: {result['students_upserted']}")
        print(f"  - Academic logs created: {result['logs_created']}")
    finally:
        db.close()


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else None
    run_standalone_loader(target)

