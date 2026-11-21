"""
Database Migration Utility: Migrate from Local SQLite to Remote Cloud Database (PostgreSQL / Neon / Supabase)
Copies all student profiles, academic logs, predictions, and intervention logs.
"""

import sys
import logging
from pathlib import Path
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.models import Base, Student, AcademicLog, RiskPrediction, InterventionLog

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("DB-Migrator")


def migrate_sqlite_to_remote(remote_url: str, sqlite_path: Path = None):
    """
    Migrates all data from local SQLite database to target remote database URL.
    """
    if sqlite_path is None:
        sqlite_path = ROOT_DIR / "retention_system.db"

    if not sqlite_path.exists():
        raise FileNotFoundError(f"Source SQLite database not found at: {sqlite_path}")

    # Fix postgres:// format for SQLAlchemy 2.0+
    if remote_url.startswith("postgres://"):
        remote_url = remote_url.replace("postgres://", "postgresql://", 1)

    logger.info(f"Connecting to source SQLite: {sqlite_path}")
    source_engine = create_engine(f"sqlite:///{sqlite_path}")
    SourceSession = sessionmaker(bind=source_engine)
    source_db = SourceSession()

    logger.info("Connecting to destination remote database...")
    dest_engine = create_engine(
        remote_url,
        pool_pre_ping=True,
        pool_recycle=300,
    )

    # Create tables in target database
    logger.info("Initializing schema tables in remote database...")
    Base.metadata.create_all(bind=dest_engine)

    DestSession = sessionmaker(bind=dest_engine)
    dest_db = DestSession()

    try:
        # 1. Migrate Students
        students = source_db.query(Student).all()
        logger.info(f"Found {len(students)} students in source SQLite. Transferring...")
        for s in students:
            existing = dest_db.query(Student).filter(Student.student_id == s.student_id).first()
            if not existing:
                dest_db.add(Student(
                    student_id=s.student_id,
                    name=s.name,
                    department=s.department,
                    enrollment_year=s.enrollment_year,
                    scholarship_status=s.scholarship_status
                ))
        dest_db.commit()

        # 2. Migrate Academic Logs
        logs = source_db.query(AcademicLog).all()
        logger.info(f"Found {len(logs)} academic logs in source SQLite. Transferring...")
        for l in logs:
            dest_db.add(AcademicLog(
                student_id=l.student_id,
                attendance_pct=l.attendance_pct,
                internal_marks=l.internal_marks,
                lms_logins=l.lms_logins,
                recorded_date=l.recorded_date
            ))
        dest_db.commit()

        # 3. Migrate Risk Predictions
        preds = source_db.query(RiskPrediction).all()
        logger.info(f"Found {len(preds)} risk predictions in source SQLite. Transferring...")
        for p in preds:
            dest_db.add(RiskPrediction(
                student_id=p.student_id,
                risk_score=p.risk_score,
                risk_level=p.risk_level,
                top_risk_factor=p.top_risk_factor,
                predicted_at=p.predicted_at
            ))
        dest_db.commit()

        # 4. Migrate Intervention Logs
        interventions = source_db.query(InterventionLog).all()
        logger.info(f"Found {len(interventions)} intervention logs in source SQLite. Transferring...")
        for i in interventions:
            dest_db.add(InterventionLog(
                student_id=i.student_id,
                advisor_notes=i.advisor_notes,
                ai_recovery_plan=i.ai_recovery_plan,
                status=i.status,
                created_at=i.created_at
            ))
        dest_db.commit()

        logger.info("=== Migration Completed Successfully! ===")
        print(f"Successfully migrated to remote database:")
        print(f"  - Students: {len(students)}")
        print(f"  - Academic Logs: {len(logs)}")
        print(f"  - Risk Predictions: {len(preds)}")
        print(f"  - Intervention Logs: {len(interventions)}")

    except Exception as e:
        dest_db.rollback()
        logger.error(f"Migration error: {e}")
        raise e
    finally:
        source_db.close()
        dest_db.close()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python etl/migrate_to_remote.py <REMOTE_DATABASE_URL>")
        sys.exit(1)
    remote_db_url = sys.argv[1]
    migrate_sqlite_to_remote(remote_db_url)

