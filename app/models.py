from datetime import datetime, timezone
from sqlalchemy import Column, String, Integer, Float, DateTime, ForeignKey, Text
from sqlalchemy.orm import relationship
from app.database import Base


def utc_now():
    return datetime.now(timezone.utc)


class Student(Base):
    __tablename__ = "students"

    student_id = Column(String(50), primary_key=True, index=True)
    name = Column(String(150), nullable=False)
    department = Column(String(100), nullable=False, index=True)
    enrollment_year = Column(Integer, nullable=False)
    scholarship_status = Column(String(10), default="No", nullable=False)  # "Yes" or "No"

    # Relationships
    academic_logs = relationship(
        "AcademicLog",
        back_populates="student",
        cascade="all, delete-orphan",
        order_by="desc(AcademicLog.recorded_date)",
    )
    risk_predictions = relationship(
        "RiskPrediction",
        back_populates="student",
        cascade="all, delete-orphan",
        order_by="desc(RiskPrediction.predicted_at)",
    )
    intervention_logs = relationship(
        "InterventionLog",
        back_populates="student",
        cascade="all, delete-orphan",
        order_by="desc(InterventionLog.created_at)",
    )

    @property
    def latest_academic_log(self):
        return self.academic_logs[0] if self.academic_logs else None

    @property
    def latest_risk_prediction(self):
        return self.risk_predictions[0] if self.risk_predictions else None

    @property
    def latest_intervention(self):
        return self.intervention_logs[0] if self.intervention_logs else None

    def __repr__(self):
        return f"<Student(id={self.student_id}, name='{self.name}', dept='{self.department}')>"


class AcademicLog(Base):
    __tablename__ = "academic_logs"

    id = Column(Integer, primary_key=True, autoincrement=True, index=True)
    student_id = Column(
        String(50),
        ForeignKey("students.student_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    attendance_pct = Column(Float, nullable=False)
    internal_marks = Column(Float, nullable=False)
    lms_logins = Column(Integer, nullable=False, default=0)
    recorded_date = Column(DateTime, default=utc_now, nullable=False)

    student = relationship("Student", back_populates="academic_logs")

    def __repr__(self):
        return (
            f"<AcademicLog(student_id={self.student_id}, "
            f"attendance={self.attendance_pct}%, marks={self.internal_marks})>"
        )


class RiskPrediction(Base):
    __tablename__ = "risk_predictions"

    id = Column(Integer, primary_key=True, autoincrement=True, index=True)
    student_id = Column(
        String(50),
        ForeignKey("students.student_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    risk_score = Column(Float, nullable=False)  # 0.0 to 1.0
    risk_level = Column(String(20), nullable=False, index=True)  # "High", "Medium", "Low"
    top_risk_factor = Column(String(150), nullable=False)  # e.g., "Low Attendance (<70%)"
    predicted_at = Column(DateTime, default=utc_now, nullable=False)

    student = relationship("Student", back_populates="risk_predictions")

    def __repr__(self):
        return (
            f"<RiskPrediction(student_id={self.student_id}, "
            f"level='{self.risk_level}', score={self.risk_score:.2f})>"
        )


class InterventionLog(Base):
    __tablename__ = "intervention_logs"

    id = Column(Integer, primary_key=True, autoincrement=True, index=True)
    student_id = Column(
        String(50),
        ForeignKey("students.student_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    advisor_notes = Column(Text, nullable=True)
    ai_recovery_plan = Column(Text, nullable=True)
    status = Column(String(50), default="Pending", nullable=False)  # "Pending", "In Progress", "Completed"
    created_at = Column(DateTime, default=utc_now, nullable=False)

    student = relationship("Student", back_populates="intervention_logs")

    def __repr__(self):
        return f"<InterventionLog(student_id={self.student_id}, status='{self.status}')>"

