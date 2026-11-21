from datetime import datetime
from typing import Optional, List
from pydantic import BaseModel, ConfigDict, Field


# -----------------------------------------------------------------------------
# Academic Log Schemas
# -----------------------------------------------------------------------------
class AcademicLogBase(BaseModel):
    attendance_pct: float = Field(..., ge=0.0, le=100.0, description="Attendance percentage (0-100)")
    internal_marks: float = Field(..., ge=0.0, le=100.0, description="Internal assessment marks (0-100)")
    lms_logins: int = Field(..., ge=0, description="Number of logins into the LMS portal")


class AcademicLogCreate(AcademicLogBase):
    student_id: str


class AcademicLogResponse(AcademicLogBase):
    id: int
    student_id: str
    recorded_date: datetime

    model_config = ConfigDict(from_attributes=True)


# -----------------------------------------------------------------------------
# Risk Prediction Schemas
# -----------------------------------------------------------------------------
class RiskPredictionBase(BaseModel):
    risk_score: float = Field(..., ge=0.0, le=1.0, description="Probability of dropout/academic failure (0.0 - 1.0)")
    risk_level: str = Field(..., description="High, Medium, or Low")
    top_risk_factor: str = Field(..., description="Primary risk driver identified by ML/heuristics")


class RiskPredictionCreate(RiskPredictionBase):
    student_id: str


class RiskPredictionResponse(RiskPredictionBase):
    id: int
    student_id: str
    predicted_at: datetime

    model_config = ConfigDict(from_attributes=True)


# -----------------------------------------------------------------------------
# Intervention Log Schemas
# -----------------------------------------------------------------------------
class InterventionLogBase(BaseModel):
    advisor_notes: Optional[str] = None
    ai_recovery_plan: Optional[str] = None
    status: str = Field(default="Pending", description="Pending, In Progress, Completed")


class InterventionLogCreate(InterventionLogBase):
    student_id: str


class InterventionLogResponse(InterventionLogBase):
    id: int
    student_id: str
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


# -----------------------------------------------------------------------------
# Student Schemas
# -----------------------------------------------------------------------------
class StudentBase(BaseModel):
    student_id: str = Field(..., description="Unique university student identification code (e.g., STU1001)")
    name: str = Field(..., min_length=2, max_length=150, description="Full student name")
    department: str = Field(..., description="Academic department or major")
    enrollment_year: int = Field(..., ge=2000, le=2100, description="Year student matriculated")
    scholarship_status: str = Field(default="No", description="Yes or No")


class StudentCreate(StudentBase):
    pass


class StudentResponse(StudentBase):
    academic_logs: List[AcademicLogResponse] = []
    risk_predictions: List[RiskPredictionResponse] = []
    intervention_logs: List[InterventionLogResponse] = []

    model_config = ConfigDict(from_attributes=True)


# -----------------------------------------------------------------------------
# Composite & Dashboard Schemas
# -----------------------------------------------------------------------------
class StudentDashboardItem(BaseModel):
    student_id: str
    name: str
    department: str
    enrollment_year: int
    scholarship_status: str
    attendance_pct: Optional[float] = None
    internal_marks: Optional[float] = None
    lms_logins: Optional[int] = None
    risk_score: Optional[float] = None
    risk_level: Optional[str] = "Unassessed"
    top_risk_factor: Optional[str] = "Pending Prediction"
    predicted_at: Optional[datetime] = None
    intervention_status: Optional[str] = "None"

    model_config = ConfigDict(from_attributes=True)


class KPISummary(BaseModel):
    total_students: int
    high_risk_count: int
    medium_risk_count: int
    low_risk_count: int
    unassessed_count: int
    avg_attendance: float
    avg_internal_marks: float
    high_risk_pct: float


class PredictionBatchResponse(BaseModel):
    status: str
    total_processed: int
    high_risk: int
    medium_risk: int
    low_risk: int
    message: str


class UploadResponse(BaseModel):
    status: str
    filename: str
    total_rows: int
    students_upserted: int
    logs_created: int
    message: str

