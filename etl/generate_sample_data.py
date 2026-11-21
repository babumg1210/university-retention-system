"""
ETL Pipeline: Sample Student Academic Data Generator
Generates realistic university student academic profiles using Faker and Pandas.
"""

import os
import random
from pathlib import Path
import pandas as pd
from faker import Faker

# Set random seeds for reproducibility
SEED = 42
random.seed(SEED)
fake = Faker()
Faker.seed(SEED)

# Academic configuration constants
DEPARTMENTS = [
    "Computer Science & Engineering",
    "Electrical & Electronics Engineering",
    "Mechanical Engineering",
    "Business Administration",
]

ENROLLMENT_YEARS = [2021, 2022, 2023, 2024]
TOTAL_STUDENTS = 200


def generate_student_record(index: int) -> dict:
    """
    Generates a single student record with correlated risk characteristics.
    """
    student_id = f"STU{1001 + index}"
    name = fake.name()
    department = random.choice(DEPARTMENTS)
    enrollment_year = random.choice(ENROLLMENT_YEARS)
    scholarship_status = "Yes" if random.random() < 0.30 else "No"

    # Multi-modal distribution to create realistic academic profiles
    profile_type = random.choices(
        ["safe", "attendance_risk", "academic_risk", "severe_compound_risk"],
        weights=[0.60, 0.15, 0.15, 0.10],
        k=1,
    )[0]

    if profile_type == "safe":
        # Safe students: Good attendance, solid internal marks, active LMS logins
        attendance_pct = round(random.uniform(75.0, 98.5), 1)
        internal_marks = round(random.uniform(60.0, 96.0), 1)
        lms_logins = random.randint(25, 90)

    elif profile_type == "attendance_risk":
        # Chronic absenteeism: Attendance < 70% but marks may still be marginally passing
        attendance_pct = round(random.uniform(38.0, 68.5), 1)
        internal_marks = round(random.uniform(48.0, 72.0), 1)
        lms_logins = random.randint(8, 28)

    elif profile_type == "academic_risk":
        # Failing internal assessments: Attendance ok, but marks < 45
        attendance_pct = round(random.uniform(71.0, 89.0), 1)
        internal_marks = round(random.uniform(22.0, 44.0), 1)
        lms_logins = random.randint(10, 35)

    else:
        # Severe compound risk: Low attendance (<70%), failing marks (<45), disengaged LMS
        attendance_pct = round(random.uniform(30.0, 65.0), 1)
        internal_marks = round(random.uniform(18.0, 42.0), 1)
        lms_logins = random.randint(1, 12)

    return {
        "student_id": student_id,
        "name": name,
        "department": department,
        "enrollment_year": enrollment_year,
        "scholarship_status": scholarship_status,
        "attendance_pct": attendance_pct,
        "internal_marks": internal_marks,
        "lms_logins": lms_logins,
    }


def generate_sample_dataset(num_students: int = TOTAL_STUDENTS, output_path: str = None) -> pd.DataFrame:
    """
    Generates synthetic dataset and writes to CSV.
    """
    records = [generate_student_record(i) for i in range(num_students)]
    df = pd.DataFrame(records)

    if output_path is None:
        base_dir = Path(__file__).resolve().parent
        output_path = base_dir / "sample_students.csv"
    else:
        output_path = Path(output_path)

    # Ensure parent directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Generated {len(df)} student records saved to: {output_path}")

    # Also save a copy to project root for easy user discovery
    root_csv = output_path.parent.parent / "sample_students.csv"
    try:
        df.to_csv(root_csv, index=False)
        print(f"Convenience copy saved to: {root_csv}")
    except Exception as e:
        print(f"Could not copy to root: {e}")

    # Print summary statistics
    high_risk_rule = (df["attendance_pct"] < 70) | (df["internal_marks"] < 45)
    print("\nDataset Summary Statistics:")
    print(f"  - Total Students: {len(df)}")
    print(f"  - Departments: {df['department'].nunique()}")
    print(f"  - Potential At-Risk Count (Attendance < 70 or Marks < 45): {high_risk_rule.sum()} ({high_risk_rule.mean():.1%})")
    print(f"  - Mean Attendance: {df['attendance_pct'].mean():.2f}%")
    print(f"  - Mean Internal Marks: {df['internal_marks'].mean():.2f}")
    print(f"  - Mean LMS Logins: {df['lms_logins'].mean():.1f}")

    return df


if __name__ == "__main__":
    generate_sample_dataset()

