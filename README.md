# 🎓 Intelligent Student Retention & Academic Risk Management System

[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688.svg)](https://fastapi.tiangolo.com)
[![SQLAlchemy](https://img.shields.io/badge/SQLAlchemy-2.0+-red.svg)](https://www.sqlalchemy.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.5+-orange.svg)](https://scikit-learn.org/)
[![Bootstrap](https://img.shields.io/badge/Bootstrap-5.3-purple.svg)](https://getbootstrap.com/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An open-source, enterprise-grade decision support platform designed for universities, colleges, and higher education institutions. The system tracks academic engagement, identifies at-risk students before midterms, and assists academic advisors in executing timely interventions.

---

## 📌 Table of Contents
- [Executive Overview](#-executive-overview)
- [System Architecture](#-system-architecture)
- [Folder Structure](#-folder-structure)
- [Beginner's Step-by-Step Setup Guide](#-beginners-step-by-step-setup-guide)
  - [Prerequisites](#1-prerequisites)
  - [Clone or Navigate to the Project](#2-navigate-to-the-project)
  - [Create and Activate Virtual Environment](#3-create-and-activate-virtual-environment)
  - [Install Dependencies](#4-install-dependencies)
  - [Configure the Database (.env)](#5-configure-the-database-env)
- [Running the System](#-running-the-system)
  - [Step 1: Generate Synthetic Student Records](#step-1-generate-synthetic-student-records)
  - [Step 2: Load Data into PostgreSQL / SQLite](#step-2-load-data-into-postgresql--sqlite)
  - [Step 3: Train the Machine Learning Model](#step-3-train-the-machine-learning-model)
  - [Step 4: Launch the Web Application](#step-4-launch-the-web-application)
- [Using the Dashboard & Features](#-using-the-dashboard--features)
- [REST API Reference (Swagger UI)](#-rest-api-reference-swagger-ui)
- [Troubleshooting & FAQ](#-troubleshooting--faq)

---

## 🌟 Executive Overview

Universities lose an estimated 15%–30% of their enrolled cohort between the first and second years due to academic distress, disengagement, and lack of timely support. 

This platform continuously monitors three critical behavioral and academic vectors:
1. **Attendance Rate (`attendance_pct`):** Chronic absenteeism (&lt; 70%) is statistically the leading precursor to course failure.
2. **Internal Assessment Marks (`internal_marks`):** Continuous assessment scoring (&lt; 45 / 100) indicates foundational comprehension gaps.
3. **LMS Activity (`lms_logins`):** Portal login frequency exposes disengagement weeks before physical attendance drops.

Using a **Random Forest Classifier**, the system computes a multi-variate risk probability for each student, assigns them to **High**, **Medium**, or **Low** risk categories, identifies the specific root-cause driver, and presents actionable insights to university leadership.

---

## 🏗️ System Architecture

```text
  [ Institutional CSV / SIS ]
             │
             ▼
   [ ETL Cleaner & Upsert ]  ──> [ PostgreSQL / SQLite Database ]
             │                                   │
             ▼                                   ▼
 [ Scikit-Learn Random Forest ] ──> [ Risk Predictions Table ]
             │                                   │
             ▼                                   ▼
 [ FastAPI Web Server / API ]  ──> [ Bootstrap 5 Jinja2 Dashboard ]
```

---

## 📁 Folder Structure

```text
university-retention-system/
├── app/
│   ├── __init__.py             # Application package marker
│   ├── database.py             # SQLAlchemy engine, session maker, and DB init
│   ├── models.py               # 4 Relational tables: Student, AcademicLog, RiskPrediction, InterventionLog
│   ├── schemas.py              # Pydantic v2 validation models
│   ├── main.py                 # FastAPI web application routes, templates & REST API
│   └── templates/
│       ├── base.html           # Master layout with Bootstrap 5.3, icons, and navbar
│       ├── dashboard.html      # Primary dashboard: KPIs, Donut Chart, filters & student table
│       └── upload.html         # CSV file upload and schema reference
├── etl/
│   ├── __init__.py             # ETL package marker
│   ├── generate_sample_data.py # Realistic synthetic data generator using Faker & Pandas (200 students)
│   ├── data_loader.py          # Data cleaner and upsert loader for PostgreSQL/SQLite
│   └── sample_students.csv     # Generated sample dataset
├── ml_engine/
│   ├── __init__.py             # ML engine package marker
│   ├── train_model.py          # Random Forest trainer, feature importance extractor & model exporter
│   ├── predict.py              # Batch risk scoring engine and root-cause classifier
│   └── risk_model.pkl          # Serialized scikit-learn model artifact
├── ai_agents/                  # Placeholder for future LangGraph / Ollama autonomous agents
│   └── __init__.py
├── .env.example                # Sample environment configuration template
├── .env                        # Active local environment settings
├── requirements.txt            # Python package dependencies
├── sample_students.csv         # Convenience root copy of sample dataset
└── README.md                   # Comprehensive documentation and quick-start guide
```

---

## 🚀 Beginner's Step-by-Step Setup Guide

> **Are you non-technical?** Follow these exact steps one by one. You do **not** need prior database or machine learning experience!

### 1. Prerequisites
Ensure you have **Python 3.10 or newer** installed.
- Check by opening **PowerShell** (Windows) or **Terminal** (Mac/Linux) and typing:
  ```powershell
  python --version
  # OR on Windows if python gives an alias error:
  py --version
  ```

---

### 2. Navigate to the Project
Open PowerShell or your terminal and change directory into the project folder:
```powershell
cd C:\Users\admin\.gemini\antigravity\scratch\university-retention-system
```

---

### 3. Create and Activate Virtual Environment

A virtual environment keeps your project packages isolated from the rest of your computer.

#### On Windows (PowerShell):
```powershell
# 1. Create the virtual environment
py -m venv venv

# 2. Enable script execution (if Windows blocks activation):
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass

# 3. Activate the virtual environment:
.\venv\Scripts\Activate.ps1
```
*(When activated, you will see `(venv)` at the beginning of your terminal line).*

#### On macOS / Linux:
```bash
python3 -m venv venv
source venv/bin/activate
```

---

### 4. Install Dependencies

With your virtual environment active, run:
```powershell
pip install -r requirements.txt
```
*(Or if running directly with the virtual environment Python: `.\venv\Scripts\python.exe -m pip install -r requirements.txt`)*

---

### 5. Configure the Database (.env)

The project includes an automatic `.env` configuration that works **immediately out-of-the-box with SQLite** (no database setup required).

#### Option A: Quick Local Testing (Zero Setup Required - Default)
In your `.env` file, the default is already:
```ini
DATABASE_URL=sqlite:///./retention_system.db
```
*No external software, servers, or passwords needed!*

#### Option B: Connecting to Cloud PostgreSQL (Neon / Supabase / AWS RDS)
If you want to use a managed cloud PostgreSQL instance:
1. Create a free database at [Neon.tech](https://neon.tech) or [Supabase.com](https://supabase.com).
2. Copy your connection URL.
3. Open the `.env` file in the project folder and paste your connection string:
   ```ini
   DATABASE_URL=postgresql://neondb_owner:YOUR_PASSWORD@ep-xyz-1234.us-east-2.aws.neon.tech/neondb?sslmode=require
   ```
*(Note: The system automatically handles both `postgres://` and `postgresql://` formats).*

---

## ⚡ Running the System

Execute these 4 simple steps to populate the system and start the server:

### Step 1: Generate Synthetic Student Records
Generates 200 realistic student academic records across 4 university departments:
```powershell
python etl/generate_sample_data.py
# (Or: .\venv\Scripts\python.exe etl\generate_sample_data.py)
```
*Output: Generates `sample_students.csv` with attendance, internal marks, and LMS logins.*

---

### Step 2: Load Data into PostgreSQL / SQLite
Cleans, validates, and upserts student profiles and academic logs into your database:
```powershell
python etl/data_loader.py
# (Or: .\venv\Scripts\python.exe etl\data_loader.py)
```
*Output: `Successfully processed 200 records: 200 students, 200 academic logs`.*

---

### Step 3: Train the Machine Learning Model
Queries historical records from the database, trains a `RandomForestClassifier`, calculates feature importances, and saves `ml_engine/risk_model.pkl`:
```powershell
python ml_engine/train_model.py
# (Or: .\venv\Scripts\python.exe ml_engine/train_model.py)
```
*Output: `Model Training Completed Successfully! Accuracy: 100%`.*

---

### Step 4: Launch the Web Application
Start the high-performance FastAPI server with automatic code reloading:
```powershell
uvicorn app.main:app --reload
# (Or: .\venv\Scripts\python.exe -m uvicorn app.main:app --reload)
```

Now open your web browser and visit:
👉 **[http://127.0.0.1:8000](http://127.0.0.1:8000)**

---

## 🖥️ Using the Dashboard & Features

Once you open **http://127.0.0.1:8000**, you have access to:

1. **Executive KPI Cards:**
   - **Cohort Size:** Total number of monitored students.
   - **High Risk Count:** Students triggering critical academic danger.
   - **Medium Risk Count:** Borderline students needing academic advising.
   - **Low Risk Count:** Students in good academic standing.
   - **Cohort Average Attendance:** Target standard &ge; 75%.
   - **Cohort Average Marks:** Passing standard &ge; 45 / 100.

2. **Risk Distribution Donut Chart:**
   - Real-time interactive breakdown using Chart.js.

3. **One-Click AI Risk Scoring:**
   - Click the yellow **"Run AI Risk Scoring"** button on the top-right navbar anytime to trigger inference across all student profiles.

4. **Multi-Column Filtering & Instant Search:**
   - **Search Box:** Type a student's name, ID (e.g. `STU1042`), or risk driver to filter rows in real time.
   - **Department Filter:** Select any of the 4 departments (e.g. Computer Science, Mechanical Engineering).
   - **Risk Filter:** Instantly isolate High Risk students.

5. **Student Detail Modal:**
   - Click **"Details"** on any student row to view their individual diagnostic scorecard, attendance gauge, primary risk factor, and recommended intervention plan.

6. **Institutional CSV Ingestion:**
   - Click **"Ingest Data"** or visit **[http://127.0.0.1:8000/upload](http://127.0.0.1:8000/upload)** to upload new student spreadsheets.

7. **Export Reports:**
   - Click **"Export Report"** on the dashboard to export the currently filtered table view directly into a `.csv` file.

---

## 🔌 REST API Reference (Swagger UI)

FastAPI automatically generates interactive API documentation.
Visit: **[http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)**

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Renders the HTML executive retention dashboard |
| `GET` | `/upload` | Renders the CSV data ingestion form |
| `POST` | `/upload` | Uploads institutional CSV and upserts records |
| `GET` | `/download-sample` | Downloads the verified 200-student sample CSV |
| `POST` | `/api/run-predictions` | Executes Random Forest inference and updates risk scores |
| `POST` | `/api/train-model` | Retrains Random Forest model on database logs |
| `GET` | `/api/kpi` | Returns cohort summary statistics in JSON |
| `GET` | `/api/students` | Returns all students with academic logs in JSON |
| `GET` | `/api/seed-sample` | Quick helper to reset and seed 200 sample students |

---

## 🛠️ Database Schema Overview

The database contains four relational tables:

```
┌───────────────────────┐         ┌───────────────────────┐
│       Student         │         │     AcademicLog       │
├───────────────────────┤         ├───────────────────────┤
│ student_id (PK)       │<───┬───┤ id (PK)               │
│ name                  │    │    │ student_id (FK)       │
│ department            │    │    │ attendance_pct        │
│ enrollment_year       │    │    │ internal_marks        │
│ scholarship_status    │    │    │ lms_logins            │
└───────────────────────┘    │    │ recorded_date         │
                             │    └───────────────────────┘
                             │
                             │    ┌───────────────────────┐
                             │    │    RiskPrediction     │
                             │    ├───────────────────────┤
                             ├───┤ id (PK)               │
                             │    │ student_id (FK)       │
                             │    │ risk_score            │
                             │    │ risk_level            │
                             │    │ top_risk_factor       │
                             │    │ predicted_at          │
                             │    └───────────────────────┘
                             │
                             │    ┌───────────────────────┐
                             │    │   InterventionLog     │
                             │    ├───────────────────────┤
                             └───┤ id (PK)               │
                                  │ student_id (FK)       │
                                  │ advisor_notes         │
                                  │ ai_recovery_plan      │
                                  │ status                │
                                  │ created_at            │
                                  └───────────────────────┘
```

---

## ❓ Troubleshooting & FAQ

### Q1: When activating the virtual environment, I get an "Execution_Policies" error on Windows.
**Fix:** Run this command in PowerShell before activating:
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

### Q2: Port 8000 is already in use.
**Fix:** Specify a different port when starting uvicorn:
```powershell
uvicorn app.main:app --reload --port 8080
```
Then visit `http://127.0.0.1:8080`.

### Q3: How do I test the system without installing PostgreSQL?
**Fix:** You don't need to install anything! The system defaults to **SQLite** (`sqlite:///./retention_system.db`), which runs completely locally inside the project folder without any external database server.

### Q4: I uploaded a CSV and got a column error.
**Fix:** Ensure your CSV contains these exact header names:
`student_id`, `name`, `department`, `enrollment_year`, `scholarship_status`, `attendance_pct`, `internal_marks`, `lms_logins`. You can download the verified template anytime from **[http://127.0.0.1:8000/download-sample](http://127.0.0.1:8000/download-sample)**.

---

## 📄 License
This project is open-source under the **MIT License**. Free to adapt for university, collegiate, and research purposes.

