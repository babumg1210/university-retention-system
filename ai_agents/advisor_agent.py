"""
AI Academic Advisor Agent: LangGraph Workflow with Local Ollama and Resilient Fallback
Retrieves student risk profile, queries university handbook RAG, and generates
a structured 3-step recovery plan + compassionate email draft.
"""

import sys
import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
from typing_extensions import TypedDict
from datetime import datetime, timezone

from langgraph.graph import StateGraph, START, END
from sqlalchemy.orm import Session

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.database import SessionLocal, init_db
from app.models import Student, AcademicLog, RiskPrediction, InterventionLog, utc_now
from ai_agents.handbook_rag import query_handbook, format_policies_for_prompt

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("AIAdvisorAgent")


class AdvisorState(TypedDict):
    student_id: str
    student_data: Optional[Dict[str, Any]]
    risk_data: Optional[Dict[str, Any]]
    retrieved_policies: Optional[List[Dict[str, Any]]]
    policy_context: Optional[str]
    recovery_plan: Optional[Dict[str, Any]]
    email_draft: Optional[Dict[str, str]]
    advisor_notes: Optional[str]
    intervention_id: Optional[int]
    used_llm_mode: Optional[str]  # "ollama" or "fallback_mock"
    error: Optional[str]


def fetch_student_profile(state: AdvisorState, db: Session) -> Dict[str, Any]:
    """
    Step 1: Queries student demographics, latest academic log, and risk predictions.
    """
    student_id = state["student_id"]
    logger.info(f"Fetching academic profile for student ID: {student_id}")

    student = db.query(Student).filter(Student.student_id == student_id).first()
    if not student:
        err = f"Student '{student_id}' not found in database."
        logger.error(err)
        return {"error": err}

    log = student.latest_academic_log
    pred = student.latest_risk_prediction

    student_data = {
        "student_id": student.student_id,
        "name": student.name,
        "department": student.department,
        "enrollment_year": student.enrollment_year,
        "scholarship_status": student.scholarship_status,
        "attendance_pct": float(log.attendance_pct) if log else 0.0,
        "internal_marks": float(log.internal_marks) if log else 0.0,
        "lms_logins": int(log.lms_logins) if log else 0,
    }

    risk_data = {
        "risk_score": float(pred.risk_score) if pred else 0.0,
        "risk_level": pred.risk_level if pred else "Unassessed",
        "top_risk_factor": pred.top_risk_factor if pred else "No current prediction",
        "predicted_at": pred.predicted_at.isoformat() if pred and pred.predicted_at else None,
    }

    return {
        "student_data": student_data,
        "risk_data": risk_data,
        "error": None,
    }


def retrieve_policies_node(state: AdvisorState) -> Dict[str, Any]:
    """
    Step 2: Semantic search against the handbook vector store tailored to student risks.
    """
    s_data = state.get("student_data") or {}
    r_data = state.get("risk_data") or {}

    attendance = s_data.get("attendance_pct", 100.0)
    marks = s_data.get("internal_marks", 100.0)
    scholarship = s_data.get("scholarship_status", "No")
    risk_level = r_data.get("risk_level", "Low")

    # Construct targeted search query
    query_parts = []
    if attendance < 75.0:
        query_parts.append("attendance condonation debarment warning below 70%")
    if marks < 45.0:
        query_parts.append("continuous assessment internal marks below 45 remedial tutoring classes")
    if scholarship == "Yes" and (marks < 60.0 or attendance < 80.0):
        query_parts.append("merit scholarship retention 30-day grace period observation")
    if risk_level == "High":
        query_parts.append("academic probation early intervention restricted credit load recovery plan")

    search_query = " ".join(query_parts) if query_parts else "academic recovery plan advising"
    logger.info(f"Querying Handbook RAG with query: '{search_query}'")

    policies = query_handbook(search_query, n_results=3)
    policy_context = format_policies_for_prompt(policies)

    return {
        "retrieved_policies": policies,
        "policy_context": policy_context,
    }


def generate_plan_with_ollama(student: dict, risk: dict, policies: str) -> Optional[Dict[str, Any]]:
    """
    Attempts to generate the plan using local Ollama model (e.g. llama3).
    Returns None if Ollama is not installed or unreachable.
    """
    try:
        from langchain_ollama import ChatOllama
        from langchain_core.messages import SystemMessage, HumanMessage

        # Short timeout so we quickly failover if Ollama server is offline
        llm = ChatOllama(model="llama3", temperature=0.3, timeout=6.0)

        sys_msg = SystemMessage(
            content=(
                "You are an empathetic, expert University Academic Advisor at Apex University. "
                "You assist at-risk undergraduate students by creating realistic recovery plans and polite emails. "
                "Always output valid JSON containing: 'summary', 'step_1', 'step_2', 'step_3', and 'email_draft'."
            )
        )

        user_content = f"""
Student Profile:
- ID: {student['student_id']}
- Name: {student['name']}
- Department: {student['department']}
- Current Attendance: {student['attendance_pct']}% (Target: >= 75%)
- Internal Marks: {student['internal_marks']}/100 (Passing: >= 45)
- LMS Logins: {student['lms_logins']} (Target: >= 15)
- Scholarship Status: {student['scholarship_status']}
- Composite Risk: {risk['risk_level']} (Score: {risk['risk_score']})
- Primary Risk Factor: {risk['top_risk_factor']}

Relevant University Policies:
{policies}

Please output ONLY a JSON object with this exact structure:
{{
  "summary": "<Concise diagnostic summary of student's academic standing>",
  "step_1": {{
    "title": "<Immediate Week 1-2 Action: e.g. Attendance / Remedial Tutoring>",
    "action": "<Specific actionable instructions for the student>",
    "timeline": "Weeks 1-2",
    "policy_reference": "<Cited policy clause>"
  }},
  "step_2": {{
    "title": "<Mid-term Academic Strengthening: e.g. Coursework / LMS Engagement>",
    "action": "<Specific actionable instructions for the student>",
    "timeline": "Weeks 3-5",
    "policy_reference": "<Cited policy clause>"
  }},
  "step_3": {{
    "title": "<Evaluation Milestone & Probation Clearance>",
    "action": "<Specific targets to achieve good standing>",
    "timeline": "Weeks 6-8",
    "policy_reference": "<Cited policy clause>"
  }},
  "email_draft": {{
    "subject": "<Empathetic and clear subject line>",
    "body": "<Compassionate, formal, and supportive email text addressing the student by name>"
  }}
}}
"""
        response = llm.invoke([sys_msg, HumanMessage(content=user_content)])
        raw_text = response.content.strip()

        # Clean JSON if model output markdown code blocks
        if "```json" in raw_text:
            raw_text = raw_text.split("```json")[1].split("```")[0].strip()
        elif "```" in raw_text:
            raw_text = raw_text.split("```")[1].split("```")[0].strip()

        parsed = json.loads(raw_text)
        logger.info("Successfully generated recovery plan using local Ollama model!")
        return parsed

    except Exception as e:
        logger.info(f"Ollama local model not active ({e}). Proceeding to high-quality fallback engine.")
        return None


def generate_plan_fallback(student: dict, risk: dict, policies: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Deterministic, high-quality fallback plan generator when Ollama is offline.
    Tailors the 3 steps, policy citations, and personalized email to the student's exact metrics.
    """
    name = student["name"]
    first_name = name.split()[0]
    dept = student["department"]
    stu_id = student["student_id"]
    att = student["attendance_pct"]
    marks = student["internal_marks"]
    logins = student["lms_logins"]
    scholarship = student["scholarship_status"]
    risk_factor = risk.get("top_risk_factor", "Academic Distress")

    # Step 1: Immediate Action (Attendance / Absenteeism / Remedial)
    if att < 70.0:
        step_1 = {
            "title": "Immediate Attendance Condonation & Class Restoration",
            "action": (
                f"Schedule an urgent meeting with the Department Chair to review attendance shortfall ({att}%). "
                f"Submit any valid medical certificates within 7 business days under Section 1.3 to avoid final exam debarment. "
                f"Commit to 100% attendance across all upcoming lecture and practical sessions."
            ),
            "timeline": "Immediate (Days 1 - 7)",
            "policy_reference": "Section 1.2 (Chronic Absenteeism Warning) & Section 1.3 (Medical Condonation)",
        }
    else:
        step_1 = {
            "title": "Mandatory Remedial Tutoring Enrollment",
            "action": (
                f"Enroll in bi-weekly peer tutoring and faculty office hours for courses with internal scores below 45. "
                f"Active attendance in remedial sessions qualifies for up to 10 compensatory recovery marks under Section 2.2."
            ),
            "timeline": "Immediate (Days 1 - 7)",
            "policy_reference": "Section 2.2 (Mandatory Remedial Classes for Struggling Students)",
        }

    # Step 2: Coursework & LMS Engagement
    if logins < 15:
        step_2 = {
            "title": "LMS Portal Engagement & Problem Set Submissions",
            "action": (
                f"Increase portal engagement from current {logins} logins to a minimum of 20 logins per evaluation cycle. "
                f"Download all missed lecture slides, complete online weekly quizzes, and actively participate in department discussion boards."
            ),
            "timeline": "Weeks 2 - 5",
            "policy_reference": "Section 2.3 (LMS Portal Engagement Guidelines)",
        }
    else:
        step_2 = {
            "title": "Structured Study Schedule & Midterm Preparation",
            "action": (
                f"Form a 3-member study group within {dept} and complete at least 3 practice midterm problem sets. "
                f"Review laboratory notebook entries weekly with teaching assistants to secure full coursework credit."
            ),
            "timeline": "Weeks 2 - 5",
            "policy_reference": "Section 2.1 (Continuous Internal Evaluation Standards)",
        }

    # Step 3: Long-term Milestone & Scholarship / Probation Clearance
    if scholarship == "Yes":
        step_3 = {
            "title": "Financial Aid Retention & 30-Day Grace Period Milestone",
            "action": (
                f"Execute the Academic Recovery Commitment Agreement to safeguard your scholarship under Section 3.2. "
                f"Target a minimum internal assessment average of >= 60 marks and attendance >= 80% before the end-of-term audit."
            ),
            "timeline": "Weeks 6 - 8 (End of Grace Period)",
            "policy_reference": "Section 3.1 & 3.2 (Emergency Scholarship Grace Period & Appeals)",
        }
    else:
        step_3 = {
            "title": "Probation Review & Return to Good Standing",
            "action": (
                f"Meet with your designated Academic Advisor for the bi-weekly progress review. "
                f"Achieve internal exam scores >= 50 and cumulative attendance >= 75% to officially lift academic warning status under Section 4.3."
            ),
            "timeline": "Weeks 6 - 8",
            "policy_reference": "Section 4.2 & 4.3 (Academic Probation Terms & Removal)",
        }

    # Diagnostic Summary
    summary = (
        f"{name} ({stu_id}, {dept}) currently holds an academic risk score of {risk.get('risk_score', 0):.2f} "
        f"({risk.get('risk_level', 'High')} Risk). The primary catalyst is '{risk_factor}', with attendance at "
        f"{att}%, internal evaluation at {marks}/100, and {logins} LMS logins. Institutional policy mandates targeted "
        f"intervention to maintain enrollment eligibility."
    )

    # Polite Email Draft
    email_subject = f"Support & Academic Success Consultation | Department of {dept}"
    email_body = (
        f"Dear {first_name},\n\n"
        f"I hope this note finds you well. I am reaching out from the Department of {dept} Academic Advising Office.\n\n"
        f"As part of our mid-semester academic progress check, our automated advisory system identified that you may be facing "
        f"some challenges in your coursework, particularly regarding your current {risk_factor.lower()}.\n\n"
        f"We understand that balancing university coursework can be demanding, and we are fully committed to ensuring you have "
        f"every resource needed to succeed. We have formulated a customized 3-Step Academic Recovery Plan tailored to your profile:\n\n"
        f"1. {step_1['title']}: {step_1['action']}\n"
        f"2. {step_2['title']}: {step_2['action']}\n"
        f"3. {step_3['title']}: {step_3['action']}\n\n"
        f"Please schedule a brief 15-minute consultation with me this week so we can review this plan together and ensure "
        f"any necessary institutional accommodations or tutoring sessions are set up for you.\n\n"
        f"You are a valued member of our university community, and we look forward to supporting your progress.\n\n"
        f"Warm regards,\n\n"
        f"Academic Advising Committee\n"
        f"Department of {dept}\n"
        f"Apex University"
    )

    return {
        "summary": summary,
        "step_1": step_1,
        "step_2": step_2,
        "step_3": step_3,
        "email_draft": {
            "subject": email_subject,
            "body": email_body,
        },
    }


def generate_plan_node(state: AdvisorState) -> Dict[str, Any]:
    """
    Step 3: Dispatches generation to local Ollama model with seamless fallback.
    """
    student = state.get("student_data") or {}
    risk = state.get("risk_data") or {}
    policy_ctx = state.get("policy_context") or ""
    policies = state.get("retrieved_policies") or []

    # Try Ollama first
    plan = generate_plan_with_ollama(student, risk, policy_ctx)
    llm_mode = "ollama"

    if not plan:
        # Fallback to high quality deterministic engine
        plan = generate_plan_fallback(student, risk, policies)
        llm_mode = "fallback_mock"

    email_draft = plan.get("email_draft", {})
    advisor_notes = (
        f"AI Advisor Recovery Plan generated via {llm_mode}. "
        f"Risk Category: {risk.get('risk_level')}, Factor: {risk.get('top_risk_factor')}. "
        f"Handbook clauses cited: {', '.join([p.get('title', '') for p in policies[:2]])}."
    )

    return {
        "recovery_plan": plan,
        "email_draft": email_draft,
        "advisor_notes": advisor_notes,
        "used_llm_mode": llm_mode,
    }


def save_intervention_node(state: AdvisorState, db: Session) -> Dict[str, Any]:
    """
    Step 4: Persists the intervention log into the database.
    """
    student_id = state["student_id"]
    advisor_notes = state.get("advisor_notes", "")
    recovery_plan = state.get("recovery_plan", {})
    email_draft = state.get("email_draft", {})

    full_payload = {
        "student_id": student_id,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "llm_engine": state.get("used_llm_mode", "fallback_mock"),
        "summary": recovery_plan.get("summary", ""),
        "steps": [
            recovery_plan.get("step_1", {}),
            recovery_plan.get("step_2", {}),
            recovery_plan.get("step_3", {}),
        ],
        "email_draft": email_draft,
        "policies_retrieved": [
            {"title": p.get("title"), "section": p.get("section")}
            for p in state.get("retrieved_policies", [])
        ],
    }

    try:
        # Create new InterventionLog record
        intervention = InterventionLog(
            student_id=student_id,
            advisor_notes=advisor_notes,
            ai_recovery_plan=json.dumps(full_payload, indent=2),
            status="Active Recovery Plan",
            created_at=utc_now(),
        )
        db.add(intervention)
        db.commit()
        db.refresh(intervention)
        logger.info(f"Persisted InterventionLog ID #{intervention.id} for student {student_id}")

        return {
            "intervention_id": intervention.id,
        }
    except Exception as e:
        db.rollback()
        logger.error(f"Failed to persist InterventionLog: {e}")
        return {"error": f"Database save failed: {str(e)}"}


def build_advisor_workflow(db: Session):
    """
    Builds and compiles the LangGraph workflow for the AI Advisor Agent.
    """
    workflow = StateGraph(AdvisorState)

    # Define nodes with db session closure
    def node_fetch(state: AdvisorState) -> AdvisorState:
        res = fetch_student_profile(state, db)
        return {**state, **res}

    def node_retrieve(state: AdvisorState) -> AdvisorState:
        if state.get("error"):
            return state
        res = retrieve_policies_node(state)
        return {**state, **res}

    def node_generate(state: AdvisorState) -> AdvisorState:
        if state.get("error"):
            return state
        res = generate_plan_node(state)
        return {**state, **res}

    def node_save(state: AdvisorState) -> AdvisorState:
        if state.get("error"):
            return state
        res = save_intervention_node(state, db)
        return {**state, **res}

    workflow.add_node("fetch_profile", node_fetch)
    workflow.add_node("retrieve_policies", node_retrieve)
    workflow.add_node("generate_plan", node_generate)
    workflow.add_node("save_intervention", node_save)

    workflow.add_edge(START, "fetch_profile")
    workflow.add_edge("fetch_profile", "retrieve_policies")
    workflow.add_edge("retrieve_policies", "generate_plan")
    workflow.add_edge("generate_plan", "save_intervention")
    workflow.add_edge("save_intervention", END)

    return workflow.compile()


def run_advisor_agent(student_id: str, db: Session = None) -> Dict[str, Any]:
    """
    Convenience orchestrator function to execute the LangGraph AI Advisor Agent.
    """
    close_db = False
    if db is None:
        init_db()
        db = SessionLocal()
        close_db = True

    try:
        app = build_advisor_workflow(db)
        initial_state: AdvisorState = {
            "student_id": student_id,
            "student_data": None,
            "risk_data": None,
            "retrieved_policies": None,
            "policy_context": None,
            "recovery_plan": None,
            "email_draft": None,
            "advisor_notes": None,
            "intervention_id": None,
            "used_llm_mode": None,
            "error": None,
        }

        final_state = app.invoke(initial_state)

        if final_state.get("error"):
            return {
                "status": "error",
                "message": final_state["error"],
                "student_id": student_id,
            }

        return {
            "status": "success",
            "student_id": student_id,
            "intervention_id": final_state.get("intervention_id"),
            "used_llm_mode": final_state.get("used_llm_mode"),
            "recovery_plan": final_state.get("recovery_plan"),
            "email_draft": final_state.get("email_draft"),
            "retrieved_policies": final_state.get("retrieved_policies"),
            "advisor_notes": final_state.get("advisor_notes"),
        }
    finally:
        if close_db:
            db.close()


if __name__ == "__main__":
    test_id = sys.argv[1] if len(sys.argv) > 1 else "STU1002"
    print(f"\n=======================================================")
    print(f"Executing AI Advisor Agent for student: {test_id}")
    print(f"=======================================================\n")
    result = run_advisor_agent(test_id)
    print(f"Status: {result['status']}")
    if result["status"] == "success":
        print(f"Engine Mode: {result['used_llm_mode']}")
        print(f"Intervention ID: {result['intervention_id']}")
        plan = result["recovery_plan"]
        print(f"\nSummary:\n{plan['summary']}\n")
        print("3-Step Recovery Plan:")
        print(f"  Step 1: {plan['step_1']['title']} ({plan['step_1']['timeline']})")
        print(f"          Action: {plan['step_1']['action']}")
        print(f"  Step 2: {plan['step_2']['title']} ({plan['step_2']['timeline']})")
        print(f"          Action: {plan['step_2']['action']}")
        print(f"  Step 3: {plan['step_3']['title']} ({plan['step_3']['timeline']})")
        print(f"          Action: {plan['step_3']['action']}")
        print(f"\nEmail Draft:\nSubject: {result['email_draft']['subject']}\n")
        print(f"{result['email_draft']['body'][:200]}...")
    else:
        print(f"Error: {result['message']}")

