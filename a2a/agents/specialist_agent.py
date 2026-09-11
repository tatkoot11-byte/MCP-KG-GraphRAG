
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)

LOG_FILE = LOG_DIR / "a2a_trace.jsonl"


def write_log(event: str, trace_id: str, **data):
    """Write one structured A2A event to the shared JSONL log."""
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "trace_id": trace_id,
        "agent": "specialist",
        "event": event,
        **data,
    }

    with LOG_FILE.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=False) + "\n")


def run_specialist(
    task: str,
    context: str = "",
    trace_id: str | None = None,
) -> dict:
    """
    Specialist worker for TaharaCo Project Aswan questions.
    Returns a bounded, traceable A2A response.
    """

    trace_id = trace_id or str(uuid.uuid4())

    write_log(
        "specialist_started",
        trace_id,
        task=task,
    )

    try:
        task_lower = task.lower()

        if "aswan" in task_lower or "project" in task_lower:
            answer = (
                "The specialist reviewed the TaharaCo Project Aswan context. "
                "The project involves the Strategy Team and the Data & Operations Team."
            )
        else:
            answer = (
                "The specialist received the task and reviewed the available context."
            )

        result = {
            "trace_id": trace_id,
            "agent": "specialist",
            "status": "success",
            "task": task,
            "answer": answer,
            "context_used": bool(context),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        write_log(
            "specialist_completed",
            trace_id,
            status="success",
        )

        return result

    except Exception as exc:
        result = {
            "trace_id": trace_id,
            "agent": "specialist",
            "status": "error",
            "task": task,
            "error": str(exc),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        write_log(
            "specialist_error",
            trace_id,
            error=str(exc),
        )

        return result


if __name__ == "__main__":
    result = run_specialist(
        task="Summarize the Project Aswan team structure.",
        context="Project Aswan involves Strategy and Data & Operations teams.",
    )

    print(result)

