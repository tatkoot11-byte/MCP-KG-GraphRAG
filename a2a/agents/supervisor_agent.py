import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

from a2a.agents.specialist_agent import run_specialist


MAX_SPECIALIST_CALLS = 2

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)

LOG_FILE = LOG_DIR / "a2a_trace.jsonl"


def write_log(event: str, trace_id: str, **data):
    """Write one structured A2A event to the shared JSONL log."""

    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "trace_id": trace_id,
        "agent": "supervisor",
        "event": event,
        **data,
    }

    with LOG_FILE.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=False) + "\n")


def run_supervisor(
    task: str,
    context: str = "",
    trace_id: str | None = None,
) -> dict:
    """Supervisor agent that delegates work to the specialist."""

    trace_id = trace_id or str(uuid.uuid4())

    write_log(
        "supervisor_started",
        trace_id,
        task=task,
    )

    if not task.strip():
        error_message = "Task cannot be empty."

        result = {
            "trace_id": trace_id,
            "agent": "supervisor",
            "status": "error",
            "error": error_message,
            "tool_input": task,
            "reason": error_message,
        }

        write_log(
            "supervisor_error",
            trace_id,
            error=error_message,
            tool_input=task,
            reason=error_message,
        )

        return result

    specialist_calls = 0

    try:
        specialist_calls += 1

        if specialist_calls > MAX_SPECIALIST_CALLS:
            raise RuntimeError(
                "Maximum specialist call limit exceeded."
            )

        write_log(
            "delegating_to_specialist",
            trace_id,
            specialist_calls=specialist_calls,
            tool_input=task,
        )

        result = run_specialist(
            task=task,
            context=context,
            trace_id=trace_id,
        )

        # Propagate specialist errors instead of treating them as success.
        if result.get("status") == "error":
            error_message = result.get(
                "error",
                "Specialist returned an unknown error.",
            )

            tool_input = result.get(
                "tool_input",
                task,
            )

            reason = result.get(
                "reason",
                error_message,
            )

            output = {
                "trace_id": trace_id,
                "agent": "supervisor",
                "status": "error",
                "specialist_calls": specialist_calls,
                "error": error_message,
                "tool_input": tool_input,
                "reason": reason,
                "worker": result,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

            write_log(
                "supervisor_error",
                trace_id,
                error=error_message,
                tool_input=tool_input,
                reason=reason,
                specialist_calls=specialist_calls,
            )

            return output

        output = {
            "trace_id": trace_id,
            "agent": "supervisor",
            "status": "success",
            "specialist_calls": specialist_calls,
            "review": "Specialist result received and reviewed successfully.",
            "worker": result,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        write_log(
            "supervisor_completed",
            trace_id,
            status="success",
            specialist_calls=specialist_calls,
        )

        return output

    except Exception as exc:
        error_message = str(exc)

        output = {
            "trace_id": trace_id,
            "agent": "supervisor",
            "status": "error",
            "specialist_calls": specialist_calls,
            "error": error_message,
            "tool_input": task,
            "reason": error_message,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        write_log(
            "supervisor_error",
            trace_id,
            error=error_message,
            tool_input=task,
            reason=error_message,
            specialist_calls=specialist_calls,
        )

        return output


if __name__ == "__main__":
    result = run_supervisor(
        task="Review the teams involved in Project Aswan.",
        context="TaharaCo Project Aswan documentation.",
    )

    print(result)