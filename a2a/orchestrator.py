import argparse
import json
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from a2a.agents.supervisor_agent import run_supervisor


# Safety limits required by the A2A task
MAX_DELEGATIONS_PER_RUN = 3
MAX_TOOL_CALLS_PER_AGENT = 5
MAX_TOOL_CALLS_PER_RUN = 10


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)

LOG_FILE = LOG_DIR / "a2a_trace.jsonl"


def write_log(event: str, trace_id: str, **data):
    """Write one structured A2A event to the JSONL log."""
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "trace_id": trace_id,
        "agent": "orchestrator",
        "event": event,
        **data,
    }

    with LOG_FILE.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=False) + "\n")


def run_direct(task: str, context: str, trace_id: str) -> dict:
    """Run the direct A2A supervisor/specialist path."""

    tool_calls = 0

    write_log(
        "delegating_to_supervisor",
        trace_id,
        mode="direct",
        delegation_count=1,
        tool_calls=tool_calls,
    )

    if tool_calls >= MAX_TOOL_CALLS_PER_AGENT:
        return {
            "trace_id": trace_id,
            "agent": "orchestrator",
            "mode": "direct",
            "status": "error",
            "error": "Maximum tool-call limit exceeded for orchestrator.",
            "tool_calls": tool_calls,
        }

    return run_supervisor(
        task=task,
        context=context,
        trace_id=trace_id,
    )


def run_mcp(task: str, trace_id: str) -> dict:
    """
    Run the MCP path using the MCP search_docs tool.
    """

    try:
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client
        import asyncio

        tool_calls = 0

        async def call_mcp():
            nonlocal tool_calls

            if tool_calls >= MAX_TOOL_CALLS_PER_AGENT:
                raise RuntimeError(
                    "Maximum tool-call limit exceeded for MCP agent."
                )

            server_params = StdioServerParameters(
                command="python",
                args=[str(PROJECT_ROOT / "mcp_tool_server.py")],
            )

            async with stdio_client(server_params) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()

                    tool_calls += 1

                    search_result = await session.call_tool(
                        "search_docs",
                        {
                            "query": task,
                            "top_k": 3,
                        },
                    )

                    return search_result

        result = asyncio.run(call_mcp())

        return {
            "trace_id": trace_id,
            "agent": "orchestrator",
            "mode": "mcp",
            "status": "success",
            "tool": "search_docs",
            "tool_calls": tool_calls,
            "result": str(result),
        }

    except Exception as exc:
        return {
            "trace_id": trace_id,
            "agent": "orchestrator",
            "mode": "mcp",
            "status": "error",
            "tool_calls": locals().get("tool_calls", 0),
            "error": str(exc),
        }


def run_orchestrator(
    task: str,
    context: str = "",
    mode: str = "direct",
) -> dict:
    """
    Main A2A orchestrator.

    Supported modes:
    - direct: Orchestrator -> Supervisor -> Specialist
    - mcp: Orchestrator -> MCP server tools
    """

    trace_id = str(uuid.uuid4())
    start_time = time.perf_counter()

    write_log(
        "orchestrator_started",
        trace_id,
        task=task,
        mode=mode,
    )

    if not task.strip():
        result = {
            "trace_id": trace_id,
            "agent": "orchestrator",
            "status": "error",
            "error": "Task cannot be empty.",
        }

        write_log(
            "orchestrator_error",
            trace_id,
            error=result["error"],
        )

        return result

    if mode not in {"direct", "mcp"}:
        result = {
            "trace_id": trace_id,
            "agent": "orchestrator",
            "status": "error",
            "error": "Mode must be either 'direct' or 'mcp'.",
        }

        write_log(
            "orchestrator_error",
            trace_id,
            error=result["error"],
        )

        return result

    delegation_count = 0
    tool_calls = 0
    error_count = 0

    try:
        delegation_count += 1

        if delegation_count > MAX_DELEGATIONS_PER_RUN:
            raise RuntimeError(
                "Maximum delegation limit exceeded for this run."
            )

        if tool_calls >= MAX_TOOL_CALLS_PER_RUN:
            raise RuntimeError(
                "Maximum tool-call limit exceeded for this run."
            )

        if mode == "direct":
            child_result = run_direct(
                task=task,
                context=context,
                trace_id=trace_id,
            )

        else:
            write_log(
                "calling_mcp",
                trace_id,
                tool="search_docs",
            )

            child_result = run_mcp(
                task=task,
                trace_id=trace_id,
            )

            tool_calls = child_result.get("tool_calls", 0)

        status = child_result.get("status", "unknown")

        if status == "error":
            error_count += 1

        latency_ms = round(
            (time.perf_counter() - start_time) * 1000,
            2,
        )

        result = {
            "trace_id": trace_id,
            "agent": "orchestrator",
            "mode": mode,
            "status": status,
            "delegation_count": delegation_count,
            "tool_calls": tool_calls,
            "error_count": error_count,
            "latency_ms": latency_ms,
            "result": child_result,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        write_log(
            "orchestrator_completed"
            if status != "error"
            else "orchestrator_error",
            trace_id,
            mode=mode,
            status=status,
            delegation_count=delegation_count,
            tool_calls=tool_calls,
            error_count=error_count,
            latency_ms=latency_ms,
            error=child_result.get("error")
            if status == "error"
            else None,
        )

        return result

    except Exception as exc:
        error_count += 1

        latency_ms = round(
            (time.perf_counter() - start_time) * 1000,
            2,
        )

        result = {
            "trace_id": trace_id,
            "agent": "orchestrator",
            "mode": mode,
            "status": "error",
            "delegation_count": delegation_count,
            "tool_calls": tool_calls,
            "error_count": error_count,
            "latency_ms": latency_ms,
            "error": str(exc),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        write_log(
            "orchestrator_error",
            trace_id,
            mode=mode,
            error=str(exc),
            delegation_count=delegation_count,
            tool_calls=tool_calls,
            error_count=error_count,
            latency_ms=latency_ms,
        )

        return result


def main():
    parser = argparse.ArgumentParser(
        description="TaharaCo A2A orchestrator"
    )

    parser.add_argument(
        "--direct",
        action="store_true",
        help="Run the direct A2A path.",
    )

    parser.add_argument(
        "--mcp",
        action="store_true",
        help="Run the MCP tool path.",
    )

    parser.add_argument(
        "--task",
        default="What teams are involved in Project Aswan?",
        help="Task to execute.",
    )

    args = parser.parse_args()

    if args.mcp and args.direct:
        parser.error("Use either --direct or --mcp, not both.")

    mode = "mcp" if args.mcp else "direct"

    result = run_orchestrator(
        task=args.task,
        context="TaharaCo Project Aswan documentation.",
        mode=mode,
    )

    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()