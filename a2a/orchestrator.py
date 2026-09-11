import argparse
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

from a2a.agents.supervisor_agent import run_supervisor


MAX_DELEGATIONS = 3

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
    """Run the normal direct A2A supervisor/specialist path."""

    write_log(
        "delegating_to_supervisor",
        trace_id,
        mode="direct",
        delegation_count=1,
    )

    return run_supervisor(
        task=task,
        context=context,
        trace_id=trace_id,
    )


def run_mcp(task: str, trace_id: str) -> dict:
    """
    Run the MCP path.

    The MCP server is started separately and its tools are used
    to retrieve TaharaCo documentation.
    """

    try:
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client

        import asyncio

        async def call_mcp():
            server_params = StdioServerParameters(
                command="python",
                args=[str(PROJECT_ROOT / "mcp_tool_server.py")],
            )

            async with stdio_client(server_params) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()

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
            "result": str(result),
        }

    except Exception as exc:
        return {
            "trace_id": trace_id,
            "agent": "orchestrator",
            "mode": "mcp",
            "status": "error",
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

    try:
        delegation_count += 1

        if delegation_count > MAX_DELEGATIONS:
            raise RuntimeError("Maximum delegation limit exceeded.")

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

        result = {
            "trace_id": trace_id,
            "agent": "orchestrator",
            "mode": mode,
            "status": child_result.get("status", "unknown"),
            "delegation_count": delegation_count,
            "result": child_result,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        if result["status"] == "error":
            write_log(
                "orchestrator_error",
                trace_id,
                mode=mode,
                error=child_result.get(
                    "error",
                    "Child agent returned an error.",
                ),
                delegation_count=delegation_count,
            )
        else:
            write_log(
                "orchestrator_completed",
                trace_id,
                mode=mode,
                status=result["status"],
                delegation_count=delegation_count,
            )

        return result

    except Exception as exc:
        result = {
            "trace_id": trace_id,
            "agent": "orchestrator",
            "mode": mode,
            "status": "error",
            "delegation_count": delegation_count,
            "error": str(exc),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        write_log(
            "orchestrator_error",
            trace_id,
            mode=mode,
            error=str(exc),
            delegation_count=delegation_count,
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