import asyncio
import sys
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SERVER_FILE = PROJECT_ROOT / "mcp_tool_server.py"


async def main():
    server_params = StdioServerParameters(
        command=sys.executable,
        args=[str(SERVER_FILE)],
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:

            await session.initialize()

            # 1. List available tools
            tools_result = await session.list_tools()

            print("\n=== AVAILABLE MCP TOOLS ===")

            for tool in tools_result.tools:
                print(f"- {tool.name}")

            # 2. Test search_docs
            print("\n=== TEST: search_docs ===")

            search_result = await session.call_tool(
                "search_docs",
                {
                    "query": "Project Aswan teams reporting chain",
                    "top_k": 2,
                },
            )

            print(search_result)

            # 3. Test fetch_doc
            print("\n=== TEST: fetch_doc ===")

            fetch_result = await session.call_tool(
                "fetch_doc",
                {
                    "doc_id": "doc1",
                },
            )

            print(fetch_result)

            # 4. Test summarize_text
            print("\n=== TEST: summarize_text ===")

            summarize_result = await session.call_tool(
                "summarize_text",
                {
                    "text": (
                        "Project Aswan involves the Strategy Team and "
                        "the Data & Operations Team. Yara Hassan leads "
                        "the project and reports to Omar Nabil."
                    ),
                    "max_words": 30,
                },
            )

            print(summarize_result)


if __name__ == "__main__":
    asyncio.run(main())