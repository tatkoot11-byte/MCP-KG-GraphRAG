import asyncio

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


async def main():
    server_params = StdioServerParameters(
        command="python",
        args=["mcp_tool_server.py"],
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            # 1. List available tools
            tools = await session.list_tools()

            print("AVAILABLE MCP TOOLS:")
            for tool in tools.tools:
                print(f"- {tool.name}")

            # 2. Test search_docs
            search_result = await session.call_tool(
                "search_docs",
                {
                    "query": "Project Aswan teams",
                    "top_k": 3,
                },
            )

            print("\nSEARCH_DOCS RESULT:")
            print(search_result)
            print("SEARCH_DOCS isError:", search_result.isError)

            # 3. Test fetch_doc
            fetch_result = await session.call_tool(
                "fetch_doc",
                {
                    "doc_id": "doc1",
                },
            )

            print("\nFETCH_DOC RESULT:")
            print(fetch_result)
            print("FETCH_DOC isError:", fetch_result.isError)

            # 4. Test summarize_text
            summarize_result = await session.call_tool(
                "summarize_text",
                {
                    "text": (
                        "Project Aswan involves the Strategy Team "
                        "and the Data & Operations Team. "
                        "Yara Hassan leads the project, while "
                        "Salma Farouk oversees operational and "
                        "data validation."
                    ),
                    "max_words": 20,
                },
            )

            print("\nSUMMARIZE_TEXT RESULT:")
            print(summarize_result)
            print(
                "SUMMARIZE_TEXT isError:",
                summarize_result.isError,
            )


if __name__ == "__main__":
    asyncio.run(main())