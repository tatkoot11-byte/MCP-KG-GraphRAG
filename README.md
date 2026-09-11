\# TaharaCo MCP \& A2A Multi-Agent System with Neo4j GraphRAG



A local multi-agent GraphRAG system built for the Sprints AI bonus task.



The project combines:



\* MCP tools for document search, document fetching, and text summarization.

\* A2A-style orchestrator, supervisor, and specialist agents.

\* Neo4j knowledge graph for people, teams, projects, documents, and vendors.

\* Hybrid GraphRAG retrieval using vector search and Neo4j graph traversal.

\* Chainlit UI showing retrieval, graph search, answer generation, and provenance.

\* DeepEval evaluation setup with Faithfulness and ProvenanceCompleteness metrics.

\* Automated Neo4j graph quality checks.



\## Project Scenario



The system is based on TaharaCo's fictional Project Aswan.



The knowledge base contains information about:



\* Project Aswan

\* Strategy Team

\* Data \& Operations Team

\* Employees and reporting relationships

\* Delta Analytics vendor

\* Project documentation and review requirements



\## Architecture



```text

User

&#x20;|

&#x20;v

Chainlit App

&#x20;|

&#x20;v

GraphRAG Pipeline

&#x20;|

&#x20;+--------------------+

&#x20;|                    |

&#x20;v                    v

FAISS Vector Search   Neo4j Graph Search

&#x20;|                    |

&#x20;+---------+----------+

&#x20;          |

&#x20;          v

&#x20;    Gemini Answer

&#x20;          |

&#x20;          v

&#x20;Provenance / Evidence

```



The project also includes an A2A workflow:



```text

Orchestrator

&#x20;    |

&#x20;    v

&#x20;Supervisor

&#x20;    |

&#x20;    v

&#x20;Specialist

```



The orchestrator supports both:



```text

\--direct

\--mcp

```



\## MCP Server



The MCP server is implemented in:



```text

mcp\_tool\_server.py

```



It exposes three tools.



\### search\_docs



Searches the TaharaCo document collection.



Arguments:



\* `query`

\* `top\_k`



\### fetch\_doc



Fetches a complete document by document ID.



Arguments:



\* `doc\_id`



\### summarize\_text



Creates a bounded summary of supplied text.



Arguments:



\* `text`

\* `max\_words`



Pydantic validation is used for the tool inputs.



\## MCP Smoke Test



Run:



```bash

python test\_mcp.py

```



The smoke test verifies:



\* MCP initialization

\* Available tools

\* `search\_docs`

\* `fetch\_doc`

\* `summarize\_text`

\* MCP error status



\## A2A System



The A2A implementation is located under:



```text

a2a/

```



Main components:



```text

a2a/orchestrator.py

a2a/agents/supervisor\_agent.py

a2a/agents/specialist\_agent.py

```



The orchestrator provides:



\* Shared `trace\_id`

\* Delegation limits

\* Error propagation

\* Structured JSONL logging

\* Direct execution mode

\* MCP execution mode



\### Direct mode



```bash

python -m a2a.orchestrator --direct

```



\### MCP mode



```bash

python -m a2a.orchestrator --mcp

```



A custom task can be supplied with:



```bash

python -m a2a.orchestrator --direct --task "Who works on Project Aswan?"

```



Logs are written to:



```text

logs/a2a\_trace.jsonl

```



\## Neo4j Knowledge Graph



Neo4j Community runs locally through Docker.



Example:



```bash

docker run -d --name neo4j-taharaco -p 7474:7474 -p 7687:7687 -e NEO4J\_AUTH=neo4j/<password> neo4j:community

```



The graph contains the following node labels:



```text

Person

Team

Project

Doc

Vendor

```



Required relationships include:



```text

WORKS\_ON

MEMBER\_OF

REPORTS\_TO

MENTIONS

VENDOR\_OF

```



The graph also contains supervision relationships used by the reporting structure.



\## Neo4j Scripts



Reset the graph:



```bash

python scripts/neo4j\_reset.py

```



Normalize names and remove duplicate entities where applicable:



```bash

python scripts/neo4j\_normalize.py

```



The reset script is destructive and should only be used when rebuilding the graph.



\## Graph Quality Checks



Run:



```bash

python evaluation/graph\_quality\_checks.py

```



The current graph quality checks verify:



\* Required node labels

\* Required relationship types

\* Duplicate Person names

\* Project Aswan workers

\* Project Aswan vendor



The current check result is:



```text

Overall passed: True

```



Results are saved to:



```text

evaluation/graph\_quality\_results.json

```



\## GraphRAG Pipeline



The main GraphRAG components are:



```text

graphrag\_pipeline.py

graphrag\_answer.py

scripts/hybrid\_retriever.py

```



The retrieval pipeline combines:



1\. Vector retrieval using FAISS.

2\. Graph traversal using Neo4j.

3\. Gemini answer generation.

4\. Provenance information.



Passages contain stable IDs such as:



```text

doc1\_p1

doc2\_p2

doc3\_p1

```



Graph provenance is represented using paths such as:



```text

Yara Hassan -> WORKS\_ON -> Project Aswan

```



\## Chainlit Application



Run the application with:



```bash

chainlit run chainlit\_app.py

```



The Chainlit interface displays:



1\. Hybrid Retrieval

2\. Vector Search

3\. Neo4j Graph Search

4\. GraphRAG Answer Generation

5\. Provenance



The demo covers scenarios including:



\### Scenario 1



```text

Who works on Project Aswan?

```



Expected entities include:



\* Yara Hassan

\* Salma Farouk

\* Karim Adel

\* Nour Ibrahim



\### Scenario 2



```text

Who is the vendor for Project Aswan, and which team uses its logistics data?

```



The answer identifies:



\* Delta Analytics

\* Data \& Operations Team

\* Relevant employees using the datasets



\## Evaluation



The evaluation script is:



```text

scripts/evaluate\_graphrag.py

```



It contains six test cases covering:



\* Project workers

\* Vendor identification

\* Vendor/team relationship

\* Reporting relationships



The configured metrics are:



\### Faithfulness



Threshold:



```text

0.70

```



\### ProvenanceCompleteness



Implemented using DeepEval GEval with:



\* Input

\* Actual output

\* Retrieval context



Threshold:



```text

0.70

```



Evaluation artifacts are stored in:



```text

evaluation/evaluation\_results.json

```



The project also includes graph quality results:



```text

evaluation/graph\_quality\_results.json

```



\## Evaluation Note



The DeepEval evaluation was executed against Gemini.



The evaluation successfully initialized the configured metrics and began evaluating all six test cases. The run was stopped by the Gemini Free Tier request quota:



```text

429 RESOURCE\_EXHAUSTED

GenerateRequestsPerDayPerProject-FreeTier

```



The evaluation artifact containing the six test cases, expected outputs, actual outputs, and retrieval contexts is preserved in:



```text

evaluation/evaluation\_results.json

```



No API keys are stored in the repository.



\## Project Structure



```text

MCP-KG-GraphRAG/

│

├── a2a/

│   ├── agents/

│   │   ├── specialist\_agent.py

│   │   └── supervisor\_agent.py

│   └── orchestrator.py

│

├── data/

│   ├── docs/

│   │   ├── doc1.txt

│   │   ├── doc2.txt

│   │   └── doc3.txt

│   └── faiss/

│

├── docs/

│

├── evaluation/

│   ├── evaluation\_results.json

│   ├── graph\_quality\_checks.py

│   └── graph\_quality\_results.json

│

├── logs/

│   └── a2a\_trace.jsonl

│

├── scripts/

│   ├── evaluate\_graphrag.py

│   ├── hybrid\_retriever.py

│   ├── neo4j\_normalize.py

│   └── neo4j\_reset.py

│

├── chainlit\_app.py

├── graphrag\_answer.py

├── graphrag\_pipeline.py

├── mcp\_tool\_server.py

├── test\_mcp.py

├── requirements.txt

├── .env.example

└── README.md

```



\## Environment Variables



Create a `.env` file locally:



```text

GEMINI\_API\_KEY=your\_gemini\_api\_key

NEO4J\_PASSWORD=your\_neo4j\_password

NEO4J\_URI=neo4j://localhost:7687

NEO4J\_USER=neo4j

```



Never commit `.env` or real API keys.



A template is provided in:



```text

.env.example

```



\## Installation



Create and activate a Python virtual environment:



```bash

python -m venv .venv

```



Windows:



```bash

.venv\\Scripts\\activate

```



Install dependencies:



```bash

pip install -r requirements.txt

```



Python 3.10+ is supported.



\## Main Dependencies



The project uses:



\* MCP

\* LangChain

\* LangGraph

\* Gemini

\* Neo4j

\* FAISS

\* Chainlit

\* DeepEval

\* Pydantic

\* structlog

\* python-dotenv



\## Security



The following files are excluded from Git:



```text

.env

.venv/

\_\_pycache\_\_/

\*.pyc

```



API credentials must remain in the local `.env` file.



\## Acceptance Checklist



\* \[x] MCP server

\* \[x] Three MCP tools

\* \[x] Pydantic input validation

\* \[x] MCP smoke test

\* \[x] A2A orchestrator

\* \[x] Supervisor and specialist agents

\* \[x] Shared trace IDs

\* \[x] Delegation limits

\* \[x] Error propagation

\* \[x] Direct mode

\* \[x] MCP mode

\* \[x] Structured A2A logs

\* \[x] Neo4j Docker setup

\* \[x] Required graph schema

\* \[x] Entity normalization

\* \[x] Duplicate checks

\* \[x] Cypher graph queries

\* \[x] Hybrid vector + graph retrieval

\* \[x] Passage provenance

\* \[x] Graph-path provenance

\* \[x] Chainlit application

\* \[x] Two demo scenarios

\* \[x] DeepEval test cases

\* \[x] Faithfulness metric configured

\* \[x] ProvenanceCompleteness metric configured

\* \[x] Evaluation artifacts

\* \[x] Graph quality checks

\* \[x] Graph quality results

\* \[x] Public repository ready for final review











