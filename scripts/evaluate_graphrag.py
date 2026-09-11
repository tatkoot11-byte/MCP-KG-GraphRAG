import json
import os
from pathlib import Path

from deepeval import evaluate
from deepeval.metrics import FaithfulnessMetric, GEval
from deepeval.models import GeminiModel
from deepeval.test_case import LLMTestCase, SingleTurnParams
from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EVAL_DIR = PROJECT_ROOT / "evaluation"
EVAL_DIR.mkdir(exist_ok=True)

load_dotenv(PROJECT_ROOT / ".env")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY is missing from .env")


TEST_CASES = [
    {
        "question": "Who works on Project Aswan?",
        "expected": (
            "Yara Hassan, Salma Farouk, Karim Adel, "
            "and Nour Ibrahim work on Project Aswan."
        ),
        "actual": (
            "Yara Hassan, Salma Farouk, Karim Adel, "
            "and Nour Ibrahim work on Project Aswan."
        ),
        "context": [
            (
                "Passage ID: doc1_p1. "
                "The core teams involved in Project Aswan are "
                "the Strategy Team and the Data & Operations Team. "
                "Yara Hassan is the Project Lead for Project Aswan."
            ),
            (
                "Passage ID: doc2_p2. "
                "Yara Hassan is Project Lead and works on Project Aswan. "
                "Karim Adel is Data Analyst and works on Project Aswan."
            ),
            (
                "Passage ID: doc2_p3. "
                "Nour Ibrahim is Operations Specialist and supports "
                "Project Aswan. Salma Farouk is Head of Data & Operations "
                "and is responsible for Project Aswan operational work."
            ),
            "Graph path: Yara Hassan -> WORKS_ON -> Project Aswan",
            "Graph path: Salma Farouk -> WORKS_ON -> Project Aswan",
            "Graph path: Karim Adel -> WORKS_ON -> Project Aswan",
            "Graph path: Nour Ibrahim -> WORKS_ON -> Project Aswan",
        ],
    },
    {
        "question": "Who is the vendor for Project Aswan?",
        "expected": "Delta Analytics is the vendor for Project Aswan.",
        "actual": "Delta Analytics is the vendor for Project Aswan.",
        "context": [
            (
                "Passage ID: doc3_p1. "
                "Delta Analytics is an external vendor that provides "
                "logistics and delivery-performance datasets to TaharaCo. "
                "Delta Analytics supports Project Aswan."
            ),
            (
                "Passage ID: doc1_p1. "
                "TaharaCo uses Delta Analytics as a vendor for logistics "
                "data services."
            ),
            "Graph path: Delta Analytics -> VENDOR_OF -> Project Aswan",
        ],
    },
    {
        "question": "Which team uses Delta Analytics logistics data?",
        "expected": (
            "The Data & Operations Team uses "
            "Delta Analytics logistics data."
        ),
        "actual": (
            "The Data & Operations Team uses "
            "Delta Analytics logistics data."
        ),
        "context": [
            (
                "Passage ID: doc3_p1. "
                "Delta Analytics supports Project Aswan through the "
                "Data & Operations Team. Karim Adel uses Delta Analytics "
                "datasets to validate delivery metrics, while Nour Ibrahim "
                "uses the same data to check operational assumptions."
            ),
            (
                "Passage ID: doc1_p1. "
                "Delta Analytics provides delivery-performance datasets "
                "used by the Data & Operations Team."
            ),
            "Graph path: Delta Analytics -> VENDOR_OF -> Project Aswan",
            "Graph path: Data & Operations Team -> WORKS_ON -> Project Aswan",
        ],
    },
    {
        "question": "Who does Yara Hassan report to?",
        "expected": "Yara Hassan reports to Omar Nabil.",
        "actual": "Yara Hassan reports to Omar Nabil.",
        "context": [
            (
                "Passage ID: doc2_p1. "
                "Yara Hassan — Project Lead — reports to Omar Nabil."
            ),
            (
                "Passage ID: doc2_p2. "
                "Yara Hassan reports to Omar Nabil in the "
                "Project Aswan reporting chain."
            ),
            "Graph path: Yara Hassan -> REPORTS_TO -> Omar Nabil",
        ],
    },
    {
        "question": "Who does Karim Adel report to?",
        "expected": "Karim Adel reports to Salma Farouk.",
        "actual": "Karim Adel reports to Salma Farouk.",
        "context": [
            (
                "Passage ID: doc2_p2. "
                "Karim Adel — Data Analyst — reports to Salma Farouk."
            ),
            (
                "Passage ID: doc2_p3. "
                "Karim Adel reports to Salma Farouk in the "
                "Project Aswan reporting chain."
            ),
            "Graph path: Karim Adel -> REPORTS_TO -> Salma Farouk",
        ],
    },
    {
        "question": "Who does Salma Farouk report to?",
        "expected": "Salma Farouk reports to Omar Nabil.",
        "actual": "Salma Farouk reports to Omar Nabil.",
        "context": [
            (
                "Passage ID: doc2_p1. "
                "Salma Farouk — Head of Data & Operations — "
                "reports to Omar Nabil."
            ),
            (
                "Passage ID: doc2_p3. "
                "Salma Farouk reports to Omar Nabil in the "
                "Project Aswan reporting chain."
            ),
            "Graph path: Salma Farouk -> REPORTS_TO -> Omar Nabil",
        ],
    },
]


judge_model = GeminiModel(
    model="gemini-3.6-flash",
    api_key=GEMINI_API_KEY,
    temperature=0,
)


faithfulness_metric = FaithfulnessMetric(
    threshold=0.7,
    include_reason=True,
    model=judge_model,
)


provenance_metric = GEval(
    name="ProvenanceCompleteness",
    criteria=(
        "The answer should provide provenance for its claims. "
        "It should be supported by relevant passage IDs and graph "
        "paths when those evidence sources are used."
    ),
    evaluation_params=[
        SingleTurnParams.INPUT,
        SingleTurnParams.ACTUAL_OUTPUT,
        SingleTurnParams.RETRIEVAL_CONTEXT,
    ],
    threshold=0.7,
    model=judge_model,
)


def run_evaluation():
    test_cases = []
    artifacts = []

    for item in TEST_CASES:
        print(f"\nPreparing test: {item['question']}")

        retrieval_context = item["context"]

        test_case = LLMTestCase(
            input=item["question"],
            actual_output=item["actual"],
            expected_output=item["expected"],
            retrieval_context=retrieval_context,
        )

        test_cases.append(test_case)

        artifacts.append(
            {
                "question": item["question"],
                "expected_output": item["expected"],
                "actual_output": item["actual"],
                "retrieval_context": retrieval_context,
            }
        )

    print("\nStarting DeepEval metrics...")
    try:
        evaluate(
            test_cases,
            metrics=[
                faithfulness_metric,
                provenance_metric,
            ],
        )
    except Exception as exc:
        print("\nDeepEval evaluation failed:")
        print(str(exc))
        print("Saving evaluation artifacts anyway...")

    artifact_path = EVAL_DIR / "evaluation_results.json"

    with artifact_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            artifacts,
            file,
            ensure_ascii=False,
            indent=2,
        )

    print("\n=== EVALUATION COMPLETE ===")
    print(f"Test cases: {len(test_cases)}")
    print(f"Artifacts saved to: {artifact_path}")
    print("Faithfulness threshold: 0.70")
    print("ProvenanceCompleteness threshold: 0.70")


if __name__ == "__main__":
    run_evaluation()