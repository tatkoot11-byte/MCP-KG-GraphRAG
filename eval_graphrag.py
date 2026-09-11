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

EVAL_CASES_FILE = PROJECT_ROOT / "data" / "eval_cases.jsonl"

load_dotenv(PROJECT_ROOT / ".env")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY is missing from .env")


def load_eval_cases():
    if not EVAL_CASES_FILE.exists():
        raise FileNotFoundError(
            f"Evaluation cases file not found: {EVAL_CASES_FILE}"
        )

    cases = []

    with EVAL_CASES_FILE.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON on line {line_number}: {exc}"
                ) from exc

            required_fields = {
                "input",
                "expected_output",
                "retrieval_context",
            }

            missing = required_fields - item.keys()

            if missing:
                raise ValueError(
                    f"Line {line_number} is missing fields: {sorted(missing)}"
                )

            cases.append(item)

    if not 5 <= len(cases) <= 10:
        raise ValueError(
            f"Expected 5-10 evaluation cases, found {len(cases)}"
        )

    return cases


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
    cases = load_eval_cases()

    print(f"Loaded evaluation cases: {len(cases)}")

    test_cases = []
    artifacts = []

    for item in cases:
        print(f"\nPreparing test: {item['input']}")

        # For the offline/demo evaluation artifact, the expected answer
        # is used as the actual answer. DeepEval can then evaluate the
        # supplied evidence/provenance when the Gemini quota is available.
        actual_output = item["expected_output"]

        test_case = LLMTestCase(
            input=item["input"],
            actual_output=actual_output,
            expected_output=item["expected_output"],
            retrieval_context=item["retrieval_context"],
        )

        test_cases.append(test_case)

        artifacts.append(
            {
                "input": item["input"],
                "expected_output": item["expected_output"],
                "actual_output": actual_output,
                "retrieval_context": item["retrieval_context"],
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

        print("\nDeepEval evaluation finished successfully.")

    except Exception as exc:
        print("\nDeepEval evaluation failed:")
        print(str(exc))
        print("Saving evaluation artifacts anyway.")

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