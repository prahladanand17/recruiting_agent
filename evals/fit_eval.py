"""Run from repo root: python -m evals.fit_eval  OR  python evals/fit_eval.py"""

from __future__ import annotations

import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# Running as a script sets sys.path to evals/ — add repo root so `agent` resolves.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from anthropic import Anthropic
from tqdm import tqdm

from agent.tool_impl import evaluate_fit, research_company
from evals.utils import extract_company as extract_company_name


def _tqdm_disabled() -> bool:
    return os.getenv("TQDM_DISABLE", "").strip().lower() in ("1", "true", "yes")


def _eval_one_email(email: dict) -> str:
    """
    Classify one example. Uses a dedicated Anthropic client per task so thread pools
    do not share a single HTTP client.
    """
    client = Anthropic()
    ground_truth = email.get("_fit", False)

    company_name = extract_company_name(
        email["subject"], email["body"], client=client
    )
    research = research_company(company_name, client=client)
    email_body = email.get("body", "")
    result = evaluate_fit(
        company_name, email["subject"], research, client=client, email_body=email_body
    )
    predicted = result.get("fit", False)

    if predicted and ground_truth:
        bucket = "tp"
    elif predicted and not ground_truth:
        bucket = "fp"
    elif not predicted and ground_truth:
        bucket = "fn"
    else:
        bucket = "tn"

    # log failures only
    if bucket in ("fp", "fn"):
        print(f"\n{bucket.upper()}: {email['subject']}")
        print(f"  Extracted company: {company_name}")
        print(f"  AI focus: {research.get('ai_focus', 'unknown')}")
        print(f"  Funding stage: {research.get('funding_stage', 'unknown')}")
        print(f"  NYC office: {research.get('nyc_office', 'unknown')}")
        print(f"  Fit decision: {result.get('fit')} | {result.get('reason')}")
        print(f"  Flags: {result.get('flags', [])}")
        print(f"  Ground truth: {ground_truth}")

    return bucket


def run_eval(emails_path: str | Path = "email_handler/emails.json") -> None:
    path = Path(emails_path)
    if not path.is_file():
        path = _ROOT / emails_path

    tp = fp = fn = tn = 0

    with path.open(encoding="utf-8") as f:
        emails = json.load(f)

    max_workers = max(1, int(os.getenv("EVAL_MAX_WORKERS", "8")))
    print(
        f"Parallel eval: {max_workers} workers "
        f"(set EVAL_MAX_WORKERS to tune; reduce if you hit rate limits)"
    )

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(_eval_one_email, e) for e in emails]
        for fut in tqdm(
            as_completed(futures),
            total=len(futures),
            desc="Fit eval",
            unit="email",
            disable=_tqdm_disabled(),
        ):
            bucket = fut.result()

            
            # Counts updated only on the main thread (workers return labels); no shared-counter races.
            if bucket == "tp":
                tp += 1
            elif bucket == "fp":
                fp += 1
            elif bucket == "fn":
                fn += 1
            else:
                tn += 1

    print(f"True Positives: {tp}")
    print(f"True Negatives: {tn}")
    print(f"False Positives: {fp}")
    print(f"False Negatives: {fn}")

    total = tp + fp + fn + tn
    if total:
        print(f"Accuracy: {(tp + tn) / total}")

    denom_p = tp + fp
    denom_r = tp + fn
    precision = tp / denom_p if denom_p else 0.0
    recall = tp / denom_r if denom_r else 0.0

    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    if precision + recall:
        print(f"F1 Score: {2 * precision * recall / (precision + recall)}")
    else:
        print("F1 Score: undefined (no positive predictions / no positives in gold)")


if __name__ == "__main__":
    run_eval()
