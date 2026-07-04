import argparse
import csv
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agents.article_intel_parser import ArticleIntelParser


DEBUG_PATH = PROJECT_ROOT / "reports" / "article_body_parse_debug.csv"
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument("--debug-csv", type=Path, default=DEBUG_PATH)
    args = parser.parse_args()

    article_parser = ArticleIntelParser()
    article = article_parser.fetch(args.url)
    intel = article_parser.extract_intel(article)

    args.debug_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.debug_csv.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "url",
                "status_code",
                "html_length",
                "body_char_count",
                "fetch_success",
                "parser_success",
                "intel_has_content",
                "confidence",
                "failure_reason",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "url": args.url,
                "status_code": article.get("status_code"),
                "html_length": article.get("html_length"),
                "body_char_count": article.get("body_char_count"),
                "fetch_success": article.get("fetch_success"),
                "parser_success": article.get("parser_success"),
                "intel_has_content": intel.get("intel_has_content"),
                "confidence": intel.get("confidence"),
                "failure_reason": article.get("failure_reason"),
            }
        )

    print(
        json.dumps(
            {
                "title": article.get("title"),
                "meta_description": article.get("meta_description"),
                "article_body": article.get("article_body"),
                "body_char_count": article.get("body_char_count"),
                "fetch_success": article.get("fetch_success"),
                "intel": intel,
                "debug_csv": str(args.debug_csv),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
