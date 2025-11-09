"""
Command line interface for the agent log parsing system.
"""

import argparse
from pathlib import Path

from core.orchestrator import LogParsingOrchestrator

DEFAULTS = {
    "log_path": Path("datasets/University/University_full.log"),
    "config": Path("config.json"),
    "model": "gemini-2.0-flash",
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Log parsing agent orchestrator")
    parser.set_defaults(**DEFAULTS)
    parser.add_argument(
        "log_path",
        type=Path,
        nargs="?",
        default=DEFAULTS["log_path"],
        help="Path to the log file to parse",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULTS["config"],
        help="Path to configuration file containing API keys",
    )
    parser.add_argument(
        "--model",
        default=DEFAULTS["model"],
        help="LLM model identifier to use via AIML API",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    orchestrator = LogParsingOrchestrator(
        config_path=args.config,
        model=args.model,
    )
    report = orchestrator.process_log_file(args.log_path)

    print("\n" + "=" * 70)
    print("PARSING REPORT")
    print("=" * 70)
    print(f"\nRouting: {report.routing.device_type} / {report.routing.vendor}")
    print(f"\nResults:")
    print(f"  Processed: {report.processed_lines} lines")
    print(f"  Matched: {report.matched_lines}")
    print(f"  Unmatched: {report.unmatched_lines}")
    print(f"  New templates: {len(report.new_templates)}")

    if report.anomalies:
        print(f"\nAnomalies: {len(report.anomalies)}")
        for item in report.anomalies[:5]:
            print(f"  - {item}")

    print(f"\nOutputs:")
    print(f"  {report.structured_output}")
    print(f"  {report.template_snapshot}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
