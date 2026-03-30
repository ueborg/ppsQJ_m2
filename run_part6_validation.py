from __future__ import annotations

import json
from pathlib import Path

from pps_qj.part6_validation import Part6ValidationConfig, run_part6_validation


def _format_summary(report: dict) -> str:
    lines = []
    lines.append("Part 6 Doob WTMC validation")
    lines.append(f"all_passed: {report['all_passed']}")
    lines.append("")
    for name, result in report["tests"].items():
        lines.append(f"{name}: {'PASS' if result['passed'] else 'FAIL'}")
        for key, value in result.get("metrics", {}).items():
            if isinstance(value, dict):
                lines.append(f"  {key}:")
                for sub_key, sub_value in value.items():
                    lines.append(f"    {sub_key}: {sub_value}")
            else:
                lines.append(f"  {key}: {value}")
        artifacts = result.get("artifacts", {})
        for key, value in artifacts.items():
            lines.append(f"  {key}: {value}")
        lines.append("")
    lines.append("notes:")
    for note in report.get("notes", []):
        lines.append(f"  - {note}")
    return "\n".join(lines) + "\n"


def main() -> None:
    config = Part6ValidationConfig()
    report = run_part6_validation(config)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / "part6_validation_report.json"
    txt_path = output_dir / "part6_validation_summary.txt"

    json_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    txt_path.write_text(_format_summary(report), encoding="utf-8")

    print(txt_path)
    print(json_path)
    print(f"all_passed={report['all_passed']}")


if __name__ == "__main__":
    main()
