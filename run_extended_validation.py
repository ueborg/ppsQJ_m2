from __future__ import annotations

import json
from pathlib import Path

from pps_qj.extended_validation import ExtendedValidationConfig, run_extended_validation


def _format_section(name: str, result: dict) -> list[str]:
    lines = [f"{name}: {'PASS' if result.get('passed') else 'FAIL'}"]
    for key, value in result.get("metrics", {}).items():
        if isinstance(value, dict):
            lines.append(f"  {key}:")
            for sub_key, sub_value in value.items():
                lines.append(f"    {sub_key}: {sub_value}")
        else:
            lines.append(f"  {key}: {value}")
    for key, value in result.get("artifacts", {}).items():
        lines.append(f"  {key}: {value}")
    lines.append("")
    return lines


def _format_summary(report: dict) -> str:
    lines: list[str] = []
    lines.append("Extended Doob WTMC validation")
    lines.append(f"all_passed: {report['all_passed']}")
    lines.append(f"elapsed_seconds: {report.get('elapsed_seconds', 0.0)}")
    lines.append("")
    lines.extend(_format_section("overlap_microtest", report["overlap_microtest"]))
    for name, result in report["tests"].items():
        lines.extend(_format_section(name, result))
    if report.get("artifacts"):
        lines.append("report_plots:")
        for key, value in report["artifacts"].items():
            lines.append(f"  {key}: {value}")
        lines.append("")
    lines.append("notes:")
    for note in report.get("notes", []):
        lines.append(f"  - {note}")
    return "\n".join(lines) + "\n"


def main() -> None:
    config = ExtendedValidationConfig()
    report = run_extended_validation(config)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / "extended_validation_report.json"
    txt_path = output_dir / "extended_validation_summary.txt"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    txt_path.write_text(_format_summary(report), encoding="utf-8")

    print(txt_path)
    print(json_path)
    print(f"all_passed={report['all_passed']}")


if __name__ == "__main__":
    main()
