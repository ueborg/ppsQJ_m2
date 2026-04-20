from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pps_qj.exact_left_edge_scan import ExactLeftEdgeScanConfig, run_exact_left_edge_scan


def _format_summary(report: dict) -> str:
    lines: list[str] = []
    params = report["parameters"]
    metrics = report["metrics"]
    artifacts = report["artifacts"]

    lines.append("Exact Left-Edge No-Click Scan")
    lines.append("")
    lines.append(f"alpha mapping: {params['alpha_mapping']}")
    lines.append(f"implemented ratio: {params['implemented_scan_ratio']}")
    lines.append(f"alpha: {params['alpha']}")
    lines.append(f"alpha_eff: {params['alpha_eff']}")
    lines.append(f"L values: {params['L_values']}")
    lines.append(f"ratios: {params['ratios']}")
    lines.append("")
    lines.append(f"max public-endpoint state error: {metrics['max_public_endpoint_state_error']}")
    lines.append(f"largest-L entropy gain: {metrics['largest_L_entropy_gain']}")
    lines.append("crossover estimates:")
    for key, value in metrics["crossover_estimates"].items():
        lines.append(f"  L={key}: {value}")
    lines.append("")
    lines.append("artifacts:")
    for key, value in artifacts.items():
        lines.append(f"  {key}: {value}")
    lines.append("")
    lines.append("notes:")
    for note in report.get("notes", []):
        lines.append(f"  - {note}")
    return "\n".join(lines) + "\n"


def main() -> None:
    config = ExactLeftEdgeScanConfig()
    report = run_exact_left_edge_scan(config)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / "exact_left_edge_scan_report.json"
    txt_path = output_dir / "exact_left_edge_scan_summary.txt"

    json_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    txt_path.write_text(_format_summary(report), encoding="utf-8")

    print(txt_path)
    print(json_path)


if __name__ == "__main__":
    main()
