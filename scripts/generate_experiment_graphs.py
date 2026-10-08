"""
Smart Education Project — Standalone Experiment Graph & Report Generator
=========================================================================
CLI tool to generate all 16 standardized experiment comparison figures
(in high-res 300 DPI PNG and vector SVG) and the 14-section comprehensive
evaluation report directly from empirical results in reports/experiment_results.csv.

Usage:
  python scripts/generate_experiment_graphs.py
  python scripts/generate_experiment_graphs.py --output-dir reports/figures
"""

import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

import os
import argparse
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import REPORTS_DIR, FIGURES_DIR, ensure_dirs
from src.experiment_visualizer import ExperimentVisualizer

def main():
    parser = argparse.ArgumentParser(description="Generate experiment comparison figures and report")
    parser.add_argument("--results-csv", type=str, default=str(REPORTS_DIR / "experiment_results.csv"),
                        help="Path to experiment_results.csv")
    parser.add_argument("--output-dir", type=str, default=str(FIGURES_DIR),
                        help="Output directory for generated figures")
    parser.add_argument("--report-path", type=str, default=str(REPORTS_DIR / "comprehensive_experiment_report.md"),
                        help="Path for generated academic report")
    args = parser.parse_args()

    ensure_dirs()
    print("=" * 80)
    print("  SMART EDUCATION — EXPERIMENT GRAPH & REPORT GENERATION ENGINE")
    print("=" * 80)

    visualizer = ExperimentVisualizer(
        results_csv_path=Path(args.results_csv),
        output_dir=Path(args.output_dir)
    )

    if not visualizer.is_data_available():
        print(f"\n[!] Notice: Results file '{args.results_csv}' not found or empty.")
        print("    Status: NOT RUN")
        print("    Please run 'python scripts/run_experiments.py' first to execute experiments.")
        sys.exit(0)

    # Generate all 16 figures
    results = visualizer.generate_all_figures()

    # Generate comprehensive 14-section report
    report_file = visualizer.generate_comprehensive_report(output_path=Path(args.report_path))

    successful = [k for k, v in results.items() if v is not None]
    print("\n" + "=" * 80)
    print(f"  GENERATION SUMMARY:")
    print(f"  - Figures Generated: {len(successful)} / {len(results)} (PNG @ 300 DPI + Vector SVG)")
    print(f"  - Destination:       {visualizer.output_dir}")
    print(f"  - Academic Report:   {report_file}")
    print("=" * 80)

if __name__ == "__main__":
    main()
