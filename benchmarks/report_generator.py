from pathlib import Path

# Importers for each module
from benchmarks import (
    run_dense_performance,
    run_ibsi_compliance,
    run_memory_efficiency,
    run_numerical_parity,
    run_reproducibility_stability,
    run_robustness,
    run_runtime_performance,
)


def build_report():
    print("==================================================")
    print("  Initializing fastrad Scientific Report Generator  ")
    print("==================================================\n")

    project_root = Path(__file__).parent.parent
    report_path = project_root / "fastrad_scientific_report.md"

    headers = [
        "# Fastrad Automated Scientific Benchmark Report\n",
        "This report is generated dynamically by evaluating the inner library tensor array routines against PyRadiomics directly.",
        "It constructs the complete analytical footprint mandated for clinical physics paper publications.\n\n",
    ]

    sections = [
        run_ibsi_compliance.run,
        run_numerical_parity.run,
        run_runtime_performance.run,
        run_memory_efficiency.run,
        run_reproducibility_stability.run,
        run_robustness.run,
        run_dense_performance.run,
    ]

    with open(report_path, "w", encoding="utf-8") as f:
        f.writelines(headers)

        for section_func in sections:
            try:
                res = section_func()
                f.write(res)
            except Exception as e:
                import traceback

                print(f"Error executing section {section_func.__name__}: {e}")
                f.write(
                    f"\n\n**Error evaluating section {section_func.__module__}**: `{e}`\n\n"
                )
                f.write(f"```python\n{traceback.format_exc()}\n```\n\n")

    print("\n==================================================")
    print("  Report generated successfully at:")
    print(f"  {report_path}")
    print("==================================================")


if __name__ == "__main__":
    build_report()
