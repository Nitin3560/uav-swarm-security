from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from paper_sim.metrics import confidence_interval_95


CONTROLLER_ALIASES = {
    "pid_baseline": "pid",
    "prior_supervisory": "generic",
    "proposed_ids_mpc": "failure_aware",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--scenarios", nargs="+", required=True)
    parser.add_argument("--controllers", nargs="+", default=["pid", "generic", "failure_aware"])
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--output-root", default="paper_sim/outputs")
    parser.add_argument("--ablations", nargs="*", default=["full"])
    return parser.parse_args()


def run_all(
    config: str,
    scenarios: list[str],
    controllers: list[str],
    seeds: list[int],
    output_root: Path,
    ablations: list[str],
) -> None:
    for scenario in scenarios:
        for controller in controllers:
            actual_controller = CONTROLLER_ALIASES.get(controller, controller)
            controller_ablations = ["full"] if actual_controller != "failure_aware" else ablations
            for ablation in controller_ablations:
                for seed in seeds:
                    subprocess.check_call(
                        [
                            "python",
                            "-m",
                            "paper_sim.run_study",
                            "--config",
                            config,
                            "--scenario",
                            scenario,
                            "--controller",
                            actual_controller,
                            "--seed",
                            str(seed),
                            "--output-root",
                            str(output_root),
                            "--ablation",
                            ablation,
                        ]
                    )


def aggregate(output_root: Path, scenarios: list[str]) -> None:
    for scenario in scenarios:
        scen_root = output_root / scenario
        summary_dir = scen_root / "summary"
        rows = []
        for path in sorted(summary_dir.glob("*.csv")):
            rows.append(pd.read_csv(path))
        if not rows:
            continue
        df = pd.concat(rows, ignore_index=True)
        group_col = "controller_label" if "controller_label" in df.columns else "controller"
        agg = (
            df.groupby(group_col, as_index=False)
            .agg(
                pre_fault_error_m=("pre_fault_error_m", "mean"),
                fault_error_m=("fault_error_m", "mean"),
                post_fault_error_m=("post_fault_error_m", "mean"),
                degradation_pct=("degradation_pct", "mean"),
                post_degradation_pct=("post_degradation_pct", "mean"),
                peak_error_spike_m=("peak_error_spike_m", "mean"),
                recovery_time_s=("recovery_time_s", "mean"),
                settling_time_s=("settling_time_s", "mean"),
                rmse_m=("rmse_m", "mean"),
                max_formation_deformation_m=("max_formation_deformation_m", "mean"),
                spacing_violation_count=("spacing_violation_count", "mean"),
                time_above_safety_s=("time_above_safety_s", "mean"),
                control_effort_total=("control_effort_total", "mean"),
                mean_connectivity=("mean_connectivity", "mean"),
                mean_comm_health=("mean_comm_health", "mean"),
                stable_run_fraction=("stable_run", "mean"),
                security_DR=("security_DR", "mean"),
                security_FAR=("security_FAR", "mean"),
                security_AA=("security_AA", "mean"),
                security_raw_AA=("security_raw_AA", "mean"),
                security_TTD_s=("security_TTD_s", "mean"),
                security_DisR=("security_DisR", "mean"),
                security_FCA_m=("security_FCA_m", "mean"),
            )
        )
        ci_fault = df.groupby(group_col)["fault_error_m"].apply(lambda s: confidence_interval_95(s.to_numpy()))
        ci_rmse = df.groupby(group_col)["rmse_m"].apply(lambda s: confidence_interval_95(s.to_numpy()))
        std_fault = df.groupby(group_col)["fault_error_m"].std(ddof=1).fillna(0.0)
        agg["fault_error_std"] = agg[group_col].map(std_fault)
        agg["fault_error_ci95"] = agg[group_col].map(ci_fault)
        agg["rmse_ci95"] = agg[group_col].map(ci_rmse)
        agg.to_csv(scen_root / "aggregate_summary.csv", index=False)

        plt.figure(figsize=(8, 4))
        plt.bar(agg[group_col], agg["degradation_pct"])
        plt.ylabel("Degradation (%)")
        plt.title(f"{scenario}: Degradation")
        plt.tight_layout()
        plt.savefig(scen_root / "degradation_bar.png", dpi=200, bbox_inches="tight")
        plt.close()

        plt.figure(figsize=(8, 4))
        plt.bar(agg[group_col], agg["recovery_time_s"])
        plt.ylabel("Recovery Time (s)")
        plt.title(f"{scenario}: Recovery")
        plt.tight_layout()
        plt.savefig(scen_root / "recovery_bar.png", dpi=200, bbox_inches="tight")
        plt.close()

        concise_cols = [
            group_col,
            "fault_error_m",
            "fault_error_ci95",
            "recovery_time_s",
            "rmse_m",
            "stable_run_fraction",
            "security_DR",
            "security_FAR",
            "security_AA",
            "security_TTD_s",
            "security_FCA_m",
        ]
        agg[concise_cols].to_csv(scen_root / "aggregate_concise_summary.csv", index=False)


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root)
    run_all(args.config, args.scenarios, args.controllers, args.seeds, output_root, args.ablations)
    aggregate(output_root, args.scenarios)


if __name__ == "__main__":
    main()
