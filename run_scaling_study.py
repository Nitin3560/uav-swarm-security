"""run_scaling_study.py — Scalability study across N = 4, 8, 16 agents.

Extension 2 — scales the existing simulation to larger swarms and
aggregates security + control metrics as a function of N.

Usage
-----
    python -m paper_sim.run_scaling_study \
        --config-base configs/final_hybrid.yaml \
        --scenarios jamming_full spoofing_strong replay compound wind \
        --controllers failure_aware pid \
        --swarm-sizes 4 8 16 \
        --seeds 1 2 3 4 5 \
        --output-root paper_sim/outputs_scaling \
        --enable-timing

The script:
  1. Generates a temporary config for each N by patching num_drones.
  2. Calls run_once() from run_study_v2.py for each (N, scenario, controller, seed).
  3. Aggregates across seeds and writes per-N summary CSVs.
  4. Produces the two scaling figures used in the journal paper:
       figure_scaling_security.png  — DR, AA, FAR, TTD vs N
       figure_scaling_control.png   — fault-window RMSE, formation error vs N
"""
from __future__ import annotations

import argparse
import copy
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from paper_sim.run_study_v2 import run_once, controller_label, ATTACK_CLASS_BY_SCENARIO


SWARM_SIZES = [4, 8, 16]

COLORS_N = {4: "#1d4ed8", 8: "#16a34a", 16: "#dc2626"}
MARKERS_N = {4: "o", 8: "s", 16: "^"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-base",  required=True,
                        help="Base YAML config (n=4 baseline)")
    parser.add_argument("--scenarios",    nargs="+",
                        default=["jamming_full", "spoofing_strong", "replay", "compound", "wind"])
    parser.add_argument("--controllers",  nargs="+", default=["failure_aware", "pid"])
    parser.add_argument("--swarm-sizes",  nargs="+", type=int, default=SWARM_SIZES)
    parser.add_argument("--seeds",        nargs="+", type=int, default=list(range(1, 6)))
    parser.add_argument("--output-root",  default="paper_sim/outputs_scaling")
    parser.add_argument("--enable-timing", action="store_true")
    return parser.parse_args()


def patch_config_for_n(base_cfg: dict, n: int) -> dict:
    """Deep-copy and patch num_drones in the config."""
    cfg = copy.deepcopy(base_cfg)
    cfg["sim"]["num_drones"] = n
    return cfg


def load_cfg(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def run_scaling(
    base_cfg: dict,
    scenarios: list[str],
    controllers: list[str],
    swarm_sizes: list[int],
    seeds: list[int],
    output_root: Path,
    enable_timing: bool,
) -> pd.DataFrame:
    """Run full matrix and return a flat DataFrame of all summaries."""
    all_summaries: list[dict] = []

    for n in swarm_sizes:
        cfg_n = patch_config_for_n(base_cfg, n)
        print(f"\n{'='*60}")
        print(f"  Swarm size N = {n}")
        print(f"{'='*60}")
        for scenario in scenarios:
            for ctrl in controllers:
                for seed in seeds:
                    label = f"N={n} | {scenario} | {ctrl} | seed={seed}"
                    print(f"  Running: {label}")
                    out_dir = output_root / f"N{n}" / scenario
                    try:
                        _, summary = run_once(
                            cfg_n, scenario, ctrl, seed, out_dir,
                            ablation="full",
                            enable_timing=enable_timing,
                        )
                        summary["n_agents_run"] = n
                        all_summaries.append(summary)
                    except Exception as exc:
                        print(f"    ERROR: {exc}")
                        all_summaries.append({
                            "n_agents_run": n, "scenario": scenario,
                            "controller": ctrl, "seed": seed,
                            "error": str(exc),
                        })

    df = pd.DataFrame(all_summaries)
    df.to_csv(output_root / "all_scaling_summaries.csv", index=False)
    return df


def aggregate_scaling(df: pd.DataFrame, output_root: Path) -> pd.DataFrame:
    """Aggregate across seeds and return mean+CI table."""
    key_metrics = [
        "security_DR", "security_AA", "security_FAR", "security_TTD_s",
        "fault_error_m", "rmse_m", "max_formation_deformation_m",
        "mean_sensor_gate_rate", "mean_qcomm_physical",
    ]
    group_cols = ["n_agents_run", "scenario", "controller"]
    valid_metrics = [m for m in key_metrics if m in df.columns]
    agg = df.groupby(group_cols)[valid_metrics].agg(["mean", "std"]).reset_index()
    agg.columns = [
        "_".join(c).strip("_") if c[1] else c[0]
        for c in agg.columns
    ]
    agg.to_csv(output_root / "scaling_aggregate.csv", index=False)
    return agg


def plot_security_scaling(agg: pd.DataFrame, output_root: Path) -> None:
    """Figure: security metrics (DR, AA, TTD) vs N per scenario."""
    scenarios    = agg["scenario"].unique()
    n_sizes      = sorted(agg["n_agents_run"].unique())
    metrics      = ["security_DR_mean", "security_AA_mean", "security_TTD_s_mean"]
    ylabels      = ["Detection Rate", "Attribution Accuracy", "Time to Detect (s)"]
    ctrl         = "failure_aware"
    n_metrics    = len(metrics)
    n_scenarios  = len(scenarios)

    fig, axes = plt.subplots(
        n_metrics, n_scenarios,
        figsize=(4.0 * n_scenarios, 3.5 * n_metrics),
        squeeze=False,
    )
    for col, scenario in enumerate(scenarios):
        sub = agg[(agg["scenario"] == scenario) & (agg["controller"] == ctrl)]
        for row, (metric, ylabel) in enumerate(zip(metrics, ylabels)):
            ax = axes[row][col]
            if metric not in sub.columns:
                ax.text(0.5, 0.5, "N/A", ha="center", va="center", transform=ax.transAxes)
                continue
            y   = sub.set_index("n_agents_run")[metric].reindex(n_sizes)
            std_col = metric.replace("_mean", "_std")
            err = sub.set_index("n_agents_run").get(std_col, pd.Series(0, index=n_sizes)).reindex(n_sizes).fillna(0)
            ax.errorbar(
                n_sizes, y.values, yerr=err.values,
                marker="o", linewidth=2, capsize=4,
                color="#1d4ed8",
            )
            ax.set_title(f"{scenario}\n{ylabel}", fontsize=9, fontweight="bold")
            ax.set_xlabel("N (agents)")
            ax.set_xticks(n_sizes)
            ax.set_xticklabels([str(n) for n in n_sizes])
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.grid(True, alpha=0.2)

    fig.suptitle("Security Metrics vs Swarm Size (Proposed Framework)", fontsize=13, fontweight="bold")
    fig.tight_layout()
    path = output_root / "figure_scaling_security.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_control_scaling(agg: pd.DataFrame, output_root: Path) -> None:
    """Figure: control metrics (fault error, RMSE, formation deformation) vs N."""
    scenarios   = agg["scenario"].unique()
    n_sizes     = sorted(agg["n_agents_run"].unique())
    ctrl_list   = ["failure_aware", "pid"]
    ctrl_colors = {"failure_aware": "#1d4ed8", "pid": "#6c757d"}
    ctrl_labels = {"failure_aware": "Proposed", "pid": "PID"}
    metrics     = ["fault_error_m_mean", "rmse_m_mean", "max_formation_deformation_m_mean"]
    ylabels     = ["Fault-Window Error (m)", "Full-Run RMSE (m)", "Max Formation Deformation (m)"]

    fig, axes = plt.subplots(
        len(metrics), len(scenarios),
        figsize=(4.0 * len(scenarios), 3.5 * len(metrics)),
        squeeze=False,
    )
    for col, scenario in enumerate(scenarios):
        for row, (metric, ylabel) in enumerate(zip(metrics, ylabels)):
            ax = axes[row][col]
            for ctrl in ctrl_list:
                sub = agg[(agg["scenario"] == scenario) & (agg["controller"] == ctrl)]
                if sub.empty or metric not in sub.columns:
                    continue
                y   = sub.set_index("n_agents_run")[metric].reindex(n_sizes)
                std_col = metric.replace("_mean", "_std")
                err = sub.set_index("n_agents_run").get(std_col, pd.Series(0, index=n_sizes)).reindex(n_sizes).fillna(0)
                ax.errorbar(
                    n_sizes, y.values, yerr=err.values,
                    marker="o", linewidth=2, capsize=4,
                    color=ctrl_colors[ctrl], label=ctrl_labels[ctrl],
                )
            ax.set_title(f"{scenario}\n{ylabel}", fontsize=9, fontweight="bold")
            ax.set_xlabel("N (agents)")
            ax.set_xticks(n_sizes)
            ax.set_xticklabels([str(n) for n in n_sizes])
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.grid(True, alpha=0.2)
            if col == 0:
                ax.legend(fontsize=8, frameon=False)

    fig.suptitle("Control Metrics vs Swarm Size: Proposed vs PID Baseline", fontsize=13, fontweight="bold")
    fig.tight_layout()
    path = output_root / "figure_scaling_control.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_sensor_gate_benefit(agg: pd.DataFrame, output_root: Path) -> None:
    """Figure: sensor gate activation rate and fault-window error for sensor scenario."""
    if "sensor" not in agg["scenario"].unique():
        return
    n_sizes = sorted(agg["n_agents_run"].unique())
    sub = agg[(agg["scenario"] == "sensor") & (agg["controller"] == "failure_aware")]
    if sub.empty:
        return

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Gate activation rate vs N
    ax = axes[0]
    if "mean_sensor_gate_rate_mean" in sub.columns:
        y = sub.set_index("n_agents_run")["mean_sensor_gate_rate_mean"].reindex(n_sizes)
        ax.bar([str(n) for n in n_sizes], y.values, color="#1d4ed8", alpha=0.8)
        ax.set_xlabel("N (agents)")
        ax.set_ylabel("Mean Gate Rejection Rate")
        ax.set_title("Sensor Gate Activation\n(Extension 1)", fontweight="bold")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # Fault-window error vs N: proposed (with gate) vs pid
    ax = axes[1]
    for ctrl, color, label in [("failure_aware", "#1d4ed8", "Proposed (gated)"),
                                ("pid",           "#6c757d", "PID (no gate)")]:
        s = agg[(agg["scenario"] == "sensor") & (agg["controller"] == ctrl)]
        if s.empty or "fault_error_m_mean" not in s.columns:
            continue
        y = s.set_index("n_agents_run")["fault_error_m_mean"].reindex(n_sizes)
        ax.plot([str(n) for n in n_sizes], y.values,
                marker="o", linewidth=2, color=color, label=label)
    ax.set_xlabel("N (agents)")
    ax.set_ylabel("Fault-Window Error (m)")
    ax.set_title("Sensor Scenario: Error vs N\n(Extension 1 benefit)", fontweight="bold")
    ax.legend(fontsize=9, frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, alpha=0.2)

    fig.suptitle("Sensor Integrity Gate (Extension 1) — Scaling Behaviour", fontsize=12, fontweight="bold")
    fig.tight_layout()
    path = output_root / "figure_sensor_gate_scaling.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def main() -> None:
    args       = parse_args()
    base_cfg   = load_cfg(args.config_base)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    # Run all simulations
    df_all = run_scaling(
        base_cfg,
        args.scenarios,
        args.controllers,
        args.swarm_sizes,
        args.seeds,
        output_root,
        args.enable_timing,
    )

    # Aggregate
    agg = aggregate_scaling(df_all, output_root)

    # Figures
    print("\nGenerating scaling figures...")
    plot_security_scaling(agg, output_root)
    plot_control_scaling(agg, output_root)
    plot_sensor_gate_benefit(agg, output_root)

    print(f"\nAll outputs written to: {output_root}")


if __name__ == "__main__":
    main()
