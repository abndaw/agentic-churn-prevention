"""
Threshold optimisation for retention system.

- Thresholds tuned on validation set
- Final reporting on test set
- ML and LLM share identical targeting policy
"""

import json
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from dataclasses import dataclass
from preprocess import preprocess

sns.set_style('whitegrid')


@dataclass
class Params:
    """
    Economic parameters for cost-sensitive retention optimization.
    
    Treatment effects (Ascarza, 2018):
    - High-risk customers (p ≥ 0.90) show LOWER responsiveness (0.90)
    - Medium-risk customers (0.30 ≤ p < 0.90) show HIGHER responsiveness (1.05)
    
    Rationale: High-risk customers are already decided to leave and may perceive aggressive offers as desperation signals, 
    resulting in lower incremental retention despite higher intervention costs.
    """
    
    # Customer economics
    monthly_revenue: float = 65.0  # Dataset mean: $64.76
    lifetime_months: int = 24      # Conservative vs median (29 months)
    
    # Intervention costs
    offer_standard: float = 30.0   # 46% of monthly revenue
    offer_aggressive: float = 60.0  # 92% of monthly revenue
    email_generic: float = 5.0
    email_llm: float = 8.0
    
    # Acceptance rates (empirical estimates)
    acc_baseline: float = 0.12  # Mass campaign
    acc_ml: float = 0.18        # ML-targeted
    acc_llm: float = 0.22       # ML + LLM personalization
    
    # Retention parameters
    retention_base: float = 0.60  # P(stay|accept offer)
    
    # Treatment effect heterogeneity (Ascarza, 2018)
    # High-risk customers exhibit LOWER responsiveness
    treatment_effect_high: float = 0.90   # p ≥ t_high
    treatment_effect_medium: float = 1.05  # t_med ≤ p < t_high
    
    # Causal attribution (Uplift modeling)
    # Only 40% of retention is incremental (vs natural retention)
    attribution: float = 0.40  # Validated in sensitivity analysis

    @property
    def ltv(self):
        return self.monthly_revenue * self.lifetime_months


P = Params()


def expected_value(y, p, t_high, t_med, acc, email_cost):
    """Calculate expected value for threshold optimization."""
    y, p = np.asarray(y), np.asarray(p)

    actions = np.zeros_like(y)
    actions[p >= t_med] = 1
    actions[p >= t_high] = 2

    cost = np.zeros_like(y, dtype=float)
    benefit = np.zeros_like(y, dtype=float)

    for tier, off, mult in [
        (1, P.offer_standard, P.treatment_effect_medium),
        (2, P.offer_aggressive, P.treatment_effect_high)
    ]:
        mask = actions == tier

        cost[mask] = email_cost + off * acc

        churners = mask & (y == 1)
        benefit[churners] = (
            acc *
            P.retention_base *
            mult *
            P.ltv *
            P.attribution
        )

    return benefit.sum() - cost.sum()


def compute_metrics(y, p, t_high, t_med, acc, email_cost):
    """Compute detailed metrics for a given policy."""
    y, p = np.asarray(y), np.asarray(p)

    actions = np.zeros_like(y)
    actions[p >= t_med] = 1
    actions[p >= t_high] = 2

    total_cost = 0.0
    total_benefit = 0.0

    for tier, off, mult in [
        (1, P.offer_standard, P.treatment_effect_medium),
        (2, P.offer_aggressive, P.treatment_effect_high)
    ]:
        mask = actions == tier

        cost = email_cost + off * acc
        total_cost += mask.sum() * cost

        churners = mask & (y == 1)
        total_benefit += churners.sum() * (
            acc *
            P.retention_base *
            mult *
            P.ltv *
            P.attribution
        )

    coverage = (actions > 0).mean()

    return {
        "profit": total_benefit - total_cost,
        "total_cost": total_cost,
        "total_benefit": total_benefit,
        "coverage": coverage
    }


def baseline_mass_campaign(y, email_cost):
    """Baseline: mass email campaign to all customers."""
    y = np.asarray(y)

    n = len(y)

    total_cost = n * email_cost

    churners = y == 1

    total_benefit = churners.sum() * (
        P.acc_baseline *
        P.retention_base *
        P.treatment_effect_medium *
        P.ltv *
        P.attribution
    )

    return {
        "profit": total_benefit - total_cost,
        "total_cost": total_cost,
        "total_benefit": total_benefit,
        "coverage": 1.0
    }


def grid_search(y, p, acc, email_cost):
    """Grid search over threshold space to find optimal policy."""
    best = None

    for t_high in np.arange(0.50, 0.91, 0.05):
        for t_med in np.arange(0.20, 0.71, 0.05):
            if t_med >= t_high:
                continue

            profit = expected_value(y, p, t_high, t_med, acc, email_cost)

            if best is None or profit > best["profit"]:
                best = {
                    "profit": profit,
                    "threshold_high": t_high,
                    "threshold_medium": t_med
                }

    return best


def plot_results(base, ml, llm,
                 save_path="../outputs/business_impact_comparison.png"):
    """
    Create 3 panel visualization of retention strategies.
    
    Panels:
    1. Profit by strategy
    2. Churner coverage
    3. Cost vs benefit breakdown
    """
    
    fig, axes = plt.subplots(1, 3, figsize=(20, 6.5))

    # PANEL 1: PROFIT
    ax = axes[0]

    labels = ["Baseline\n(Mass Campaign)", "ML", "ML + LLM"]

    profits = [
        base["profit"],
        ml["profit"],
        llm["profit"]
    ]

    colors = ["#808080", "#ff7f0e", "#2ca02c"]

    bars = ax.bar(labels, profits, color=colors, alpha=0.85, edgecolor="black")

    top = max(profits + [1])

    for b, v in zip(bars, profits):
        ax.text(
            b.get_x() + b.get_width() / 2,
            b.get_height() + top * 0.01,
            f"${v:,.0f}",
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=12
        )

    ax.set_title("Profit by Strategy", fontweight="bold", fontsize=14)
    ax.set_ylabel("Profit ($)", fontsize=12)
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, top * 1.15)

    # PANEL 2: COVERAGE
    ax = axes[1]

    coverage_bars = ax.bar(
        ["Baseline", "ML", "ML + LLM"],
        [base["coverage"], ml["coverage"], llm["coverage"]],
        color=["#808080", "#ff7f0e", "#2ca02c"],
        alpha=0.85,
        edgecolor="black"
    )

    # Adding percentage labels on bars
    for b, v in zip(coverage_bars, [base["coverage"], ml["coverage"], llm["coverage"]]):
        ax.text(
            b.get_x() + b.get_width() / 2,
            b.get_height() + 0.02,
            f"{v:.1%}",
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=12
        )

    ax.set_title("Customer Contact Rate", fontweight="bold", fontsize=14)
    ax.set_ylabel("Fraction contacted", fontsize=12)
    ax.set_ylim(0, 1.15)
    ax.grid(axis="y", alpha=0.3)

    # PANEL 3: COST VS BENEFIT
    ax = axes[2]

    scenarios = ["Baseline", "ML", "ML + LLM"]

    costs = [
        base["total_cost"],
        ml["total_cost"],
        llm["total_cost"]
    ]

    benefits = [
        base["total_benefit"],
        ml["total_benefit"],
        llm["total_benefit"]
    ]

    x = np.arange(len(scenarios))
    width = 0.35

    bars1 = ax.bar(x - width/2, costs, width, label="Cost", 
                   color="#d62728", alpha=0.85, edgecolor="black")
    bars2 = ax.bar(x + width/2, benefits, width, label="Benefit", 
                   color="#2ca02c", alpha=0.85, edgecolor="black")

    # Add value labels on bars
    for bars in [bars1, bars2]:
        for b in bars:
            height = b.get_height()
            ax.text(
                b.get_x() + b.get_width() / 2,
                height + max(costs + benefits) * 0.01,
                f"${height:,.0f}",
                ha="center",
                va="bottom",
                fontsize=10
            )

    ax.set_xticks(x)
    ax.set_xticklabels(scenarios)
    ax.set_title("Cost vs Benefit", fontweight="bold", fontsize=14)
    ax.set_ylabel("Amount ($)", fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"\nPlot saved to: {save_path}")

    return fig


def main():
    """Main execution pipeline."""
    
    print("Loading data...")
    _, X_val, X_test, _, y_val, y_test, _ = preprocess(scale_numeric=False)

    with open("../models/best_model.pkl", "rb") as f:
        model = pickle.load(f)

    #  VALIDATION: FINDING OPTIMAL THRESHOLDS 
    p_val = model.predict_proba(X_val)[:, 1]
    y_val = y_val.values

    print(f"Validation size: {len(y_val)}")
    print("\nOptimizing thresholds on validation set")

    policy = grid_search(y_val, p_val, P.acc_ml, P.email_generic)

    print("\nOptimal policy (shared by ML and ML+LLM):")
    print(f"  Threshold high:   {policy['threshold_high']:.2f}")
    print(f"  Threshold medium: {policy['threshold_medium']:.2f}")

    # TEST SET: FINAL EVALUATION 
    p_test = model.predict_proba(X_test)[:, 1]
    y_test = y_test.values

    print(f"\nTest size: {len(y_test)}")
    print("Evaluating on test set")

    # BASELINE (mass campaign)
    base = baseline_mass_campaign(y_test, P.email_generic)

    # ML (targeted with optimal thresholds)
    ml_metrics = compute_metrics(
        y_test,
        p_test,
        policy["threshold_high"],
        policy["threshold_medium"],
        P.acc_ml,
        P.email_generic
    )

    # ML + LLM (targeted + personalized)
    llm_metrics = compute_metrics(
        y_test,
        p_test,
        policy["threshold_high"],
        policy["threshold_medium"],
        P.acc_llm,
        P.email_llm
    )

    # RESULTS 
    print("\n" + "=" * 60)
    print("FINAL RESULTS (Test Set)")
    print("=" * 60)
    print(f"Baseline (Mass Campaign):")
    print(f"  Profit:   ${base['profit']:>10,.0f}")
    print(f"  Cost:     ${base['total_cost']:>10,.0f}")
    print(f"  Benefit:  ${base['total_benefit']:>10,.0f}")
    print(f"  Coverage: {base['coverage']:>10.1%}")
    
    print(f"\nML (Targeted):")
    print(f"  Profit:   ${ml_metrics['profit']:>10,.0f}  (+{(ml_metrics['profit']/base['profit']-1)*100:>5.1f}%)")
    print(f"  Cost:     ${ml_metrics['total_cost']:>10,.0f}")
    print(f"  Benefit:  ${ml_metrics['total_benefit']:>10,.0f}")
    print(f"  Coverage: {ml_metrics['coverage']:>10.1%}")
    
    print(f"\nML + LLM (Targeted + Personalized):")
    print(f"  Profit:   ${llm_metrics['profit']:>10,.0f}  (+{(llm_metrics['profit']/base['profit']-1)*100:>5.1f}%)")
    print(f"  Cost:     ${llm_metrics['total_cost']:>10,.0f}")
    print(f"  Benefit:  ${llm_metrics['total_benefit']:>10,.0f}")
    print(f"  Coverage: {llm_metrics['coverage']:>10.1%}")
    print("=" * 60)

    # Saving
    output = {
        "policy": policy,
        "baseline": base,
        "ml": ml_metrics,
        "llm": llm_metrics
    }

    with open("../outputs/threshold_results.json", "w") as f:
        json.dump(output, f, indent=2)

    print("\nSaved results to: ../outputs/threshold_results.json")

    # PLOT
    plot_results(base, ml_metrics, llm_metrics)


if __name__ == "__main__":
    main()