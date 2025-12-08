#!/usr/bin/env python3

"""
Logistic Regression Analysis for Igbo Acceptability Judgments.

Research question:
    Can LM probability predict which sentence humans find more acceptable?

Input:
    simulated_judgments.jsonl

Outputs (in 'results' directory):
    - regression_summary.txt           
    - figures/score_distribution.png   (distribution of LM score differences)
    - figures/probability_curve.png    (logistic curve + observed choices)
    - figures/confusion_matrix.png     
"""


import json
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import statsmodels.api as sm
from scipy.special import expit  # sigmoid

from sklearn.metrics import (confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score)


INPUT_PATH = Path("simulation_output/simulated_judgments.jsonl")
OUTPUT_DIR = Path("results")


# 1. Load and prepare data

def load_data(path: Path) -> pd.DataFrame:
    print(f"Loading data from {path}...")
    data = []
    with open(path, "r") as f:
        for line in f:
            data.append(json.loads(line))
    df = pd.DataFrame(data)
    print(f"  Loaded {len(df)} observations")
    return df


def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # LM score difference: B - A
    df["lm_score_diff_ba"] = df["lm_score_b"] - df["lm_score_a"]

    # Target: 1 = chose B, 0 = chose A ie the choice_numeric column
    print(f"  Predictor range (Score B-A): {df['lm_score_diff_ba'].min():.2f} to "
          f"{df['lm_score_diff_ba'].max():.2f}")
    print(f"  Outcome distribution: {df['choice_numeric'].mean():.1%} chose B")

    return df


# 2. Descriptive statistics

def compute_descriptive_stats(df: pd.DataFrame) -> dict:
    stats_dict = {
        "n_observations": len(df),
        "n_chose_a": int((df["choice_numeric"] == 0).sum()),
        "n_chose_b": int((df["choice_numeric"] == 1).sum()),
        "pct_chose_b": float(df["choice_numeric"].mean()),
        "lm_score_diff": {
            "mean": float(df["lm_score_diff_ba"].mean()),
            "std": float(df["lm_score_diff_ba"].std()),
            "min": float(df["lm_score_diff_ba"].min()),
            "max": float(df["lm_score_diff_ba"].max()),
            "median": float(df["lm_score_diff_ba"].median()),
        },
        "by_pair_type": {},
    }

    for pair_type in df["pair_type"].unique():
        subset = df[df["pair_type"] == pair_type]
        stats_dict["by_pair_type"][pair_type] = {
            "n": len(subset),
            "pct_chose_b": float(subset["choice_numeric"].mean()),
        }

    return stats_dict


# 3. Logistic regression

def run_logistic_regression(df: pd.DataFrame):
    """
    Model: logit(P(choose B)) = β0 + β1 * (lm_score_b - lm_score_a)
    """
    print("Running logistic regression...")
    y = df["choice_numeric"].values
    X = df["lm_score_diff_ba"].values

    X_with_const = sm.add_constant(X)
    model = sm.Logit(y, X_with_const)
    result = model.fit(disp=0)

    # Extract key numbers in a simple dict
    summary = {
        "n_observations": int(result.nobs),
        "intercept": {
            "estimate": float(result.params[0]),
            "std_err": float(result.bse[0]),
            "z_value": float(result.tvalues[0]),
            "p_value": float(result.pvalues[0]),
        },
        "lm_score_diff": {
            "estimate": float(result.params[1]),
            "std_err": float(result.bse[1]),
            "z_value": float(result.tvalues[1]),
            "p_value": float(result.pvalues[1]),
            "odds_ratio": float(np.exp(result.params[1])),
        },
        "model_fit": {
            "pseudo_r_squared": float(result.prsquared),
            "log_likelihood": float(result.llf),
            "aic": float(result.aic),
            "bic": float(result.bic),
        },
        "significant": bool(result.pvalues[1] < 0.05),
    }

    return result, summary


# 4. Evaluation

def evaluate_model(df, result):
    print("Evaluating model...")

    y_true = df["choice_numeric"].values
    X = df["lm_score_diff_ba"].values
    X_with_const = sm.add_constant(X)

    # Predicted probabilities and classes
    y_prob = result.predict(X_with_const)
    y_pred = (y_prob >= 0.5).astype(int)

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
    }

    cm = confusion_matrix(y_true, y_pred)
    cm_dict = {
        "true_negative": int(cm[0, 0]),
        "false_positive": int(cm[0, 1]),
        "false_negative": int(cm[1, 0]),
        "true_positive": int(cm[1, 1]),
    }

    clf_report = classification_report(
        y_true, y_pred, target_names=["Chose A", "Chose B"]
    )

    return {
        "y_prob": y_prob,
        "y_pred": y_pred,
        "metrics": metrics,
        "confusion_matrix": cm_dict,
        "classification_report": clf_report,
    }


# 5. Visualizations

def create_visualizations(df, result, evaluation, output_dir) -> list[Path]:
    print("Creating visualizations...")

    figures_dir = output_dir / "figures"
    figures_dir.mkdir(exist_ok=True)

    created_figures: list[Path] = []

    plt.style.use("seaborn-v0_8-whitegrid")
    sns.set_palette("husl")

    # 5.1 Score distribution and by choice
    print("  - score_distribution.png")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Histogram of LM score differences
    ax1 = axes[0]
    ax1.hist(df["lm_score_diff_ba"], bins=50, edgecolor="black", alpha=0.7)
    ax1.axvline(x=0, color="red", linestyle="--", label="Equal scores")
    ax1.axvline(
        x=df["lm_score_diff_ba"].mean(),
        color="green",
        linestyle="-",
        label=f"Mean = {df['lm_score_diff_ba'].mean():.1f}",
    )
    ax1.set_xlabel("LM Score Difference (B - A)")
    ax1.set_ylabel("Frequency")
    ax1.set_title("Distribution of LM Score Differences")
    ax1.legend()

    # Boxplot by human choice
    ax2 = axes[1]
    df_plot = df.copy()
    df_plot["Choice"] = df_plot["choice_numeric"].map({0: "Chose A", 1: "Chose B"})
    sns.boxplot(data=df_plot, x="Choice", y="lm_score_diff_ba", ax=ax2)
    ax2.axhline(y=0, color="red", linestyle="--", alpha=0.5)
    ax2.set_xlabel("Human Choice")
    ax2.set_ylabel("LM Score Difference (B - A)")
    ax2.set_title("Score Differences by Choice")

    plt.tight_layout()
    path = figures_dir / "score_distribution.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    created_figures.append(path)

    # 5.2 Probability curve (logistic function)
    print("  - probability_curve.png")
    fig, ax = plt.subplots(figsize=(10, 6))

    b0 = result.params[0]
    b1 = result.params[1]

    x_range = np.linspace(df["lm_score_diff_ba"].min(), df["lm_score_diff_ba"].max(), 200)
    y_prob_curve = expit(b0 + b1 * x_range)

    # Fitted curve
    ax.plot(x_range, y_prob_curve, "b-", linewidth=2, label="Fitted logistic curve")

    # Observed points
    jitter = np.random.normal(0, 0.02, len(df))
    ax.scatter(
        df["lm_score_diff_ba"],
        df["choice_numeric"] + jitter,
        alpha=0.3,
        s=20,
        c="gray",
        label="Observed choices",
    )

    ax.axhline(y=0.5, color="red", linestyle="--", alpha=0.5, label="P = 0.5")
    ax.axvline(x=0, color="green", linestyle="--", alpha=0.5, label="Equal LM scores")

    ax.set_xlabel("LM Score Difference (B - A)")
    ax.set_ylabel("P(Choose B)")
    ax.set_title("Logistic Regression: Probability of Choosing B")
    ax.set_ylim(-0.1, 1.1)
    ax.legend(loc="lower right")

    eq_text = f"logit(P) = {b0:.4f} + {b1:.6f} × ΔScore"
    ax.text(
        0.05,
        0.95,
        eq_text,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat"),
    )

    plt.tight_layout()
    path = figures_dir / "probability_curve.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    created_figures.append(path)

    # 5.3 Confusion matrix heatmap (if metrics available)
    if "confusion_matrix" in evaluation:
        print("  - confusion_matrix.png")
        fig, ax = plt.subplots(figsize=(8, 6))

        cm = evaluation["confusion_matrix"]
        cm_array = np.array(
            [
                [cm["true_negative"], cm["false_positive"]],
                [cm["false_negative"], cm["true_positive"]],
            ]
        )

        sns.heatmap(
            cm_array,
            annot=True,
            fmt="d",
            cmap="Blues",
            ax=ax,
            xticklabels=["Predicted A", "Predicted B"],
            yticklabels=["Actual A", "Actual B"],
        )
        ax.set_xlabel("Predicted Choice")
        ax.set_ylabel("Actual Choice")
        ax.set_title("Confusion Matrix")

        acc = evaluation["metrics"]["accuracy"]
        ax.text(
            0.5,
            -0.12,
            f"Accuracy: {acc:.1%}",
            transform=ax.transAxes,
            fontsize=12,
            ha="center",
        )

        plt.tight_layout()
        path = figures_dir / "confusion_matrix.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        created_figures.append(path)

    return created_figures


# 6. Text report

def generate_report(
    descriptive_stats: dict,
    regression_summary: dict,
    evaluation: dict,
    output_dir: Path,
) -> Path:
    print("Generating text report...")

    lines: list[str] = []

    def add(line: str = ""):
        lines.append(line)

    add("=" * 80)
    add("LOGISTIC REGRESSION ANALYSIS: IGBO ACCEPTABILITY JUDGMENTS")
    add("=" * 80)
    add()

    # Research question
    add("Research question:")
    add("  Can language model scores predict human acceptability judgments?")
    add()
    add("Model:")
    add("  logit(P(choose B)) = β0 + β1 × (LM_score_B - LM_score_A)")
    add()

    # Descriptive statistics
    add("-" * 80)
    add("DESCRIPTIVE STATISTICS")
    add("-" * 80)
    add(f"Total observations: {descriptive_stats['n_observations']}")
    add(
        f"Chose A: {descriptive_stats['n_chose_a']} "
        f"({100*(1 - descriptive_stats['pct_chose_b']):.1f}%)"
    )
    add(
        f"Chose B: {descriptive_stats['n_chose_b']} "
        f"({100*descriptive_stats['pct_chose_b']:.1f}%)"
    )
    add()

    lm_stats = descriptive_stats["lm_score_diff"]
    add("LM Score Difference (B - A):")
    add(f"  Mean:   {lm_stats['mean']:.2f}")
    add(f"  Std:    {lm_stats['std']:.2f}")
    add(f"  Min:    {lm_stats['min']:.2f}")
    add(f"  Max:    {lm_stats['max']:.2f}")
    add(f"  Median: {lm_stats['median']:.2f}")
    add()

    # Regression results
    add("-" * 80)
    add("LOGISTIC REGRESSION RESULTS")
    add("-" * 80)

    add(f"N observations (used in model): {regression_summary['n_observations']}")
    add()

    int_est = regression_summary["intercept"]["estimate"]
    int_se = regression_summary["intercept"]["std_err"]
    int_z = regression_summary["intercept"]["z_value"]
    int_p = regression_summary["intercept"]["p_value"]

    coef_est = regression_summary["lm_score_diff"]["estimate"]
    coef_se = regression_summary["lm_score_diff"]["std_err"]
    coef_z = regression_summary["lm_score_diff"]["z_value"]
    coef_p = regression_summary["lm_score_diff"]["p_value"]
    odds_ratio = regression_summary["lm_score_diff"]["odds_ratio"]

    add("Coefficients:")
    add("  Intercept (β0):")
    add(f"    Estimate: {int_est:.6f}")
    add(f"    Std Err:  {int_se:.6f}")
    add(f"    z-value:  {int_z:.4f}")
    add(f"    p-value:  {int_p:.2e}")
    add()
    add("  LM Score Difference (β1):")
    add(f"    Estimate:   {coef_est:.6f}")
    add(f"    Std Err:    {coef_se:.6f}")
    add(f"    z-value:    {coef_z:.4f}")
    add(f"    p-value:    {coef_p:.2e}")
    add(f"    Odds Ratio: {odds_ratio:.6f}")
    add()

    fit = regression_summary["model_fit"]
    add("Model fit:")
    add(f"  Pseudo R²:       {fit['pseudo_r_squared']:.4f}")
    add(f"  Log-Likelihood:  {fit['log_likelihood']:.2f}")
    add(f"  AIC:             {fit['aic']:.2f}")
    add(f"  BIC:             {fit['bic']:.2f}")
    add()

    add("Statistical significance (β1):")
    if regression_summary["significant"]:
        add("  ✓ Significant (p < 0.05)")
        if coef_est > 0:
            add("  Interpretation:")
            add("    Higher LM score for B predicts higher probability of humans choosing B.")
        else:
            add("  Interpretation:")
            add("    Higher LM score for B predicts lower probability of humans choosing B (opposite direction).")
    else:
        add("  ✗ Not significant (p ≥ 0.05)")
        add("  Interpretation:")
        add("    No clear evidence that LM scores predict human choices.")
    add()

    # Evaluation metrics
    add("-" * 80)
    add("MODEL EVALUATION (THRESHOLD = 0.5)")
    add("-" * 80)
    metrics = evaluation["metrics"]
    add(f"Accuracy:  {metrics['accuracy']:.3f}")
    add(f"Precision: {metrics['precision']:.3f}")
    add(f"Recall:    {metrics['recall']:.3f}")
    add(f"F1 Score:  {metrics['f1_score']:.3f}")
    add()

    cm = evaluation["confusion_matrix"]
    add("Confusion matrix:")
    add("                  Predicted A    Predicted B")
    add(f"  Actual A:         {cm['true_negative']:>6}         {cm['false_positive']:>6}")
    add(f"  Actual B:         {cm['false_negative']:>6}         {cm['true_positive']:>6}")
    add()
    add("Classification report:")
    add(evaluation["classification_report"])
    add()

    add("-" * 80)
    add("END OF REPORT")
    add("-" * 80)

    report_path = output_dir / "regression_summary.txt"
    with open(report_path, "w") as f:
        f.write("\n".join(lines))

    return report_path



def main():
    print("=" * 80)
    print("LOGISTIC REGRESSION ANALYSIS")
    print("Igbo Acceptability Judgments")
    print("=" * 80)
    print()

    OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
    print(f"Output directory: {OUTPUT_DIR}")
    print()

    # 1. Load data
    print("[1/5] Loading data...")
    df = load_data(INPUT_PATH)
    print()

    # 2. Prepare data
    print("[2/5] Preparing data...")
    df = prepare_data(df)
    print()

    # 3. Descriptive stats
    print("[3/5] Computing descriptive statistics...")
    descriptive_stats = compute_descriptive_stats(df)
    print()

    # 4. Logistic regression and evaluation
    print("[4/5] Running logistic regression and evaluation...")
    result, regression_summary = run_logistic_regression(df)
    evaluation = evaluate_model(df, result)
    print(f"  Coefficient (β1): {regression_summary['lm_score_diff']['estimate']:.6f}")
    print(f"  p-value:          {regression_summary['lm_score_diff']['p_value']:.2e}")
    print(f"  Odds Ratio:       {regression_summary['lm_score_diff']['odds_ratio']:.4f}")
    print(f"  Accuracy:         {evaluation['metrics']['accuracy']:.3f}")
    print()

    # 5. Visualizations and report
    print("[5/5] Creating visualizations and report...")
    figures = create_visualizations(df, result, evaluation, OUTPUT_DIR)
    for fig in figures:
        print(f" {fig}")
    report_path = generate_report(descriptive_stats, regression_summary, evaluation, OUTPUT_DIR)
    print(f" {report_path}")
    print()

    print("=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
