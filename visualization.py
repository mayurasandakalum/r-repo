import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
from typing import List, Dict
import numpy as np
from scipy import stats

# Create visualizations directory if it doesn't exist
os.makedirs("visualizations", exist_ok=True)


def plot_learning_style_distribution(classifications: List[Dict], output_path: str):
    """Creates a bar plot showing the distribution of learning styles."""
    styles = [c["learning_style"] for c in classifications]
    style_counts = pd.Series(styles).value_counts()

    plt.figure(figsize=(10, 6))
    sns.barplot(x=style_counts.index, y=style_counts.values)
    plt.title("Distribution of Learning Styles")
    plt.xlabel("Learning Style")
    plt.ylabel("Number of Students")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join("visualizations", output_path))
    plt.close()


def plot_score_distributions(classifications: List[Dict], output_path: str):
    """Creates box plots for score distributions across modalities."""
    scores_data = []
    for c in classifications:
        for modality, score in c["scores"].items():
            scores_data.append({"Modality": modality, "Score": score})

    df = pd.DataFrame(scores_data)

    plt.figure(figsize=(8, 6))
    sns.boxplot(x="Modality", y="Score", data=df)
    plt.title("Score Distributions by Modality")
    plt.tight_layout()
    plt.savefig(os.path.join("visualizations", output_path))
    plt.close()


def plot_threshold_comparison(
    classifications: List[Dict], thresholds: Dict[str, float], output_path: str
):
    """Creates a violin plot with thresholds marked."""
    scores_data = []
    for c in classifications:
        for modality, score in c["scores"].items():
            scores_data.append({"Modality": modality, "Score": score})

    df = pd.DataFrame(scores_data)

    plt.figure(figsize=(10, 6))
    sns.violinplot(x="Modality", y="Score", data=df)

    # Add threshold markers
    for i, modality in enumerate(["V", "A", "R", "K"]):
        plt.hlines(
            y=thresholds[modality],
            xmin=i - 0.2,
            xmax=i + 0.2,
            colors="red",
            linestyles="dashed",
            label="Threshold" if i == 0 else "",
        )

    plt.title("Score Distributions with Classification Thresholds")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join("visualizations", output_path))
    plt.close()


def plot_heatmap_correlation(classifications: List[Dict], output_path: str):
    """Creates a heatmap showing correlations between different modalities."""
    scores_df = pd.DataFrame([c["scores"] for c in classifications])
    correlation_matrix = scores_df.corr()

    plt.figure(figsize=(8, 6))
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", center=0)
    plt.title("Correlation between Learning Modalities")
    plt.tight_layout()
    plt.savefig(os.path.join("visualizations", output_path))
    plt.close()


def plot_style_radar(classifications: List[Dict], output_path: str):
    """Creates a radar chart showing average scores for each learning style."""
    style_scores = {}
    for c in classifications:
        style = c["learning_style"]
        if style not in style_scores:
            style_scores[style] = {
                "count": 0,
                "scores": {"V": 0, "A": 0, "R": 0, "K": 0},
            }
        style_scores[style]["count"] += 1
        for modality, score in c["scores"].items():
            style_scores[style]["scores"][modality] += score

    # Calculate averages
    for style in style_scores:
        for modality in style_scores[style]["scores"]:
            style_scores[style]["scores"][modality] /= style_scores[style]["count"]

    # Create radar chart
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection="polar"))
    angles = np.linspace(0, 2 * np.pi, 4, endpoint=False)
    angles = np.concatenate((angles, [angles[0]]))  # complete the circle

    for style, data in style_scores.items():
        if style == "Undefined":
            continue
        values = [data["scores"][m] for m in ["V", "A", "R", "K"]]
        values = np.concatenate((values, [values[0]]))
        ax.plot(angles, values, "-o", linewidth=2, label=style)
        ax.fill(angles, values, alpha=0.25)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(["Visual", "Auditory", "Reading", "Kinesthetic"])
    plt.title("Average Scores by Learning Style")
    plt.legend(bbox_to_anchor=(0.95, 0.95))
    plt.tight_layout()
    plt.savefig(os.path.join("visualizations", output_path))
    plt.close()


def plot_style_distribution_pie(classifications: List[Dict], output_path: str):
    """Creates a pie chart showing the distribution of learning styles."""
    styles = [c["learning_style"] for c in classifications]
    style_counts = pd.Series(styles).value_counts()

    plt.figure(figsize=(10, 8))
    plt.pie(style_counts.values, labels=style_counts.index, autopct="%1.1f%%")
    plt.title("Learning Style Distribution")
    plt.axis("equal")
    plt.tight_layout()
    plt.savefig(os.path.join("visualizations", output_path))
    plt.close()


def create_summary_report(
    classifications: List[Dict], thresholds: Dict[str, float], output_path: str
):
    """Creates a text file with summary statistics."""
    total_students = len(classifications)
    styles = [c["learning_style"] for c in classifications]
    style_counts = pd.Series(styles).value_counts()

    with open(os.path.join("visualizations", output_path), "w") as f:
        f.write("VARK Analysis Summary Report\n")
        f.write("==========================\n\n")

        f.write(f"Total Students: {total_students}\n\n")

        f.write("Learning Style Distribution:\n")
        for style, count in style_counts.items():
            percentage = (count / total_students) * 100
            f.write(f"{style}: {count} ({percentage:.1f}%)\n")

        f.write("\nThresholds Used:\n")
        for modality, threshold in thresholds.items():
            f.write(f"{modality}: {threshold:.2f}\n")


def plot_modality_distributions(classifications: List[Dict], output_path: str):
    """Creates a distribution plot comparing all modalities."""
    scores_data = []
    for c in classifications:
        for modality, score in c["scores"].items():
            scores_data.append({"Modality": modality, "Score": score})

    df = pd.DataFrame(scores_data)

    plt.figure(figsize=(12, 6))
    for modality in ["V", "A", "R", "K"]:
        sns.kdeplot(
            data=df[df["Modality"] == modality]["Score"],
            label=modality,
            fill=True,
            alpha=0.3,
        )

    plt.title("Distribution Comparison of Learning Modalities")
    plt.xlabel("Score")
    plt.ylabel("Density")
    plt.legend(title="Modality")
    plt.tight_layout()
    plt.savefig(os.path.join("visualizations", output_path))
    plt.close()


def create_statistical_analysis(
    classifications: List[Dict], thresholds: Dict[str, float], output_path: str
):
    """Creates a detailed statistical analysis report."""
    scores_df = pd.DataFrame([c["scores"] for c in classifications])

    with open(os.path.join("visualizations", output_path), "w") as f:
        f.write("Statistical Analysis Report\n")
        f.write("=========================\n\n")

        # Basic statistics
        f.write("1. Basic Statistics\n")
        f.write("-----------------\n")
        for modality in ["V", "A", "R", "K"]:
            stats_dict = scores_df[modality].describe()
            f.write(f"\n{modality} Modality:\n")
            f.write(f"Mean: {stats_dict['mean']:.2f}\n")
            f.write(f"Median: {scores_df[modality].median():.2f}\n")
            f.write(f"Std Dev: {stats_dict['std']:.2f}\n")
            f.write(f"Min: {stats_dict['min']:.2f}\n")
            f.write(f"Max: {stats_dict['max']:.2f}\n")

        # Normality tests
        f.write("\n2. Normality Tests (Shapiro-Wilk)\n")
        f.write("--------------------------------\n")
        for modality in ["V", "A", "R", "K"]:
            stat, p_value = stats.shapiro(scores_df[modality])
            f.write(f"\n{modality} Modality:\n")
            f.write(f"Statistic: {stat:.4f}\n")
            f.write(f"P-value: {p_value:.4f}\n")
            f.write(f"Normal distribution? {p_value >= 0.05}\n")

        # Correlations
        f.write("\n3. Correlation Analysis\n")
        f.write("----------------------\n")
        corr_matrix = scores_df.corr()
        for i in ["V", "A", "R", "K"]:
            for j in ["V", "A", "R", "K"]:
                if i < j:  # avoid duplicates
                    correlation = corr_matrix.loc[i, j]
                    f.write(f"\n{i}-{j} Correlation: {correlation:.4f}\n")

        # Threshold Analysis
        f.write("\n4. Threshold Analysis\n")
        f.write("-------------------\n")
        for modality in ["V", "A", "R", "K"]:
            scores = scores_df[modality]
            threshold = thresholds[modality]
            percent_above = (scores >= threshold).mean() * 100
            f.write(f"\n{modality} Modality:\n")
            f.write(f"Threshold: {threshold:.2f}\n")
            f.write(f"% Above Threshold: {percent_above:.1f}%\n")
            f.write(
                f"Z-score of threshold: {(threshold - scores.mean()) / scores.std():.2f}\n"
            )
