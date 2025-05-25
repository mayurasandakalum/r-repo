import json
import numpy as np
from typing import Dict, List
from statistics import mean, stdev
from dataclasses import dataclass
from scipy.stats import shapiro
from collections import Counter


@dataclass
class StudentScore:
    id: str
    name: str = None  # Add name field with None default
    visual: float = 0.0
    auditory: float = 0.0
    reading: float = 0.0
    kinesthetic: float = 0.0


class VARKClassifier:
    def __init__(
        self,
        z_score_threshold: float = 0.5,
        percentile_threshold: float = 70.0,
        alpha: float = 0.05,
        max_iterations: int = 3,
        balance_tolerance: float = 0.10,
    ):
        """
        :param z_score_threshold: Initial z-score multiplier for normal distributions.
        :param percentile_threshold: Initial percentile threshold for non-normal distributions.
        :param alpha: Significance level for Shapiro-Wilk normality test.
        :param max_iterations: Max number of times to re-adjust thresholds if distribution is unbalanced.
        :param balance_tolerance: Tolerance for 'Undefined' or single categories; used to tweak thresholds.
        """
        self.z_score_threshold = z_score_threshold
        self.percentile_threshold = percentile_threshold
        self.alpha = alpha
        self.max_iterations = max_iterations
        self.balance_tolerance = balance_tolerance

    def process_data(self, input_file: str, output_file: str):
        """
        Main pipeline:
          1. Load scores from JSON.
          2. Calculate initial thresholds (z-score or percentile).
          3. Optionally fine-tune thresholds if distribution is unbalanced.
          4. Classify each student.
          5. Save results to JSON.
        """
        with open(input_file, "r") as f:
            data = json.load(f)

        students = [StudentScore(**student) for student in data["students"]]
        thresholds = self.calculate_thresholds(students)
        thresholds = self.fine_tune_thresholds(students, thresholds)
        classifications = [
            self.classify_student(student, thresholds) for student in students
        ]

        # Create visualizations
        from visualization import (
            plot_learning_style_distribution,
            plot_score_distributions,
            plot_threshold_comparison,
            plot_heatmap_correlation,
            plot_style_radar,
            plot_style_distribution_pie,
            create_summary_report,
            plot_modality_distributions,
            create_statistical_analysis,
        )

        # Generate all visualizations
        plot_learning_style_distribution(classifications, "learning_styles_bar.png")
        plot_score_distributions(classifications, "score_distributions_box.png")
        plot_threshold_comparison(
            classifications, thresholds, "threshold_comparison_violin.png"
        )
        plot_heatmap_correlation(classifications, "modality_correlations.png")
        plot_style_radar(classifications, "learning_style_radar.png")
        plot_style_distribution_pie(classifications, "learning_style_pie.png")
        plot_modality_distributions(classifications, "modality_distributions.png")
        create_summary_report(classifications, thresholds, "analysis_summary.txt")
        create_statistical_analysis(
            classifications, thresholds, "statistical_analysis.txt"
        )

        results = {
            "thresholds": thresholds,
            "classifications": classifications,
        }

        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)

    def calculate_thresholds(self, scores: List[StudentScore]) -> Dict[str, float]:
        """
        Computes a threshold for each modality, either:
         - mean + z*std (if normal by Shapiro-Wilk), or
         - percentile (if not normal or stdev is 0).
        """
        all_scores = {
            "V": [s.visual for s in scores],
            "A": [s.auditory for s in scores],
            "R": [s.reading for s in scores],
            "K": [s.kinesthetic for s in scores],
        }

        thresholds = {}
        for modality, values in all_scores.items():
            # Handle edge case: zero or near-zero variance or all zeros
            if all(v == 0 for v in values):
                # All scores are zero - set threshold to -1 to ensure no one passes
                thresholds[modality] = -1
                continue
                
            std_val = stdev(values) if len(values) > 1 else 0.0
            if std_val < 1e-9:
                # If stdev is effectively 0 but values aren't all 0, use a slightly lower value
                avg = mean(values)
                thresholds[modality] = avg * 0.95 if avg > 0 else -1
                continue

            # Use Shapiro-Wilk for normality check
            # If p-value >= alpha => cannot reject normality => treat as normal
            stat, p_value = shapiro(values)
            is_normal = p_value >= self.alpha

            if is_normal:
                # normal distribution => use mean + (z_score_threshold * stdev)
                thresholds[modality] = mean(values) + (self.z_score_threshold * std_val)
            else:
                # not normal => use percentile
                thresholds[modality] = float(
                    np.percentile(values, self.percentile_threshold)
                )

        return thresholds

    def classify_student(
        self, scores: StudentScore, thresholds: Dict[str, float]
    ) -> Dict:
        """
        Compares each score to the threshold and appends the modality label if >= threshold.
        Sorts the modalities to ensure consistent naming for multimodal learners.
        """
        preferences = []
        scores_dict = {
            "V": scores.visual,
            "A": scores.auditory,
            "R": scores.reading,
            "K": scores.kinesthetic,
        }

        for modality, score in scores_dict.items():
            if score >= thresholds[modality]:
                preferences.append(modality)

        learning_style = "".join(sorted(preferences)) if preferences else "Undefined"

        return {
            "student_id": scores.id,
            "student_name": scores.name,  # Include name in the output
            "scores": scores_dict,
            "learning_style": learning_style,
        }

    def fine_tune_thresholds(
        self, students: List[StudentScore], thresholds: Dict[str, float]
    ) -> Dict[str, float]:
        """
        A simple iterative approach to adjust thresholds if the classification
        distribution is too skewed or too many "Undefined" cases appear.

        *This is a demo-level approach*. In practice, you'd define your own rules
        for what "skewed" means and how to adjust thresholds.
        """
        current_thresholds = thresholds.copy()

        for _ in range(self.max_iterations):
            # Classify students with current thresholds
            classifications = [
                self.classify_student(s, current_thresholds) for s in students
            ]
            styles = [c["learning_style"] for c in classifications]

            # Count how many times each style occurs
            style_counter = Counter(styles)
            total_students = len(students)

            # If "Undefined" is too large a fraction => we might be too strict
            undefined_fraction = style_counter["Undefined"] / total_students
            if undefined_fraction > self.balance_tolerance:
                # Lower thresholds: reduce z-score threshold & percentile threshold
                self.z_score_threshold -= 0.1
                self.percentile_threshold -= 2.0
                self.z_score_threshold = max(
                    self.z_score_threshold, 0.0
                )  # don't go below 0
                self.percentile_threshold = max(self.percentile_threshold, 0.0)
                current_thresholds = self.calculate_thresholds(students)
                continue

            # Check if any single category is too high (e.g., > 60%)
            # or too low (e.g., < 5%)
            # You can define your own criteria for "skewed."
            distribution_issues = False
            for style, count in style_counter.items():
                frac = count / total_students
                if style != "Undefined" and (frac > 0.60 or frac < 0.05):
                    distribution_issues = True
                    break

            if distribution_issues:
                # Nudging thresholds up or down to reduce skew
                # (This is just an example heuristic.)
                self.z_score_threshold += 0.1
                self.percentile_threshold += 2.0
                current_thresholds = self.calculate_thresholds(students)
            else:
                # Distribution seems reasonable; stop iterating
                break

        return current_thresholds
