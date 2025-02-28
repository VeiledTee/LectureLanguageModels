import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import re
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


def extract_question_number(filename):
    """Extracts question number from filename using pattern matching"""
    match = re.search(r"q(\d+)", filename, re.IGNORECASE)
    return f"Q{match.group(1)}" if match else None


def correlation_analysis(csv_path):
    # Load data and preprocess
    df = pd.read_csv(csv_path)

    # Extract question number from filename
    df["question"] = df["filename"].apply(extract_question_number)

    # Handle files without question numbers
    if df["question"].isnull().any():
        print("Warning: Some files don't contain question numbers in filename")
        df = df.dropna(subset=["question"])

    # Metrics to analyze
    metrics = [
        "bleu",
        "rouge1",
        "rougeL",
        "token_f1",
        "bert_f1",
        "jaccard",
        "quiz_score",
    ]

    # Create output directory for plots
    output_dir = Path(csv_path).parent / "correlation_analysis"
    output_dir.mkdir(exist_ok=True)

    # Perform analysis per question
    for question, group in df.groupby("question"):
        print(f"\n=== Correlation Analysis for {question} ===")

        # Calculate correlation matrix
        corr_matrix = group[metrics].corr()

        # Print numerical results
        print(f"\nCorrelation Matrix for {question}:")
        print(corr_matrix.round(2))

        # Visualize correlations
        plt.figure(figsize=(12, 8))
        sns.heatmap(
            corr_matrix,
            annot=True,
            cmap="coolwarm",
            vmin=-1,
            vmax=1,
            fmt=".2f",
            linewidths=0.5,
        )
        plt.title(f"Metric Correlations - {question}")
        plt.tight_layout()

        # Save visualization
        plot_path = output_dir / f"{question}_correlations.png"
        plt.savefig(plot_path)
        plt.close()

        print(f"Saved visualization to {plot_path}")

    # Add cross-question summary
    print("\n=== Cross-Question Summary ===")
    summary = df.groupby("question")[metrics].mean().T
    print("\nAverage Metrics per Question:")
    print(summary.round(2))


if __name__ == "__main__":
    # Configure paths
    answer_dir = Path("AI_Course/Exams/generated_answers")  # Update with your path
    path_to_stats_csv = answer_dir / "evaluation_results.csv"

    if not path_to_stats_csv.exists():
        raise FileNotFoundError(f"CSV file not found at {path_to_stats_csv}")

    correlation_analysis(path_to_stats_csv)
    print("\nCorrelation analysis complete!")
