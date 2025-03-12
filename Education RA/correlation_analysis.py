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


def is_rag(filename):
    """Determines if the file is a RAG version"""
    return "_rag_" in filename.lower()


def correlation_analysis(csv_path):
    # Load data and preprocess
    df = pd.read_csv(csv_path)

    # Extract metadata from filename
    df["question"] = df["filename"].apply(extract_question_number)
    df["is_rag"] = df["filename"].apply(is_rag)

    # Handle files without question numbers
    if df["question"].isnull().any():
        print(f"Warning: Some files in {csv_path} don't contain question numbers")
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

    # Create output directory
    csv_path = Path(csv_path)
    output_dir = csv_path.parent / f"{csv_path.stem}_correlation_analysis"
    output_dir.mkdir(exist_ok=True)

    # Perform analysis per question and RAG status
    for (question, rag_flag), group in df.groupby(["question", "is_rag"]):
        analysis_type = "RAG" if rag_flag else "non-RAG"
        print(f"\n=== {question} ({analysis_type}) ===")

        # Calculate correlation matrix
        corr_matrix = group[metrics].corr()

        # Print numerical results
        print(f"\nCorrelation Matrix for {question} ({analysis_type}):")
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
        plt.title(f"Metric Correlations - {question} ({analysis_type})")
        plt.tight_layout()

        # Save visualization
        plot_path = output_dir / f"{question}_{analysis_type}_correlations.png"
        plt.savefig(plot_path)
        plt.close()

        print(f"Saved visualization to {plot_path}")

    # Add cross-question summary with RAG distinction
    print(f"\n=== Cross-Question Summary for {csv_path.name} ===")
    summary = df.groupby(["question", "is_rag"])[metrics].mean().unstack()
    print("\nAverage Metrics per Question:")
    print(summary.round(2).T.to_string())


if __name__ == "__main__":
    answer_dir = Path("Artificial_Intelligence/Exams/generated_answers")
    csv_files = list(answer_dir.glob("*.csv"))

    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {answer_dir}")

    for csv_file in csv_files:
        print(f"\n{'=' * 40}\nProcessing file: {csv_file.name}\n{'=' * 40}")
        correlation_analysis(csv_file)

    print("\nCorrelation analysis complete for all CSVs!")
