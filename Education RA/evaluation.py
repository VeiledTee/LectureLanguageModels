import re
import time
from collections import Counter
from pathlib import Path
from typing import Any

from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge import Rouge
from bert_score import score

LOCAL_EXAM_DIR = Path("/home/penguins/Documents/PhD/LectureLanguageModels/Education RA/AI_Course/Exams")


def extract_answers_from_markdown(md_file: Path) -> list[str]:
    """
    Extracts all answers from a markdown file that begin with the marker "//// ANSWER:".

    Each answer is assumed to start on a new line with "//// ANSWER:" and continue until
    the next occurrence of that marker or the end of the file.

    Args:
        md_file (Path): The path to the markdown file.

    Returns:
        List[str]: A list of answer strings extracted from the file.
    """
    content = md_file.read_text(encoding="utf-8")
    # Regex pattern to capture text after "//// ANSWER:" until the next marker or end-of-file.
    pattern = re.compile(r"^//// ANSWER:(.*?)(?=^//// ANSWER:|\Z)", re.DOTALL | re.MULTILINE)
    answers = pattern.findall(content)
    return [ans.strip() for ans in answers]


def extract_gold_answers(gold_file: Path) -> list[str]:
    """
    Extract gold answers from a solution file.

    The function uses a regular expression to detect the beginning of new answers
    (e.g., lines starting with "- 1." or "- 2.") and collects subsequent lines until
    a new question or section header (e.g., lines starting with "##") is encountered.

    Args:
        gold_file (Path): The path to the file containing the gold standard answers.

    Returns:
        list[str]: A list of extracted gold answer strings.
    """
    with gold_file.open("r", encoding="utf-8") as f:
        content = f.read()

    answers: list[str] = []
    current_answer: list[str] = []
    in_answer: bool = False

    # Regex to detect question lines (e.g., "- 1.", "- 2.")
    question_pattern = re.compile(r"^-\s*\d+\.")

    for line in content.split("\n"):
        line = line.strip()
        if question_pattern.match(line):
            # Save the previous answer if one is in progress
            if in_answer and current_answer:
                answers.append("\n".join(current_answer).strip())
                current_answer = []
            in_answer = True
        elif in_answer:
            # Capture lines until a section header is encountered
            if line.startswith("##"):
                in_answer = False
                if current_answer:
                    answers.append("\n".join(current_answer).strip())
                    current_answer = []
            else:
                current_answer.append(line)

    # Append the final answer if still in an answer block
    if in_answer and current_answer:
        answers.append("\n".join(current_answer).strip())

    return answers


def compute_token_f1(candidate: str, gold: str) -> float:
    """
    Compute the token-level F1 score between a candidate answer and a gold answer.

    The F1 score is calculated as the harmonic mean of token-level precision and recall.

    Args:
        candidate (str): The candidate answer text.
        gold (str): The gold standard answer text.

    Returns:
        float: The token-level F1 score.
    """
    candidate_counts = Counter(candidate.split())
    gold_counts = Counter(gold.split())

    common = sum(min(candidate_counts[token], gold_counts.get(token, 0)) for token in candidate_counts)
    total_candidate = sum(candidate_counts.values())
    total_gold = sum(gold_counts.values())

    if total_candidate == 0 or total_gold == 0:
        return 0.0

    precision = common / total_candidate
    recall = common / total_gold

    if (precision + recall) == 0:
        return 0.0

    return 2 * (precision * recall) / (precision + recall)


def evaluate_answers(generated_answers: list[str], gold_answers: list[str]) -> dict[str, float]:
    """
    Compare generated answers with gold answers and compute evaluation metrics.

    The function computes BLEU, ROUGE, token-level F1, and BERTScore metrics for each pair of
    generated and gold answers, then aggregates the results by averaging each metric.

    Args:
        generated_answers (list[str]): A list of generated answer strings.
        gold_answers (list[str]): A list of gold standard answer strings.

    Returns:
        dict[str, float]: A dictionary containing the averaged evaluation metrics.
    """
    smoothing = SmoothingFunction().method1
    rouge_evaluator = Rouge()
    results: list[dict[str, float]] = []

    if len(generated_answers) != len(gold_answers):
        raise ValueError(f"Generated answers and gold answers count mismatch! {len(generated_answers)} vs {len(gold_answers)}")

    for gen, gold in zip(generated_answers, gold_answers):
        gen_tokens = gen.split()
        gold_tokens = gold.split()
        bleu = sentence_bleu([gold_tokens], gen_tokens, smoothing_function=smoothing)

        # Compute ROUGE scores
        try:
            rouge_scores = rouge_evaluator.get_scores(gen, gold)[0]
        except Exception:
            rouge_scores = {"rouge-1": {"f": 0.0}, "rouge-l": {"f": 0.0}}

        token_f1 = compute_token_f1(gen, gold)

        # Compute BERTScore
        P, R, F1 = score([gen], [gold], lang="en")

        results.append({
            "bleu": bleu,
            "rouge1": rouge_scores.get("rouge-1", {}).get("f", 0.0),
            "rougeL": rouge_scores.get("rouge-l", {}).get("f", 0.0),
            "token_f1": token_f1,
            "bert_f1": F1[0].item(),
        })

    # Aggregate results by averaging each metric
    avg_results = {k: sum(r[k] for r in results) / len(results) for k in results[0]}
    return avg_results


if __name__ == '__main__':
    answer_directory = Path("AI_Course/Exams/generated_answers")
    gold_answers: list[str] = extract_answers_from_markdown(Path("/home/penguins/Documents/PhD/LectureLanguageModels/Education RA/AI_Course/Exams/q1_soln_parsed.txt"))
    for file in answer_directory.glob("*.txt"):
        print(f"Processing file: {file.name}")
        model: str = file.name.split('_')[0]
        generated_answers: list[str] = extract_answers_from_markdown(file)

        # Evaluate and print aggregated metrics
        metrics: dict[str, float] = evaluate_answers(generated_answers, gold_answers)

        print(f"\n=== Final Metrics for {model} ===")
        print(f"BLEU: {metrics['bleu']:.4f}")
        print(f"ROUGE-1: {metrics['rouge1']:.4f}")
        print(f"ROUGE-L: {metrics['rougeL']:.4f}")
        print(f"Token F1: {metrics['token_f1']:.4f}")
        print(f"BERTScore F1: {metrics['bert_f1']:.4f}")
