import os
import re
import csv
from collections import Counter
from pathlib import Path
from tqdm import tqdm

import ollama
import torch
from bert_score import score
from dotenv import load_dotenv
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from rich.console import Console
from rich.table import Table
from rouge import Rouge
from transformers import logging as transformers_logging

load_dotenv()

transformers_logging.set_verbosity_error()
console = Console()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def print_metrics(question_number: int, evaluation_metrics: dict):
    table = Table(title=f"Question {question_number} Metrics")
    table.add_column("Metric", style="cyan", no_wrap=True)
    table.add_column("Score", style="magenta", justify="right")

    table.add_row("BLEU", f"{evaluation_metrics['bleu']:.4f}")
    table.add_row("ROUGE-1", f"{evaluation_metrics['rouge1']:.4f}")
    table.add_row("ROUGE-L", f"{evaluation_metrics['rougeL']:.4f}")
    table.add_row("Token F1", f"{evaluation_metrics['token_f1']:.4f}")
    table.add_row("BERTScore F1", f"{evaluation_metrics['bert_f1']:.4f}")
    table.add_row("Jaccard", f"{evaluation_metrics['jaccard']:.4f}")
    table.add_row("Rubric Score", evaluation_metrics["rubric_score"])

    console.print(table)


def extract_answers(md_file: Path) -> list[str]:
    content = md_file.read_text(encoding="utf-8")
    pattern = re.compile(
        r"^//// ANSWER:(.*?)(?=^//// ANSWER:|^#|\Z)", re.DOTALL | re.MULTILINE
    )
    answers = pattern.findall(content)
    return [ans.strip() for ans in answers]


def compute_token_f1(candidate: str, gold: str) -> float:
    candidate_counts = Counter(candidate.split())
    gold_counts = Counter(gold.split())
    common = sum(
        min(candidate_counts[token], gold_counts.get(token, 0))
        for token in candidate_counts
    )
    total_candidate = sum(candidate_counts.values())
    total_gold = sum(gold_counts.values())
    if total_candidate == 0 or total_gold == 0:
        return 0.0
    precision = common / total_candidate
    recall = common / total_gold
    if (precision + recall) == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)


def jaccard_similarity(candidate: str, gold: str) -> float:
    candidate_tokens = set(candidate.split())
    gold_tokens = set(gold.split())
    if not candidate_tokens and not gold_tokens:
        return 0.0
    intersection = candidate_tokens.intersection(gold_tokens)
    union = candidate_tokens.union(gold_tokens)
    return len(intersection) / len(union)


def extract_rubric(rubric_file: Path, question_number: int) -> str:
    content = rubric_file.read_text(encoding="utf-8")
    rubrics = re.split(r"^//// RUBRIC:", content, flags=re.MULTILINE)[1:]
    try:
        return rubrics[question_number - 1].strip()
    except IndexError:
        raise ValueError(f"No rubric found for question {question_number}")


def evaluate_with_rubric(
    rubric_text: str, query: str, student_answer: str
) -> tuple[float, float]:
    total_match = re.search(r"Total Points:\s*(\d+)", rubric_text, re.IGNORECASE)
    if not total_match:
        return 0.0, 0.0
    total_available = float(total_match.group(1))

    prompt = f"""Evaluate this answer based on the following rubric. Return only the numerical score awarded as a number.

Rubric:
{rubric_text}

Question:
{query}

Student Answer:
{student_answer}

Provide your evaluation score:"""

    response = ollama.generate(
        model=str(os.getenv("EVALUATION_MODEL")),
        prompt=prompt,
        options={
            "temperature": 0.0,
            "max_tokens": 50,
            "top_p": 0.9,
            "stop": ["</s>", "\n\n\n"],
        },
    )

    answer = response.get("response", "").strip()
    try:
        awarded = float(answer.split()[0])
    except (ValueError, IndexError):
        awarded = 0.0

    return awarded, total_available


def evaluate_answers(
    answers_to_evaluate: list[str],
    gold_standard_answers: list[str],
    rubric_file: Path,
    verbose: bool = False,
) -> dict[str, float]:
    smoothing = SmoothingFunction().method1
    rouge_evaluator = Rouge()
    results: list[dict[str, float]] = []
    total_awarded = 0.0
    total_possible = 0.0

    if len(answers_to_evaluate) != len(gold_standard_answers):
        raise ValueError(
            f"Generated answers and gold answers count mismatch! {len(answers_to_evaluate)} vs {len(gold_standard_answers)}"
        )

    for idx, (gen, gold) in enumerate(tqdm(
        zip(answers_to_evaluate, gold_standard_answers),
        total=len(answers_to_evaluate),
        desc="Evaluating questions"
    )):
        question_num = idx + 1
        try:
            rubric_text = extract_rubric(rubric_file, question_num)
            awarded, possible = evaluate_with_rubric(
                rubric_text, f"Question {question_num}", gen
            )
        except Exception as e:
            print(f"Error evaluating rubric for Q{question_num}: {str(e)}")
            awarded, possible = 0.0, 0.0

        total_awarded += awarded
        total_possible += possible

        gen_tokens = gen.split()
        gold_tokens = gold.split()
        bleu = sentence_bleu([gold_tokens], gen_tokens, smoothing_function=smoothing)

        try:
            rouge_scores = rouge_evaluator.get_scores(gen, gold)[0]
        except Exception:
            rouge_scores = {"rouge-1": {"f": 0.0}, "rouge-l": {"f": 0.0}}

        token_f1 = compute_token_f1(gen, gold)
        P, R, F1 = score([gen], [gold], lang="en")
        bert_f1 = F1[0].item()
        jaccard = jaccard_similarity(gen, gold)

        results.append({
            "bleu": bleu,
            "rouge1": rouge_scores.get("rouge-1", {}).get("f", 0.0),
            "rougeL": rouge_scores.get("rouge-l", {}).get("f", 0.0),
            "token_f1": token_f1,
            "bert_f1": bert_f1,
            "jaccard": jaccard,
            "rubric_score": f"{awarded:.1f}/{possible:.0f}",
        })

        if verbose:
            print_metrics(question_num, results[-1])

    avg_metrics = ["bleu", "rouge1", "rougeL", "token_f1", "bert_f1", "jaccard"]
    avg_results = {
        metric: sum(r[metric] for r in results) / len(results)
        for metric in avg_metrics
    }

    avg_results["total_awarded"] = total_awarded
    avg_results["total_possible"] = total_possible

    return avg_results


if __name__ == "__main__":
    ANSWER_DIR = Path(os.getenv("ANSWER_DIR", "AI_Course/Exams/generated_answers"))
    EXAM_DIR = Path(os.getenv("EXAM_DIR", "AI_Course/Exams"))
    RUBRIC_DIR = Path(os.getenv("RUBRIC_DIR", "AI_Course/Rubrics"))

    # Get all answer files
    answer_files = list(ANSWER_DIR.glob("*_answers.txt")) + list(ANSWER_DIR.glob("*_rag_answers.txt"))

    # CSV path and processed file check
    csv_path = ANSWER_DIR / "evaluation_results.csv"
    processed_files = set()
    csv_mode = "w"
    if csv_path.exists():
        with open(csv_path, "r", newline="", encoding="utf-8") as csvfile:
            reader = csv.DictReader(csvfile)
            processed_files = {row["filename"] for row in reader if 'deep' not in row["filename"]}
        csv_mode = "a"

    with open(csv_path, csv_mode, newline="", encoding="utf-8") as csvfile:
        fieldnames = [
            "filename", "bleu", "rouge1", "rougeL",
            "token_f1", "bert_f1", "jaccard",
            "total_awarded", "total_possible", "quiz_score"
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if csv_mode == "w":
            writer.writeheader()

        for answer_file in answer_files:
            if answer_file.name in processed_files:
                print(f"Skipping file: {answer_file.name} (already graded)")
                continue

            print(f"\nProcessing file: {answer_file.name}")
            filename_parts = answer_file.stem.split("_")
            exam_name = filename_parts[0]
            model_name = "_".join(filename_parts[1:-1])

            gold_path = EXAM_DIR / f"{exam_name}_soln_parsed.txt"
            rubric_path = RUBRIC_DIR / f"{exam_name}_rubric.txt"

            if not gold_path.exists():
                print(f"Gold file {gold_path} not found, skipping")
                continue

            if not rubric_path.exists():
                print(f"Rubric {rubric_path} not found, skipping")
                continue

            try:
                gold_answers = extract_answers(gold_path)
                generated_answers = extract_answers(answer_file)

                metrics = evaluate_answers(
                    answers_to_evaluate=generated_answers,
                    gold_standard_answers=gold_answers,
                    rubric_file=rubric_path,
                    verbose=False,
                )

                score_pct = round(
                    (metrics["total_awarded"] / metrics["total_possible"]) * 100, 2
                ) if metrics["total_possible"] > 0 else 0.0

                writer.writerow({
                    "filename": answer_file.name,
                    "bleu": round(metrics["bleu"], 4),
                    "rouge1": round(metrics["rouge1"], 4),
                    "rougeL": round(metrics["rougeL"], 4),
                    "token_f1": round(metrics["token_f1"], 4),
                    "bert_f1": round(metrics["bert_f1"], 4),
                    "jaccard": round(metrics["jaccard"], 4),
                    "total_awarded": round(metrics["total_awarded"], 1),
                    "total_possible": int(metrics["total_possible"]),
                    "quiz_score": score_pct
                })

                print(f"| Score: {score_pct}% | BERTScore: {metrics['bert_f1']:.4f} |")

            except Exception as e:
                print(f"Error processing {answer_file}: {str(e)}")
                continue

    print(f"\nEvaluation complete! Results saved to {csv_path}")
