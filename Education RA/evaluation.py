import re
from collections import Counter
from pathlib import Path
import torch
from bert_score import score
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from rich.console import Console
from rich.table import Table
from rouge import Rouge
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import logging as transformers_logging
transformers_logging.set_verbosity_error()

# Console to print question metrics
console = Console()

# Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize the NLI model and tokenizer
nli_model_name = "facebook/bart-large-mnli"
nli_tokenizer = AutoTokenizer.from_pretrained(nli_model_name)
nli_model = AutoModelForSequenceClassification.from_pretrained(nli_model_name).to(device)  # Move model to GPU

def print_metrics(question_number: int, metrics: dict):
    table = Table(title=f"Question {question_number} Metrics")
    table.add_column("Metric", style="cyan", no_wrap=True)
    table.add_column("Score", style="magenta", justify="right")

    table.add_row("BLEU", f"{metrics['bleu']:.4f}")
    table.add_row("ROUGE-1", f"{metrics['rouge1']:.4f}")
    table.add_row("ROUGE-L", f"{metrics['rougeL']:.4f}")
    table.add_row("Token F1", f"{metrics['token_f1']:.4f}")
    table.add_row("BERTScore F1", f"{metrics['bert_f1']:.4f}")
    table.add_row("NLI Entailment", f"{metrics['nli_entailment']:.4f}")
    table.add_row("Exact Match", f"{metrics['exact_match']:.4f}")
    table.add_row("Jaccard", f"{metrics['jaccard']:.4f}")

    console.print(table)

def extract_answers_from_markdown(md_file: Path) -> list[str]:
    content = md_file.read_text(encoding="utf-8")
    pattern = re.compile(
        r"^//// ANSWER:(.*?)(?=^//// ANSWER:|^#|\Z)", re.DOTALL | re.MULTILINE
    )
    answers = pattern.findall(content)
    print(len(answers))
    return [ans.strip() for ans in answers]

def extract_gold_answers(gold_file: Path) -> list[str]:
    with gold_file.open("r", encoding="utf-8") as f:
        content = f.read()

    answers: list[str] = []
    current_answer: list[str] = []
    in_answer: bool = False
    question_pattern = re.compile(r"^-\s*\d+\.")

    for line in content.split("\n"):
        line = line.strip()
        if question_pattern.match(line):
            if in_answer and current_answer:
                answers.append("\n".join(current_answer).strip())
                current_answer = []
            in_answer = True
        elif in_answer:
            if line.startswith("##"):
                in_answer = False
                if current_answer:
                    answers.append("\n".join(current_answer).strip())
                    current_answer = []
            else:
                current_answer.append(line)

    if in_answer and current_answer:
        answers.append("\n".join(current_answer).strip())

    return answers

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

def exact_match(candidate: str, gold: str) -> float:
    return 1.0 if candidate.strip() == gold.strip() else 0.0

def jaccard_similarity(candidate: str, gold: str) -> float:
    candidate_tokens = set(candidate.split())
    gold_tokens = set(gold.split())
    if not candidate_tokens and not gold_tokens:
        return 0.0
    intersection = candidate_tokens.intersection(gold_tokens)
    union = candidate_tokens.union(gold_tokens)
    return len(intersection) / len(union)

def nli_score(premise: str, hypothesis: str) -> float:
    inputs = nli_tokenizer.encode_plus(
        premise, hypothesis, return_tensors="pt", truncation=True
    ).to(device)  # Move input tensors to GPU
    logits = nli_model(**inputs).logits
    probs = torch.softmax(logits, dim=1).detach().cpu().numpy()[0]
    return float(probs[2])

def evaluate_answers(
    answers_to_evaluate: list[str],
    gold_standard_answers: list[str],
    verbose: bool = False,
) -> dict[str, float]:
    smoothing = SmoothingFunction().method1
    rouge_evaluator = Rouge()
    results: list[dict[str, float]] = []

    if len(answers_to_evaluate) != len(gold_standard_answers):
        raise ValueError(
            f"Generated answers and gold answers count mismatch! {len(answers_to_evaluate)} vs {len(gold_standard_answers)}"
        )

    question_count: int = 0
    for gen, gold in zip(answers_to_evaluate, gold_standard_answers):
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
        nli_entailment = nli_score(gold, gen)
        em = exact_match(gen, gold)
        jaccard = jaccard_similarity(gen, gold)

        results.append(
            {
                "bleu": bleu,
                "rouge1": rouge_scores.get("rouge-1", {}).get("f", 0.0),
                "rougeL": rouge_scores.get("rouge-l", {}).get("f", 0.0),
                "token_f1": token_f1,
                "bert_f1": bert_f1,
                "nli_entailment": nli_entailment,
                "exact_match": em,
                "jaccard": jaccard,
            }
        )

        if verbose:
            print_metrics(question_count + 1, results[-1])
        question_count += 1

    avg_results = {k: sum(r[k] for r in results) / len(results) for k in results[0]}
    return avg_results

if __name__ == "__main__":
    answer_directory = Path("AI_Course/Exams/generated_answers")
    question_dirs = ["q1", "q2"]  # List of question directories to process

    for question in question_dirs:
        # Load gold standard answers for the current question
        gold_path = Path(
            f"/home/penguins/Documents/PhD/LectureLanguageModels/Education RA/AI_Course/Exams/{question}_soln_parsed.txt"
        )
        gold_answers: list[str] = extract_answers_from_markdown(gold_path)

        # Process generated answers for the current question
        question_dir = answer_directory / question
        for file in question_dir.glob("*.txt"):
            print(f"\nProcessing file: {file.name} (Question: {question})")
            model = file.name.split("_")[0]
            generated_answers: list[str] = extract_answers_from_markdown(file)

            metrics: dict[str, float] = evaluate_answers(
                answers_to_evaluate=generated_answers,
                gold_standard_answers=gold_answers,
                verbose=False,
            )

            print(f"\n=== Final Metrics for {model} (Question: {question}) ===")
            print(f"BLEU: {metrics['bleu']:.4f}")
            print(f"ROUGE-1: {metrics['rouge1']:.4f}")
            print(f"ROUGE-L: {metrics['rougeL']:.4f}")
            print(f"Token F1: {metrics['token_f1']:.4f}")
            print(f"BERTScore F1: {metrics['bert_f1']:.4f}")
            print(f"NLI Entailment: {metrics['nli_entailment']:.4f}")
            print(f"Exact Match: {metrics['exact_match']:.4f}")
            print(f"Jaccard: {metrics['jaccard']:.4f}")
