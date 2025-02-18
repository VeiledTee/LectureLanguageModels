import json
import jsonlines
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from bert_score import score as bert_score
from sklearn.metrics import f1_score

def load_json(file_path):
    """Loads a JSON file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def load_jsonl(file_path):
    """Loads responses from a JSONL file."""
    responses = {}
    with jsonlines.open(file_path, 'r') as reader:
        for obj in reader:
            responses[obj["custom_id"]] = obj
    return responses

def extract_llm_responses(llm_jsonl):
    """Extracts responses from the LLM batch API output."""
    responses = {}
    for entry in llm_jsonl:
        request_id = entry.get('custom_id', '')
        response_text = entry.get('response', {}).get('body', {}).get('choices', [])[0].get('message', {}).get('content', '')
        responses[request_id] = response_text
    return responses

def calculate_bleu(reference, hypothesis):
    """Calculates BLEU score between reference and hypothesis."""
    smoothing = SmoothingFunction().method1
    return sentence_bleu([reference.split()], hypothesis.split(), smoothing_function=smoothing) * 100

def calculate_bertscore(reference, hypothesis):
    """Calculates BERTScore between reference and hypothesis."""
    P, R, F1 = bert_score([hypothesis], [reference], lang="en")
    return F1.item() * 100  # Return F1 score as percentage

def calculate_f1(reference, hypothesis):
    """Calculates token-level F1 score."""
    ref_tokens = set(reference.split())
    hyp_tokens = set(hypothesis.split())
    common_tokens = ref_tokens.intersection(hyp_tokens)
    if not ref_tokens or not hyp_tokens:
        return 0.0
    precision = len(common_tokens) / len(hyp_tokens)
    recall = len(common_tokens) / len(ref_tokens)
    if precision + recall == 0:
        return 0.0
    return (2 * precision * recall / (precision + recall)) * 100

def evaluate_responses(llm_responses, ground_truth):
    """Evaluates the LLM responses against ground truth using ROUGE-L, BLEU, BERTScore, and F1."""
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    scores = {"rougeL": {}, "bleu": {}, "bertscore": {}, "f1": {}}
    
    for i, item in enumerate(ground_truth):
        for j, sub in enumerate(item.get("subquestions", [])):
            sub_question = sub.get("question", "")
            gt_answer = sub.get("answer", "")
            llm_answer = llm_responses.get(sub["custom_id"], "").get('response', {}).get('body', {}).get('choices', [])[0].get('message', {}).get('content', '')
            
            # ROUGE-L
            rouge_score = scorer.score(gt_answer, llm_answer)['rougeL'].fmeasure * 100
            scores["rougeL"][sub["custom_id"]] = rouge_score
            
            # BLEU
            bleu_score = calculate_bleu(gt_answer, llm_answer)
            scores["bleu"][sub["custom_id"]] = bleu_score
            
            # BERTScore
            bert_score_value = calculate_bertscore(gt_answer, llm_answer)
            scores["bertscore"][sub["custom_id"]] = bert_score_value
            
            # F1 Score
            f1_score_value = calculate_f1(gt_answer, llm_answer)
            scores["f1"][sub["custom_id"]] = f1_score_value

    return scores

def plot_results(scores):
    """Visualizes the evaluation scores with histograms."""
    metrics = ["rougeL", "bleu", "bertscore", "f1"]
    titles = {
        "rougeL": "ROUGE-L Score Distribution",
        "bleu": "BLEU Score Distribution",
        "bertscore": "BERTScore Distribution",
        "f1": "F1 Score Distribution"
    }
    plt.figure(figsize=(15, 10))
    
    for idx, metric in enumerate(metrics, 1):
        plt.subplot(2, 2, idx)
        sns.histplot(list(scores[metric].values()), bins=10, color="skyblue", kde=True)
        plt.xlim(0, 100)
        plt.title(titles[metric])
        plt.xlabel(f"{metric.upper()} Score (%)")
        plt.ylabel("Frequency")
        
        mean_score = np.mean(list(scores[metric].values()))
        plt.text(0.5, 0.95, f'Average {metric.upper()} Score= {mean_score:.2f}%', 
                 ha='left', va='center', transform=plt.gca().transAxes, fontsize=10, color='black')
    
    plt.tight_layout()
    plt.show()

def main(llm_response_file, ground_truth_file):
    llm_jsonl = load_jsonl(llm_response_file)
    ground_truth_json = load_json(ground_truth_file)
    
    scores = evaluate_responses(llm_jsonl, ground_truth_json)
    
    print("Evaluation Results:")
    print("| Question ID       | ROUGE-L Score | BLEU Score | BERTScore | F1 Score |")
    print("|--------------------|---------------|------------|-----------|----------|")
    all_q_ids = list(scores["rougeL"].keys())  # Get all question IDs from one metric
    for q_id in all_q_ids:
        rouge_l = scores["rougeL"].get(q_id, 0)
        bleu = scores["bleu"].get(q_id, 0)
        bert = scores["bertscore"].get(q_id, 0)
        f1 = scores["f1"].get(q_id, 0)
        print(f"| {q_id} | {rouge_l:.2f}% | {bleu:.2f}% | {bert:.2f}% | {f1:.2f}% |")

    avg_rouge_l = np.mean(list(scores["rougeL"].values()))
    avg_bleu = np.mean(list(scores["bleu"].values()))
    avg_bert = np.mean(list(scores["bertscore"].values()))
    avg_f1 = np.mean(list(scores["f1"].values()))
    # Print averages in the last row
    print(f"| **Average**            | {avg_rouge_l:.2f}% | {avg_bleu:.2f}% | {avg_bert:.2f}% | {avg_f1:.2f}% |")

if __name__ == "__main__":
    main("gpt4o_mini_2_shot.jsonl", "ground_truth.json")