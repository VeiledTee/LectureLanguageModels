import json
import jsonlines
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from rouge_score import rouge_scorer

def load_json(file_path):
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

def evaluate_responses(llm_responses, ground_truth):
    """Evaluates the LLM responses against ground truth using ROUGE-L."""
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    scores = {}
    for i, item in enumerate(ground_truth):
        for j, sub in enumerate(item.get("subquestions", [])):
            sub_question = sub.get("question", "")
            gt_answer = sub.get("answer", "")
            llm_answer = llm_responses.get(sub["custom_id"], "").get('response', {}).get('body', {}).get('choices', [])[0].get('message', {}).get('content', '')
            score = scorer.score(gt_answer, llm_answer)
            scores[sub["custom_id"]] = score['rougeL'].fmeasure * 100  # Convert to percentage
    return scores

def plot_results(scores):
    """Visualizes the evaluation scores with a histogram for large datasets."""
    plt.figure(figsize=(10, 5))
    
    if len(scores) > 20:  # If too many questions, use bins
        sns.histplot(x = list(scores.values()), bins=10, palette="viridis")
        plt.xlim(0, 50)  
        plt.xlabel("ROUGE-L Score (%)")
        plt.ylabel("Frequency")
        plt.title("LLM Evaluation Score Distribution")
    else:
        # sns.barplot(x=list(scores.keys()), y=list(scores.values()), palette="viridis")
        sns.histplot(x=list(scores.values()), palette="viridis")
        plt.xticks(rotation=90)
        plt.xlim(0, 50)  
        plt.ylabel("Frequency")
        plt.xlabel("ROUGE-L Score (%)")
        plt.title("LLM Evaluation Results")

    mean_score = np.mean(list(scores.values()))

    plt.text(0.5, 0.95, f'Average ROUGE-L Score= {mean_score:.2f}%', ha='left', va='center', transform=plt.gca().transAxes, fontsize=12, color='black')

    plt.show()

def main(llm_response_file, ground_truth_file):
    llm_jsonl = load_jsonl(llm_response_file)
    ground_truth_json = load_json(ground_truth_file)
    
    evaluation_scores = evaluate_responses(llm_jsonl, ground_truth_json)
    
    print("Evaluation Results:")
    for q_num, score in evaluation_scores.items():
        print(f"Question ID: {q_num}: ROUGE-L Score = {score:.2f}%")
    
    plot_results(evaluation_scores)

if __name__ == "__main__":
    main("llm_response_ns.jsonl", "ground_truth_ns.json")
