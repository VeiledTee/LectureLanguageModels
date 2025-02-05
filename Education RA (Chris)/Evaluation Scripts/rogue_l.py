import json
import jsonlines
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
            print(llm_answer)
            score = scorer.score(gt_answer, llm_answer)
            scores[sub["custom_id"]] = score['rougeL'].fmeasure

    return scores

def main(llm_response_file, ground_truth_file):
    llm_jsonl = load_jsonl(llm_response_file)
    ground_truth_json = load_json(ground_truth_file)
    
    evaluation_scores = evaluate_responses(llm_jsonl, ground_truth_json)
    
    print("Evaluation Results:")
    for q_num, score in evaluation_scores.items():
        print(f"Q{q_num}: ROUGE-L Score = {score:.4f}")

if __name__ == "__main__":
    main("llm_response_ns.jsonl", "ground_truth_ns.json")
