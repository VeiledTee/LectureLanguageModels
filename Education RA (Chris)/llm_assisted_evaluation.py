from pydantic import BaseModel
from openai import OpenAI
import jsonlines
import json
import os
from typing import Dict, Tuple, List

client = OpenAI()

class ResponseEvaluation(BaseModel):
    score: int  # 0 (wrong), 1 (partially correct), 2 (fully correct)
    comment: str

def load_jsonl(file_path: str) -> Dict:
    """Loads responses from a JSONL file."""
    responses = {}
    with jsonlines.open(file_path, 'r') as reader:
        for obj in reader:
            responses[obj["custom_id"]] = obj
    return responses

def load_json(file_path: str) -> List[Dict]:
    """Loads a JSON file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def clean_comment(comment: str) -> str:
    """Replace newlines with spaces in comments."""
    return comment.replace('\n', ' ').strip()

def evaluate_llm_assisted(llm_responses: Dict, ground_truth: List[Dict]) -> Tuple[Dict, int, float]:
    """Evaluates the LLM responses using the LLM itself to classify answers and provide structured feedback."""
    evaluations = {}
    total_score = 0
    total_questions = 0

    for item in ground_truth:
        question = item.get("question", "")
        context = item.get("context", "")
        gt_answer = item.get("answer", "")
        llm_answer = llm_responses.get(item["custom_id"], {}).get('response', {}).get('body', {}).get('choices', [])[0].get('message', {}).get('content', '')
        
        # Ask the LLM to evaluate the response
        completion = client.beta.chat.completions.parse(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are an expert evaluator assessing the quality of answers to technical questions. Evaluate based on the provided ground truth answer."},
                {"role": "user", "content": f"Context: {context} \nQuestion: {question}\n Ground truth: {gt_answer}\nLLM response: {llm_answer}\n\nCategorize the answer as 'fully correct' (2), 'partially correct' (1), or 'wrong' (0) based on the ground truth and provide brief justification comment for the score, noting specific matches/mismatches with ground truth."},
            ],
            response_format=ResponseEvaluation,
        )
        
        evaluation = completion.choices[0].message.parsed
        evaluation.comment = clean_comment(evaluation.comment)  # Clean the comment
        evaluations[item["custom_id"]] = evaluation
        total_score += evaluation.score
        total_questions += 1

    percentage_score = (total_score / (total_questions * 2)) * 100  # Max score per question is 2
    return evaluations, total_score, percentage_score

def save_markdown_report(llm_response_file: str, evaluations: Dict, total_score: int, percentage_score: float) -> str:
    """Saves evaluation results to a markdown file in evaluation_results folder."""
    # Create output directory if it doesn't exist
    os.makedirs("evaluation_results", exist_ok=True)
    
    # Generate output filename
    base_name = os.path.basename(llm_response_file).split('.')[0]
    output_file = os.path.join("evaluation_results", f"{base_name}_llm_assisted.md")
    
    # Write markdown content
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("# LLM Response Evaluation Report\n\n")
        f.write("## Summary\n")
        f.write(f"- **Total Score**: {total_score}/{len(evaluations) * 2}\n")
        f.write(f"- **Percentage**: {percentage_score:.2f}%\n\n")
        
        f.write("## Detailed Results\n")
        f.write("| Question ID | Score | Comments |\n")
        f.write("|-------------|-------|----------|\n")
        
        for q_id, evaluation in evaluations.items():
            f.write(f"| {q_id} | {evaluation.score} | {evaluation.comment} |\n")
    
    return output_file

def evaluate(llm_response_file: str, ground_truth_file: str):
    """Main execution function."""
    llm_responses = load_jsonl(llm_response_file)
    ground_truth = load_json(ground_truth_file)

    evaluations, total_score, percentage_score = evaluate_llm_assisted(llm_responses, ground_truth)
    
    # Print results to console
    print("LLM-Assisted Evaluation Results:")
    print("| Question ID       | Score | Comments |")
    print("|-------------------|-------|----------|")

    for q_id, evaluation in evaluations.items():
        print(f"| {q_id} | {evaluation.score} | {evaluation.comment} |")
    
    print(f"\nTotal Score: {total_score}/{len(evaluations) * 2} ({percentage_score:.2f}%)")
    
    # Save results to markdown file
    output_path = save_markdown_report(llm_response_file, evaluations, total_score, percentage_score)
    print(f"\nEvaluation report saved to: {output_path}")

# if __name__ == "__main__":
#     main("llm_response_ns_ft.jsonl", "ground_truth_ns.json")