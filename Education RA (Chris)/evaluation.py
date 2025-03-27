import json
import jsonlines
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from bert_score import score as bert_score
from typing import Dict, List, Tuple
from dataclasses import dataclass

@dataclass
class EvaluationMetrics:
    rouge1: float
    rougeL: float
    bleu: float
    bertscore: float
    f1: float
    jaccard: float

class NLPEvaluator:
    """
    A comprehensive NLP evaluation class that computes multiple metrics
    (ROUGE, BLEU, BERTScore, F1, Jaccard) and generates reports.
    """
    
    def __init__(self):
        self.scorer = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)
        self.smoothing = SmoothingFunction().method1
    
    def load_json(self, file_path: str) -> Dict:
        """Loads a JSON file."""
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def load_jsonl(self, file_path: str) -> Dict:
        """Loads responses from a JSONL file."""
        responses = {}
        with jsonlines.open(file_path, 'r') as reader:
            for obj in reader:
                responses[obj["custom_id"]] = obj
        return responses
    
    def extract_responses(self, llm_jsonl: Dict) -> Dict[str, str]:
        """Extracts response text from LLM output."""
        return {
            custom_id: entry.get('response', {}).get('body', {}).get('choices', [{}])[0].get('message', {}).get('content', '')
            for custom_id, entry in llm_jsonl.items()
        }
    
    def _calculate_bleu(self, reference: str, hypothesis: str) -> float:
        """Calculates BLEU score between reference and hypothesis."""
        return sentence_bleu([reference.split()], hypothesis.split(), 
                            smoothing_function=self.smoothing) * 100
    
    def _calculate_bertscore(self, reference: str, hypothesis: str) -> float:
        """Calculates BERTScore between reference and hypothesis."""
        _, _, F1 = bert_score([hypothesis], [reference], lang="en")
        return F1.item() * 100
    
    def _calculate_f1(self, reference: str, hypothesis: str) -> float:
        """Calculates token-level F1 score."""
        ref_tokens = set(reference.split())
        hyp_tokens = set(hypothesis.split())
        common_tokens = ref_tokens.intersection(hyp_tokens)
        
        if not ref_tokens or not hyp_tokens:
            return 0.0
        
        precision = len(common_tokens) / len(hyp_tokens)
        recall = len(common_tokens) / len(ref_tokens)
        return (2 * precision * recall / (precision + recall)) * 100 if (precision + recall) > 0 else 0.0
    
    def _calculate_jaccard(self, reference: str, hypothesis: str) -> float:
        """Calculates Jaccard similarity between reference and hypothesis."""
        ref_tokens = set(reference.split())
        hyp_tokens = set(hypothesis.split())
        intersection = ref_tokens.intersection(hyp_tokens)
        union = ref_tokens.union(hyp_tokens)
        return (len(intersection) / len(union)) * 100 if union else 0.0
    
    def evaluate_pair(self, reference: str, hypothesis: str) -> EvaluationMetrics:
        """Evaluates a single reference-hypothesis pair."""
        if not reference or not hypothesis:
            return EvaluationMetrics(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        
        rouge_scores = self.scorer.score(reference, hypothesis)
        return EvaluationMetrics(
            rouge1=rouge_scores['rouge1'].fmeasure * 100,
            rougeL=rouge_scores['rougeL'].fmeasure * 100,
            bleu=self._calculate_bleu(reference, hypothesis),
            bertscore=self._calculate_bertscore(reference, hypothesis),
            f1=self._calculate_f1(reference, hypothesis),
            jaccard=self._calculate_jaccard(reference, hypothesis)
        )
    
    def evaluate_all(self, llm_responses: Dict[str, str], ground_truth: List[Dict]) -> Dict[str, EvaluationMetrics]:
        """Evaluates all responses against ground truth."""
        evaluations = {}
        
        for item in ground_truth:
            custom_id = item.get("custom_id", "")
            gt_answer = item.get("answer", "")
            llm_answer = llm_responses.get(custom_id, "")
            
            evaluations[custom_id] = self.evaluate_pair(gt_answer, llm_answer)
        
        return evaluations
    
    def save_markdown_report(self, evaluations: Dict[str, EvaluationMetrics], 
                           llm_response_file: str) -> str:
        """
        Saves evaluation results as a markdown report.
        
        Args:
            evaluations: Dictionary of evaluation metrics
            llm_response_file: Path to the input LLM response file
            
        Returns:
            Path to the saved markdown file
        """
        os.makedirs("evaluation_results", exist_ok=True)
        
        # Generate output filename
        base_name = os.path.basename(llm_response_file).split('.')[0]
        output_file = os.path.join("evaluation_results", f"{base_name}_evaluation.md")
        
        # Calculate averages
        avg_scores = EvaluationMetrics(
            rouge1=np.mean([m.rouge1 for m in evaluations.values()]),
            rougeL=np.mean([m.rougeL for m in evaluations.values()]),
            bleu=np.mean([m.bleu for m in evaluations.values()]),
            bertscore=np.mean([m.bertscore for m in evaluations.values()]),
            f1=np.mean([m.f1 for m in evaluations.values()]),
            jaccard=np.mean([m.jaccard for m in evaluations.values()])
        )
        
        # Write markdown content
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("# NLP Evaluation Report\n\n")
            f.write("## Summary\n")
            f.write(f"- **Average ROUGE-1**: {avg_scores.rouge1:.2f}%\n")
            f.write(f"- **Average ROUGE-L**: {avg_scores.rougeL:.2f}%\n")
            f.write(f"- **Average BLEU**: {avg_scores.bleu:.2f}%\n")
            f.write(f"- **Average BERTScore**: {avg_scores.bertscore:.2f}%\n")
            f.write(f"- **Average F1**: {avg_scores.f1:.2f}%\n")
            f.write(f"- **Average Jaccard**: {avg_scores.jaccard:.2f}%\n\n")
            
            f.write("## Detailed Results\n")
            f.write("| Question ID | ROUGE-1 | ROUGE-L | BLEU | BERTScore | F1 | Jaccard |\n")
            f.write("|-------------|---------|---------|------|-----------|----|---------|\n")
            
            for q_id, metrics in evaluations.items():
                f.write(f"| {q_id} | {metrics.rouge1:.2f}% | {metrics.rougeL:.2f}% | "
                       f"{metrics.bleu:.2f}% | {metrics.bertscore:.2f}% | "
                       f"{metrics.f1:.2f}% | {metrics.jaccard:.2f}% |\n")
        
        return output_file
    
    def print_results(self, evaluations: Dict[str, EvaluationMetrics]) -> None:
        """Prints evaluation results to console."""
        print("Evaluation Results:")
        print("| Question ID       | ROUGE-1 | ROUGE-L | BLEU | BERTScore | F1 | Jaccard |")
        print("|-------------------|---------|---------|------|-----------|----|---------|")
        
        for q_id, metrics in evaluations.items():
            print(f"| {q_id:<17} | {metrics.rouge1:>7.2f}% | {metrics.rougeL:>7.2f}% | "
                  f"{metrics.bleu:>6.2f}% | {metrics.bertscore:>9.2f}% | "
                  f"{metrics.f1:>4.2f}% | {metrics.jaccard:>7.2f}% |")
        
        # Calculate and print averages
        avg_scores = EvaluationMetrics(
            rouge1=np.mean([m.rouge1 for m in evaluations.values()]),
            rougeL=np.mean([m.rougeL for m in evaluations.values()]),
            bleu=np.mean([m.bleu for m in evaluations.values()]),
            bertscore=np.mean([m.bertscore for m in evaluations.values()]),
            f1=np.mean([m.f1 for m in evaluations.values()]),
            jaccard=np.mean([m.jaccard for m in evaluations.values()])
        )
        
        print(f"| **Average**        | {avg_scores.rouge1:>7.2f}% | {avg_scores.rougeL:>7.2f}% | "
              f"{avg_scores.bleu:>6.2f}% | {avg_scores.bertscore:>9.2f}% | "
              f"{avg_scores.f1:>4.2f}% | {avg_scores.jaccard:>7.2f}% |")
    
    def plot_results(self, evaluations: Dict[str, EvaluationMetrics]) -> None:
        """Visualizes the evaluation scores with histograms."""
        metrics = ["rouge1", "rougeL", "bleu", "bertscore", "f1", "jaccard"]
        titles = {
            "rouge1": "ROUGE-1 Score Distribution",
            "rougeL": "ROUGE-L Score Distribution",
            "bleu": "BLEU Score Distribution",
            "bertscore": "BERTScore Distribution",
            "f1": "F1 Score Distribution",
            "jaccard": "Jaccard Similarity Distribution"
        }
        
        plt.figure(figsize=(18, 12))
        
        for idx, metric in enumerate(metrics, 1):
            plt.subplot(3, 2, idx)
            scores = [getattr(m, metric) for m in evaluations.values()]
            sns.histplot(scores, bins=10, color="skyblue", kde=True)
            plt.xlim(0, 100)
            plt.title(titles[metric])
            plt.xlabel(f"{metric.upper()} Score (%)")
            plt.ylabel("Frequency")
            
            mean_score = np.mean(scores)
            plt.text(0.5, 0.95, f'Average {metric.upper()}= {mean_score:.2f}%', 
                     ha='left', va='center', transform=plt.gca().transAxes, fontsize=10)
        
        plt.tight_layout()
        plt.show()
    
    def evaluate(self, llm_response_file: str, ground_truth_file: str) -> Dict[str, EvaluationMetrics]:
        """
        Main evaluation pipeline.
        
        Args:
            llm_response_file: Path to LLM responses JSONL file
            ground_truth_file: Path to ground truth JSON file
            
        Returns:
            Dictionary containing all evaluation metrics
        """
        # Load data
        llm_jsonl = self.load_jsonl(llm_response_file)
        ground_truth = self.load_json(ground_truth_file)
        
        # Extract responses
        llm_responses = self.extract_responses(llm_jsonl)
        
        # Get ground truth questions (handle both formats)
        gt_questions = ground_truth.get("questions", ground_truth) if isinstance(ground_truth, dict) else ground_truth
        
        # Evaluate responses
        evaluations = self.evaluate_all(llm_responses, gt_questions)
        
        # Print results
        self.print_results(evaluations)
        
        # Save markdown report
        report_path = self.save_markdown_report(evaluations, llm_response_file)
        print(f"\nEvaluation report saved to: {report_path}")
   
        return evaluations

# # Example usage
# if __name__ == "__main__":
#     evaluator = NLPEvaluator()
#     evaluator.evaluate(
#         llm_response_file="llm_response_ns_ft.jsonl",
#         ground_truth_file="ground_truth_ns.json"
#     )