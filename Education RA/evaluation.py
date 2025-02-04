from datasets import Dataset
from ragas import evaluate

# Sample exam question test case
test_data = {
    "question": ["Explain gradient descent with momentum vs. Nesterov acceleration"],
    "answer": ["Gradient descent with momentum applies..."],  # System's answer
    "contexts": [
        ["Lecture 8 Slides: Momentum helps accelerate..."]
    ],  # Retrieved chunks
    "ground_truth": [
        "Nesterov momentum evaluates gradient at lookahead position..."
    ],  # Professor's rubric
}

dataset = Dataset.from_dict(test_data)

# RAGAS metrics for academic context
score = evaluate(
    dataset, metrics=["answer_correctness", "context_recall", "faithfulness"]
)

print(f"Answer matches rubric: {score['answer_correctness']:.2f}")
print(f"Key points retrieved: {score['context_recall']:.2f}")
print(f"Minimal hallucinations: {score['faithfulness']:.2f}")
