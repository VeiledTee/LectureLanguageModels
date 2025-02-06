from datasets import Dataset
from ragas import evaluate
from ragas.metrics import AnswerCorrectness, ContextRecall, Faithfulness  # import metric classes
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge import Rouge

# Define your texts.
question_text = (
    "The standard alpha-beta algorithm performs a depth-first exploration (to a pre-specified depth) of the game tree. "
    "1. Can alpha-beta be generalized to do a breadth-first exploration of the game tree and still get the optimal answer? "
    "Explain how or why not. If it can be generalized, indicate any advantages or disadvantages of using breadth-first search in this application. "
    "2. Can alpha-beta be generalized to do a progressive-deepening exploration of the game tree and still get the optimal answer? "
    "Explain how or why not. If it can be generalized, indicate any advantages or disadvantages of using progressive-deepening search in this application."
)

candidate_answer = (
    "Q: The standard alpha-beta algorithm performs a depth-first exploration (to a pre-specified depth) of the game tree. "
    "1. Can alpha-beta be generalized to do a breadth-first exploration of the game tree and still get the optimal answer? "
    "Explain how or why not. If it can be generalized, indicate any advantages or disadvantages of using breadth-first search in this application. "
    "Some answers suggested that min-max inherently needs to go all the way down to the leaves of the game tree, where the outcome of the game is known. "
    "This is not true. Typically one picks some depth of look-ahead and searches to that depth, using the static evaluator to compute a score for the board position at that depth. "
    "2. Can alpha-beta be generalized to do a progressive-deepening exploration of the game tree and still get the optimal answer? "
    "Explain how or why not. If it can be generalized, indicate any advantages or disadvantages of using progressive-deepening search in this application.\n\n"
    "A: 1. Yes, alpha-beta can be generalized to do a breadth-first exploration of the game tree and still get the optimal answer. "
    "In fact, alpha-beta can be used as a component in a breadth-first search algorithm. This is because alpha-beta can be used to determine the best move at each level of the search tree, "
    "which is similar to the way breadth-first search works.\n\n"
    "Alpha-beta is typically used to evaluate the best move at a given depth of the search tree, but it can also be used to determine the best move at a higher level of the search tree. "
    "This is because alpha-beta is designed to evaluate the best move at a given depth, so it can be used to determine the best move at a higher level of the search tree.\n\n"
    "The advantages of using alpha-beta in this application include:\n"
    "* The ability to use alpha-beta to evaluate the best move at a given depth, which can be more efficient than using a separate search algorithm.\n"
    "* The ability to use alpha-beta to determine the best move at a higher level of the search tree, which can be more efficient than using a separate search algorithm.\n\n"
    "The disadvantages of using alpha-beta in this application include:\n"
    "* The need to use alpha-beta to evaluate the best move at a given depth, which can be more complex than using a separate search algorithm.\n"
    "* The need to use alpha-beta to determine the best move at a higher level of the search tree, which can be more complex than using a separate search algorithm.\n\n"
    "2. Yes, alpha-beta can be generalized to do a progressive-deepening exploration of the game tree and still get the optimal answer. "
    "In fact, alpha-beta can be used as a component in a progressive-deepening search algorithm. This is because alpha-beta can be used to evaluate the best move at a given depth of the search tree, "
    "which is similar to the way progressive-deepening search works.\n\n"
    "Progressive-deepening search is a type of search algorithm that uses a combination of depth and breadth to explore the game tree. "
    "The idea is to start with a small number of nodes and gradually increase the depth of the search tree as you explore the game tree. "
    "This allows you to explore the game tree in a more efficient way.\n\n"
    "The advantages of using alpha-beta in a progressive-deepening search algorithm include:\n"
    "* The ability to use alpha-beta to evaluate the best move at a given depth, which can be more efficient than using a separate search algorithm.\n"
    "* The ability to use alpha-beta to determine the best move at a higher level of the search tree, which can be more efficient than using a separate search algorithm.\n\n"
    "The disadvantages of using alpha-beta in a progressive-deepening search algorithm include:\n"
    "* The need to use alpha-beta to evaluate the best move at a given depth, which can be more complex than using a separate search algorithm.\n"
    "* The need to use alpha-beta to determine the best move at a higher level of the search tree, which can be more complex than using a separate search algorithm."
)

gold_standard_answer = (
    "The standard alpha-beta algorithm performs a depth-first exploration (to a pre-specified depth) of the game tree.\n"
    "- 1. Can alpha-beta be generalized to do a breadth-first exploration of the game tree and still get the optimal answer? "
    "Explain how or why not. If it can be generalized, indicate any advantages or disadvantages of using breadth-first search in this application.\n"
    "- No. The alpha-beta algorithm is an optimization on min-max. Min-max inherently needs to look at the game-tree nodes below the current node "
    "(down to some pre-determined depth) in order to assign a value to that node. A breadth-first version of min-max does not make much sense. "
    "Thinking about alpha-beta instead of min-max only makes it worse, since the whole point of alpha-beta is to use min-max values from one of the earlier "
    "(left-most) sub-trees to decide that we do not need to explore some later (right-most) subtrees.\n"
    "Some answers suggested that min-max inherently needs to go all the way down to the leaves of the game tree, where the outcome of the game is known. "
    "This is not true. Typically one picks some depth of look-ahead and searches to that depth, using the static evaluator to compute a score for the board position at that depth.\n"
    "- 2. Can alpha-beta be generalized to do a progressive-deepening exploration of the game tree and still get the optimal answer? "
    "Explain how or why not. If it can be generalized, indicate any advantages or disadvantages of using progressive-deepening search in this application.\n"
    "- Yes. Progressive-deepening involves repeated depth-first searches to increasing depths. This can be done trivially with min-max and alpha-beta as well, "
    "which also involve picking a maximum depth of lookahead in the tree. PD does waste some work, but as we saw in the notes, the extra work is a small fraction of the work "
    "that you would do anyways, especially when the branching factor is high, as it is in game trees. The advantage is that in timed situations you guarantee that you always have a reasonable move available."
)

# Build the test data dictionary.
test_data = {
    "question": [question_text],
    "answer": [candidate_answer],
    "ground_truth": [gold_standard_answer],
    "contexts": [[]],  # Optional: include supporting context if needed.
}

# Create a Hugging Face Dataset.
dataset = Dataset.from_dict(test_data)

# # Pass metric objects, not strings.
# metrics = [AnswerCorrectness(), ContextRecall(), Faithfulness()]
# scores = evaluate(dataset, metrics=metrics)
#
# print("RAGAS Metrics:")
# print(f"Answer matches rubric: {scores['answer_correctness']:.2f}")
# print(f"Key points retrieved: {scores['context_recall']:.2f}")
# print(f"Minimal hallucinations: {scores['faithfulness']:.2f}")

# Optional: Compute BLEU and ROUGE metrics.
smoothing = SmoothingFunction().method1
candidate_tokens = candidate_answer.split()
gold_tokens = gold_standard_answer.split()
bleu = sentence_bleu([gold_tokens], candidate_tokens, smoothing_function=smoothing)
rouge = Rouge()
rouge_scores = rouge.get_scores(candidate_answer, gold_standard_answer)[0]

print("\nOverlap Metrics:")
print(f"BLEU: {bleu:.4f}")
print(f"ROUGE-1: {rouge_scores['rouge-1']['f']:.4f}")
print(f"ROUGE-2: {rouge_scores['rouge-2']['f']:.4f}")
print(f"ROUGE-L: {rouge_scores['rouge-l']['f']:.4f}")
