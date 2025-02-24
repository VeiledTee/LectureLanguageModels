import logging
import os
import re
import time
from pathlib import Path

import ollama
from dotenv import load_dotenv

from preprocessing import process_exam_file

load_dotenv()

# Configuration
EXAM_DIR = Path(r"AI_Course/Exams").resolve()
OUTPUT_DIR = Path("AI_Course/Exams/generated_answers").resolve()

# Initialize logging
logging.basicConfig(level=logging.INFO)


class LLMQASystem:
    """A class for direct question answering using Ollama models without RAG context."""

    def __init__(
            self,
            generation_model_name: str = "llama3.2",
            temperature: float = 0.3,
            max_tokens: int = 2048,
    ) -> None:
        """Initializes the direct answering instance.

        Args:
            generation_model_name: Name of the Ollama model for answer generation.
            temperature: Controls randomness (0.0-1.0, lower means more factual).
            max_tokens: Maximum length of the generated response.
        """
        self.generation_model = generation_model_name
        self.temperature = temperature
        self.max_tokens = max_tokens

    def generate_answer(self, question: str) -> str:
        """Generates an answer to a user question directly using Ollama.

        Args:
            question: The user question to answer.

        Returns:
            The generated answer string.
        """
        try:
            # Create simplified prompt without context
            prompt = f"""<|system|>
You are an AI teaching assistant. Answer the following question to the best of your knowledge.
Provide a detailed answer based on your training.
Be definitive in your answer if the question calls for a yes or no response.
Answer all parts of the question completely.
</s>
<|user|>
{question}
</s>
<|assistant|>"""

            # Generate response through Ollama
            response = ollama.generate(
                model=self.generation_model,
                prompt=prompt,
                options={
                    "temperature": self.temperature,
                    "max_tokens": self.max_tokens,
                    "top_p": 0.9,
                    "stop": ["</s>", "\n\n\n"],
                },
            )

            # Extract and format response
            if "deepseek" not in self.generation_model:
                answer = response.get("response", "").strip()
            else:
                answer = response.get("response", "").strip()
                answer = re.sub(r"<think>.*?</think>\n?", "", answer, flags=re.DOTALL)

            return answer

        except Exception as e:
            logging.error(f"Answer generation failed: {str(e)}")
            return "Error generating answer"


def process_exams(ollama_pipeline: LLMQASystem):
    """
    Processes exam files to generate answers using direct Ollama queries.

    Args:
        ollama_pipeline: An instance of LLMQASystem used to generate answers.
    """
    for exam_file in EXAM_DIR.glob("*_answerless.txt"):
        try:
            start_time = time.time()
            exam_name = exam_file.stem.replace("_answerless", "")
            output_directory = OUTPUT_DIR / exam_name
            output_directory.mkdir(parents=True, exist_ok=True)
            output_path = OUTPUT_DIR / exam_name / f"{exam_name}_{ollama_pipeline.generation_model}_answers.txt"

            questions = process_exam_file(exam_file)

            with open(output_path, "w", encoding="utf-8") as f:
                for question in questions:
                    answer = ollama_pipeline.generate_answer(question)
                    f.write(f"QUESTION: {question}\n//// ANSWER: {answer}\n\n")

            duration = time.time() - start_time
            logging.info(
                f"Processed {exam_name} in {duration:.2f}s ({len(questions)} questions)"
            )
        except Exception as e:
            logging.error(f"Error processing {exam_file}: {str(e)}")


if __name__ == "__main__":
    models: list[str] = [
        "phi4",
        "llama3.2",
        "mistral",
        "qwen2.5",
        "deepseek-r1",
    ]
    for model in models:
        # Initialize direct answering pipeline
        quiz_taking_system = LLMQASystem(
            generation_model_name=model,
            temperature=0.3,
            max_tokens=2048
        )

        # Process exams directly
        logging.info(f"Processing exams with Ollama queries and {model}...")
        process_exams(quiz_taking_system)
        logging.info(f"{model} exam complete!")
