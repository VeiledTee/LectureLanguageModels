import logging
import os
import re
import time
from pathlib import Path
from tqdm import tqdm

import ollama
from dotenv import load_dotenv

from preprocessing import process_exam_file

load_dotenv()

# Initialize logging
logging.basicConfig(level=logging.INFO)


class LLMQASystem:
    """A class for direct question answering using Ollama models without RAG context."""

    def __init__(self, generation_model_name: str):
        """Initializes the direct answering instance.

        Args:
            generation_model_name: Name of the Ollama model for answer generation.
        """
        self.generation_model = generation_model_name
        self.temperature = float(os.getenv("GENERATION_TEMPERATURE", "0.3"))
        self.max_tokens = int(os.getenv("MAX_TOKENS", "2048"))

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
    exam_dir = Path(os.getenv("EXAM_DIR", "AI_Course/Exams"))
    output_dir = Path(os.getenv("ANSWER_DIR", "AI_Course/Exams/generated_answers"))
    for exam_file in exam_dir.glob("*_answerless.txt"):
        try:
            start_time = time.time()
            exam_name = exam_file.stem.replace("_answerless", "")
            output_path = output_dir / f"{exam_name}_{ollama_pipeline.generation_model.split(':')[0]}_answers.txt"

            questions = process_exam_file(exam_file)

            with open(output_path, "w", encoding="utf-8") as f:
                for question in tqdm(questions, desc="Generating answers", unit="question"):
                    answer = ollama_pipeline.generate_answer(question)
                    f.write(f"QUESTION: {question}\n//// ANSWER: {answer}\n\n")
                    f.flush()  # write answer immediately

            duration = time.time() - start_time

            # Convert duration to hh:mm:ss
            hours = int(duration // 3600)
            minutes = int((duration % 3600) // 60)
            seconds = int(duration % 60)

            logging.info(
                f"Processed {exam_name} in {hours:02}:{minutes:02}:{seconds:02} ({len(questions)} questions)"
            )
        except Exception as e:
            logging.error(f"Error processing {exam_file}: {str(e)}")


if __name__ == "__main__":
    # Configuration
    env_models: str = os.getenv("GENERATION_MODELS")
    models: list[str] = [m.strip() for m in env_models.split(",")]
    for model in models:
        # Initialize direct answering pipeline
        quiz_taking_system = LLMQASystem(generation_model_name=model)

        # Process exams directly
        logging.info(f"Processing exams with Ollama queries and {model}...")
        process_exams(quiz_taking_system)
        logging.info(f"{model} exam complete!")
