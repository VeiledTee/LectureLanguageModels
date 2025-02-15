import gc
import logging
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List

import torch
from dotenv import load_dotenv
from transformers import pipeline

from data_preprocessing import load_markdown_sections

# Configuration constants
LOCAL_EXAM_DIR: Path = Path(
    "/home/penguins/Documents/PhD/LectureLanguageModels/Education RA/AI_Course/Exams"
).resolve()
OUTPUT_DIR: Path = LOCAL_EXAM_DIR / "generated_answers"
load_dotenv()  # Load variables from .env
hf_token = os.getenv("HF_TOKEN")

# Verify and create directories
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
print(f"EXAM DIR: {LOCAL_EXAM_DIR.exists()}")


def format_question(header: str, content: str) -> str:
    """Format a question from its header and content sections.

    Args:
        header: Hierarchical header path from markdown sections
        content: Question content text

    Returns:
        Formatted question string with proper punctuation
    """
    question = f"{header} > {content}" if content else header
    return question.rstrip(":") + ":" if not question.endswith(":") else question


def process_exam_file(file_path: Path) -> List[str]:
    """Process an answerless exam file into individual questions.

    Args:
        file_path: Path to the answerless exam markdown file

    Returns:
        List of formatted question strings

    Raises:
        FileNotFoundError: If the specified file doesn't exist
    """
    questions: List[str] = []
    sections: Dict[str, str] = load_markdown_sections(str(file_path))

    for header, content in sections.items():
        questions.append(format_question(header, content))

    print(f"Processed {len(questions)} questions from {file_path.name}")
    return questions


def generate_answers(questions: List[str], pipe: pipeline) -> List[str]:
    """Generate answers for a list of questions using a text generation pipeline.

    Args:
        questions: List of formatted question strings
        pipe: Initialized Hugging Face text generation pipeline

    Returns:
        List of generated answer strings with error handling

    Raises:
        RuntimeError: If text generation fails catastrophically
    """
    answers: List[str] = []
    for i, question in enumerate(questions):
        try:
            # Memory management
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            if i % 5 == 0:
                gc.collect()

            # Generate response
            response: List[Dict[str, Any]] = pipe(
                f"Answer concisely in English without code: {question}",
                max_new_tokens=256,
                do_sample=True,
                temperature=0.3,
                top_p=0.7,
                truncation=True,
                pad_token_id=pipe.tokenizer.eos_token_id,

            )

            # Clean and format response
            raw_text: str = response[0]["generated_text"]
            # Remove the initial prompt from the response
            cleaned = raw_text.replace(f"Answer concisely in English without code: {question}", "", 1).strip()
            # Proceed with existing regex cleaning and punctuation checks

            # Ensure proper sentence endings
            last_punct: int = max(
                cleaned.rfind("."), cleaned.rfind("!"), cleaned.rfind("?")
            )
            if last_punct != -1 and not cleaned.endswith((".", "!", "?")):
                cleaned = cleaned[: last_punct + 1]

            answers.append(cleaned.strip())
            print(f"Generated answer {i + 1}/{len(questions)}")

        except Exception as e:
            answers.append(f"Error: {str(e)}")

    return answers


def run_exams_for_model(model_id: str) -> None:
    """Process all exam files for a single language model.

    Args:
        model_id: Hugging Face model identifier (e.g., "gpt2")

    Raises:
        ValueError: If model initialization fails
        OSError: If file operations fail
    """
    model_name: str = model_id.replace("/", "-")
    logging.info(f"Starting exam processing for {model_id}")

    # Initialize text generation pipeline
    pipe: pipeline = pipeline(
        "text-generation",
        model=model_id,
        device_map="cpu",
        token=hf_token,
        torch_dtype=torch.float32,
        trust_remote_code=True,
    )

    # Process each exam file
    for exam_file in LOCAL_EXAM_DIR.glob("*_answerless.txt"):
        start_time = time.time()
        try:
            exam_name = exam_file.stem.replace("_answerless", "")
            exam_dir = exam_name.split("_")[0]

            # Create both directory levels
            output_dir = OUTPUT_DIR / exam_dir
            output_dir.mkdir(
                parents=True, exist_ok=True
            )  # <-- This creates both directories

            output_path = output_dir / f"{model_name}_{exam_name}_answers.txt"

            questions = process_exam_file(exam_file)
            answers = generate_answers(questions, pipe)

            with open(output_path, "w", encoding="utf-8") as f:
                for q, a in zip(questions, answers):
                    f.write(f"QUESTION: {q}\n//// ANSWER: {a}\n\n")

            # Log performance
            duration: float = time.time() - start_time
            logging.info(
                f"Completed {exam_name} in {duration:.2f}s ({len(questions)} questions)"
            )

        except Exception as e:
            logging.error(f"Failed processing {exam_file.name}: {str(e)}")
            continue


if __name__ == "__main__":
    """Main execution block for processing exams with multiple models."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    models: list[str] = [
        # "meta-llama/Llama-3.3-70B-Instruct",
        # "deepseek-ai/DeepSeek-R1",
        # "simplescaling/s1-32B",
        # "mistralai/Mistral-Small-24B-Instruct-2501",
        # ":ibm-granite/granite-3.2-8b-instruct-preview",
        # "HuggingFaceH4/zephyr-7b-beta",
        # "mistralai/Mistral-7B-Instruct-v0.3",
        # "bigscience/bloom",
        # "Qwen/Qwen2.5-VL-3B-Instruct",
        # "meta-llama/Llama-3.2-1B",
        # "open-thoughts/OpenThinker-7B",
        "HuggingFaceTB/SmolLM-135M-Instruct",
        "HuggingFaceTB/SmolLM-360M-Instruct",
        "HuggingFaceTB/SmolLM-1.7B-Instruct",
        "facebook/opt-1.3b",
        "gpt2-medium",
    ]

    try:
        for model in models:
            run_exams_for_model(model)
            logging.info(f"Completed all exams for {model}")
    except KeyboardInterrupt:
        logging.error("Process interrupted by user")
    finally:
        print("\n=== All model processing completed ===")
