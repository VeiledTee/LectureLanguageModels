import gc
import logging
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"

import time
from pathlib import Path

import torch
from dotenv import load_dotenv
from transformers import pipeline

from data_preprocessing import parse_quiz

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
    """Format a question from its header and content sections."""
    question = f"{header} > {content}" if content else header
    return question.rstrip(":") + ":" if not question.endswith(":") else question

def process_exam_file(file_path: Path) -> list[str]:
    """Process an answerless exam file into individual questions."""
    with open(file_path, "r", encoding="utf-8") as f:
        input_text = f.read()
        questions: list[str] = parse_quiz(str(input_text))
    print(f"Processed {len(questions)} questions from {file_path.name}")
    return questions

def generate_answers(questions: list[str], pipe: pipeline, batch_size: int = 8) -> list[str]:
    """Generate answers for a list of questions using a text generation pipeline."""
    answers: list[str] = []
    for i in range(0, len(questions), batch_size):
        batch = questions[i:i + batch_size]
        try:
            # Memory management
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            if i % 5 == 0:
                gc.collect()

            # Generate responses
            responses = pipe(
                [f"Answer concisely in English without code: {q}" for q in batch],
                max_new_tokens=256,
                do_sample=True,
                temperature=0.3,
                top_p=0.7,
                truncation=True,
                pad_token_id=pipe.tokenizer.eos_token_id,
            )

            # Clean and format responses
            for j, response in enumerate(responses):

                raw_text = response[0]["generated_text"]
                cleaned = raw_text.replace(
                    f"Answer concisely in English without code: {batch[j]}", "", 1
                ).strip()
                last_punct = max(
                    cleaned.rfind("."), cleaned.rfind("!"), cleaned.rfind("?")
                )
                if last_punct != -1 and not cleaned.endswith((".", "!", "?")):
                    cleaned = cleaned[: last_punct + 1]
                answers.append(cleaned.strip())
            print(f"Generated answers {i + 1}/{len(questions)}")

        except Exception as e:
            answers.extend([f"Error: {str(e)}"] * len(batch))

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

    # Attempt to initialize the model on GPU
    try:
        # Check if GPU is available
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # Initialize text generation pipeline with the correct device
        pipe: pipeline = pipeline(
            "text-generation",
            model=model_id,
            device=0 if device == "cuda" else -1,  # Use GPU if available, otherwise CPU
            token=hf_token,
            torch_dtype=torch.float32,
            trust_remote_code=True,
        )
        logging.info(f"Device set to use {device}:0")

    except RuntimeError as e:
        if 'CUDA out of memory' in str(e):
            logging.warning(f"CUDA out of memory error: {e}. Falling back to CPU.")
            # Initialize the pipeline on CPU
            pipe: pipeline = pipeline(
                "text-generation",
                model=model_id,
                device=-1,  # Use CPU
                token=hf_token,
                torch_dtype=torch.float32,
                trust_remote_code=True,
            )
            logging.info("Device set to use CPU.")
        else:
            raise  # Re-raise the exception if it's not a CUDA out of memory error

    # Process each exam file
    for exam_file in LOCAL_EXAM_DIR.glob("*_answerless.txt"):
        start_time = time.time()
        try:
            exam_name = exam_file.stem.replace("_answerless", "")
            exam_dir = exam_name.split("_")[0]

            # Create both directory levels
            output_dir = OUTPUT_DIR / exam_dir
            output_dir.mkdir(parents=True, exist_ok=True)  # <-- This creates both directories

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
        # "open-thoughts/OpenThinker-7B",
        "HuggingFaceTB/SmolLM2-135M-Instruct",  # v2
        "HuggingFaceTB/SmolLM2-360M-Instruct",  # v2
        "HuggingFaceTB/SmolLM2-1.7B-Instruct",  # v2
        "HuggingFaceTB/SmolLM-135M-Instruct",  # v1
        "HuggingFaceTB/SmolLM-360M-Instruct",  # v1
        "HuggingFaceTB/SmolLM-1.7B-Instruct",  # v1
        # "facebook/opt-1.3b",
        # "gpt2-medium",
        # "Qwen/Qwen2.5-7B-Instruct-1M",
        # "meta-llama/Llama-3.1-8B-Instruct",
    ]

    try:
        for model in models:
            torch.cuda.empty_cache()
            run_exams_for_model(model)
            logging.info(f"Completed all exams for {model}")
    except KeyboardInterrupt:
        logging.error("Process interrupted by user")
    finally:
        print("\n=== All model processing completed ===")
