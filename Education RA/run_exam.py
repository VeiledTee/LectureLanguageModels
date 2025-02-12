import logging
from pathlib import Path
from transformers import pipeline
import re
from data_preprocessing import extract_questions, load_markdown_sections
import torch

MODEL_NAME = "HuggingFaceTB/SmolLM-1.7B-Instruct"
LOCAL_EXAM_DIR = Path(
    "/home/penguins/Documents/PhD/LectureLanguageModels/Education RA/AI_Course/Exams"
).resolve()
LOCAL_NOTES_DIR = Path(
    "/home/penguins/Documents/PhD/LectureLanguageModels/Education RA/AI_Course/Lecture_Notes"
).resolve()
OUTPUT_DIR = LOCAL_EXAM_DIR / "generated_answers"
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

# Verify directories exist
print(f"LOCAL_EXAM_DIR exists: {LOCAL_EXAM_DIR.exists()}")
print(f"LOCAL_NOTES_DIR exists: {LOCAL_NOTES_DIR.exists()}")


def process_quiz_files() -> list[list[str]]:
    question_list = []

    # Process all answerless quiz files in exam directory
    for file_path in LOCAL_EXAM_DIR.glob("*_answerless.txt"):
        print(f"Processing file: {file_path}")
        start_count: int = len(question_list)
        with open(file_path, "r") as f:
            sections = load_markdown_sections(str(file_path))
            for header, content in sections.items():
                if len(content) >= len(header):
                    question_list.append(extract_questions(content))
        print(
            f"Processed {len(question_list) - start_count} questions from {file_path}"
        )

    return question_list


def generate_answer(query: str) -> str:
    # Initialize pipeline once
    if not hasattr(generate_answer, "pipe"):
        generate_answer.pipe = pipeline(
            "text-generation",
            model=MODEL_NAME,
            device=0 if torch.cuda.is_available() else -1,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        )

    try:
        # Format using chat template
        messages = [
            {
                "role": "user",
                "content": f"""
                Please provide a comprehensive answer to the following question.
                Answer in detail while maintaining accuracy. No code - English only.

                Question: {query}
            """,
            }
        ]

        # Generate with optimized parameters
        response = generate_answer.pipe(
            messages,
            max_new_tokens=2048,
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
            truncation=True,
            pad_token_id=generate_answer.pipe.tokenizer.eos_token_id,
            return_full_text=False,
        )[0]["generated_text"]

        # Clean up the response
        cleaned = re.sub(
            r"(<|endoftext|>|<\/s>|\[.*?\]|�+)",
            "",
            response.split("assistant")[-1].strip(),
        )

        # Ensure proper sentence boundaries
        if not cleaned.endswith((".", "!", "?")):
            last_punct = max(cleaned.rfind("."), cleaned.rfind("!"), cleaned.rfind("?"))
            if last_punct != -1:
                cleaned = cleaned[: last_punct + 1]

        return cleaned.strip()

    except Exception as e:
        return f"Error: {str(e)}"


def main(lm: str, output_directory: Path):
    question_list = process_quiz_files()

    for i, questions in enumerate(question_list):
        output_path = output_directory / f"output_{i}_{lm.split('/')[-1]}.txt"
        with open(output_path, "w", encoding="utf-8") as f:
            for j, query in enumerate(questions):
                print(f"Generating answer for question {i + 1}")
                response = generate_answer(query)
                f.write(f"Q: {query}\nA: {response}\n\n")


if __name__ == "__main__":
    # question = "What's the capital of France?"
    # answer = generate_answer(question)
    # print(f"Q: {question}\nA: {answer}")

    for model in [
        "HuggingFaceTB/SmolLM-135M-Instruct",
        "HuggingFaceTB/SmolLM-360M-Instruct",
        "HuggingFaceTB/SmolLM-1.7B-Instruct",
        "mistralai/Mistral-7B-Instruct-v0.3",
        "bigscience/bloom",
        "Qwen/Qwen2.5-VL-3B-Instruct",
    ]:

        # Configure logging
        logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

        # Log messages
        logging.warning(f"{model} has started the exam")
        output_dir = LOCAL_EXAM_DIR / model.split('/')[-1] / "generated_answers"
        output_dir.mkdir(exist_ok=True, parents=True)
        main(lm=model, output_directory=output_dir)
        logging.warning(f"{model} has finished the exam")