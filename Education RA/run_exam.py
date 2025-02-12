import logging
import time

from pathlib import Path
from transformers import pipeline
import re
from data_preprocessing import load_markdown_sections
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


def process_quiz_files() -> list[str]:
    """
    Processes all answerless quiz files in the exam directory by splitting each Markdown file
    into sections using header tags (via split_markdown_sections) and combining the header and
    associated content to form the full question.

    The hierarchical header path (the key) is combined with the section content (the value).
    For example:
        [H1] 6.034 Quiz 1, Spring 2005 > [H2] 1 Search Algorithms (16 points) > [H3] 1.1 Games
        > [H4] 1. Can alpha-beta be generalized to do a breadth-first exploration ...:

    Returns:
        list[str]: A list of question strings.
    """
    question_list = []

    # Process all answerless quiz files in the exam directory.
    for file_path in LOCAL_EXAM_DIR.glob("*_answerless.txt"):
        print(f"Processing file: {file_path}")
        start_count = len(question_list)

        # Use the new function to split the Markdown file into sections.
        sections = load_markdown_sections(str(file_path))
        for header, content in sections.items():
            # Combine the hierarchical header (key) and its content (if any) to form the question.
            if content:
                question = f"{header} > {content}"
            else:
                question = header

            # Append a colon at the end if not already present.
            if not question.endswith(":"):
                question += ":"
            question_list.append(question)

        print(f"Processed {len(question_list) - start_count} questions from {file_path}")

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
                Answer succinctly but in detail. Maintain accuracy. 
                Do not provide code. 
                English only.
                Begin answering immediately.
                Answer as succinctly as possible.
                Do not repeat the question

                Question: {query}
                Answer: 
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


def main(lm: str, output_directory: Path) -> dict:
    question_list = process_quiz_files()
    model_name = lm.split('/')[-1]
    output_path = output_directory / f"{model_name}_combined_answers.txt"

    start_time = time.time()  # Start timing

    with open(output_path, "w", encoding="utf-8") as f:
        for i, question in enumerate(question_list):
            print(f"Generating answer for question {i + 1}")
            if '[H2]' in question:
                response = generate_answer(question)
                f.write(f"//// ANSWER: {response}\n\n")

    total_time = time.time() - start_time

    return {
        "model": model_name,
        "total_time": total_time,
        "questions": len(question_list),
        "avg_time_per_question": total_time/len(question_list)
    }


if __name__ == "__main__":
    timing_report = {}

    for model in [
        "HuggingFaceTB/SmolLM-135M-Instruct",
        "HuggingFaceTB/SmolLM-360M-Instruct",
        "HuggingFaceTB/SmolLM-1.7B-Instruct",
        "mistralai/Mistral-7B-Instruct-v0.3",
        "bigscience/bloom",
        "Qwen/Qwen2.5-VL-3B-Instruct",
    ]:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
        model_name = model.split('/')[-1]

        logging.warning(f"{model_name} has started the exam")
        output_dir = LOCAL_EXAM_DIR / "generated_answers"
        output_dir.mkdir(exist_ok=True, parents=True)

        # Run main and collect timing data
        metrics = main(lm=model, output_directory=output_dir)
        timing_report[model_name] = metrics

        logging.warning(f"{model_name} has finished the exam in {metrics['total_time']:.2f}s")

    # Print final report
    print("\n=== Timing Report ===")
    for model, data in timing_report.items():
        print(f"""Model: {model}
                Total time: {data['total_time']:.2f} seconds
                Questions answered: {data['questions']}
                Average time per question: {data['avg_time_per_question']:.2f}s
                """)
