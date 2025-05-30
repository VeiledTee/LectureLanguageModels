import os
import re

import modal
from transformers import pipeline

MODEL_NAME = "distilgpt2"  # or "gpt2"
MODEL_CACHE = "/vol/cache"
QUESTION_FILE: str = "Artificial_Intelligence/Exams/q1_soln_answerless.txt"


def create_cache_dir():
    os.makedirs("/vol/cache", exist_ok=True)


image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install(
        "torch",
        "transformers",
        "accelerate",
        "bitsandbytes",
    )
    .run_function(
        create_cache_dir, secrets=[modal.Secret.from_name("huggingface-secret")]
    )
    # Add the file to the image with explicit paths
    .add_local_file(
        local_path=QUESTION_FILE, remote_path="/root/q1_soln_answerless.txt"
    )
)


def extract_questions(extract_from) -> list[str]:
    extracted_questions = []
    for line in extract_from.split("\x1e"):  # split by delimiter
        for query in line.split("\x1f"):  # split questions
            if line != "":
                extracted_questions.append(query.strip())
    return extracted_questions


def load_markdown_sections(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        text = file.read()

    # Splitting sections by headers (##) while keeping them
    sections = re.split(r"(## .*\n)", text)

    extracted_sections = {}
    for i in range(1, len(sections), 2):
        header = sections[i].strip("# \n")
        content = sections[i + 1].strip()
        # Replace only actual section breaks with \x1E, keep the rest as is
        extracted_sections[header] = content.replace("\n", " ") + "\x1e"

    return extracted_sections


app = modal.App(
    name="huggingface-testing",
    image=image,
    secrets=[modal.Secret.from_name("huggingface-secret")],
)


@app.function()
def process_quiz_file(input_file: str) -> list[list[str]]:
    question_list: list[list[str]] = []
    with open("/root/q1_soln_answerless.txt", "r") as f:
        sections = load_markdown_sections("/root/q1_soln_answerless.txt")
        print(sections)
        for header, content in sections.items():
            print(f"Header: {len(header)} | Content: {len(content)}")
            if len(content) >= len(header):
                extracted = extract_questions(content)
                print(extracted)
                question_list.append(extracted)
    return question_list  # Return the list


@app.function(gpu="any")  # Request any available GPU type
def generate_answer(prompt: str) -> str:
    generator = pipeline(
        "text-generation", model=MODEL_NAME, truncation=True, max_length=2048
    )
    result = generator(prompt)
    return result[0]["generated_text"]


@app.local_entrypoint()
def main():
    prompt: str = "Answer this question to the best of your ability: "
    # Get questions remotely
    question_list = process_quiz_file.remote()
    # Query the LLM remotely for each question
    for q in question_list:
        for query in q:
            full_prompt = prompt + query
            response = generate_answer.remote(full_prompt)
            print(response + "\n---")
