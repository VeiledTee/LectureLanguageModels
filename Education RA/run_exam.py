from pathlib import Path
import os
import subprocess

import modal
from transformers import pipeline

from data_preprocessing import extract_questions, load_markdown_sections

VOLUME_NAME: str = "llm-volume"

MODEL_NAME: str = "distilgpt2"
MODEL_CACHE: str = "/vol/cache"
MODAL_EXAM_DIR: str = "/root/AI_Course/Exams"
MODAL_NOTES_DIR: str = "/root/AI_Course/Lecture_Notes"

LOCAL_EXAM_DIR: Path = Path("/home/penguins/Documents/PhD/LectureLanguageModels/Education RA/AI_Course/Exams").resolve()
LOCAL_NOTES_DIR: Path = Path("/home/penguins/Documents/PhD/LectureLanguageModels/Education RA/AI_Course/Lecture_Notes").resolve()
print(f"LOCAL_EXAM_DIR exists: {os.path.isdir(LOCAL_EXAM_DIR)}")
print(f"LOCAL_NOTES_DIR exists: {os.path.isdir(LOCAL_NOTES_DIR)}")


def create_cache_dir() -> None:
    os.makedirs(MODEL_CACHE, exist_ok=True)


def validate_paths():
    exam_dir = Path(LOCAL_EXAM_DIR)
    if not exam_dir.exists():
        exam_dir.mkdir(parents=True, exist_ok=True)
    print(f"Volume contains: {volume.listdir('/')}")

image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install("torch", "transformers", "accelerate", "bitsandbytes", "docling")
    .run_function(create_cache_dir, secrets=[modal.Secret.from_name("huggingface-secret")])
)

app = modal.App(
    name="huggingface-testing",
    image=image,
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
# Upload local files to volume (before app definition)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
validate_paths()


if LOCAL_EXAM_DIR.is_dir() and LOCAL_NOTES_DIR.is_dir():  # will only run when executing locally
    with volume.batch_upload(force=True) as batch:
        batch.put_directory(local_path=str(LOCAL_EXAM_DIR), remote_path=MODAL_EXAM_DIR, recursive=True)
        batch.put_directory(local_path=str(LOCAL_NOTES_DIR), remote_path=MODAL_NOTES_DIR, recursive=True)


@app.function(volumes={MODAL_EXAM_DIR: volume})
def process_quiz_file() -> list[list[str]]:
    question_list: list[list[str]] = []

    for entry in volume.iterdir(""):
        print(f"Found: {entry.path}")  # Debugging

    for entry in volume.iterdir(""):
        print(f"Attempting to open: {entry.path}")  # Debugging

        if entry.path.endswith("_answerless.txt"):
            print(f"Opening file: {entry.path}")  # Debugging

            with open(f"/{entry.path}", "r") as f:
                sections = load_markdown_sections(f"/{entry.path}")
                for header, content in sections.items():
                    if len(content) >= len(header):
                        question_list.append(extract_questions(content))

    return question_list


@app.function(gpu="any", volumes={MODAL_EXAM_DIR: volume})
def generate_answer(prompt: str) -> str:
    try:
        generator = pipeline("text-generation", model=MODEL_NAME, max_length=1024)
        inputs = generator.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        result = generator.model.generate(**inputs, max_new_tokens=512)
        return generator.tokenizer.decode(result[0])
    except RuntimeError as e:
        return f"Error: {str(e)}"


@app.local_entrypoint()
def main() -> None:

    for directory in volume.iterdir('root/'):
        print(f"In root/: {directory.path}")

    # Process files
    question_list = process_quiz_file.remote()

    # Write outputs directly to volume
    output_dir = Path(MODAL_EXAM_DIR)
    for i, questions in enumerate(question_list):
        output_path = output_dir / f"output_{i}.txt"
        with open(output_path, "w", encoding="utf-8") as f:
            for j, query in enumerate(questions):
                response = generate_answer.remote(f"Answer: {query}")
                f.write(f"Q{j + 1}: {query}\nA: {response}\n\n")

    # Commit changes and download
    volume.commit()
    subprocess.run([
        "modal", "volume", "get",
        VOLUME_NAME,
        MODAL_EXAM_DIR,
        LOCAL_EXAM_DIR
    ], check=True)