import os
from pathlib import Path

import asyncio
import time
import os
import inspect
from lightrag.llm.ollama import ollama_model_complete, ollama_embed
from lightrag.lightrag import LightRAG, QueryParam
from lightrag.utils import EmbeddingFunc
import logging
from preprocessing import process_exam_file

logging.basicConfig(level=logging.DEBUG)

# Path to documents
KNOWLEDGE_DIR = Path(r"Artificial_Intelligence/Lecture_Notes").resolve()
# Path to exams
EXAM_DIR = Path(r"Artificial_Intelligence/Exams").resolve()
# Output directory
OUTPUT_DIR = Path("Artificial_Intelligence/Exams/generated_rag_answers").resolve()

# Define the working directory for LightRAG
WORKING_DIR = "./lightrag"
Path(WORKING_DIR).mkdir(exist_ok=True)


# Function to load documents from the knowledge directory
def load_documents(directory: Path):
    all_documents = []
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            file_path = directory / filename
            with open(file_path, "r", encoding="utf-8") as file:
                content = file.read()
                all_documents.append({"id": filename, "text": content})
    return all_documents


def process_exams(exam_dir: Path):
    for exam_file in exam_dir.glob("*_answerless.txt"):  # find all quiz files
        start_time = time.time()  # start timer
        try:
            exam_name = exam_file.stem.replace("_answerless", "")
            exam_dir = exam_name.split("_")[0]
            output_dir = OUTPUT_DIR / exam_dir
            output_dir.mkdir(
                parents=True, exist_ok=True
            )  # creates both directories if they don't exist
            output_path = (
                output_dir / f"{exam_name}_rag_answers.txt"
            )  # build output dir

            questions = process_exam_file(exam_file)  # load questions

            # Generate answers for each question
            with open(output_path, "w", encoding="utf-8") as f:
                for question in questions:
                    response = rag.query(
                        question, param=QueryParam(mode="hybrid")
                    )  # use LightRAG for answer
                    print(response)
                    f.write(f"QUESTION: {question}\n//// ANSWER: {response}\n\n")

            duration: float = time.time() - start_time  # end timer
            logging.info(
                f"Completed {exam_name} in {duration:.2f}s ({len(questions)} questions)"
            )

        except Exception as e:
            print(f"Failed processing {exam_file.name}: {str(e)}")


if __name__ == "__main__":
    rag = LightRAG(
        working_dir=WORKING_DIR,
        # Ollama model for text generation
        llm_model_func=ollama_model_complete,
        llm_model_name="deepseek-r1",
        llm_model_max_token_size=512,
        llm_model_kwargs={"reasoning_tag": "think"},
        # Use Ollama embedding function
        embedding_func=EmbeddingFunc(
            embedding_dim=768,
            max_token_size=512,
            func=lambda embedded_texts: ollama_embed(
                embedded_texts, embed_model="nomic-embed-text"
            ),
        ),
    )

    # Load and insert documents into LightRAG
    documents = load_documents(KNOWLEDGE_DIR)
    texts = [doc["text"] for doc in documents]
    for text in texts:
        print(text)
        rag.insert([text])

    # Process the exams
    process_exams(EXAM_DIR)
