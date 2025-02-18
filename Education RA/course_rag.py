import os
from pathlib import Path

import asyncio
import os
import inspect
from lightrag.llm.ollama import ollama_model_complete, ollama_embed
from lightrag.lightrag import LightRAG, QueryParam
from lightrag.utils import EmbeddingFunc
import logging

logging.basicConfig(level=logging.DEBUG)

# Path to documents
KNOWLEDGE_DIR = Path(r"AI_Course/Lecture_Notes").resolve()

# Path to exams
EXAM_DIR = Path(r"AI_Course/Exams").resolve()

# Output directory
OUTPUT_DIR = Path("AI_Course/Exams/generated_rag_answers").resolve()
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

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


# Function to generate answers using LightRAG
def generate_answer(question: str) -> str:
    try:
        # Query LightRAG with the question
        response = rag.query(question, param=QueryParam(mode="hybrid"))
        return response["result"]
    except Exception as e:
        return f"Error: {str(e)}"


# Function to process exam files and generate answers
def process_exams(exam_dir: Path, output_dir: Path):
    for exam_file in exam_dir.glob("*_answerless.txt"):
        try:
            exam_name = exam_file.stem.replace("_answerless", "")
            output_path = output_dir / f"{exam_name}_rag_answers.txt"

            # Load questions from the exam file
            with open(exam_file, "r", encoding="utf-8") as f:
                questions = [line.strip() for line in f if line.strip()]

            # Generate answers for each question
            with open(output_path, "w", encoding="utf-8") as f:
                for question in questions:
                    answer = generate_answer(question)
                    f.write(f"QUESTION: {question}\n//// ANSWER: {answer}\n\n")
        except Exception as e:
            print(f"Failed processing {exam_file.name}: {str(e)}")


rag = LightRAG(
    working_dir=WORKING_DIR,
    # Ollama model for text generation
    llm_model_func=ollama_model_complete,
    llm_model_name="deepseek-r1",
    llm_model_kwargs={"reasoning_tag": "think"},
    # Use Ollama embedding function
    embedding_func=EmbeddingFunc(
        embedding_dim=768,
        max_token_size=8192,
        func=lambda embedded_texts: ollama_embed(
            embedded_texts, embed_model="nomic-embed-text"
        ),
    ),
)

# Load and insert documents into LightRAG
documents = load_documents(KNOWLEDGE_DIR)
texts = [doc["text"] for doc in documents[:2]]
rag.insert(texts)

# Process the exams
process_exams(EXAM_DIR, OUTPUT_DIR)
