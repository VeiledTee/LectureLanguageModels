import os
from pathlib import Path

from lightrag import LightRAG, QueryParam
from lightrag.llm.hf import hf_embed, hf_model_complete
from lightrag.utils import EmbeddingFunc
from transformers import AutoModel, AutoTokenizer

# Path to documents
KNOWLEDGE_DIR = Path(
    "/home/penguins/Documents/PhD/LectureLanguageModels/Education RA/AI_Course/Lecture_Notes"
).resolve()

# Path to exams
EXAM_DIR = Path(
    "/home/penguins/Documents/PhD/LectureLanguageModels/Education RA/AI_Course/Exams"
).resolve()


# Output directory
OUTPUT_DIR = Path(
    "/home/penguins/Documents/PhD/LectureLanguageModels/Education RA/AI_Course/Exams/generated_rag_answers"
).resolve()
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)


# Define the working directory for LightRAG
WORKING_DIR = "./lightrag_working_dir"
Path(WORKING_DIR).mkdir(exist_ok=True)


# Function to load documents from the knowledge directory
def load_documents(directory: Path):
    documents = []
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            file_path = directory / filename
            with open(file_path, "r", encoding="utf-8") as file:
                content = file.read()
                documents.append({"id": filename, "text": content})
    return documents


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


# Initialize the Hugging Face embedding model
embedding_model_name = "sentence-transformers/all-MiniLM-L6-v2"
tokenizer = AutoTokenizer.from_pretrained(embedding_model_name)
embed_model = AutoModel.from_pretrained(embedding_model_name)

# Initialize LightRAG with Hugging Face models
rag = LightRAG(
    working_dir=WORKING_DIR,
    llm_model_func=hf_model_complete,
    llm_model_name="HuggingFaceTB/SmolLM-1.7B-Instruct",
    embedding_func=EmbeddingFunc(
        embedding_dim=384,
        max_token_size=5000,
        func=lambda texts: hf_embed(
            texts, tokenizer=tokenizer, embed_model=embed_model
        ),
    ),
)

# Load and insert documents into LightRAG
documents = load_documents(KNOWLEDGE_DIR)
texts = [doc["text"] for doc in documents]
rag.insert(texts)

# Process the exams
process_exams(EXAM_DIR, OUTPUT_DIR)
