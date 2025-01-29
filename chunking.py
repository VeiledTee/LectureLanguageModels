import time
import os
import re
from transformers import pipeline
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.cluster import DBSCAN


def fixed_size_chunking(text: str, chunk_size: int) -> list[str]:
    """Chunks text into fixed-size segments."""
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]


def context_aware_chunking(text: str) -> list[str]:
    """Chunks text into segments based on sentences or paragraphs."""
    paragraphs: list[str] = text.split('\n')
    chunks: list[str] = []
    for para in paragraphs:
        para = para.replace('\n', ' ').replace('<!-- image -->', '')
        sentences: str = re.split(r'(?<=[.!?]) +', para)
        chunks.extend(sentences)
    return chunks


def recursive_chunking(text: str) -> list[str]:
    """Chunks text hierarchically based on paragraphs, then sentences."""
    paragraphs: list[str] = text.split('\n')
    chunks: list[str] = []
    for para in paragraphs:
        para = para.replace('\n', ' ').replace('<!-- image -->', '')
        sentences = re.split(r'(?<=[.!?]) +', para)
        chunks.extend(sentences)
    return chunks


def semantic_chunking(text: str, model_name: str) -> list[str]:
    """Chunks text based on semantic similarity using a provided model."""
    tokenizer: AutoTokenizer = AutoTokenizer.from_pretrained(model_name)
    model: AutoModel = AutoModel.from_pretrained(model_name)
    sentences: list[str] = context_aware_chunking(text)
    inputs: dict[str, torch.Tensor] = tokenizer(sentences, return_tensors='pt', padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings: torch.Tensor = outputs.last_hidden_state.mean(dim=1)
    clustering = DBSCAN(eps=0.75, min_samples=1, metric='cosine').fit(embeddings.numpy())
    clusters: dict[int, list[str]] = {}
    for idx, label in enumerate(clustering.labels_):
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(sentences[idx])
    return [' '.join(cluster) for cluster in clusters.values()]


def adaptive_chunking(text: str) -> list[str]:
    """Dynamically determines chunk sizes and boundaries based on context."""
    summarizer = pipeline("summarization")
    chunks = context_aware_chunking(text)
    adaptive_chunks = []
    for chunk in chunks:
        summary = summarizer(chunk, max_length=50, min_length=25, do_sample=False)
        adaptive_chunks.append(summary[0]['summary_text'])
    return adaptive_chunks


def overlap_chunking(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Chunks text with overlapping segments."""
    chunks = []
    for i in range(0, len(text) - chunk_size + 1, chunk_size - overlap):
        chunks.append(text[i:i + chunk_size])
        if i + chunk_size < len(text):
            chunks.append(text[i + chunk_size - overlap:])
    return chunks


# DIRECTORY = "/home/penguins/Documents/PhD/Education RA/AI Course/Exams"
DIRECTORY = "/home/penguins/Documents/PhD/Education RA/AI Course/Lecture Notes"

if __name__ == '__main__':
    for filename in os.listdir(DIRECTORY):
        if filename.endswith(".txt"):
            file_path: str = '/home/penguins/Documents/PhD/Education RA/AI Course/Lecture Notes/ch2_search1_parsed.txt'
            with open(file_path, 'r') as file:
                file_text: str = file.read().strip()

            CHUNK_SIZE: int = 100
            OVERLAP: int = 20
            MODEL_NAME: str = 'sentence-transformers/all-MiniLM-L6-v2'

            print(f"Processing file: {filename}")
            print("Fixed Size Chunks:", fixed_size_chunking(file_text, CHUNK_SIZE))
            time.sleep(10)
            print("Context Aware Chunks:", context_aware_chunking(file_text))
            time.sleep(10)
            print("Recursive Chunks:", recursive_chunking(file_text))
            time.sleep(10)
            print("Overlap Chunks:", overlap_chunking(file_text, CHUNK_SIZE, OVERLAP))
            time.sleep(10)
            print("Semantic Chunks:", semantic_chunking(file_text, MODEL_NAME))
            time.sleep(10)
            print("Adaptive Chunks:", adaptive_chunking(file_text))

            print("\n" + "=" * 80 + "\n")
            break
