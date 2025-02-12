import re

import torch
from langchain.text_splitter import (
    CharacterTextSplitter,
    MarkdownTextSplitter,
    RecursiveCharacterTextSplitter,
)
from nltk.tokenize import sent_tokenize
from sklearn.cluster import DBSCAN
from transformers import AutoModel, AutoTokenizer, pipeline


class RAGChunking:
    """
    A collection of text chunking methods for Retrieval-Augmented Generation (RAG) systems.

    Provides various strategies for splitting documents into chunks suitable for processing
    with LLMs and semantic search systems.
    """

    def __init__(self) -> None:
        """
        Initialize RAGChunking instance.
        """
        pass

    @staticmethod
    def fixed_size_chunk(
        text: str, chunk_size: int = 512, overlap: int = 50
    ) -> list[str]:
        """
        Split text into fixed-size chunks with specified overlap.

        Args:
            text: Input text to be chunked
            chunk_size: Desired character length of each chunk (default: 512)
            overlap: Number of overlapping characters between chunks (default: 50)

        Returns:
            list of text chunks with specified size and overlap

        Note:
            Uses langchain's CharacterTextSplitter with newline separator
        """
        splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=chunk_size,
            chunk_overlap=overlap,
            length_function=len,
        )
        return splitter.split_text(text)

    @staticmethod
    def sentence_chunk(text: str, sentences_per_chunk: int = 5) -> list[str]:
        """
        Split text into chunks containing a fixed number of sentences.

        Args:
            text: Input text to be chunked
            sentences_per_chunk: Number of sentences per chunk (default: 5)

        Returns:
            list of text chunks containing specified number of sentences

        Note:
            Uses NLTK's sentence tokenizer for sentence splitting
        """
        sentences = sent_tokenize(text)
        chunks = []
        for i in range(0, len(sentences), sentences_per_chunk):
            chunk = " ".join(sentences[i : i + sentences_per_chunk])
            chunks.append(chunk)
        return chunks

    @staticmethod
    def paragraph_chunk(text: str) -> list[str]:
        """
        Split text into paragraphs using multiple newlines as separators.

        Args:
            text: Input text to be chunked

        Returns:
            list of paragraphs with preserved formatting

        Note:
            Splits on two or more consecutive newlines, markdown-friendly
        """
        paragraphs = re.split(r"\n{2,}", text)
        return [p.strip() for p in paragraphs if p.strip()]

    @staticmethod
    def recursive_chunk(text: str, chunk_size: int = 512) -> list[str]:
        """
        Split text recursively using hierarchical separators.

        Args:
            text: Input text to be chunked
            chunk_size: Target character length for chunks (default: 512)

        Returns:
            list of recursively split text chunks

        Note:
            Uses langchain's RecursiveCharacterTextSplitter which tries to preserve
            paragraph/sentence boundaries
        """
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=20, length_function=len
        )
        return splitter.split_text(text)

    @staticmethod
    def markdown_header_chunk(text: str) -> list[str]:
        """
        Split markdown text based on header boundaries.

        Args:
            text: Markdown-formatted text input

        Returns:
            list of chunks split at header boundaries

        Note:
            Uses langchain's MarkdownTextSplitter for header-aware splitting
        """
        splitter = MarkdownTextSplitter()
        return splitter.split_text(text)

    @staticmethod
    def semantic_chunk(
        text: str, threshold: float = 0.85, model_name: str = "all-MiniLM-L6-v2"
    ) -> list[str]:
        """
        Split text into semantically coherent chunks using sentence embeddings.

        Args:
            text: Input text to be chunked
            threshold: Cosine similarity threshold for grouping sentences (default: 0.85)
            model_name: Name of sentence-transformers model (default: 'all-MiniLM-L6-v2')

        Returns:
            list of chunks grouped by semantic similarity

        Note:
            Requires sentence-transformers and numpy packages installed.
            Uses cosine similarity between consecutive sentence embeddings.
        """
        import numpy as np
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(model_name)
        sentences = sent_tokenize(text)
        embeddings = model.encode(sentences)

        chunks = []
        current_chunk = []

        for i in range(1, len(sentences)):
            similarity = np.dot(embeddings[i - 1], embeddings[i])
            if similarity < threshold:
                chunks.append(" ".join(current_chunk))
                current_chunk = [sentences[i]]
            else:
                current_chunk.append(sentences[i])
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        return chunks

    @staticmethod
    def sliding_window_chunk(
        text: str, window_size: int = 512, stride: int = 256
    ) -> list[str]:
        """
        Split text using sliding window approach with specified stride.

        Args:
            text: Input text to be chunked
            window_size: Character length of each window (default: 512)
            stride: Number of characters to slide the window (default: 256)

        Returns:
            list of overlapping text chunks

        Note:
            Implemented using CharacterTextSplitter with space separator
        """
        splitter = CharacterTextSplitter(
            chunk_size=window_size, chunk_overlap=stride, separator=" "
        )
        return splitter.split_text(text)

    @staticmethod
    def hybrid_markdown_chunk(text: str) -> list[str]:
        """
        Split markdown text into header-content sections.

        Args:
            text: Markdown-formatted text input

        Returns:
            list of chunks where each header is followed by its content

        Note:
            Creates separate chunks for each header line and its subsequent content.
            Headers are defined by lines starting with '#' characters.
        """
        sections = []
        current_section = []

        for line in text.split("\n"):
            if line.startswith("#"):
                if current_section:
                    sections.append("\n".join(current_section))
                    current_section = []
                sections.append(line)
            else:
                current_section.append(line)

        if current_section:
            sections.append("\n".join(current_section))

        return sections

    @staticmethod
    def dbscan_semantic_chunk(
        text: str,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        eps: float = 0.75,
        min_samples: int = 1,
    ) -> list[str]:
        """
        Cluster sentences using DBSCAN based on semantic embeddings.

        Args:
            text: Input text to chunk
            model_name: HuggingFace model for embeddings (default: 'all-MiniLM-L6-v2')
            eps: DBSCAN epsilon parameter (default: 0.75)
            min_samples: DBSCAN min_samples parameter (default: 1)

        Returns:
            list of semantically coherent text chunks

        Note:
            Requires sklearn and torch installed. Uses mean-pooled embeddings.
        """
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)

        sentences = RAGChunking.sentence_chunk(text, sentences_per_chunk=1)
        inputs = tokenizer(
            sentences, return_tensors="pt", padding=True, truncation=True
        )

        with torch.no_grad():
            outputs = model(**inputs)

        embeddings = outputs.last_hidden_state.mean(dim=1).numpy()
        clustering = DBSCAN(eps=eps, min_samples=min_samples, metric="cosine").fit(
            embeddings
        )

        clusters: dict[int, list[str]] = {}
        for idx, label in enumerate(clustering.labels_):
            clusters.setdefault(label, []).append(sentences[idx])

        return [" ".join(cluster) for cluster in clusters.values()]

    @staticmethod
    def summarization_adaptive_chunk(
        text: str,
        model_name: str = "sshleifer/distilbart-cnn-12-6",
        max_length: int = 50,
        min_length: int = 25,
    ) -> list[str]:
        """
        Create chunks through content-aware summarization.

        Args:
            text: Input text to chunk
            model_name: Summarization model name (default: distilbart-cnn)
            max_length: Maximum summary length per chunk
            min_length: Minimum summary length per chunk

        Returns:
            list of summary-based chunks

        Note:
            Requires transformers pipeline. Better for extractive chunking.
        """
        summarizer = pipeline("summarization", model=model_name)
        base_chunks = RAGChunking.context_aware_chunk(text)

        summaries = []
        for chunk in base_chunks:
            try:
                summary = summarizer(
                    chunk, max_length=max_length, min_length=min_length, do_sample=False
                )
                summaries.append(summary[0]["summary_text"])
            except Exception as e:
                summaries.append(chunk[:min_length])

        return summaries

    @staticmethod
    def enhanced_overlap_chunk(
        text: str, chunk_size: int = 512, overlap: int = 128
    ) -> list[str]:
        """
        Improved overlap chunking with edge case handling.

        Args:
            text: Input text to chunk
            chunk_size: Desired chunk length
            overlap: Overlap between chunks

        Returns:
            list of overlapping text chunks

        Note:
            Ensures complete coverage of text with proper overlap handling
        """
        if overlap >= chunk_size:
            raise ValueError("Overlap must be smaller than chunk size")

        chunks = []
        pos = 0
        while pos < len(text):
            end = pos + chunk_size
            chunks.append(text[pos:end])
            pos += chunk_size - overlap

            # Handle final partial chunk
            if end >= len(text):
                break

        return chunks

    @staticmethod
    def context_aware_chunk(text: str) -> list[str]:
        """
        Clean text and split into context-aware chunks (paragraphs/sentences).

        Args:
            text: Input text to process

        Returns:
            list of cleaned context-based chunks
        """
        cleaned = re.sub(r"\n{2,}", "\n", text).replace("<!-- image -->", "")
        return RAGChunking.paragraph_chunk(cleaned)


if __name__ == "__main__":
    DBSCAN_EPSILON: float = 0.6
    COSINE_THRESHOLD: float = 0.9

    chunker = RAGChunking()
    with open("AI_Course/Lecture_Notes/ch3_csp_games1_parsed.txt", "r") as f:
        markdown_content = f.read()

    # Create strategy demonstrations with different parameter combinations
    strategies = {
        # Core strategies
        "Fixed Size (200c/50o)": chunker.fixed_size_chunk(markdown_content, 200, 50),
        "Sentence-Based (3 sent)": chunker.sentence_chunk(markdown_content, 3),
        "Paragraph-Based": chunker.paragraph_chunk(markdown_content),
        # Structural strategies
        "Recursive (150c)": chunker.recursive_chunk(markdown_content, 150),
        "Markdown Header": chunker.markdown_header_chunk(markdown_content),
        "Hybrid Markdown": chunker.hybrid_markdown_chunk(markdown_content),
        # Semantic strategies
        f"Semantic Similarity ({COSINE_THRESHOLD})": chunker.semantic_chunk(
            markdown_content, COSINE_THRESHOLD
        ),
        f"DBSCAN Semantic ({DBSCAN_EPSILON})": chunker.dbscan_semantic_chunk(
            markdown_content, eps=DBSCAN_EPSILON
        ),
        # Window/Overlap strategies
        "Sliding Window (100/50)": chunker.sliding_window_chunk(
            markdown_content, 100, 50
        ),
        "Enhanced Overlap (300/100)": chunker.enhanced_overlap_chunk(
            markdown_content, 300, 100
        ),
        # Adaptive strategies
        "Summarization Adaptive": chunker.summarization_adaptive_chunk(
            markdown_content
        ),
        "Context Aware": chunker.context_aware_chunk(markdown_content),
    }

    # Display formatted results
    for strategy_name, chunks in strategies.items():
        print(
            f"\n{'-' * 60}\n🏷️ {strategy_name} Chunks ({len(chunks)} total)\n{'-' * 60}"
        )
        for i, chunk in enumerate(chunks[:3], 1):  # Show first 3 chunks
            preview = chunk[:150] + "..." if len(chunk) > 150 else chunk
            print(f"📦 Chunk {i} [Length: {len(chunk):,} chars]\n{preview}\n")
