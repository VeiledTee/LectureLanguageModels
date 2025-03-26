import logging
import os
import re
import time
from pathlib import Path
from tqdm import tqdm

import ollama
import pinecone
from dotenv import load_dotenv
from pinecone import Pinecone

from chunking import RAGChunking
from preprocessing import load_markdown_sections, parse_quiz, process_exam_file

load_dotenv()

# Initialize logging
logging.basicConfig(level=logging.INFO)


class PineconeRAG:
    """
    A class that integrates Pinecone vector indexing with Ollama's language models to enable
    Retrieval Augmented Generation (RAG). It provides methods for processing and indexing
    documents, generating embeddings, retrieving relevant context, and generating answers
    with optional source citations.
    """

    def __init__(
        self,
        pinecone_client: pinecone.Pinecone,
        index_name: str,
        ollama_generation_model_name: str,
    ) -> None:
        """Initializes the PineconeRAG instance by setting up the Pinecone connection and index.

        Args:
            pinecone_client: Authenticated Pinecone client instance.
            index_name: Name of the index to create or use.
            ollama_generation_model_name: Name of the Ollama model for answer generation.
        """
        self.chunker = RAGChunking()
        self.generation_model = ollama_generation_model_name
        self.pinecone = pinecone_client
        self.top_k = int(os.getenv("RAG_TOP_K", "5"))
        self.index_name = os.getenv("RAG_INDEX_NAME", "ai-course-rag")
        self.embedding_model = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")
        self.sources = bool(int(os.getenv("INCLUDE_SOURCES", 0)))

        # Initialize index
        existing_indexes = [i.name for i in self.pinecone.list_indexes()]
        if self.index_name not in existing_indexes:
            self.pinecone.create_index(
                name=self.index_name,
                dimension=int(os.getenv("RAG_EMBEDDING_DIM", "768")),
                metric="cosine",
                spec=pinecone.ServerlessSpec(
                    cloud="aws", region=os.getenv("PINECONE_REGION", "us-east-1")
                ),
            )
        self.index = self.pinecone.Index(self.index_name)

        logging.info(f"Initialized Pinecone index {index_name}")
        logging.info(f"Initialized Pinecone RAG system with {self.generation_model}")

    def upsert_documents(self, directory: Path, batch_size: int = 50) -> None:
        """Processes and indexes markdown documents while preserving their structure.

        Args:
            directory: Path containing markdown documents.
            batch_size: Number of vectors per upsert batch.
        """
        logging.info("Indexing documents...")
        if not directory.exists():
            raise FileNotFoundError(f"Directory {directory} not found")
        start_time = time.time()
        batch = []
        for file_path in directory.glob("*.txt"):
            print(file_path)
            try:
                # Get filename for metadata
                filename = file_path.name

                # Use preprocessing.py's markdown parser
                sections = load_markdown_sections(str(file_path))

                if not sections:  # Empty dictionary case
                    with open(str(file_path), 'r', encoding='utf-8') as notes_file:
                        content = notes_file.read()
                        lines = content.split('\n')

                        if len(lines) > 1:  # Multi-line content
                            sections = {
                                           f"Path NA - Line {i + 1}": line.strip()
                                           for i, line in enumerate(lines)
                                           if line.strip()  # Skip empty lines
                                       } or {'Path NA': 'Empty file'}  # Fallback if all lines empty
                        else:  # Single-line or empty content
                            stripped = content.strip()
                            sections = {'Path NA': stripped or 'Empty file'}

                # Process hierarchical sections
                for section_path, content in sections.items():
                    # Generate chunks with filename in metadata
                    chunks = self._process_section(
                        content=content,
                        section_path=section_path,
                        filename=filename,
                    )
                    print(section_path, content)

                    # Add to batch with embeddings
                    batch.extend(self._create_vectors(chunks, filename))
                    print(len(batch))

                    # Upsert in configured batches
                    if len(batch) >= batch_size:
                        self._upsert_batch(batch)
                        batch = []

            except Exception as e:
                logging.error(f"Failed processing {filename}: {str(e)}")

        if batch:
            self._upsert_batch(batch)

        duration = time.time() - start_time

        # Convert duration to hh:mm:ss
        hours = int(duration // 3600)
        minutes = int((duration % 3600) // 60)
        seconds = int(duration % 60)
        logging.info(
            f"Document indexing complete in {hours:02}:{minutes:02}:{seconds:02}!"
        )

    def _process_section(
        self, content: str, section_path: str, filename: str
    ) -> list[dict]:
        """Processes a markdown section into chunks with associated structural metadata.

        Args:
            content: The markdown content of the section.
            section_path: The hierarchical path of the section.
            filename: Name of the file containing the section.

        Returns:
            A list of dictionaries, each representing a text chunk with metadata.
        """
        chunks = []
        content_type = self._detect_content_type(content)

        base_metadata = {
            "filename": filename,  # Explicit filename field
            "section_path": section_path,
            "content_type": content_type,
            "header_level": section_path.count(">") + 1,
        }

        # Split long sections using preprocessing-aware chunking
        chunk_size = int(os.getenv("RAG_CHUNK_SIZE", "512"))
        if len(content) > 1000:
            sub_chunks = self.chunker.recursive_chunk(content, chunk_size=chunk_size)
            chunks.extend(
                {
                    "text": chunk,
                    "metadata": {
                        **base_metadata,
                        "is_subchunk": True,
                        "parent_content": content[:200],
                    },
                }
                for chunk in sub_chunks
            )
        else:
            chunks.append({"text": content, "metadata": base_metadata})

        return chunks

    def _create_vectors(self, chunks: list[dict], filename: str) -> list[dict]:
        """Converts text chunks into Pinecone vector representations with embeddings.

        Args:
            chunks: List of text chunks with metadata.
            filename: Name of the file from which the chunks were generated.

        Returns:
            A list of dictionaries representing vectors with associated metadata.
        """
        texts = [chunk["text"] for chunk in chunks]
        embeddings = self.generate_embeddings(texts)

        return [
            {
                "id": f"{filename}-{idx}",  # Filename in ID
                "values": emb,
                "metadata": {
                    **chunk["metadata"],
                    "text": chunk["text"],
                    "document_source": filename,
                },
            }
            for idx, (chunk, emb) in enumerate(zip(chunks, embeddings))
        ]

    def generate_embeddings(self, texts: list[str]) -> list[list[float]]:
        """Generates embeddings for given texts using the Ollama model.

        Args:
            texts: A list of strings to generate embeddings for.

        Returns:
            A list of embeddings corresponding to the input texts.
        """
        if type(texts) == str:
            texts = [texts]
        try:
            response = ollama.embed(model=self.embedding_model, input=texts)
            embeddings = response.get("embeddings", [])
            if not embeddings:
                raise ValueError("No embeddings returned in the response.")
            return embeddings
        except Exception as e:
            logging.error(f"Embedding generation failed: {str(e)}")
            return []

    def _upsert_batch(self, batch: list[dict]) -> None:
        """Executes a batch upsert of vectors into the Pinecone index with error handling.

        Args:
            batch: List of vector dictionaries to be upserted.
        """
        try:
            self.index.upsert(vectors=batch)
            logging.info(f"Upserted batch of {len(batch)} vectors")
        except Exception as e:
            logging.error(f"Pinecone upsert failed: {str(e)}")

    def _detect_content_type(self, text: str) -> str:
        """Identifies the content type of the text based on preprocessing patterns.

        Args:
            text: The text to analyze.

        Returns:
            A string representing the detected content type.
        """
        if "```" in text:
            return "code_block"
        if re.match(r"^(#{1,6} |\* )", text):
            return "header_content"
        if any(q in text.lower() for q in ["question", "problem"]):
            return "potential_question"
        return "text_content"

    def retrieve_context(self, question: str, top_k: int = 3) -> tuple[list, list]:
        """Retrieves relevant context chunks and associated source filenames from the Pinecone index.

        Args:
            question: The user question to retrieve context for.
            top_k: The number of top matching vectors to retrieve.

        Returns:
            A tuple containing:
                - A list of context text segments.
                - A list of filenames corresponding to the sources of the context.
        """
        embedding = self.generate_embeddings([question])
        result = self.index.query(
            vector=embedding[0], top_k=top_k, include_metadata=True
        )

        # Extract both text and metadata for citations
        context_parts = []
        sources = set()
        for match in result.matches:
            meta = match.metadata
            context_parts.append(f"{meta['text']}")
            sources.add(meta["filename"])

        return context_parts, list(sources)

    def generate_answer(self, question: str) -> str:
        """
        Generates an answer to a user question using an Ollama LLM with context-aware prompting.

        Args:
            question: The user question to answer.
            temperature: Controls randomness (0.0-1.0, lower means more factual).
            max_tokens: Maximum length of the generated response.

        Returns:
            A formatted answer string with citations if source inclusion is enabled,
            otherwise the plain answer.
        """
        temperature = float(os.getenv("GENERATION_TEMPERATURE", "0.3"))
        max_tokens = int(os.getenv("MAX_TOKENS", "2048"))
        # Get context and sources for the question
        context, sources = self.retrieve_context(question)
        # Create structured prompt with context instructions
        prompt = f"""<|system|>
    You are an AI teaching assistant. 
    Use the following context to answer the question to the best of your knowledge.
    Provide a detailed answer from the context.
    If unsure, make your best guess. 
    Be definitive in your answer if the question calls for a yes or no response.
    Answer all parts of the question definitively.

    Background: """
        for content in context:
            prompt += f"{content}\n"
        prompt += f""" 
    </s>
    <|user|>
    {question}
    </s>
    <|assistant|>"""

        # Generate response through Ollama
        response = ollama.generate(
            model=self.generation_model,
            prompt=prompt,
            options={
                "temperature": temperature,
                "max_tokens": max_tokens,
                "top_p": 0.9,
                "stop": ["</s>", "\n\n\n"],
            },
        )

        # Extract and format response
        if "deepseek" not in self.generation_model:
            answer = response.get("response", "").strip()
        else:
            answer = response.get("response", "").strip()
            answer = re.sub(r"<think>.*?</think>\n?", "", answer, flags=re.DOTALL)

        return self._add_citations(answer, sources) if self.sources else answer

    def _add_citations(self, answer: str, sources: list) -> str:
        """Appends source citations (filenames) to the generated answer.

        Args:
            answer: The generated answer text.
            sources: List of source filenames used to retrieve context.

        Returns:
            The answer string appended with source citations.
        """
        if not sources:
            return answer

        source_list = "\n".join(src for src in sorted(sources))
        return f"{answer}\n\nSources:\n{source_list}"

    def delete_index(self) -> None:
        """Disables deletion protection (if enabled) and deletes the Pinecone index.

        Raises:
            Logs an error if deletion fails.
        """
        logging.info(f"Deleting index {self.index_name}...")

        try:
            # Disable deletion protection
            self.pinecone.configure_index(
                self.index_name, deletion_protection="disabled"
            )
            logging.info(f"Deletion protection disabled for index '{self.index_name}'.")

            # Delete the index
            self.pinecone.delete_index(self.index_name)
            logging.info(f"Index '{self.index_name}' has been deleted.")
        except Exception as e:
            logging.error(f"Failed to delete index '{self.index_name}': {str(e)}")


def process_exams(rag_pipeline: PineconeRAG):
    """
    Processes exam files to generate answers using a RAG pipeline.

    Iterates over exam files in EXAM_DIR matching the pattern '*_answerless.txt',
    processes the exam file to extract questions, and uses the provided
    PineconeRAG pipeline to generate answers. The answers are written to an output
    file corresponding to each exam.

    Args:
        rag_pipeline: An instance of PineconeRAG used to generate answers.

    Side Effects:
        Creates output files in OUTPUT_DIR and logs processing duration.
    """
    logging.info(
        f"Processing exams with Pinecone and Ollama queries and {rag_pipeline.generation_model}..."
    )

    exam_dir = Path(os.getenv("EXAM_DIR", "Artificial_Intelligence/Exams"))
    output_dir = Path(
        os.getenv("ANSWER_DIR", "Artificial_Intelligence/Exams/generated_answers")
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    for exam_file in exam_dir.glob("*_answerless.txt"):
        try:
            start_time = time.time()
            exam_name = exam_file.stem.replace("_answerless", "")
            output_path = (
                output_dir
                / f"{exam_name}_{rag_pipeline.generation_model.split(':')[0]}_rag_answers.txt"
            )

            questions = process_exam_file(exam_file)

            with open(output_path, "w", encoding="utf-8") as f:
                for question in tqdm(
                    questions, desc="Generating answers", unit="question"
                ):
                    answer = rag_pipeline.generate_answer(question)
                    f.write(f"QUESTION: {question}\n//// ANSWER: {answer}\n\n")
                    f.flush()  # write answer immediately

            duration = time.time() - start_time

            # Convert duration to hh:mm:ss
            hours = int(duration // 3600)
            minutes = int((duration % 3600) // 60)
            seconds = int(duration % 60)
            logging.info(f"{model} exam complete {hours:02}:{minutes:02}:{seconds:02}!")

        except Exception as e:
            logging.error(f"Error processing {exam_file}: {str(e)}")


if __name__ == "__main__":
    pinecone_index_name: str = str(os.getenv("RAG_INDEX_NAME"))

    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    models = [m.strip() for m in os.getenv("GENERATION_MODELS", "").split(",")]

    for model in models:
        chunker: RAGChunking = RAGChunking()
        pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        # initialize
        rag = PineconeRAG(
            pinecone_client=pc,
            index_name=pinecone_index_name,
            ollama_generation_model_name=model,
        )

        # clean slate
        rag.delete_index()

        # reinitialize
        rag = PineconeRAG(
            pinecone_client=pc,
            index_name=pinecone_index_name,
            ollama_generation_model_name=model,
        )

        # Index documents
        rag.upsert_documents(Path(os.getenv("KNOWLEDGE_DIR")))

        # Process exams
        process_exams(rag)
