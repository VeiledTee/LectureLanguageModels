import logging
import os
import re
import json
from pathlib import Path
from typing import Any, Optional

import aiofiles
import anyio
import openai
from docling.document_converter import DocumentConverter
from dotenv import load_dotenv
from pdf2image import convert_from_path
from pydantic import BaseModel
from io import BytesIO
import base64
from PIL import Image

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger: logging.Logger = logging.getLogger(__name__)

load_dotenv()


class Config:
    """Centralized configuration manager with validation.

    Attributes:
        openai_api_key (str): OpenAI API key for AI operations
        model_name (str): Name of the OpenAI model to use
        image_format (str): Image format for PDF conversion
        image_quality (int): Quality setting for image conversion
        process_lectures (bool): Flag to enable lecture processing
        process_exams (bool): Flag to enable exam processing
        max_concurrency (int): Maximum concurrent processing tasks
        directories (dict[str, Path]): Paths to various processing directories
        excluded_headers (list[str]): Headers to exclude from markdown processing
        image_patterns (list[str]): Regex patterns for image detection in markdown
    """

    def __init__(self) -> None:
        """Initialize configuration from environment variables."""
        load_dotenv()

        # Core settings
        self.openai_api_key: Optional[str] = os.getenv("OPENAI_API_KEY")
        openai.api_type = os.getenv("OPENAI_API_TYPE")
        self.model_name: str = os.getenv("OPENAI_MODEL_NAME", "gpt-4o")
        self.image_format: str = os.getenv("IMAGE_FORMAT", "JPEG").upper()
        self.image_quality: int = int(os.getenv("IMAGE_QUALITY", 85))

        # Processing flags
        self.process_lectures: bool = bool(int(os.getenv("PROCESS_LECTURES", 0)))
        self.process_exams: bool = bool(int(os.getenv("PROCESS_EXAMS", 0)))
        self.max_concurrency: int = int(os.getenv("MAX_CONCURRENCY", 5))

        # Directories
        self.directories: dict[str, Path] = {
            'lecture_pdf': Path(os.getenv("LECTURE_PDF_DIR", "AI_Course/Lecture_Notes/Source_PDFs")),
            'lecture_image_output': Path(os.getenv("LECTURE_OUTPUT_DIR", "AI_Course/Lecture_Notes/Processed")),
            'exam_pdf': Path(os.getenv("EXAM_PDF_DIR", "AI_Course/Exams")),
            'exam_image_output': Path(os.getenv("EXAM_OUTPUT_DIR", "AI_Course/Exams")),
            'knowledge_dir': Path(os.getenv("KNOWLEDGE_DIR")),
            'exam_dir': Path(os.getenv("EXAM_DIR")),
        }

        # Processing parameters
        self.excluded_headers: list[str] = os.getenv(
            "EXCLUDED_HEADERS", "## Slide,## 6.034").split(",")
        self.image_patterns: list[str] = [
            r"(?:!\[.*?\]\(.*?\))",  # Markdown images
            r"(?:<!--\s*image[^>]*-->)",  # Image comments
            r"(?:<img.*?>)",  # HTML images
        ]

        self.validate()
        self.create_directories()

    def validate(self) -> None:
        """Validate configuration values.

        Raises:
            ValueError: For invalid image formats or quality values
            RuntimeError: If required API key is missing
        """
        if not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY is required in environment")
        if self.image_format not in {"JPEG", "PNG"}:
            raise ValueError(f"Invalid IMAGE_FORMAT: {self.image_format}. Must be JPEG or PNG")
        if self.image_quality < 1 or self.image_quality > 100:
            raise ValueError("IMAGE_QUALITY must be between 1 and 100")

    def create_directories(self) -> None:
        """Create all required directories if they don't exist."""
        for dir_path in self.directories.values():
            dir_path.mkdir(parents=True, exist_ok=True)


config: Config = Config()


def image_to_base64(image: Image.Image) -> str:
    """Convert a PIL Image to a base64 encoded string using config settings."""
    buffered = BytesIO()
    image.save(buffered, format=config.image_format, quality=config.image_quality)
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return f"data:image/{config.image_format.lower()};base64,{img_str}"


class ClassNotes(BaseModel):
    text: str


async def convert_pdf_to_images(pdf_path: Path, file_type: str) -> int:
    """Convert PDF pages to images with parallel processing.

    Args:
        pdf_path: Path to source PDF file
        file_type: Type of PDF ('lecture' or 'exam')

    Returns:
        int: Number of new pages processed (0 if file was quarantined)

    Raises:
        RuntimeError: If conversion fails and file is quarantined
    """
    # Determine output directory based on file type
    if file_type == 'lecture':
        output_dir = config.directories['lecture_image_output'] / pdf_path.stem
    else:
        output_dir = config.directories['exam_image_output'] / pdf_path.stem
    output_dir.mkdir(parents=True, exist_ok=True)

    existing_pages: set[str] = {f.name for f in output_dir.glob("page_*.*")}
    processed: int = 0

    try:
        images: list[Any] = convert_from_path(
            str(pdf_path),
            dpi=200,
            thread_count=4,
            fmt=config.image_format.lower(),
            poppler_path=r"C:\poppler\Library\bin"
        )

        for i, image in enumerate(images, 1):
            page_name: str = f"page_{i:03d}.{config.image_format.lower()}"
            if page_name not in existing_pages:
                image.save(output_dir / page_name, config.image_format, quality=config.image_quality)
                processed += 1
                logger.info(f"Converted page {i} of {pdf_path.name}")

        return processed

    except Exception as e:
        logger.error(f"Error processing {pdf_path}: {str(e)}")
        quarantine_path: Path = config.directories['quarantine'] / pdf_path.name
        pdf_path.rename(quarantine_path)
        logger.warning(f"Moved {pdf_path.name} to quarantine")
        return 0


async def process_pdfs() -> None:
    """Orchestrate PDF processing based on configuration flags."""
    pdf_files: list[tuple[Path, str]] = []  # (PDF path, type)

    # Collect lecture PDFs if enabled
    if config.process_lectures:
        lecture_files: list[Path] = sorted(config.directories['lecture_pdf'].glob("*.pdf"))
        pdf_files += [(path, 'lecture') for path in lecture_files]
        logger.debug(f"Found {len(lecture_files)} lecture PDFs")

    # Collect exam PDFs if enabled
    if config.process_exams:
        exam_files: list[Path] = sorted(config.directories['exam_pdf'].glob("*.pdf"))
        pdf_files += [(path, 'exam') for path in exam_files]
        logger.debug(f"Found {len(exam_files)} exam PDFs")

    logger.info(f"Total PDFs to process: {len(pdf_files)}")

    async with anyio.create_task_group() as tg:
        for pdf_file, file_type in pdf_files:
            tg.start_soon(convert_pdf_to_images, pdf_file, file_type)


def format_question(header: str, content: str) -> str:
    """Format a question string from header and content sections.

    Args:
        header: Section header text
        content: Associated content text

    Returns:
        str: Formatted question string with proper punctuation

    Example:
        format_question("Section 1", "What is AI?")
        "Section 1 > What is AI:"
    """
    question: str = f"{header} > {content}" if content else header
    return question.rstrip(":") + ":" if not question.endswith(":") else question


async def process_exam_file(file_path: Path) -> list[str]:
    """Process exam file without answers into formatted questions.

    Args:
        file_path: Path to exam markdown file

    Returns:
        list[str]: list of parsed questions or empty list on error
    """
    try:
        async with aiofiles.open(file_path, "r", encoding="utf-8") as f:
            content: str = await f.read()
            return parse_quiz(content)
    except Exception as e:
        logger.error(f"Error processing {file_path}: {str(e)}")
        return []


async def pdf_to_markdown(pdf_path: Path, output_path: Path) -> bool:
    """Convert PDF to cleaned markdown with configurable patterns.

    Args:
        pdf_path: Source PDF file path
        output_path: Target markdown file path

    Returns:
        bool: True if conversion succeeded, False otherwise

    Raises:
        IOError: If file operations fail
    """
    try:
        doc_converter: DocumentConverter = DocumentConverter()
        result: Any = doc_converter.convert(str(pdf_path))
        markdown_text: str = result.document.export_to_markdown()

        # Clean content using configured patterns
        image_pattern: re.Pattern = re.compile("|".join(config.image_patterns), re.IGNORECASE)
        processed_lines: list[str] = []
        in_ignore_block: bool = False

        for line in markdown_text.splitlines():
            line = image_pattern.sub("", line).strip()
            if not line:
                continue

            if "<!-- image" in line.lower():
                in_ignore_block = True
                continue

            if in_ignore_block and line.startswith("#"):
                in_ignore_block = False

            if not in_ignore_block and not any(line.startswith(h) for h in config.excluded_headers):
                processed_lines.append(line)

        async with aiofiles.open(output_path, "w", encoding="utf-8") as f:
            await f.write("\n".join(processed_lines))

        return True

    except Exception as e:
        logger.error(f"Failed to convert {pdf_path}: {str(e)}")
        return False


def extract_questions(extract_from: str) -> list[str]:
    """Extract deepest-level headers from markdown as questions.

    Args:
        extract_from: Markdown content to analyze

    Returns:
        list[str]: list of header texts at deepest hierarchy level

    Example:
        extract_questions("# Main\n## Sub\n### Question")
        ["Question"]
    """
    headers: list[tuple[int, str]] = []
    header_re: re.Pattern = re.compile(r"^(#+)\s+(.+?)(?:\(.*?\))?\s*$", re.MULTILINE)

    for match in header_re.finditer(extract_from):
        level: int = len(match.group(1))
        text: str = match.group(2).strip()
        headers.append((level, text))

    if not headers:
        return []

    max_level: int = max(level for level, _ in headers)
    return [text for level, text in headers if level == max_level]


def load_markdown_sections(file_path: str) -> dict[str, str]:
    """Parse markdown file into hierarchical sections.

    Args:
        file_path: Path to markdown file

    Returns:
        dict[str, str]: Mapping of section paths to content
        Format: {"Parent > Child": "content text"}

    Example:
        load_markdown_sections("doc.md")
        {"Introduction": "Welcome text", "Chapter 1 > Section 1": "Content..."}
    """
    header_regex: re.Pattern = re.compile(r"^(#{1,6})\s+(.*?)(?:\(.*?\))?\s*$")
    markdown_sections: dict[str, str] = {}
    stack: list[dict[str, Any]] = [{"level": 0, "title": "Root", "content": []}]

    with open(file_path, "r", encoding="utf-8") as markdown_file:
        for line_num, line in enumerate(markdown_file, 1):
            line = line.rstrip("\n")
            m = header_regex.match(line)

            if m:
                level: int = len(m.group(1))
                title: str = m.group(2).strip()

                while stack and stack[-1]["level"] >= level:
                    popped: dict[str, Any] = stack.pop()
                    if popped["level"] >= 2:
                        key_parts: list[str] = [s["title"] for s in stack[1:]]
                        key_parts.append(popped["title"])
                        key: str = " > ".join(key_parts)
                        markdown_sections[key] = "\n".join(popped["content"]).strip()

                new_section: dict[str, Any] = {"level": level, "title": title, "content": []}
                stack.append(new_section)
            else:
                if stack:
                    stack[-1]["content"].append(line)

    # Process remaining sections
    while stack:
        popped = stack.pop()
        if popped["level"] >= 2:
            key_parts = [s["title"] for s in stack[1:]]
            key_parts.append(popped["title"])
            key = " > ".join(key_parts)
            markdown_sections[key] = "\n".join(popped["content"]).strip()

    return markdown_sections


def parse_quiz(md_text: str) -> list[str]:
    """Parse hierarchical quiz structure from markdown content.

    Args:
        md_text: Markdown content with quiz questions

    Returns:
        list[str]: list of question paths in "Section > Subsection > Question" format

    Example:
        parse_quiz("# Quiz\n## Q1\nWhat is?")
        ["Quiz > Q1 > What is?"]
    """
    lines: list[str] = md_text.splitlines()
    root: list[dict[str, Any]] = []
    stack: list[dict[str, Any]] = []

    for line in lines:
        line = line.rstrip()
        if not line.strip():
            continue
        if line.startswith("#"):
            level: int = 0
            while level < len(line) and line[level] == "#":
                level += 1
            header: str = line[level:].strip()
            node: dict[str, Any] = {
                "level": level,
                "header": header,
                "content": "",
                "children": [],
            }
            while stack and stack[-1]["level"] >= level:
                stack.pop()
            if stack:
                stack[-1]["children"].append(node)
            else:
                root.append(node)
            stack.append(node)
        else:
            if stack:
                current: dict[str, Any] = stack[-1]
                current["content"] = (
                    (current["content"] + " " + line.strip()).strip()
                    if current["content"]
                    else line.strip()
                )

    questions: list[str] = []

    def traverse(path: list[str], graph_node: dict[str, Any]) -> None:
        """Recursively build question paths."""
        text: str = graph_node["header"]
        if graph_node["content"]:
            text += " " + graph_node["content"]
        new_path: list[str] = path + [text]
        if not graph_node["children"]:
            questions.append(" > ".join(new_path))
        else:
            for child in graph_node["children"]:
                traverse(new_path, child)

    for node in root:
        traverse([], node)

    return questions


async def process_images_directory(input_dir: Path, output_dir: Path) -> None:
    """Process all images in a given directory (including subdirectories),
    extract text via OpenAI API, and save results as Markdown.

    Args:
        input_dir: Root directory containing images to process.
        output_dir: Directory where the output Markdown file will be saved.
    """
    results = []
    output_dir.mkdir(parents=True, exist_ok=True)
    valid_extensions = {".jpeg", ".jpg"} if config.image_format.upper() == "JPEG" else {".png"}

    for image_file in input_dir.rglob("*"):
        if image_file.is_file() and image_file.suffix.lower() in valid_extensions:
            try:
                image = Image.open(image_file)
                img_base64 = image_to_base64(image)
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {"url": img_base64}
                            },
                            {
                                "type": "text",
                                "text": "Extract all text from this image, including any figures."
                            }
                        ]
                    }
                ]
                response = await anyio.to_thread.run_sync(
                    lambda: openai.beta.chat.completions.parse(
                        model=config.model_name,
                        response_format=ClassNotes,
                        messages=messages
                    )
                )
                extracted = response.choices[0].message.parsed
                results.append({
                    "file": str(image_file),
                    "text": extracted.text
                })
                logger.info(f"Processed image {image_file}")
            except Exception as e:
                logger.error(f"Error processing image {image_file}: {e}")

    output_file = output_dir / "pdf_images_to_text.md"
    markdown_output = "\n".join(f"## {entry['file']}\n\n{entry['text']}\n" for entry in results)

    try:
        async with aiofiles.open(output_file, "w", encoding="utf-8") as f:
            await f.write(markdown_output)
        logger.info(f"Saved extracted text to {output_file}")
    except Exception as e:
        logger.error(f"Error writing output file {output_file}: {e}")


async def main() -> None:
    """Main orchestration function."""
    await process_pdfs()
    # To process a separate directory of images, uncomment and adjust the following lines:
    if config.process_exams:
        await process_images_directory(config.directories['exam_image_output'], config.directories['exam_dir'])
    if config.process_lectures:
        await process_images_directory(config.directories['lecture_image_output'], config.directories['knowledge_dir'])


if __name__ == "__main__":
    anyio.run(main)
    # for directory in ["AI_Course/Exams"]:
    #     # Create a converter instance
    #     converter = DocumentConverter()
    #
    #     for filename in os.listdir(directory):
    #         source_path = os.path.join(directory, filename)  # found for every file
    #
    #         # if filename.endswith(".pdf"):  # filter by .pdf extension
    #         #     output_path = os.path.join(
    #         #         directory, filename.split(".")[0] + "_parsed.txt"
    #         #     )
    #         #     pdf_to_markdown(
    #         #         source_path, output_path
    #         #     )  # convert pdf to markdown file and save in directory with '_parsed' suffix
    #
    #         if filename.endswith("_answerless.txt"):
    #             with open(source_path, "r", encoding="utf-8") as f:
    #                 input_text = f.read()
    #                 questions = parse_quiz(input_text)
    #                 print(f"File: {filename}, Questions Found: {len(questions)}")
    #                 # for i in questions:
    #                 #     print(i)
