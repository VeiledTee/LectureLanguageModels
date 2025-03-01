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
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger: logging.Logger = logging.getLogger(__name__)

load_dotenv()
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"


class Config:
    def __init__(self) -> None:
        load_dotenv()
        self.openai_api_key: Optional[str] = os.getenv("OPENAI_API_KEY")
        openai.api_type = os.getenv("OPENAI_API_TYPE")
        self.model_name: str = os.getenv("OPENAI_MODEL_NAME", "gpt-4o")
        self.image_format: str = os.getenv("IMAGE_FORMAT", "JPEG").upper()
        self.image_quality: int = int(os.getenv("IMAGE_QUALITY", 85))
        self.process_lectures: bool = bool(int(os.getenv("PROCESS_LECTURES", 0)))
        self.process_exams: bool = bool(int(os.getenv("PROCESS_EXAMS", 0)))
        self.max_concurrency: int = int(os.getenv("MAX_CONCURRENCY", 5))
        self.directories: dict[str, Path] = {
            "lecture_pdf": Path(
                os.getenv("LECTURE_PDF_DIR", "AI_Course/Lecture_Notes/Source_PDFs")
            ),
            "lecture_image_output": Path(
                os.getenv("LECTURE_OUTPUT_DIR", "AI_Course/Lecture_Notes/Processed")
            ),
            "exam_pdf": Path(os.getenv("EXAM_PDF_DIR", "AI_Course/Exams/Source_PDFs")),
            "exam_image_output": Path(
                os.getenv("EXAM_OUTPUT_DIR", "AI_Course/Exams/Processed")
            ),
            "knowledge_dir": Path(os.getenv("KNOWLEDGE_DIR")),
            "exam_dir": Path(os.getenv("EXAM_DIR")),
        }
        self.excluded_headers: list[str] = os.getenv(
            "EXCLUDED_HEADERS", "## Slide,## 6.034"
        ).split(",")
        self.image_patterns: list[str] = [
            r"(?:!\[.*?\]\(.*?\))",
            r"(?:<!--\s*image[^>]*-->)",
            r"(?:<img.*?>)",
            r"^Page \d+$",
        ]
        self.validate()
        self.create_directories()

    def validate(self) -> None:
        if not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY is required in environment")
        if self.image_format not in {"JPEG", "PNG"}:
            raise ValueError(
                f"Invalid IMAGE_FORMAT: {self.image_format}. Must be JPEG or PNG"
            )
        if self.image_quality < 1 or self.image_quality > 100:
            raise ValueError("IMAGE_QUALITY must be between 1 and 100")

    def create_directories(self) -> None:
        for dir_path in self.directories.values():
            dir_path.mkdir(parents=True, exist_ok=True)


config: Config = Config()


def load_markdown_sections(file_path: str) -> dict[str, str]:
    header_regex = re.compile(r"^(#{1,6})\s+(.*?)(?:\(.*?\))?\s*$")  # Improved pattern

    markdown_sections = {}
    stack = [{"level": 0, "title": "Root", "content": []}]
    with open(file_path, "r", encoding="utf-8") as markdown_file:
        for line_num, line in enumerate(markdown_file, 1):
            line = line.rstrip("\n")
            m = header_regex.match(line)

            if m:
                # Extract header level and clean title
                level = len(m.group(1))
                title = m.group(2).strip()

                # Pop stack until we reach parent level
                while stack and stack[-1]["level"] >= level:
                    popped = stack.pop()
                    # Store all sections with level >= 2 (modified from original >=4)
                    if popped["level"] >= 2:
                        key_parts = [
                            f"{s['title']}" for s in stack[1:]
                        ]  # Simplified key
                        key_parts.append(popped["title"])
                        key = " > ".join(key_parts)
                        markdown_sections[key] = "\n".join(popped["content"]).strip()

                # Push new section to stack
                new_section = {"level": level, "title": title, "content": []}
                stack.append(new_section)
            else:
                # Add content to current section
                if stack:
                    stack[-1]["content"].append(line)

    # Process remaining sections in stack
    while stack:
        popped = stack.pop()
        if popped["level"] >= 2:  # Modified from original >=4
            key_parts = [f"{s['title']}" for s in stack[1:]]
            key_parts.append(popped["title"])
            key = " > ".join(key_parts)
            markdown_sections[key] = "\n".join(popped["content"]).strip()

    return markdown_sections

def image_to_base64(image: Image.Image) -> str:
    buffered = BytesIO()
    image.save(buffered, format=config.image_format, quality=config.image_quality)
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return f"data:image/{config.image_format.lower()};base64,{img_str}"


class ClassNotes(BaseModel):
    text: str


async def convert_pdf_to_images(pdf_path: Path, file_type: str) -> int:
    if file_type == "lecture":
        output_dir = config.directories["lecture_image_output"] / pdf_path.stem
    else:
        output_dir = config.directories["exam_image_output"] / pdf_path.stem
    output_dir.mkdir(parents=True, exist_ok=True)

    existing_pages: set[str] = {f.name for f in output_dir.glob("page_*.*")}
    processed: int = 0

    try:
        images: list[Any] = convert_from_path(
            str(pdf_path),
            dpi=200,
            thread_count=4,
            fmt=config.image_format.lower(),
            poppler_path=r"C:\poppler\Library\bin",
        )

        for i, image in enumerate(images, 1):
            page_name: str = f"page_{i:03d}.{config.image_format.lower()}"
            if page_name not in existing_pages:
                image.save(
                    output_dir / page_name,
                    config.image_format,
                    quality=config.image_quality,
                )
                processed += 1
                logger.info(f"Converted page {i} of {pdf_path.name}")

        return processed

    except Exception as e:
        logger.error(f"Error processing {pdf_path}: {str(e)}")
        return 0


async def pdf_to_markdown(pdf_path: Path, output_path: Path) -> bool:
    try:
        doc_converter: DocumentConverter = DocumentConverter()
        result: Any = doc_converter.convert(str(pdf_path))
        markdown_text: str = result.document.export_to_markdown()

        image_pattern: re.Pattern = re.compile(
            "|".join(config.image_patterns), re.IGNORECASE
        )
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

            if not in_ignore_block and not any(
                line.startswith(h) for h in config.excluded_headers
            ):
                processed_lines.append(line)

        async with aiofiles.open(output_path, "w", encoding="utf-8") as f:
            await f.write("\n".join(processed_lines))

        return True

    except Exception as e:
        logger.error(f"Failed to convert {pdf_path}: {str(e)}")
        return False


async def extract_text_from_image(image_path: Path) -> str:
    try:
        image = Image.open(image_path)
        img_base64 = image_to_base64(image)

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Convert this image to structured markdown following these rules:\n"
                        "1. Extract ALL text with perfect accuracy:\n"
                        "   - Preserve original structure/headings/lists\n"
                        "   - Maintain code blocks (```) and math equations ($$)\n"
                        "   - Keep exact numbering and indentation\n"
                        "2. Special element handling:\n"
                        "   a) Questions ALWAYS use ###### headers\n"
                        "   b) Answers start with '//// ANSWER:'\n"
                        "   c) Convert diagrams to:\n"
                        "      - ASCII art with └─ ├─ symbols\n"
                        "      - [X] for failed constraints\n"
                        "      - [✓] for explored nodes\n"
                        "3. Formatting requirements:\n"
                        "   - Replace images with text descriptions\n"
                        "   - Use proper markdown for:\n"
                        "     * Lists (-, 1.)\n"
                        "     * Tables\n"
                        "     * Code (language-specific)\n"
                        "     * Math equations\n"
                        "4. Example transformation:\n"
                        "   Image Content: 'Question 3: What is... diagram of A→B→C'\n"
                        "   Converted to:\n"
                        "   ###### 3. What is...\n"
                        "   ```\n"
                        "   A\n"
                        "   └─ B [✓]\n"
                        "      └─ C [X]\n"
                        "   ```\n"
                        "   //// ANSWER: Explanation...",
                    },
                    {"type": "image_url", "image_url": {"url": img_base64}},
                ],
            }
        ]

        response = await anyio.to_thread.run_sync(
            lambda: openai.chat.completions.create(
                model="gpt-4o", messages=messages, temperature=0.0
            )
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"Error extracting text from {image_path}: {e}")
        return ""


async def merge_markdowns(docling_md: str, vision_md: str, image_path: Path) -> str:
    try:
        image = Image.open(image_path)
        img_base64 = image_to_base64(image)

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"""Below are two markdown representations of the same image. "
The first is generated from a PDF conversion tool (docling), and the second from OpenAI's "
Vision API analyzing images of each page. "
Please merge them into a single markdown document, combining the best aspects of both. "
Ensure that the structure is preserved, eliminate redundancies, and correct any errors. "
Focus on maintaining accurate code blocks, mathematical expressions, tables, graphs, and formatting. "
If there are questions and answers prepend the questions with '###### ' and the "
answers with '//// ANSWER: ' \n\n"
Docling Markdown:\n"
-----------------\n"
{docling_md}\n\n"
Vision API Markdown:\n"
--------------------\n"
{vision_md}\n\n"
Merged Markdown:"
1. Extract ALL text with perfect accuracy:\n"
   - Preserve original structure/headings/lists\n"
   - Maintain code blocks (```) and math equations ($$)\n"
   - Keep exact numbering and indentation\n"
2. Special element handling:\n"
   a) Questions ALWAYS use ###### headers\n"
   b) Answers start with '//// ANSWER:'\n"
   c) Convert diagrams to:\n"
      - ASCII art with └─ ├─ symbols\n"
      - [X] for failed constraints\n"
      - [✓] for explored nodes\n"
3. Formatting requirements:\n"
   - Replace images with text descriptions\n"
   - Use proper markdown for:\n"
     * Lists (-, 1.)\n"
     * Tables\n"
     * Code (language-specific)\n"
     * Math equations\n"
4. Example transformation:\n"
   Image Content: 'Question 3: What is... diagram of A→B→C'\n"
   Converted to:\n"
   ###### 3. What is...\n"
   ```\n"
   A\n"
   └─ B [✓]\n"
      └─ C [X]\n"
   ```\n"
   //// ANSWER: Explanation...""",
                    },
                    {"type": "image_url", "image_url": {"url": img_base64}},
                ],
            }
        ]

        response = await anyio.to_thread.run_sync(
            lambda: openai.chat.completions.create(
                model="gpt-4o", messages=messages, temperature=0.0
            )
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"Error merging markdowns: {e}")
        return ""


async def process_pdfs() -> None:
    pdf_files: list[tuple[Path, str]] = []
    if config.process_lectures:
        lecture_files: list[Path] = sorted(
            config.directories["lecture_pdf"].glob("*.pdf")
        )
        pdf_files += [(path, "lecture") for path in lecture_files]
        logger.debug(f"Found {len(lecture_files)} lecture PDFs")
    if config.process_exams:
        exam_files: list[Path] = sorted(config.directories["exam_pdf"].glob("*.pdf"))
        pdf_files += [(path, "exam") for path in exam_files]
        logger.debug(f"Found {len(exam_files)} exam PDFs")
    logger.info(f"Total PDFs to process: {len(pdf_files)}")
    async with anyio.create_task_group() as tg:
        for pdf_file, file_type in pdf_files:
            tg.start_soon(convert_pdf_to_images, pdf_file, file_type)


async def main() -> None:
    # Step 1: Split PDFs into images
    await process_pdfs()

    # Step 2: Generate initial docling markdown
    for pdf_file in config.directories["exam_pdf"].glob("*.pdf"):
        docling_md_path = pdf_file.with_name(pdf_file.stem + "_docling.txt")
        success = await pdf_to_markdown(pdf_file, docling_md_path)
        if success:
            logger.info(f"Generated docling markdown for {pdf_file.name}")
        else:
            logger.error(f"Failed to generate docling markdown for {pdf_file.name}")

    # Step 3: Generate vision markdown and merge
    for pdf_file in config.directories["exam_pdf"].glob("*.pdf"):
        image_dir = config.directories["exam_image_output"] / pdf_file.stem
        vision_md_path = pdf_file.with_name(pdf_file.stem + "_vision.txt")
        docling_md_path = pdf_file.with_name(pdf_file.stem + "_docling.txt")
        merged_md_path = pdf_file.with_name(pdf_file.stem + "_merged.txt")

        # Generate vision markdown
        vision_md_sections = []
        valid_extensions = (
            {".jpeg", ".jpg"} if config.image_format.upper() == "JPEG" else {".png"}
        )
        image_files = sorted(
            [
                f
                for f in image_dir.glob("page_*.*")
                if f.suffix.lower() in valid_extensions
            ],
            key=lambda x: x.name,
        )

        merged_sections = []
        for image_file in image_files:
            try:
                text = await extract_text_from_image(image_file)
                if text:
                    vision_md_sections.append(text)
            except Exception as e:
                logger.error(f"Error processing image {image_file}: {e}")

        if vision_md_sections:
            # Write vision MD once
            async with aiofiles.open(vision_md_path, "w", encoding="utf-8") as f:
                await f.write("\n\n".join(vision_md_sections))

            # Merge FULL documents ONCE
            try:
                async with aiofiles.open(docling_md_path, "r", encoding="utf-8") as f:
                    docling_md = await f.read()
                vision_md = "\n\n".join(vision_md_sections)

                merged_md = await merge_markdowns(docling_md, vision_md, image_dir)

                if merged_md:
                    async with aiofiles.open(merged_md_path, "w", encoding="utf-8") as f:
                        await f.write(merged_md)
            except Exception as e:
                logger.error(f"Error merging markdowns: {e}")


if __name__ == "__main__":
    anyio.run(main)
