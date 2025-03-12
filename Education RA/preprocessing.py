import logging
import os
import httpx
import re
import json
from pathlib import Path
from typing import Any, Optional
import shutil
import subprocess
import pypandoc

import aiofiles
import anyio
import openai
import requests
from docling.document_converter import DocumentConverter
from dotenv import load_dotenv
from pdf2image import convert_from_path
from pydantic import BaseModel
from io import BytesIO
import base64
from PIL import Image
import ollama

pypandoc.download_pandoc()

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger: logging.Logger = logging.getLogger(__name__)

load_dotenv()
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
# Set API_MODE to either "openai" or "ollama"
API_MODE = "openai"


class Config:
    def __init__(self) -> None:
        load_dotenv()
        self.openai_api_key: Optional[str] = os.getenv("OPENAI_API_KEY")
        self.github_token: Optional[str] = os.getenv("GITHUB_TOKEN")
        openai.api_type = os.getenv("OPENAI_API_TYPE")
        self.api_mode: str = os.getenv("API_MODE", "openai").lower()
        self.model_name: str = os.getenv("OPENAI_MODEL_NAME", "gpt-4o")
        self.image_format: str = os.getenv("IMAGE_FORMAT", "JPEG").upper()
        self.image_quality: int = int(os.getenv("IMAGE_QUALITY", 85))
        self.process_lectures: bool = bool(int(os.getenv("PROCESS_LECTURES", 0)))
        self.process_exams: bool = bool(int(os.getenv("PROCESS_EXAMS", 0)))
        self.process_github: bool = bool(int(os.getenv("PROCESS_GITHUB", 0)))
        self.max_concurrency: int = int(os.getenv("MAX_CONCURRENCY", 5))

        # GitHub configuration
        self.github_repo_owner: str = os.getenv("GITHUB_REPO_OWNER", "")
        self.github_repo_name: str = os.getenv("GITHUB_REPO_NAME", "")
        self.github_branch: str = os.getenv("GITHUB_BRANCH", "main")
        self.github_target_dir: str = os.getenv("GITHUB_TARGET_DIR", "")

        self.directories: dict[str, Path] = {
            "lecture_note": Path(
                os.getenv("LECTURE_NOTE_DIR", "Lecture_Notes/Source_Notes")
            ),
            "lecture_image_output": Path(
                os.getenv("LECTURE_OUTPUT_DIR", "Lecture_Notes/Processed")
            ),
            "exam_note": Path(os.getenv("EXAM_NOTE_DIR", "Exams/Source_Notes")),
            "exam_image_output": Path(os.getenv("EXAM_OUTPUT_DIR", "Exams/Processed")),
            "knowledge_dir": Path(os.getenv("KNOWLEDGE_DIR", "Knowledge_Base")),
            "github_dir": Path(os.getenv("GITHUB_DIR", "GitHub_Content")),
            "json_output": Path(os.getenv("LECTURE_OUTPUT_DIR", "Knowledge_Base/JSON")),
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
        if self.api_mode == "openai" and not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY is required when using OpenAI")
        if self.process_github and not self.github_token:
            raise ValueError("GITHUB_TOKEN is required for GitHub processing")
        if self.image_format not in {"JPEG", "PNG"}:
            raise ValueError(
                f"Invalid IMAGE_FORMAT: {self.image_format}. Must be JPEG or PNG"
            )

    def create_directories(self) -> None:
        for dir_path in self.directories.values():
            dir_path.mkdir(parents=True, exist_ok=True)


config: Config = Config()


async def download_github_directory():
    """Async GitHub directory downloader"""
    if not config.process_github:
        return

    headers = (
        {"Authorization": f"token {config.github_token}"} if config.github_token else {}
    )
    base_url = f"https://api.github.com/repos/{config.github_repo_owner}/{config.github_repo_name}/contents/"

    async with httpx.AsyncClient() as client:

        async def download_recursive(path: str):
            url = f"{base_url}{path}?ref={config.github_branch}"
            response = await client.get(url, headers=headers)
            if response.status_code != 200:
                logger.error(f"Failed to fetch {path}: {response.status_code}")
                return

            items = response.json()
            for item in items:
                item_path = item["path"]
                relative_path = os.path.relpath(item_path, config.github_target_dir)
                local_path = config.directories["github_dir"] / relative_path

                if item["type"] == "file":
                    download_url = item["download_url"]
                    file_response = await client.get(download_url)
                    if file_response.status_code == 200:
                        local_path.parent.mkdir(parents=True, exist_ok=True)
                        with open(local_path, "wb") as f:
                            f.write(file_response.content)
                        logger.info(f"Downloaded GitHub file: {local_path}")
                elif item["type"] == "dir":
                    await download_recursive(item_path)

        await download_recursive(config.github_target_dir)


async def extract_text_from_image(image_path: Path) -> str:
    """Main entry point for image processing with API mode selection"""
    if config.api_mode == "openai":
        return await extract_text_from_image_openai(image_path)
    else:
        return await extract_text_from_image_ollama(image_path)


async def extract_text_from_image_openai(image_path: Path) -> str:
    try:
        image = Image.open(image_path)
        img_base64 = image_to_base64(image)
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Extract all text from image and put in markdown format. "
                                "To the best of your ability, convert the image to ASCII Art."
                                "Do not skip lines in graphs. Capture them all: ",
                    },
                    {"type": "image_url", "image_url": {"url": img_base64}},
                ],
            }
        ]
        response = await anyio.to_thread.run_sync(
            lambda: openai.chat.completions.create(
                model=config.model_name, messages=messages, temperature=0.0
            )
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"OpenAI processing error: {e}")
        return ""


async def extract_text_from_image_ollama(image_path: Path) -> str:
    try:
        response = ollama.chat(
            model="llava",
            messages=[
                {
                    "role": "user",
                    "content": "Extract all text from image and put in markdown format. "
                               "To the best of your ability, convert the image to ASCII Art."
                               "Do not skip lines in graphs. Capture them all: ",
                    "images": [str(image_path)],
                }
            ],
        )
        return response["message"]["content"].strip()
    except Exception as e:
        logger.error(f"Ollama processing error: {e}")
        return ""


async def process_github_content():
    """Process downloaded GitHub content"""
    if not config.process_github:
        return

    # Process images first
    for root, _, files in os.walk(config.directories["github_dir"]):
        for file in files:
            if file.endswith((".png", ".jpg", ".jpeg")):
                image_path = Path(root) / file
                text = await extract_text_from_image(image_path)
                markdown_path = image_path.with_suffix(".md")

                async with aiofiles.open(markdown_path, "w", encoding="utf-8") as f:
                    await f.write(f"# Processed GitHub Image\n\n{text}")
                logger.info(f"Processed GitHub image: {image_path}")

    # Process RST files
    for root, _, files in os.walk(config.directories["github_dir"]):
        for file in files:
            if file.endswith(".rst"):
                rst_path = Path(root) / file

                # Generate output path based on last element of RST path
                output_md_filename = f"{rst_path.stem}.md"
                output_md_path = (
                        config.directories["lecture_image_output"] / output_md_filename
                )

                if rst_path.exists():
                    await convert_rst_to_markdown(rst_path, output_md_path)
                    logger.info(f"Processed GitHub RST: {rst_path} -> {output_md_path}")


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
    """Convert PIL Image to base64 data URL"""
    from io import BytesIO
    import base64

    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return f"data:image/png;base64,{img_str}"


async def convert_note_to_images(pdf_path: Path, file_type: str) -> int:
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


async def pdf_to_markdown(pdf_path: Path, output_path: Path, file_type: str) -> bool:
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

        await convert_markdown_to_json(output_path, file_type)

        return True

    except Exception as e:
        logger.error(f"Failed to convert {pdf_path}: {str(e)}")
        return False


async def merge_markdowns_openai(
        docling_md: str, vision_md: str, image_path: Path
) -> str:
    try:
        image = Image.open(image_path)
        img_base64 = image_to_base64(image)
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "Below are two markdown representations of the same image. "
                            "The first is generated from a PDF conversion tool (docling), and the second from OpenAI's Vision API. "
                            "Please merge them into a single markdown document, combining the best aspects of both. "
                            "Ensure that the structure is preserved, eliminate redundancies, and correct any errors. "
                            "Focus on maintaining accurate code blocks, mathematical expressions, tables, graphs, and formatting. "
                            "If there are questions and answers, prepend the questions with '###### ' and the answers with '//// ANSWER: '\n\n"
                            "Docling Markdown:\n"
                            "-----------------\n"
                            f"{docling_md}\n\n"
                            "Vision API Markdown:\n"
                            "--------------------\n"
                            f"{vision_md}\n\n"
                            "Merged Markdown:\n"
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
                            "   //// ANSWER: Explanation..."
                        ),
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


def call_ollama(model: str, messages):
    response = ollama.chat(
        model=model,
        messages=messages,
    )
    print(response)
    return response


async def merge_markdowns_ollama(
        docling_md: str, vision_md: str, image_path: Path
) -> str:
    try:
        image = Image.open(image_path)
        img_base64 = image_to_base64(image)
        messages = [
            {
                "role": "user",
                "content": "Below are two markdown representations of the same image. "
                           "The first is generated from a PDF conversion tool (docling), and the second from Ollama's vision API. "
                           "Please merge them into a single markdown document, combining the best aspects of both. "
                           "Ensure that the structure is preserved, eliminate redundancies, and correct any errors. "
                           "Focus on maintaining accurate code blocks, mathematical expressions, tables, graphs, and formatting. "
                           "If there are questions and answers, prepend the questions with '###### ' and the answers with '//// ANSWER: '\n\n"
                           "Docling Markdown:\n"
                           "-----------------\n"
                           f"{docling_md}\n\n"
                           "Vision API Markdown:\n"
                           "--------------------\n"
                           f"{vision_md}\n\n"
                           "Merged Markdown:\n"
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
                "images": {"url": img_base64},
            }
        ]
        response = await anyio.to_thread.run_sync(
            lambda: call_ollama("llava", messages)
        )
        return response["choices"][0]["message"]["content"].strip()
    except Exception as e:
        logger.error(f"Error merging markdowns: {e}")
        return ""


async def merge_markdowns(docling_md: str, vision_md: str, image_path: Path) -> str:
    if API_MODE == "openai":
        return await merge_markdowns_openai(docling_md, vision_md, image_path)
    else:
        return await merge_markdowns_ollama(docling_md, vision_md, image_path)


async def process_rst_images(rst_path: Path, output_md_path: Path) -> None:
    """
    Process images referenced in an RST file and generate Markdown replacements.
    """
    # Extract image paths from RST figure directives
    figure_pattern = re.compile(r"^\.\. figure:: (.+?\.(?:png|jpg|jpeg))", re.MULTILINE)
    rst_content = rst_path.read_text(encoding="utf-8")
    image_paths = figure_pattern.findall(rst_content)

    # Process each image
    for img_rel_path in image_paths:
        img_path = (rst_path.parent / img_rel_path).resolve()
        if img_path.exists():
            # Convert image to Markdown using existing pipeline
            text = await extract_text_from_image(img_path)
            md_path = img_path.with_suffix(".md")
            md_path.write_text(text, encoding="utf-8")
            logger.info(f"Processed RST image: {img_path}")


async def convert_rst_to_markdown(rst_path: Path, output_md_path: Path) -> None:
    """
    Convert RST to Markdown and replace figure directives with processed content.
    """
    # Step 1: Process images in RST first
    await process_rst_images(rst_path, output_md_path)

    # Step 2: Convert RST to Markdown using pandoc
    pypandoc.convert_file(
        str(rst_path),
        'markdown',
        outputfile=str(output_md_path),
        extra_args=['-s', '--wrap=none']
    )

    # Step 3: Replace image links with generated .md content
    md_content = output_md_path.read_text(encoding="utf-8")

    # Find all markdown image links (generated by pandoc)
    md_image_pattern = re.compile(r"!\[(.*?)\]\((.*?\.(?:png|jpg|jpeg))\)")
    matches = md_image_pattern.findall(md_content)

    for alt_text, img_rel_path in matches:
        img_path = (rst_path.parent / img_rel_path).resolve()
        md_replacement_path = img_path.with_suffix(".md")

        if md_replacement_path.exists():
            replacement_content = md_replacement_path.read_text(encoding="utf-8")
            # Escape replacement content for regex substitution
            replacement_escaped = re.escape(replacement_content)
            # Replace the image link with the content from the .md file
            md_content = re.sub(
                re.escape(f"![{alt_text}]({img_rel_path})"),
                replacement_content,
                md_content,
                count=1  # Replace only the first occurrence per match
            )

    # Write the final Markdown
    output_md_path.write_text(md_content, encoding="utf-8")
    await convert_markdown_to_json(output_md_path, "github")
    logger.info(f"Converted RST to Markdown: {output_md_path}")


async def convert_markdown_to_json(markdown_path: Path, output_type: str) -> None:
    """Convert processed Markdown file to structured JSON"""
    try:
        # Load and parse the markdown content
        sections = load_markdown_sections(str(markdown_path))

        # Create output directory structure
        json_dir = config.directories["json_output"] / output_type
        relative_path = markdown_path.relative_to(config.directories["lecture_image_output"] if output_type == "lecture"
                                                  else config.directories["exam_image_output"]
                                                  if output_type == "exam"
                                                  else config.directories["github_dir"])
        json_path = json_dir / relative_path.with_suffix(".json")
        json_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert to JSON structure
        json_data = {
            "source": str(markdown_path),
            "sections": [
                {
                    "path": path,
                    "content": content
                } for path, content in sections.items()
            ]
        }

        # Write JSON file
        async with aiofiles.open(json_path, "w", encoding="utf-8") as f:
            await f.write(json.dumps(json_data, indent=2, ensure_ascii=False))

        logger.info(f"Converted {markdown_path} to JSON at {json_path}")

    except Exception as e:
        logger.error(f"JSON conversion failed for {markdown_path}: {str(e)}")


async def process_notes() -> None:
    pdf_files: list[tuple[Path, str]] = []
    if config.process_lectures:
        lecture_files: list[Path] = sorted(
            config.directories["lecture_note"].glob("*.pdf")
        )
        pdf_files += [(path, "lecture") for path in lecture_files]
        logger.debug(f"Found {len(lecture_files)} lecture PDFs")
    if config.process_exams:
        exam_files: list[Path] = sorted(config.directories["exam_note"].glob("*.pdf"))
        pdf_files += [(path, "exam") for path in exam_files]
        logger.debug(f"Found {len(exam_files)} exam PDFs")
    logger.info(f"Total PDFs to process: {len(pdf_files)}")
    return None
    # async with anyio.create_task_group() as tg:
    #     for pdf_file, file_type in pdf_files:
    #         tg.start_soon(convert_note_to_images, pdf_file, file_type)


async def main() -> None:
    # Existing processing
    await download_github_directory()
    await process_github_content()
    await process_notes()


if __name__ == "__main__":
    anyio.run(main)
