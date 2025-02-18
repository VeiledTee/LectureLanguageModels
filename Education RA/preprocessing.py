import os
import re
from pathlib import Path
from typing import Any

from docling.document_converter import DocumentConverter


def format_question(header: str, content: str) -> str:
    """Format a question from its header and content sections."""
    question = f"{header} > {content}" if content else header
    return question.rstrip(":") + ":" if not question.endswith(":") else question


def process_exam_file(file_path: Path) -> list[str]:
    """Process an answerless exam file into individual questions."""
    with open(file_path, "r", encoding="utf-8") as f:
        file_text = f.read()
        queries: list[str] = parse_quiz(str(file_text))
    print(f"Processed {len(queries)} questions from {file_path.name}")
    return queries


def pdf_to_markdown(pdf_path: str, output_txt_path: str) -> None:
    """
    Converts a PDF to cleaned Markdown text, removing images, specific headers, and unnecessary whitespace.

    Args:
        pdf_path (str): Path to input PDF file
        output_txt_path (str): Path for output Markdown file
    """
    # Initialize converter and process PDF
    doc_converter = DocumentConverter()
    result = doc_converter.convert(pdf_path)
    markdown_text: str = result.document.export_to_markdown()

    # Regex pattern for image detection
    image_pattern = re.compile(
        r"(?:!\[.*?\]\(.*?\))|"  # Markdown images
        r"(?:<!--\s*image[^>]*-->)|"  # Image comments
        r"(?:<img.*?>)",  # HTML images
        re.IGNORECASE | re.DOTALL,
    )

    # Patterns to exclude
    excluded_headers = ("## Slide", "## 6.034")

    processed_lines: list[str] = []
    in_ignore_block = False

    for line in markdown_text.splitlines():
        # Remove images and strip whitespace
        cleaned_line = image_pattern.sub("", line).strip()

        # Skip empty lines
        if not cleaned_line:
            continue

        # Handle block-level ignoring
        if "<!-- image" in cleaned_line.lower():
            in_ignore_block = True
            continue
        if in_ignore_block and cleaned_line.startswith("#"):
            in_ignore_block = False

        # Skip excluded headers and content in ignore blocks
        if not in_ignore_block:
            if not cleaned_line.startswith(excluded_headers):
                processed_lines.append(cleaned_line)

    # Write to file
    with open(output_txt_path, "w", encoding="utf-8") as writefile:
        writefile.write("\n".join(processed_lines))

    print(f"Converted {pdf_path} to text and saved as {output_txt_path}")


def extract_questions(extract_from: str) -> list[str]:
    """Dynamically finds and extracts deepest-level headers as questions"""
    # Find all headers and their levels
    headers = []
    header_re = re.compile(r"^(#+)\s+(.+?)(?:\(.*?\))?\s*$", re.MULTILINE)

    for match in header_re.finditer(extract_from):
        level = len(match.group(1))
        text = match.group(2).strip()
        headers.append((level, text))

    if not headers:
        return []

    # Find deepest header level used in document
    max_level = max(level for level, _ in headers)

    # Extract all headers at deepest level
    return [text for level, text in headers if level == max_level]


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


def parse_quiz(md_text: str) -> list[str]:
    """
    Parse markdown text representing a quiz with hierarchical sections.

    Headers (lines starting with '#' characters) signify sections, subsections, and questions.
    Lines between headers are considered content for the preceding header.
    The hierarchy is built dynamically based on header levels. If a header of a lower level is found,
    the current stack is popped until the header level matches. The function returns a list of
    concatenated paths (using ' > ') from the root to each leaf node.

    Args:
        md_text (str): The markdown text to parse.

    Returns:
        list[str]: A list of full question paths in the form "Section > Subsection > ... > Question".
    """
    lines: list[str] = md_text.splitlines()
    root: list[dict[str, Any]] = []
    stack: list[dict[str, Any]] = []

    for line in lines:
        line = line.rstrip()
        if not line.strip():
            continue
        if line.startswith("#"):
            # Determine header level and text.
            level = 0
            while level < len(line) and line[level] == "#":
                level += 1
            header = line[level:].strip()
            node: dict[str, Any] = {
                "level": level,
                "header": header,
                "content": "",
                "children": [],
            }
            # Pop until the top of the stack is of a lower level.
            while stack and stack[-1]["level"] >= level:
                stack.pop()
            if stack:
                stack[-1]["children"].append(node)
            else:
                root.append(node)
            stack.append(node)
        else:
            # Append non-header content to the current header.
            if stack:
                current = stack[-1]
                current["content"] = (
                    (current["content"] + " " + line.strip()).strip()
                    if current["content"]
                    else line.strip()
                )

    questions: list[str] = []

    def traverse(path: list[str], node: dict[str, Any]) -> None:
        text = node["header"]
        if node["content"]:
            text += " " + node["content"]
        new_path = path + [text]
        if not node["children"]:
            questions.append(" > ".join(new_path))
        else:
            for child in node["children"]:
                traverse(new_path, child)

    for node in root:
        traverse([], node)

    return questions


if __name__ == "__main__":
    for directory in ["AI_Course/Exams"]:
        # Create a converter instance
        converter = DocumentConverter()

        for filename in os.listdir(directory):
            source_path = os.path.join(directory, filename)  # found for every file

            # if filename.endswith(".pdf"):  # filter by .pdf extension
            #     output_path = os.path.join(
            #         directory, filename.split(".")[0] + "_parsed.txt"
            #     )
            #     pdf_to_markdown(
            #         source_path, output_path
            #     )  # convert pdf to markdown file and save in directory with '_parsed' suffix

            if filename.endswith("_answerless.txt"):
                with open(source_path, "r", encoding="utf-8") as f:
                    input_text = f.read()
                    questions = parse_quiz(input_text)
                    print(f"File: {filename}, Questions Found: {len(questions)}")
                    # for i in questions:
                    #     print(i)
