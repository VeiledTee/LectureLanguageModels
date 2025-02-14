import os
import re

from docling.document_converter import DocumentConverter


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
    """
    Extracts individual questions from a string using specific delimiters.
    First it attempts to split using the record (\x1e) and unit (\x1f) separators.
    If those aren’t present, it falls back to using a regex to pick out question lines.
    """
    # Try using the expected delimiters.
    if "\x1e" in extract_from or "\x1f" in extract_from:
        questions = []
        for block in extract_from.split("\x1e"):
            for query in block.split("\x1f"):
                if query.strip():
                    questions.append(query.strip())
        if questions:
            return questions

    pattern = re.compile(r"^#{3,6}\s*(\d+\.\s+.+)$", re.MULTILINE)
    matches = pattern.findall(extract_from)
    if matches:
        return [m.strip() for m in matches]
    else:
        # Final fallback: return all nonempty lines.
        return [line.strip() for line in extract_from.splitlines() if line.strip()]


import re


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


if __name__ == "__main__":
    # Directory containing the PDF files
    # DIRECTORY = "AI_Course/Exams"
    # DIRECTORY = "AI_Course/Lecture_Notes"
    for directory in ["AI_Course/Exams"]:
        # Create a converter instance
        converter = DocumentConverter()

        for filename in os.listdir(directory):
            source_path = os.path.join(directory, filename)  # found for every file
            print(source_path)

            # if filename.endswith(".pdf"):  # filter by .pdf extension
            #     output_path = os.path.join(
            #         directory, filename.split(".")[0] + "_parsed.txt"
            #     )
            #     pdf_to_markdown(
            #         source_path, output_path
            #     )  # convert pdf to markdown file and save in directory with '_parsed' suffix

            if filename.endswith("_answerless.txt"):  # find edited quizzes for parsing
                with open(source_path, "r", encoding="utf-8") as f:
                    input_text = f.read()
                    sections = load_markdown_sections(
                        file_path=f"{directory}/{filename}"
                    )
                    print(len(sections.items()))
                    # for header, content in sections.items():
                    #     print(f"{header}:\n{content}\n{'-' * 40}")
