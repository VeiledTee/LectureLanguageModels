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

    The input string is first split by the record separator (\\x1e), then each resulting
    substring is split by the unit separator (\\x1f) to extract individual questions.

    Args:
        extract_from (str): Input string containing questions separated by delimiters.

    Returns:
        list[str]: list of extracted and stripped questions.
    """
    extracted_questions: list[str] = []
    for line in extract_from.split("\x1e"):  # split by delimiter
        for query in line.split("\x1f"):  # split questions
            if line != "":
                extracted_questions.append(query.strip())
    return extracted_questions


def load_markdown_sections(file_path: str) -> dict[str, str]:
    """
    Loads a Markdown file and splits it into sections based on headers.

    Sections are identified by '##' headers. Content between headers is processed to
    replace newlines with spaces and marked with record separators (\\x1e).

    Args:
        file_path (str): Path to the Markdown file to process.

    Returns:
        dict[str, str]: dictionary where keys are section headers and values are
            processed content with newlines replaced by spaces and section breaks
            marked with \\x1e.
    """
    with open(file_path, "r", encoding="utf-8") as file:
        text: str = file.read()

    # Splitting sections by headers (##) while keeping them
    split_sections = re.split(r"(## .*\n)", text)
    extracted_sections: dict[str, str] = {}

    for i in range(1, len(split_sections), 2):
        section_header: list[str] = split_sections[i].strip("# \n")
        section_content: list[str] = split_sections[i + 1].strip()
        extracted_sections[section_header] = section_content.replace("\n", " ") + "\x1e"

    return extracted_sections


if __name__ == "__main__":
    # Directory containing the PDF files
    # DIRECTORY = "AI Course/Exams"
    DIRECTORY = "AI Course/Lecture Notes"

    # Create a converter instance
    converter = DocumentConverter()

    for filename in os.listdir(DIRECTORY):
        source_path = os.path.join(DIRECTORY, filename)  # found for every file
        print(source_path)

        if filename.endswith(".pdf"):  # filter by .pdf extension
            output_path = os.path.join(
                DIRECTORY, filename.split(".")[0] + "_parsed.txt"
            )
            pdf_to_markdown(
                source_path, output_path
            )  # convert pdf to markdown file and save in DIRECTORY with '_parsed' suffix

        elif filename.endswith("_answerless.txt"):  # find edited quizzes for parsing
            with open(source_path, "r", encoding="utf-8") as f:
                input_text = f.read()
                sections = load_markdown_sections(
                    "AI Course/Exams/q1_soln_answerless.txt"
                )

                # Print all sections
                for header, content in sections.items():
                    if len(content) >= len(header):
                        questions = extract_questions(content)
                        print(f"## {header}: {questions}")
