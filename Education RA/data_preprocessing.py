import os
import re

from docling.document_converter import DocumentConverter


def process_file(text_to_process: list[str]):
    output = []
    in_ignore = False

    for line in text_to_process:
        print(line)
        stripped = line.strip()
        if in_ignore:
            if stripped.startswith("#"):
                output.append(line)
                in_ignore = False
                print("ADDED")
        else:
            if stripped == "<!-- image -->":
                in_ignore = True
                print("NOT ADDED")
            else:
                output.append(line)
                print("ADDED")

    return output


def pdf_to_markdown(pdf_path, output_txt_path):
    # Initialize the converter
    doc_converter = DocumentConverter()

    # Convert PDF to markdown
    result = doc_converter.convert(pdf_path)
    markdown_text = result.document.export_to_markdown()

    # Regular expression to detect Markdown images, HTML images, and image comments
    image_pattern = re.compile(
        r"!\[.*?\]\(.*?\)|<!--\s*image\s*-->|<img.*?>", re.IGNORECASE
    )

    # Process lines: remove images and empty lines
    processed_lines = []
    for line in markdown_text.splitlines():
        stripped_line = line.strip()
        # Skip lines that are empty or contain images
        if not stripped_line or image_pattern.search(stripped_line):
            continue
        processed_lines.append(stripped_line)

    cleaned_markdown: list[str] = process_file(processed_lines)

    # Save as a text file
    with open(output_txt_path, "w", encoding="utf-8") as writefile:
        writefile.write("\n".join(cleaned_markdown))

    print(f"Converted {pdf_path} to text and saved as {output_txt_path}")


def extract_questions(extract_from):
    extracted_questions = []
    for line in extract_from.split("\x1e"):  # split by delimiter
        for query in line.split("\x1f"):  # split questions
            if line != "":
                extracted_questions.append(query.strip())
    return extracted_questions


def load_markdown_sections(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        text = file.read()
    # Splitting sections by headers (##) while keeping them
    split_sections = re.split(r"(## .*\n)", text)
    extracted_sections = {}
    for i in range(1, len(split_sections), 2):
        section_header = split_sections[i].strip("# \n")
        section_content = split_sections[i + 1].strip()
        # Replace only actual section breaks with \x1E, keep the rest as is
        extracted_sections[section_header] = section_content.replace("\n", " ") + "\x1e"
    return extracted_sections


if __name__ == "__main__":
    # Directory containing the PDF files
    # DIRECTORY = "AI Course/Exams"
    DIRECTORY = "AI Course/Lecture Notes"

    # Create a converter instance
    converter = DocumentConverter()

    for filename in os.listdir(DIRECTORY):
        source_path = os.path.join(DIRECTORY, filename)
        print(source_path)

        if filename.endswith(".pdf"):
            output_path = os.path.join(
                DIRECTORY, filename.split(".")[0] + "_parsed.txt"
            )
            pdf_to_markdown(source_path, output_path)

        elif filename.endswith("_answerless.txt"):
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
