import os
import asyncio
from lightrag import LightRAG, QueryParam
from lightrag.llm.openai import gpt_4o_mini_complete, openai_embed
from dotenv import load_dotenv

load_dotenv()

import os
import argparse
import pypandoc

# A thorough conversion prompt explaining the conversion requirements:
conversion_prompt = """
Please convert the following lecture notes from reStructuredText (rst) format into raw Markdown text.
The final Markdown should be structured to be compatible with lightrag for creating a knowledge graph.
Ensure that:
- Section headers (e.g., "Problem", "Initialization", "Evaluation", etc.) are converted to appropriate Markdown headings.
- Inline math notations such as :math:`n` are converted into Markdown math delimiters (e.g. $n$).
- Code blocks and literal includes (e.g., ".. literalinclude::" and ".. code-block::") are converted to fenced code blocks,
  preserving language hints (like "python" or "text") where specified.
- Image directives (e.g., ".. figure::") are converted to Markdown image syntax, preserving image paths, alt texts, alignment,
  and links if provided.
- Download links (e.g., ":download:`The Selection Script </path/to/file>`") are converted into proper Markdown links.
- Any rst-specific formatting is translated appropriately into Markdown.
Note: The image and rst files are stored in:
    C:\\Users\\Ethan\\Documents\\PhD\\LectureLanguageModels\\Education RA\\Evolutionary_Computation\\Lecture_Notes\\Source_Notes
and the .py files are stored in:
    C:\\Users\\Ethan\\Documents\\PhD\\LectureLanguageModels\\Education RA\\Evolutionary_Computation\\Lecture_Notes\\src
Adjust any file paths accordingly if needed.
"""


def convert_rst_to_markdown(input_file: str, output_file: str):
    """
    Convert an rst file to markdown using pypandoc.
    """
    # You can print or log the conversion prompt if needed:
    print("Conversion prompt:")
    print(conversion_prompt)

    try:
        # Convert the input reST file to Markdown
        markdown_text = pypandoc.convert_file(input_file, "markdown", format="rst")
    except Exception as e:
        print(f"Error during conversion: {e}")
        return

    # Optionally, post-process the markdown_text here (for custom replacements) if needed

    # Write the markdown text to the output file
    with open(output_file, "w", encoding="utf-8") as md_file:
        md_file.write(markdown_text)

    print(f"Conversion complete. Markdown file saved at: {output_file}")


async def init_rag():
    # Initialize LightRAG with OpenAI functions
    rag = LightRAG(
        working_dir="./rag_storage",  # change this to your preferred storage directory
        embedding_func=openai_embed,
        llm_model_func=gpt_4o_mini_complete,
    )
    return rag


def main():
    if "OPENAI_API_KEY" not in os.environ:
        print("Error: Please set the OPENAI_API_KEY environment variable.")
        return

    # Initialize the RAG system
    rag = asyncio.run(init_rag())

    # Insert some sample text
    sample_text = (
        "Once upon a time in a land of magic and mystery, "
        "brave heroes embarked on epic quests to vanquish darkness "
        "and restore harmony."
    )
    rag.insert(sample_text)

    # Query the RAG system using a hybrid retrieval mode
    question = "What themes are present in the story?"
    result = rag.query(question, param=QueryParam(mode="hybrid"))
    print("Query Result:", result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert reStructuredText lecture notes to Markdown for lightrag knowledge graph."
    )
    parser.add_argument(
        "--input",
        "-i",
        required=True,
        help="Path to the input .rst file (e.g., from Source_Notes directory)",
    )
    parser.add_argument(
        "--output", "-o", required=True, help="Path to the output .md file"
    )
    args = parser.parse_args()

    # Ensure the input file exists
    if not os.path.isfile(args.input):
        print(f"Input file not found: {args.input}")
        exit(1)

    convert_rst_to_markdown(args.input, args.output)
