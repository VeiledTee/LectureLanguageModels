import pdfplumber


def pdf_to_text(pdf_path, output_txt_path):
    with pdfplumber.open(pdf_path) as pdf:
        text_content = []
        for page in pdf.pages:
            text = page.extract_text()
            if text:  # Skip pages with no text (e.g., pure image pages)
                text_content.append(text)

        # Save to file
        with open(output_txt_path, "w", encoding="utf-8") as f:
            f.write("\n".join(text_content))

    print(f"Extracted text from {pdf_path} and saved to {output_txt_path}")


pdf_path = "/home/penguins/Documents/PhD/LectureLanguageModels/Education RA/AI Course/Lecture Notes/ch2_search1.pdf"
out_path = 'output.md'
from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
from marker.output import text_from_rendered

converter = PdfConverter(
    artifact_dict=create_model_dict(),
)
rendered = converter(pdf_path)
text, _, _ = text_from_rendered(rendered)
print(text)
with open(out_path, "w", encoding="utf-8") as f:
    f.write("".join(text))

print(f"Extracted text from {pdf_path} and saved to {out_path}")
