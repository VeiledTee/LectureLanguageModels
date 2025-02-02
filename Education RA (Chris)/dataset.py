import PyPDF2
import re
import pandas as pd

def extract_text_from_pdf(pdf_path):
    """Extracts text from the entire PDF."""
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
    return text

def extract_questions_and_answers(text):
    """Extracts questions and answers from the given text."""
    # Pattern to find questions and corresponding solutions
    qa_pairs = []
    questions = re.findall(r"([a-z]\.\s*\[.*?\].*?)\n", text)
    answers = re.split(r"[a-z]\.\s*\[.*?\].*?\n", text)[1:]

    for q, a in zip(questions, answers):
        q = re.sub(r"\s+", " ", q.strip())  # Clean whitespace in question
        a = re.sub(r"Solution:\s*", "", a.split('\n', 1)[0].strip())  # Extract answer
        qa_pairs.append({"Question": q, "Answer": a})
    
    return qa_pairs

def save_to_csv(qa_pairs, output_path):
    """Saves question-answer pairs to a CSV file."""
    df = pd.DataFrame(qa_pairs)
    df.to_csv(output_path, index=False)
    print(f"Dataset saved to {output_path}")

def save_to_json(qa_pairs, output_path):
    """Saves question-answer pairs to a JSON file."""
    df = pd.DataFrame(qa_pairs)
    df.to_json(output_path, orient="records", indent=4)
    print(f"Dataset saved to {output_path}")

# Main function
if __name__ == "__main__":
    pdf_path = "2a867820c158048636cfe8efcc10425b_38.pdf"  # Replace with your PDF file path
    text = extract_text_from_pdf(pdf_path)
    qa_pairs = extract_questions_and_answers(text)
    
    # Save dataset
    save_to_csv(qa_pairs, "qa_dataset.csv")
    save_to_json(qa_pairs, "qa_dataset.json")
