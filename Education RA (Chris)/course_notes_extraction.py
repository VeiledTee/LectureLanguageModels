from google import genai
import os
from pydantic import BaseModel, Field
from typing import List
import json
from tqdm import tqdm

# Pydantic Models
class QuestionAnswer(BaseModel):
    question: str
    answer: str

class QuestionAnswerList(BaseModel):
    questions: List[QuestionAnswer]

# Core PDF Processing Functions
class PDFToQAConverter:
    def __init__(self, api_key: str, model_id: str = "gemini-2.0-flash"):
        """Initialize with API key and optional model selection"""
        self.client = genai.Client(api_key=api_key)
        self.model_id = model_id
    
    def _extract_from_pdf(self, file_path: str) -> QuestionAnswerList:
        """Internal method to extract Q&A from single PDF"""
        file = self.client.files.upload(
            file=file_path,
            config={'display_name': os.path.basename(file_path)}
        )
        
        prompt = (
            "Convert the given pdf into JSON format by structuring each topic as question-answer pairs, "
            "ensuring that questions and answers are under 'question' and 'answer' properties. "
            "Keep the text literal as possible. Create minimal number of questions as possible, "
            "while making sure all the content course content is covered. "
            "If the image contains graphical content, describe it in text format instead. "
            "Use Latex format for any mathematical equations."
        )
        
        response = self.client.models.generate_content(
            model=self.model_id,
            contents=[prompt, file],
            config={
                'response_mime_type': 'application/json',
                'response_schema': QuestionAnswerList
            }
        )
        return response.parsed

    def process_directory(self, pdf_directory: str, output_json: str) -> dict:
        """
        Process all PDFs in directory and save as JSON
        
        Args:
            pdf_directory: Path to directory containing PDFs
            output_json: Path to output JSON file
            
        Returns:
            Dictionary containing all extracted questions/answers
        """
        if not os.path.exists(pdf_directory):
            raise FileNotFoundError(f"Directory not found: {pdf_directory}")
        
        pdf_files = [
            os.path.join(pdf_directory, f) 
            for f in os.listdir(pdf_directory) 
            if f.lower().endswith('.pdf')
        ]
        
        if not pdf_files:
            raise FileNotFoundError(f"No PDF files found in {pdf_directory}")
        
        result = {"questions": []}
        
        for file_path in tqdm(pdf_files, desc="Processing PDFs"):
            try:
                extracted = self._extract_from_pdf(file_path)
                result["questions"].extend(
                    [qa.model_dump() for qa in extracted.questions]
                )
            except Exception as e:
                print(f"Error processing {file_path}: {str(e)}")
                continue
                
        # Save results
        os.makedirs(os.path.dirname(output_json) or ".", exist_ok=True)
        with open(output_json, 'w', encoding='utf-8') as f:
            json.dump(result["questions"], f, indent=4)
            
        print(f"Successfully saved {len(result['questions'])} Q&A pairs to {output_json}")
        return result

# # Example Usage
# if __name__ == "__main__":
#     # Initialize with your API key
#     converter = PDFToQAConverter(api_key="AIzaSyBMCcDVvlasr7ffaMDkkHqE7XGfpCN9TLM")
    
#     # Process PDFs and save to JSON
#     converter.process_directory(
#         pdf_directory="path/to/your/pdfs",  # Change this
#         output_json="output.json"          # Change this
#     )