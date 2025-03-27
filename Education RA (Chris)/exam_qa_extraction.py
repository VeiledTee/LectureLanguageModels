from google import genai
import os
from pydantic import BaseModel
from typing import List
import json
from tqdm import tqdm

class QuestionAnswer(BaseModel):
    """Data model for each question-answer pair"""
    problem_number: str
    context: str
    question: str
    answer: str

class QuestionAnswerList(BaseModel):
    """Container for multiple question-answer pairs"""
    questions: List[QuestionAnswer]

class PDFQuestionExtractor:
    """Main class for extracting questions from PDFs"""
    
    def __init__(self, api_key: str, model_id: str = "gemini-2.0-flash"):
        self.client = genai.Client(api_key=api_key)
        self.model_id = model_id
    
    def extract_questions(self, file_path: str) -> QuestionAnswerList:
        """Extract structured questions from a single PDF file"""
        file = self.client.files.upload(
            file=file_path,
            config={'display_name': os.path.basename(file_path)}
        )
        
        prompt = (
            "Extract the questions and answers from the following text into a JSON format. "
            "Do not nest any questions. For each question, include the relevant context from "
            "the surrounding text that provides background or introduces the problem. "
            "Use the keys problem_number, context, question, and answer for each entry. "
            "Ensure that all parts of each question and their corresponding contexts are "
            "fully represented and accurately transcribed from the original text. "
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

class PDFProcessor:
    """Handles processing of multiple PDF files"""
    
    def __init__(self, extractor: PDFQuestionExtractor):
        self.extractor = extractor
    
    def process_directory(self, input_dir: str, output_dir: str) -> None:
        """
        Process all PDFs in a directory and save results to JSON files
        
        Args:
            input_dir: Directory containing PDF files
            output_dir: Directory to save JSON outputs
        """
        if not os.path.exists(input_dir):
            raise FileNotFoundError(f"Input directory not found: {input_dir}")
        
        os.makedirs(output_dir, exist_ok=True)
        pdf_files = [
            f for f in os.listdir(input_dir) 
            if f.lower().endswith('.pdf')
        ]
        
        if not pdf_files:
            print(f"No PDF files found in {input_dir}")
            return
        
        for filename in tqdm(pdf_files, desc="Processing PDFs"):
            file_path = os.path.join(input_dir, filename)
            try:
                result = self.extractor.extract_questions(file_path)
                output_file = os.path.join(
                    output_dir, 
                    f"{os.path.splitext(filename)[0]}_processed.json"
                )
                self._save_results(result, output_file)
            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")
    
    def _save_results(self, result: QuestionAnswerList, output_path: str) -> None:
        """Save extracted questions to JSON file"""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result.model_dump(), f, indent=4)
        print(f"Saved results to {output_path}")

# def main():
#     """Example usage with your specific directory structure"""
#     # Initialize components
#     extractor = PDFQuestionExtractor(
#         api_key="AIzaSyBMCcDVvlasr7ffaMDkkHqE7XGfpCN9TLM"
#     )
#     processor = PDFProcessor(extractor)
    
#     # Define directory mapping
#     directory_mapping = {
#         "data/course_exam_files/introduction_to_algorithms": 
#             "output_json_files/introduction_to_algorithms",
#         "data/course_exam_files/design_and_analysis_of_algorithms": 
#             "output_json_files/design_and_analysis_of_algorithms"
#     }
    
#     # Process each directory pair
#     for input_dir, output_dir in directory_mapping.items():
#         print(f"\nProcessing {input_dir} -> {output_dir}")
#         processor.process_directory(input_dir, output_dir)

# if __name__ == "__main__":
#     main()

# from your_module import PDFQuestionExtractor, PDFProcessor

# extractor = PDFQuestionExtractor(api_key="your-api-key")
# processor = PDFProcessor(extractor)
# processor.process_directory("input/pdfs", "output/jsons")
