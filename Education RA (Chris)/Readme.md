Refer [usage.ipynb](https://github.com/VeiledTee/LectureLanguageModels/blob/main/Education%20RA%20(Chris)/usage.ipynb)

#### Course Note Data Processing
```
from course_notes_extraction import PDFToQAConverter

converter = PDFToQAConverter(api_key="your-gemini-api-key")

converter.process_directory(
    pdf_directory="data/design_and_analysis_of_algorithms/course_notes/pdfs",  # input raw pdfs directory
    output_json="data/design_and_analysis_of_algorithms/course_notes/json/design_combined.json"          # output file name
)
```

#### Exam QA Data Processing
```
from exam_qa_extraction import PDFQuestionExtractor, PDFProcessor
from utils.data_util import combine_json_files

extractor = PDFQuestionExtractor(api_key="your-gemini-api-key")
processor = PDFProcessor(extractor)
processor.process_directory("data/design_and_analysis_of_algorithms/exams/pdfs", "data/design_and_analysis_of_algorithms/exams/json")
combine_json_files(input_dir="data/design_and_analysis_of_algorithms/exams/json", output_file="data/design_and_analysis_of_algorithms/exams/design_combined.json")
```
#### Create prompt datasets (n-prompt)
```
from create_prompt_dataset import create_chatgpt_input_dataset
create_chatgpt_input_dataset(input_file="data/design_and_analysis_of_algorithms/exams/design_combined.json", output_file="data/design_and_analysis_of_algorithms/0_shot.jsonl", shots=0)
```
#### Generate LLM Responses
```
import openai as client
import time
from utils.open_ai_batch_processing import upload_and_create_batch, save_batch_response

batch = upload_and_create_batch(jsonl_file_path="data/design_and_analysis_of_algorithms/0_shot.jsonl")
print(batch)
save_batch_response(batch, output_file="data/design_and_analysis_of_algorithms/llm_responses/0_shot.jsonl")
```
#### RAG with Course Notes
```
from rag import RAGPipeline
rag_pipeline = RAGPipeline()

# Load and process documents
rag_pipeline.load_and_process_documents(document_json_path="data/design_and_analysis_of_algorithms/exams/design_combined.json")

# Process input file and generate responses
rag_pipeline.process_jsonl_file(
    input_file="data/design_and_analysis_of_algorithms/0_shot.jsonl",
    output_file="data/design_and_analysis_of_algorithms/llm_responses/rag.jsonl"
)
```
#### Evaluation (Rouge-1, Rouge-L, BLEU, BertScore, F1, Jaccard)
```
from evaluation import NLPEvaluator

evaluator = NLPEvaluator()
evaluator.evaluate(
    llm_response_file="data/design_and_analysis_of_algorithms/llm_responses/rag.jsonl",
    ground_truth_file="data/design_and_analysis_of_algorithms/exams/design_combined.json"
)
```
#### LLM Assisted Evaluation
```
from llm_assisted_evaluation import evaluate

evaluate(
    llm_response_file="data/design_and_analysis_of_algorithms/llm_responses/rag.jsonl",
    ground_truth_file="data/design_and_analysis_of_algorithms/exams/design_combined.json",
)
```
#### Fine-tuning
https://platform.openai.com/finetune/


