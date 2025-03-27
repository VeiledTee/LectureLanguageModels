import json
import jsonlines
from langchain.schema import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from typing import List, Dict, Any

class RAGPipeline:
    """
    A class to handle the complete RAG pipeline from document processing to response generation.
    """
    
    def __init__(self, openai_api_key: str = None):
        """
        Initialize the RAG pipeline.
        
        Args:
            openai_api_key: Optional OpenAI API key. If None, will use environment variable.
        """
        self.embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
        self.llm = OpenAI(temperature=0, openai_api_key=openai_api_key)
        self.qa_chain = None
        self.vectorstore = None
    
    def load_and_process_documents(self, document_json_path: str) -> None:
        """
        Load and process documents for the RAG system.
        
        Args:
            document_json_path: Path to JSON file containing Q&A documents
        """
        # Step 1: Load and parse the JSON file
        with open(document_json_path, "r") as f:
            data = json.load(f)
        
        # Step 2: Convert JSON data into LangChain Documents
        documents = [{"question": item["question"], "answer": item["answer"]} for item in data]
        langchain_documents = [
            Document(page_content=item["answer"], metadata={"question": item["question"]})
            for item in documents
        ]
        
        # Step 3: Split the documents into smaller chunks
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        split_documents = text_splitter.split_documents(langchain_documents)
        
        # Step 4: Create embeddings and vector store
        self.vectorstore = FAISS.from_documents(split_documents, self.embeddings)
        
        # Step 5: Initialize the Retrieval QA chain
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vectorstore.as_retriever()
        )
    
    def generate_response(self, question: str) -> str:
        """
        Generate a response for a single question using the RAG system.
        
        Args:
            question: The input question to answer
            
        Returns:
            The generated answer
        """
        if not self.qa_chain:
            raise ValueError("RAG system not initialized. Call load_and_process_documents() first.")
        return self.qa_chain.run(question)
    
    def process_jsonl_file(self, input_file: str, output_file: str) -> None:
        """
        Process an input JSONL file and generate responses for each question.
        
        Args:
            input_file: Path to input JSONL file
            output_file: Path to output JSONL file with responses
        """
        if not self.qa_chain:
            raise ValueError("RAG system not initialized. Call load_and_process_documents() first.")
            
        with jsonlines.open(input_file, 'r') as infile, jsonlines.open(output_file, 'w') as outfile:
            for item in infile:
                try:
                    custom_id = item["custom_id"]
                    user_message = item["body"]["messages"][-1]["content"]
                    
                    # Get response from RAG system
                    response = self.generate_response(user_message)
                    
                    # Format the output
                    output = {
                        "custom_id": custom_id,
                        "response": {
                            "body": {
                                "choices": [{
                                    "message": {
                                        "role": "assistant",
                                        "content": response
                                    }
                                }]
                            }
                        }
                    }
                    
                    outfile.write(output)
                except Exception as e:
                    print(f"Error processing item {item.get('custom_id', 'unknown')}: {str(e)}")

# # Example usage
# if __name__ == "__main__":
#     # Initialize the pipeline
#     rag_pipeline = RAGPipeline()
    
#     # Load and process documents
#     rag_pipeline.load_and_process_documents("data/course_notes_files/json/design_combined.json")
    
#     # Process input file and generate responses
#     rag_pipeline.process_jsonl_file(
#         input_file="data/dataset/AI_q1_chatgpt_dataset_0_shot.jsonl",
#         output_file="AI_q1_rag.jsonl"
#     )