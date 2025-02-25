# LectureLanguageModels

A system for evaluating LLM performance on MIT OpenCourseWare materials using both direct answering and Retrieval-Augmented Generation (RAG) approaches.

## Features
- **Centralized Configuration**: All parameters controlled via `.env` file
- **RAG Integration**: Pinecone-powered document retrieval
- **Multi-Model Evaluation**: 5+ LLM support
- **Automated Processing**: From PDF conversion to answer generation

## Installation

1. Clone repository:
```bash
git clone https://github.com/VeiledTee/LectureLanguageModels.git
cd LectureLanguageModels
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create `.env` file:
```env
# Core Configuration
GENERATION_MODELS=phi4,llama3.2,mistral,qwen2.5,deepseek-r1
EMBEDDING_MODEL=nomic-embed-text
EVALUATION_MODEL=deepseek-r1

# RAG Parameters
RAG_CHUNK_SIZE=512
RAG_TOP_K=5
RAG_EMBEDDING_DIM=768
RAG_INDEX_NAME=ai-course-rag

# Generation Parameters
GENERATION_TEMPERATURE=0.3
MAX_TOKENS=2048
TOP_P=0.9

# Evaluation
BLEU_SMOOTHING=meth1
ROUGE_METRICS=rouge-1,rouge-l
JACCARD_THRESHOLD=0.25

# Pinecone
PINECONE_API_KEY=your_api_key
PINECONE_ENV=us-east1-aws

# Paths
EXAM_DIR=AI_Course/Exams
KNOWLEDGE_DIR=AI_Course/Lecture_Notes
```

## Project Structure
```
├── AI_Course/
│   ├── Exams/
│   └── Lecture_Notes/
├── .env                    # All configuration parameters
├── chunking.py             
├── evaluation.py           
├── pinecone_rag.py         
├── preprocessing.py        
└── run_exam.py
```

## Key Configuration Options

### Ollama Set-Up
Download [Ollama](https://ollama.com/).
Within project venv run the following to install models used by default in the project.
```terminal
ollama pull nomic-embed-text
ollama pull phi4
ollama pull llama3.2:3b
ollama pull mistral
ollama pull qwen2.5:7b
ollama pull deepseek-r1:7b
```

### Model Selection
```env
GENERATION_MODELS=phi4,llama3.2,mistral,qwen2.5,deepseek-r1
EMBEDDING_MODEL=nomic-embed-text
```

### RAG Settings
```env
RAG_CHUNK_SIZE=512       # Context chunk size (characters)
RAG_TOP_K=5              # Retrieved contexts per query
RAG_EMBEDDING_DIM=768    # Vector dimension size
```

### Generation Parameters
```env
GENERATION_TEMPERATURE=0.3  # 0.0-1.0 (lower = more factual)
MAX_TOKENS=2048             # Maximum response length
TOP_P=0.9                   # Nucleus sampling threshold
```

### Evaluation
```env
EVALUATION_MODEL=deepseek-r1  # Model for rubric scoring
JACCARD_THRESHOLD=0.25       # Similarity cutoff
```

## Usage

1. Generate answers (direct):
```bash
python run_exam.py
```

2. Generate answers (RAG):
```bash
python pinecone_rag.py
```

3. Evaluate results:
```bash
python evaluation.py
```

## Updating Parameters

Edit the `.env` file for any configuration changes:
- Add/remove models from `GENERATION_MODELS`
- Adjust RAG performance with `RAG_CHUNK_SIZE` and `RAG_TOP_K`
- Control answer creativity with `GENERATION_TEMPERATURE`
- Modify evaluation thresholds