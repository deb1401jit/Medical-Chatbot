# Medical Chatbot

A retrieval-augmented medical chatbot that answers questions using curated PDF sources.

## Screenshot
![UI Screenshot](assets/ui.png)

## Features
- RAG pipeline with Pinecone vector store
- Hybrid retrieval (BM25 + vector similarity)
- Multi-query rewriting for better recall
- Streaming responses
- Embedding and response caching

## Tech Stack
- Python, Flask
- LangChain
- Pinecone
- Ollama
- SentenceTransformers

## Local Setup
1. Create and activate a virtual environment.
2. Install dependencies.
3. Set environment variables.
4. Ingest PDFs.
5. Run the app.

## Commands (CMD)
1. `cd /d "E:\Coding\Resume\Medical Chatbot"`
2. `python -m venv medical`
3. `medical\Scripts\activate.bat`
4. `pip install -r requirements.txt`
5. `set PINECONE_API_KEY=your_key_here`
6. `ollama serve`
7. `ollama pull llama3.1:8b`
8. `python ingest.py`
9. `python app.py`

## Environment Variables
- `PINECONE_API_KEY` (required)
- `PINECONE_INDEX` (optional, default: medical-chatbot)
- `DATA_DIR` (optional, default: data)
- `VECTOR_K` (optional, default: 4)
- `BM25_K` (optional, default: 4)
- `ENSEMBLE_BM25_WEIGHT` (optional, default: 0.35)
- `ENSEMBLE_VECTOR_WEIGHT` (optional, default: 0.65)
- `RESPONSE_CACHE_MAX` (optional, default: 128)

## Notes
- Put your PDFs in `data/` before running `ingest.py`.
- Open the UI at `http://localhost:8080`.
