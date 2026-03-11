from flask import Flask, render_template, request, Response, stream_with_context, jsonify
from src.helper import download_hugging_face_embeddings, load_pdf_file, add_metadata, text_split
from langchain_pinecone import PineconeVectorStore
from langchain_ollama import ChatOllama
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.globals import set_llm_cache
from langchain_community.cache import SQLiteCache
from dotenv import load_dotenv
from src.prompt import *
import os
import json
from pathlib import Path
from collections import OrderedDict
import re

app = Flask(__name__)
load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY is not set in the environment.")
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY

INDEX_NAME = os.getenv("PINECONE_INDEX", "medical-chatbot")
DATA_DIR = os.getenv("DATA_DIR", "data")
VECTOR_K = int(os.getenv("VECTOR_K", "4"))
BM25_K = int(os.getenv("BM25_K", "4"))
ENSEMBLE_BM25_WEIGHT = float(os.getenv("ENSEMBLE_BM25_WEIGHT", "0.35"))
ENSEMBLE_VECTOR_WEIGHT = float(os.getenv("ENSEMBLE_VECTOR_WEIGHT", "0.65"))

CACHE_DIR = Path(".cache"); CACHE_DIR.mkdir(exist_ok=True)
set_llm_cache(SQLiteCache(database_path=str(CACHE_DIR / "llm_cache.sqlite")))

RESPONSE_CACHE = OrderedDict()
RESPONSE_CACHE_MAX = int(os.getenv("RESPONSE_CACHE_MAX", "128"))


def get_cached_response(query: str):
    # Return cached answer if present
    cached = RESPONSE_CACHE.get(query)
    if cached:
        RESPONSE_CACHE.move_to_end(query)
    return cached


def set_cached_response(query: str, answer: str):
    # Store answer in in-memory LRU cache
    RESPONSE_CACHE[query] = answer; RESPONSE_CACHE.move_to_end(query)
    if len(RESPONSE_CACHE) > RESPONSE_CACHE_MAX: RESPONSE_CACHE.popitem(last=False)


embeddings = download_hugging_face_embeddings()

# Embed each chunk and upsert the embeddings into your Pinecone index.
docsearch = PineconeVectorStore.from_existing_index(
    index_name=INDEX_NAME,
    embedding=embeddings
)

def build_hybrid_retriever():
    # Combine BM25 keyword search with vector similarity, plus query rewriting
    vector_retriever = docsearch.as_retriever(
        search_type="similarity",
        search_kwargs={"k": VECTOR_K},
    )
    try:
        bm25_retriever = BM25Retriever.from_documents(
            text_split(add_metadata(load_pdf_file(DATA_DIR)))
        )
        bm25_retriever.k = BM25_K
        base_retriever = EnsembleRetriever(
            retrievers=[bm25_retriever, vector_retriever],
            weights=[ENSEMBLE_BM25_WEIGHT, ENSEMBLE_VECTOR_WEIGHT],
        )
    except Exception as exc:
        print(f"BM25 retriever unavailable, falling back to vector only: {exc}")
        base_retriever = vector_retriever
    try:
        return MultiQueryRetriever.from_llm(
            retriever=base_retriever,
            llm=query_llm,
            include_original=True,
        )
    except TypeError:
        return MultiQueryRetriever.from_llm(
            retriever=base_retriever,
            llm=query_llm,
        )


query_llm = ChatOllama(model="llama3.1:8b", temperature=0)
retriever = build_hybrid_retriever()

chatModel = ChatOllama(model="llama3.1:8b", temperature=0.2, streaming=True)
prompt = ChatPromptTemplate.from_messages([("system", system_prompt), ("human", "{input}")])
question_answer_chain = create_stuff_documents_chain(chatModel, prompt)


def retrieve_docs(query: str):
    # Fetch relevant documents from the retriever
    return retriever.invoke(query) if hasattr(retriever, "invoke") else retriever.get_relevant_documents(query)

def format_answer(text: str, max_sentences: int = 2, max_bullets: int = 3) -> str:
    # Normalize to short sentences or short bullet list
    cleaned = " ".join(text.strip().split())
    if not cleaned:
        return ""
    cleaned = re.sub(r"\s*•\s*", "\n- ", cleaned)
    cleaned = re.sub(r"\s*(\d+)\.\s+", "\n- ", cleaned)
    lines = [line.strip() for line in cleaned.splitlines() if line.strip()]
    bullet_lines = [line for line in lines if line.startswith("-")]
    if bullet_lines:
        return "\n".join(bullet_lines[:max_bullets])
    sentences = re.split(r"(?<=[.!?])\s+", cleaned)
    return " ".join(sentences[:max_sentences])


@app.route("/")
def index():
    # Render the chat UI
    return render_template('chat.html')

@app.route("/get", methods=["GET", "POST"])
def chat():
    # Handle non-streaming chat responses
    msg = request.form.get("msg") or (request.get_json(silent=True) or {}).get("msg")
    if not msg:
        return jsonify({"answer": ""}), 400
    print(msg)
    cached = get_cached_response(msg)
    if cached:
        return jsonify({"answer": cached})
    docs = retrieve_docs(msg)
    answer = question_answer_chain.invoke({"input": msg, "context": docs})
    print("Response : ", answer)
    answer_text = format_answer(str(answer))
    set_cached_response(msg, answer_text)
    return jsonify({"answer": answer_text})


@app.route("/stream", methods=["POST"])
def stream():
    # Stream tokens as they are generated
    payload = request.get_json(silent=True) or {}
    msg = payload.get("msg", "").strip()
    if not msg:
        return Response("Missing 'msg' in request body.", status=400)

    def generate():
        cached = get_cached_response(msg)
        if cached:
            yield json.dumps({"type": "token", "content": cached}) + "\n"
            yield json.dumps({"type": "done"}) + "\n"
            return
        docs = retrieve_docs(msg)
        answer_text = ""
        for chunk in question_answer_chain.stream({"input": msg, "context": docs}):
            if isinstance(chunk, dict):
                token = chunk.get("answer", "")
            else:
                token = str(chunk)
            if token:
                answer_text += token
                yield json.dumps({"type": "token", "content": token}) + "\n"
        formatted = format_answer(answer_text)
        set_cached_response(msg, formatted)
        yield json.dumps({"type": "final", "content": formatted}) + "\n"
        yield json.dumps({"type": "done"}) + "\n"

    return Response(stream_with_context(generate()), mimetype="application/x-ndjson")
if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=True)
