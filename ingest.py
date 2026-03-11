import argparse
import os

from dotenv import load_dotenv
from langchain_pinecone import PineconeVectorStore

from src.helper import add_metadata, download_hugging_face_embeddings, load_pdf_file, text_split


def batch_iterable(items, batch_size: int):
    # Yield items in fixed-size batches
    for i in range(0, len(items), batch_size):
        yield items[i:i + batch_size]


def main():
    # Load PDFs, chunk them, and upsert embeddings in batches
    load_dotenv()

    parser = argparse.ArgumentParser(description="Ingest PDFs into Pinecone with batched embeddings.")
    parser.add_argument("--data-dir", default="data", help="Directory containing PDF files.")
    parser.add_argument("--index", default=os.environ.get("PINECONE_INDEX", "medical-chatbot"))
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    if not os.environ.get("PINECONE_API_KEY"):
        raise ValueError("PINECONE_API_KEY is not set in the environment.")

    docs = load_pdf_file(args.data_dir)
    docs = add_metadata(docs)
    chunks = text_split(docs)

    embeddings = download_hugging_face_embeddings()
    vectorstore = PineconeVectorStore.from_existing_index(
        index_name=args.index,
        embedding=embeddings,
    )

    for batch_no, batch in enumerate(batch_iterable(chunks, args.batch_size), start=1):
        vectorstore.add_documents(batch)
        print(f"Upserted batch {batch_no} ({len(batch)} chunks)")


if __name__ == "__main__":
    main()
