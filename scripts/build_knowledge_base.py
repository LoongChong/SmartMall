from app.rag.chunker import split_documents
from app.rag.embeddings import create_embeddings
from app.rag.loader import load_markdown_documents
from app.rag.metadata import enrich_metadata
from app.rag.vector_store import create_vector_store


DATA_DIR = "data/documents"


def main():
    print("1. Loading documents...")
    documents = load_markdown_documents(DATA_DIR)

    print(f"Loaded documents: {len(documents)}")

    print("2. Enriching metadata...")
    documents = [
        enrich_metadata(document)
        for document in documents
    ]

    print("3. Splitting documents...")
    chunks = split_documents(documents)

    print(f"Generated chunks: {len(chunks)}")

    print("4. Creating embeddings...")
    embeddings = create_embeddings()

    print("5. Building vector store...")
    create_vector_store(
        documents=chunks,
        embeddings=embeddings,
    )

    print("6. Knowledge base build completed.")


if __name__ == "__main__":
    main()