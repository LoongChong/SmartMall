from app.rag.embeddings import create_embeddings
from app.rag.retriever import create_retriever
from app.rag.vector_store import CHROMA_DIR, COLLECTION_NAME
from langchain_chroma import Chroma


def main():
    embeddings = create_embeddings()

    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(CHROMA_DIR),
    )

    retriever = create_retriever(vector_store)

    query = "星云 X1 Pro 支持无线充电吗？"

    documents = retriever.invoke(query)

    print(f"\nQuery: {query}")
    print(f"Retrieved: {len(documents)} documents\n")

    for index, document in enumerate(documents, start=1):
        print("=" * 60)
        print(f"Document #{index}")
        print(f"Metadata: {document.metadata}")
        print(document.page_content)
        print()


if __name__ == "__main__":
    main()