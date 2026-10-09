from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

from app.rag.embeddings import create_embeddings
from app.rag.loader import load_markdown_documents
from app.rag.metadata import enrich_metadata


DATA_DIR = "data/documents"


@dataclass
class EvalQuery:
    query: str
    source_file: str
    expected_keyword: str


EVAL_QUERIES = [
    EvalQuery(
        query="星云 X1 Pro 支持多少瓦无线充电？",
        source_file="phone_x1_pro.md",
        expected_keyword="30W 无线充电",
    ),
    EvalQuery(
        query="星云 X1 Pro 的电池容量是多少？",
        source_file="phone_x1_pro.md",
        expected_keyword="5500mAh",
    ),
    EvalQuery(
        query="星云 AirBook 14 适合什么人群？",
        source_file="laptop_air_14.md",
        expected_keyword="大学生",
    ),
    EvalQuery(
        query="商城支持哪些支付方式？",
        source_file="payment.md",
        expected_keyword="微信支付",
    ),
    EvalQuery(
        query="拆封后的商品还能七天无理由退货吗？",
        source_file="seven_day_return.md",
        expected_keyword="拆封后可能影响七天无理由退货资格",
    ),
]


# ============================================================
# A. Recursive Chunking
# ============================================================

def recursive_split(
    documents: list[Document],
) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=220,
        chunk_overlap=40,
        add_start_index=True,
        separators=[
            "\n## ",
            "\n### ",
            "\n\n",
            "\n",
            "。",
            "！",
            "？",
            "，",
            "、",
            " ",
            "",
        ],
    )

    return splitter.split_documents(documents)


# ============================================================
# B. Markdown Structure-Aware Chunking
# ============================================================

def markdown_split(
    documents: list[Document],
) -> list[Document]:
    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[
            ("#", "header_1"),
            ("##", "header_2"),
            ("###", "header_3"),
        ],
        strip_headers=False,
    )

    size_splitter = RecursiveCharacterTextSplitter(
        chunk_size=220,
        chunk_overlap=40,
    )

    results: list[Document] = []

    for document in documents:
        markdown_documents = header_splitter.split_text(
            document.page_content
        )

        for markdown_document in markdown_documents:
            markdown_document.metadata.update(
                {
                    "source": document.metadata.get("source"),
                    "file_name": document.metadata.get("file_name"),
                    "knowledge_type": document.metadata.get(
                        "knowledge_type"
                    ),
                }
            )

        # 标题切完后，再进行尺寸控制
        chunks = size_splitter.split_documents(
            markdown_documents
        )

        results.extend(chunks)

    return results


# ============================================================
# C. Parent-Child Chunking
# ============================================================

def build_parent_child_documents(
    documents: list[Document],
) -> tuple[list[Document], dict[str, Document]]:
    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=[
            "\n## ",
            "\n### ",
            "\n\n",
            "\n",
            "。",
            "！",
            "？",
            "",
        ],
    )

    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=160,
        chunk_overlap=30,
        separators=[
            "\n",
            "。",
            "！",
            "？",
            "，",
            " ",
            "",
        ],
    )

    child_documents: list[Document] = []
    parent_documents: dict[str, Document] = {}

    parent_counter = 0
    child_counter = 0

    for document in documents:
        parents = parent_splitter.split_documents([document])

        for parent in parents:
            parent_counter += 1

            parent_id = f"parent_{parent_counter:04d}"

            parent.metadata["parent_id"] = parent_id

            parent_documents[parent_id] = parent

            children = child_splitter.split_documents([parent])

            for child in children:
                child_counter += 1

                child.metadata["parent_id"] = parent_id
                child.metadata["child_id"] = (
                    f"child_{child_counter:04d}"
                )

                child_documents.append(child)

    return child_documents, parent_documents


# ============================================================
# Vector Store
# ============================================================

def create_in_memory_vector_store(
    documents: list[Document],
    embeddings,
    collection_name: str,
) -> Chroma:

    vector_store = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
    )

    vector_store.add_documents(documents)

    return vector_store


# ============================================================
# Evaluation
# ============================================================

def document_matches(
    document: Document,
    eval_query: EvalQuery,
) -> bool:

    source = document.metadata.get("source", "")
    file_name = document.metadata.get("file_name", "")
    # source_match：检查文档来源路径或文件名是否符合期望（匹配上一节写进 metadata 的路径/文件名）。
    source_match = (
        eval_query.source_file in source
        or eval_query.source_file == file_name
    )
    # keyword_match：检查文档内容中是否包含预期出现的关键特征词（expected_keyword）。
    keyword_match = (
        eval_query.expected_keyword
        in document.page_content
    )

    return source_match and keyword_match


def evaluate_vector_store(
    vector_store: Chroma,
    queries: list[EvalQuery],
    k: int = 3,
) -> tuple[float, float]:

    recall_at_k = 0
    reciprocal_rank_sum = 0.0

# 调用 Chroma 数据库的 similarity_search 方法，将当前测试用例的问题文本（item.query）进行向量化，并与数据库中存储的文档向量计算相似度。
# k=k：限定数据库只返回相似度最高的前 k 个（默认 3 个）Document 列表。

    for item in queries:
        documents = vector_store.similarity_search(
            item.query,
            k=k,
        )

        first_hit_rank = None

        for rank, document in enumerate(
            documents,
            start=1,
        ):
            if document_matches(document, item):
                first_hit_rank = rank
                break

        if first_hit_rank is not None:
            recall_at_k += 1
            reciprocal_rank_sum += 1.0 / first_hit_rank

    recall = recall_at_k / len(queries)
    mrr = reciprocal_rank_sum / len(queries)

    return recall, mrr


# ============================================================
# Parent-Child Retrieval
# ============================================================

def evaluate_parent_child(
    child_vector_store: Chroma,
    parent_documents: dict[str, Document],
    queries: list[EvalQuery],
    k: int = 3,
) -> tuple[float, float]:

    recall_at_k = 0
    reciprocal_rank_sum = 0.0

    for item in queries:

        child_hits = child_vector_store.similarity_search(
            item.query,
            k=k,
        )

        parent_ids: list[str] = []

        for child in child_hits:
            parent_id = child.metadata.get("parent_id")

            if parent_id and parent_id not in parent_ids:
                parent_ids.append(parent_id)

        parent_hits = [
            parent_documents[parent_id]
            for parent_id in parent_ids
            if parent_id in parent_documents
        ]

        first_hit_rank = None

        for rank, parent in enumerate(
            parent_hits,
            start=1,
        ):
            if document_matches(parent, item):
                first_hit_rank = rank
                break

        if first_hit_rank is not None:
            recall_at_k += 1
            reciprocal_rank_sum += 1.0 / first_hit_rank

    recall = recall_at_k / len(queries)
    mrr = reciprocal_rank_sum / len(queries)

    return recall, mrr


# ============================================================
# Debug Output
# ============================================================

def print_chunks(
    name: str,
    chunks: list[Document],
) -> None:

    print("\n" + "=" * 80)
    print(name)
    print("=" * 80)

    print(f"Chunk count: {len(chunks)}")

    for index, chunk in enumerate(
        chunks[:10],
        start=1,
    ):
        print("\n" + "-" * 60)
        print(f"Chunk #{index}")
        print(f"Metadata: {chunk.metadata}")
        print(chunk.page_content[:300])


# ============================================================
# Main
# ============================================================

def main():

    print("Loading source documents...")

    documents = load_markdown_documents(
        DATA_DIR
    )

    documents = [
        enrich_metadata(document)
        for document in documents
    ]

    print(f"Source documents: {len(documents)}")

    # --------------------------
    # A
    # --------------------------

    recursive_chunks = recursive_split(
        documents
    )

    print_chunks(
        "A. Recursive Chunking",
        recursive_chunks,
    )

    # --------------------------
    # B
    # --------------------------

    markdown_chunks = markdown_split(
        documents
    )

    print_chunks(
        "B. Markdown Structure-Aware Chunking",
        markdown_chunks,
    )

    # --------------------------
    # C
    # --------------------------

    child_chunks, parent_documents = (
        build_parent_child_documents(
            documents
        )
    )

    print_chunks(
        "C. Parent-Child Child Chunks",
        child_chunks,
    )

    print(
        f"\nParent count: "
        f"{len(parent_documents)}"
    )

    # --------------------------
    # Embeddings
    # --------------------------

    print("\nCreating embeddings...")

    embeddings = create_embeddings()

    # --------------------------
    # Build vector stores
    # --------------------------

    recursive_store = (
        create_in_memory_vector_store(
            recursive_chunks,
            embeddings,
            "lesson4_recursive",
        )
    )

    markdown_store = (
        create_in_memory_vector_store(
            markdown_chunks,
            embeddings,
            "lesson4_markdown",
        )
    )

    parent_child_store = (
        create_in_memory_vector_store(
            child_chunks,
            embeddings,
            "lesson4_parent_child",
        )
    )

    # --------------------------
    # Evaluation
    # --------------------------

    recursive_recall, recursive_mrr = (
        evaluate_vector_store(
            recursive_store,
            EVAL_QUERIES,
        )
    )

    markdown_recall, markdown_mrr = (
        evaluate_vector_store(
            markdown_store,
            EVAL_QUERIES,
        )
    )

    parent_recall, parent_mrr = (
        evaluate_parent_child(
            parent_child_store,
            parent_documents,
            EVAL_QUERIES,
        )
    )

    print("\n")
    print("=" * 80)
    print("Chunking Evaluation")
    print("=" * 80)

    print(
        f"\nRecursive:"
        f"\n  Recall@3 = {recursive_recall:.2f}"
        f"\n  MRR      = {recursive_mrr:.2f}"
    )

    print(
        f"\nMarkdown:"
        f"\n  Recall@3 = {markdown_recall:.2f}"
        f"\n  MRR      = {markdown_mrr:.2f}"
    )

    print(
        f"\nParent-Child:"
        f"\n  Recall@3 = {parent_recall:.2f}"
        f"\n  MRR      = {parent_mrr:.2f}"
    )


if __name__ == "__main__":
    main()