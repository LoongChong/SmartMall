from pathlib import Path

from langchain_core.documents import Document

"""
这段代码是 RAG / Agent 系统中非常典型的 元数据增强（Metadata Enrichment） 或 文档属性打标（Document Tagging） 实现。
"""
def enrich_metadata(document: Document) -> Document:
    # file_path = Path(document.metadata["source"])
    source_path = document.metadata.get("source")
    if not source_path:
        return document
    file_path = Path(source_path)
    # parts 属性：这是 pathlib.Path 非常好用的一个特性，
    # 它会把路径拆解成一个包含了各个目录/文件名层级的元组（tuple）。
    # 例如：如果路径是 "data/knowledge/products/electronics/iphone15.md"
    # file_path.parts 的结果就是：('data', 'knowledge', 'products', 'electronics', 'iphone15.md')
    parts = file_path.parts

    knowledge_type = "unknown"

    if "products" in parts:
        knowledge_type = "product"
    elif "faq" in parts:
        knowledge_type = "faq"
    elif "after_sale" in parts:
        knowledge_type = "after_sale_policy"

    document.metadata["knowledge_type"] = knowledge_type

    return document