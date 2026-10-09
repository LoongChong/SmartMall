"""
这段代码是 RAG（检索增强生成）系统 中极其标准且关键的 文档加载器（Document Loader） 接口实现。

在构建 Agent 系统时，Agent 常常需要访问本地的知识库（如商品说明、常见问题 FAQ、业务文档等）。
这段代码的作用就是遍历指定目录下的所有 Markdown 文档，将它们封装成 LangChain 标准的 Document 对象，
以便后续进行文本切分（Chunking）、向量化（Embedding）并存入向量数据库。

核心作用
把本地硬盘上的 .md 文本文件读取出来，
转换成 LangChain 整个生态（如文本切分器、向量数据库、检索器）都能通用处理的 Document 统一数据结构。
"""

# Path：来自 Python 标准库 pathlib，是现代 Python 推荐的面向对象的路径操作工具（比传统的 os.path 更好用）。
from pathlib import Path
# Document：来自 LangChain 的核心数据结构。
# LangChain 中所有的文档数据（不管是来自 PDF、网页、Markdown 还是数据库）最终都会被统一封装为 Document 对象。
from langchain_core.documents import Document


def load_markdown_documents(data_dir: str) -> list[Document]:
    documents: list[Document] = []

    root = Path(data_dir)
    # rglob("*.md")：核心亮点！ rglob 代表 Recursive Glob（递归匹配）。
    # 它不仅会找 root 根目录下的 .md 文件；
    # 如果 root 里面有嵌套的子文件夹（例如 root/mall/goods.md、root/rules/faq.md），
    # 它也会深度递归遍历把所有的 .md 文件全部找出来。
    for file_path in root.rglob("*.md"):
        content = file_path.read_text(encoding="utf-8")


    # Document(...) 实例化：LangChain 的 Document 类有两个核心属性：
    # page_content（必填，str）：文档的实际文本内容。Agent 在回答问题时，LLM 读的就是这段文字。
    # metadata（可选，dict）：文档的元数据。
    # "source"：记录文件的完整路径（如 "data/goods/iphone.md"）。
    # "file_name"：记录文件名（如 "iphone.md"）。
        document = Document(
            page_content=content,
            metadata={
                "source": str(file_path),
                "file_name": file_path.name,
            },
        )

        documents.append(document)

    return documents