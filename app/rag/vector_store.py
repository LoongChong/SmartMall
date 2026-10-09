from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

"""
这段代码是 RAG 系统中向量数据库构建与持久化（Vector Store Creation & Persistence） 的核心实现。

它的作用是：将之前准备好的文档块（Chunks）与 Embedding 模型结合，在内存中计算每个文档块的向量，
并将其以及元数据存储到本地磁盘上的 Chroma 向量数据库中。

核心作用
使用 Chroma 向量数据库，把前面切分并增强过的文档存入本地名为 "smartmall_knowledge" 的集合（Collection）中，
实现向量数据的本地持久化（保存到 data/vector_db 路径）。

这样一来，下次重启应用时就不需要重新计算向量，Agent 可以直接发起相似度检索。
"""
# 指定数据库在本地硬盘上的持久化存储目录。
CHROMA_DIR = Path("data/vector_db")

COLLECTION_NAME = "smartmall_knowledge"

# """
# 函数接收两个核心依赖：

# documents：包含文本内容（page_content）和元数据（metadata）的文档块列表。

# embeddings：上一节创建的 BAAI/bge-small-zh-v1.5 嵌入模型实例。
# """
def create_vector_store(
    documents: list[Document],
    embeddings: HuggingFaceEmbeddings,
) -> Chroma:
    # Chroma.from_documents 在后台默默为你完成了以下 4 个步骤：
# 提取文本：遍历 documents 中的每一个 Document 对象，提取其 page_content。
# 生成向量：调用 embeddings.embed_documents(...) 批量计算这些文本的高维数值向量（对应之前配置的 512 维向量）。
# 关联元数据：将得到的向量、对应的原始文本以及包含 knowledge_type / source 的 metadata 绑定在一起。
# 持久化落盘：将数据存储在指定的 collection_name 中，并保存到磁盘的 CHROMA_DIR（数据不会因为程序关闭而丢失）。
    # return Chroma.from_documents(
    #     documents=documents,
    #     embedding=embeddings,
    #     collection_name=COLLECTION_NAME,
    #     persist_directory=str(CHROMA_DIR),
    # )
    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(CHROMA_DIR),
    )
    vector_store.reset_collection()

    vector_store.add_documents(documents)

    return vector_store

