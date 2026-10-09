from langchain_chroma import Chroma


def create_retriever(vector_store: Chroma):
    # 转换为带有元数据过滤条件的检索器
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={
            "k": 3,  # 返回最相似的前 3 个结果
            # "filter": {"knowledge_type": "product"}  # 结合前面打了标签的元数据做过滤！
        }
    )
    return retriever 