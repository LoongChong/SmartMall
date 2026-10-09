from langchain_huggingface import HuggingFaceEmbeddings

"""
这段代码是 RAG 系统中向量化组件（Embeddings） 的初始化实现。

它的核心任务是：加载一个本地的开源文本嵌入模型，
将文本（如上一节切分好的 Chunk 或用户的提问 Prompt）转换成高维的数值向量（Vector/Embedding）。
只有转换成向量后，我们才能通过计算余弦相似度等数学方法，在向量数据库中检索出和用户提问最匹配的文档。

核心作用
初始化北京人工智能研究院（BAAI）开发的 bge-small-zh-v1.5 中文 Embedding 模型，
将其封装为 LangChain 统一的 HuggingFaceEmbeddings 对象，并且强制开启了向量归一化（Normalize Embeddings），
为后续的向量检索做准备。
"""
def create_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-zh-v1.5",
        model_kwargs={
            "device": "cpu",#指定模型在 CPU 上运行。
        },

        # """
        # normalize_embeddings=True（向量归一化）：
        # 数学意义：将生成的向量长度（模）压缩/标准化为 1。
        # 为什么这一步极其重要？
        # 在向量数据库（如 Chroma、Milvus、FAISS）中，我们通常使用 余弦相似度（Cosine Similarity） 或 点积（Dot Product） 来计算两段文本的相似程度。
        # 当向量被归一化后，点积计算（Dot Product）的结果在数值上直接等于余弦相似度！
        # 这不仅能大大加快向量数据库检索时的计算速度，还能保证不同长度文本生成的向量具有可比性。
        # """
        encode_kwargs={
            "normalize_embeddings": True,
        },
    )