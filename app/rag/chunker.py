from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# LangChain 当前的 RecursiveCharacterTextSplitter 就是通过递归尝试不同分隔符来找到合适的 chunk 边界。
def split_documents(documents: list[Document]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        # 含义：每个切分出来的文本块的最大字符数（Character Count）。

        # 为什么设为 500？ 这是一个在中文 RAG 中非常经典的平衡点。
        # 既保证了单个 Chunk 包含足够回答问题的语义上下文，又不会因为太长而引入噪声。
        chunk_size=500,

        # 含义：相邻两个 Chunk 之间重叠（重复）的字符数。
        # 为什么需要 Overlap？ 为了防止语义在切分边界处断裂！
        chunk_overlap=80,

        # 切分器会按列表中从上到下的顺序依次尝试分隔符，
        # 只有当按上一个分隔符切出来的块依然大于 chunk_size 时，才会降级使用下一个分隔符。
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