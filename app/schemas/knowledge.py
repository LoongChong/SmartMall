from typing import Literal

from pydantic import BaseModel, Field


KnowledgeType = Literal[
    "product",
    "faq",
    "after_sale_policy",
]


class KnowledgeMetadata(BaseModel):
    knowledge_id: str = Field(description="知识唯一 ID")
    knowledge_type: KnowledgeType = Field(description="知识类型")
    title: str = Field(description="知识标题")

    category: str | None = None
    brand: str | None = None
    product_id: str | None = None

    source: str = Field(description="知识来源")