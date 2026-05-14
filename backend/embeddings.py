from openai import OpenAI
from typing import List
from llm_client import EmbeddingModelEntry
import numpy as np

class EmbeddingService:
    def __init__(self, client: OpenAI, model: str = "text-embedding-3-small"):
        self.client=client
        self.model=model

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """将文本列表转为向量列表"""
        batch_size = 10
        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            # 注意： embed 方法接受字符串或字符串列表
            response = self.client.embeddings.create(
                model = self.model,
                input = batch
            )
            all_embeddings.extend([d.embedding for d in response.data])

        return all_embeddings
    
    async def embed_single(self, text: str) -> list[float]:
        results = await self.embed([text])
        return results[0]
    
    @staticmethod
    def cosine_similarity(a: list[float], b: list[float]) -> float:
        """余弦相似度"""
        a = np.array(a)
        b = np.array(b)
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
    

async def main():
    model_entry = EmbeddingModelEntry()
    service = EmbeddingService(model_entry.client, model_entry.config.model)
    texts = [
        "苹果是一种很好吃的水果",
        "苹果公司发布了新款iPhone",
        "我喜欢吃香蕉"
    ]

    embeddings = await service.embed(texts)

    print(f"苹果水果 vs 苹果公司: {EmbeddingService.cosine_similarity(embeddings[0], embeddings[1]):.4f}")
    print(f"苹果水果 vs 香蕉: {EmbeddingService.cosine_similarity(embeddings[0], embeddings[2]):.4f}")


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())