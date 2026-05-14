import chromadb
from chromadb.config import Settings
from typing import List, Optional
from embeddings import EmbeddingService
from llm_client import EmbeddingModelEntry
import uuid

class DocumentStore:
    """基于 ChromaDB 的文档存储与检索"""

    def __init__(self, persist_dir: str = "./chroma_data"):
        self.client = chromadb.PersistentClient(
            path=persist_dir,
            settings=Settings(anonymized_telemetry=False)
        )

        self.collection_name = "documents"

        # 如果集合已经存在就获取，否则创建
        try:
            self.collection = self.client.get_collection(self.collection_name)
            self.client.delete_collection(self.collection_name)
        except Exception as e:
            print(f"Process collection error: {str(e)}")
            
        self.collection = self.client.create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"} # 用余弦距离
        )

    def add_documents(self, texts: list[str], embeddings: list[list[float]], metadatas: Optional[list[dict]] = None):
        """批量添加文档"""
        ids = [str(uuid.uuid4()) for _ in texts]
        self.collection.add(
            ids = ids,
            documents=texts,
            embeddings=embeddings,
            # metadatas=metadatas or [{}] * len(texts)
        )
        return ids

    def is_empty(self) -> bool:
        """检查文档库是否为空"""
        return self.collection.count() == 0
    
    def cleanup(self):
        self.client.delete_collection(self.collection_name)

    def search(self, query_embedding: list[float], top_k: int = 1) -> dict:
        """检索相似的文章"""
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=['documents', 'metadatas', 'distances']
        )
        return {
            'documents': results['documents'][0],
            "metadatas": results["metadatas"][0],
            "distances": results["distances"][0]
        }
    

async def main():
    store = DocumentStore()
    documents=[
        "Java is used for backend development",
        "Vue.js is a frontend framework",
        "Python is widely used in AI"
    ]

    model_entry = EmbeddingModelEntry()
    embedding_service = EmbeddingService(model_entry.client, model_entry.config.model)

    embeddings = await embedding_service.embed(documents)

    store.add_documents(documents, embeddings)
    results = store.search(embeddings[2])
    print(results)

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())