from uuid import uuid5, NAMESPACE_URL

from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models


class QdrantService:
    def __init__(self, qdrant_url: str, collection_name: str):
        self.client = AsyncQdrantClient(url=qdrant_url)
        self.collection_name = collection_name

    async def health_check(self) -> bool:
        try:
            await self.client.get_collections()
            return True
        except Exception:
            return False

    async def ensure_collection(self, vector_size: int) -> None:
        collections = await self.client.get_collections()
        existing_names = {collection.name for collection in collections.collections}
        if self.collection_name in existing_names:
            return

        await self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=models.VectorParams(
                size=vector_size,
                distance=models.Distance.COSINE,
            ),
        )

    async def upsert_document_chunks(self, document: dict, chunks: list[str], vectors: list[list[float]]) -> None:
        points = []
        for index, (chunk, vector) in enumerate(zip(chunks, vectors)):
            points.append(
                models.PointStruct(
                    id=str(uuid5(NAMESPACE_URL, f"{document['documentId']}-{index}")),
                    vector=vector,
                    payload={
                        "document_id": document["documentId"],
                        "filename": document["filename"],
                        "chunk_index": index,
                        "text": chunk,
                        "source": document["filename"],
                        "created_at": document["createdAt"],
                    },
                )
            )

        await self.client.upsert(collection_name=self.collection_name, points=points)

    async def search(self, query_vector: list[float], limit: int) -> list[dict]:
        response = await self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=limit,
            with_payload=True,
        )
        results = response.points
        return [
            {
                "score": result.score,
                "documentId": result.payload.get("document_id"),
                "filename": result.payload.get("filename"),
                "chunkIndex": result.payload.get("chunk_index"),
                "text": result.payload.get("text"),
                "source": result.payload.get("source"),
            }
            for result in results
        ]

    async def delete_document(self, document_id: str) -> None:
        await self.client.delete(
            collection_name=self.collection_name,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="document_id",
                            match=models.MatchValue(value=document_id),
                        )
                    ]
                )
            ),
        )
