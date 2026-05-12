import httpx


class EmbeddingService:
    def __init__(self, ollama_url: str, model: str, timeout_seconds: float):
        self.ollama_url = ollama_url.rstrip("/")
        self.model = model
        self.timeout_seconds = timeout_seconds

    async def embed_text(self, text: str) -> list[float]:
        async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
            response = await client.post(
                f"{self.ollama_url}/api/embeddings",
                json={"model": self.model, "prompt": text},
            )
            response.raise_for_status()
            data = response.json()
            return data["embedding"]

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        vectors = []
        for text in texts:
            vectors.append(await self.embed_text(text))
        return vectors

    async def get_vector_size(self) -> int:
        return len(await self.embed_text("vector size probe"))
