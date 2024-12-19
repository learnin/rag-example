from langchain_chroma.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStoreRetriever


VECTOR_STORE_PATH = "vector_store/chroma"


class ChromaVectoreStore:
    def __init__(self, embedding_model: Embeddings) -> None:
        self._vector_store = Chroma(embedding_function=embedding_model, persist_directory=VECTOR_STORE_PATH)

    def add_documents(self, docs: list[Document]) -> None:
        self._vector_store.add_documents(docs)

    def as_retriever(self) -> VectorStoreRetriever:
        return self._vector_store.as_retriever()

    def similarity_search(self, query: str, count: int) -> list[Document]:
        return self._vector_store.similarity_search(query, count)
