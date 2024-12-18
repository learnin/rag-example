from langchain_chroma.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore, VectorStoreRetriever


VECTOR_STORE_PATH = "vector_store/chroma"


class ChromaVectoreStore:
    def __init__(self, embedding_model: Embeddings) -> None:
        self._embedding_model = embedding_model

    def add_documents(self, docs: list[Document]) -> None:
        vector_store = self._get_vector_store()
        vector_store.add_documents(docs)

    def as_retriever(self) -> VectorStoreRetriever:
        return self._get_vector_store().as_retriever()

    def open(self) -> None:
        pass

    def close(self) -> None:
        pass

    def similarity_search(self, query: str, count: int) -> list[Document]:
        return self._get_vector_store().similarity_search(query, count)

    def _get_vector_store(self) -> VectorStore:
        return Chroma(embedding_function=self._embedding_model, persist_directory=VECTOR_STORE_PATH)
