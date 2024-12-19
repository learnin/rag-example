import os
from types import TracebackType
from typing_extensions import Self

from hdbcli import dbapi
from langchain_community.vectorstores import HanaDB
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStoreRetriever


VECTORE_STORE_TABLE_NAME = "RAG_EXAMPLE_VECTOR_STORE"

MSG_VECTOR_STORE_NOT_INITIALIZED = "HanaVectorStore is not initialized. Use 'with' statement to initialize the HanaVectorStore."


class HanaVectorStore:
    """SAP HANA に接続してベクトルストアを操作するクラス
    
        Usage:
            with HanaVectorStore(embedding_model) as vector_store:
                vector_store.add_documents(docs)
    """
    def __init__(self, embedding_model: Embeddings) -> None:
        self._conn = None
        self._embedding_model = embedding_model
        self._vector_store: HanaDB | None = None

    def __enter__(self) -> Self:
        self._open()
        return self

    def __exit__(self, exc_type: type[BaseException] | None, exc_value: BaseException | None, traceback: TracebackType | None) -> None:
        self._close()

    def add_documents(self, docs: list[Document]) -> None:
        assert self._vector_store is not None, MSG_VECTOR_STORE_NOT_INITIALIZED
        self._vector_store.add_documents(docs)

    def as_retriever(self) -> VectorStoreRetriever:
        assert self._vector_store is not None, MSG_VECTOR_STORE_NOT_INITIALIZED
        return self._vector_store.as_retriever()

    def similarity_search(self, query: str, count: int) -> list[Document]:
        assert self._vector_store is not None, MSG_VECTOR_STORE_NOT_INITIALIZED
        return self._vector_store.similarity_search(query, count)

    def _open(self) -> None:
        if self._conn is not None and self._conn.isconnected():
            return

        self._conn = dbapi.connect(
            address=os.environ.get("DB_HOST"),
            port=os.environ.get("DB_PORT"),
            user=os.environ.get("DB_USER"),
            password=os.environ.get("DB_PASSWORD"),
            schema=os.environ.get("DB_SCHEMA")
        )
        self._vector_store = HanaDB(embedding=self._embedding_model, connection=self._conn, table_name=VECTORE_STORE_TABLE_NAME)

    def _close(self) -> None:
        if self._conn is not None and self._conn.isconnected():
            self._conn.close()
            self._conn = None
