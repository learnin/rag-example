import os

from hdbcli import dbapi
from langchain_community.vectorstores import HanaDB


VECTORE_STORE_TABLE_NAME="RAG_EXAMPLE_VECTOR_STORE"


class HanaVectoreStore:
    def __init__(self, embedding_model):
        self._conn = None
        self._embedding_model = embedding_model

    def add_documents(self, docs):
        conn = None
        try:
            self.open()
            vector_store = self._get_vector_store()
            vector_store.add_documents(docs)
        finally:
            self.close()

    def as_retriever(self):
        return self._get_vector_store().as_retriever()

    def open(self):
        self._conn = dbapi.connect(
            address=os.environ.get("DB_HOST"),
            port=os.environ.get("DB_PORT"),
            user=os.environ.get("DB_USER"),
            password=os.environ.get("DB_PASSWORD"),
            schema=os.environ.get("DB_SCHEMA")
        )

    def close(self):
        if self._conn is not None and self._conn.isconnected():
            self._conn.close()
            self._conn = None

    def similarity_search(self, query, count):
        return self._get_vector_store().similarity_search(query, count)

    def _get_vector_store(self):
        return HanaDB(embedding=self._embedding_model, connection=self._conn, table_name=VECTORE_STORE_TABLE_NAME)
