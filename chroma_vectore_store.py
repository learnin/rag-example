from langchain_chroma.vectorstores import Chroma


VECTOR_STORE_PATH="vector_store/chroma"


class ChromaVectoreStore:
    def __init__(self, embedding_model):
        self._embedding_model = embedding_model

    def add_documents(self, docs):
        vector_store = self._get_vector_store()
        vector_store.add_documents(docs)

    def as_retriever(self):
        return self._get_vector_store().as_retriever()

    def open(self):
        pass

    def close(self):
        pass

    def similarity_search(self, query, count):
        return self._get_vector_store().similarity_search(query, count)

    def _get_vector_store(self):
        return Chroma(embedding_function=self._embedding_model, persist_directory=VECTOR_STORE_PATH)
