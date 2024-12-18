from langchain_core.embeddings import Embeddings
from langchain_huggingface import HuggingFaceEmbeddings


EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-small"


def create_embedding_model() -> Embeddings:
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
