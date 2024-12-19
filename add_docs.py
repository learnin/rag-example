from collections.abc import Iterable

from langchain_community.document_loaders import UnstructuredHTMLLoader
from langchain_core.documents import Document
from langchain_text_splitters.character import CharacterTextSplitter

from chroma_vectore_store import ChromaVectoreStore
from embedding import create_embedding_model
from hana_vectore_store import HanaVectorStore


def main(vector_store: HanaVectorStore | ChromaVectoreStore) -> None:
    doc_path = "./docs/新語・流行語大賞-Wikipedia.html"
    loader = UnstructuredHTMLLoader(doc_path)
    docs = loader.load()
    chunked_docs = split_docs(docs, 512, 128)
    # print(chunked_docs)
    vector_store.add_documents(chunked_docs)


def split_docs(docs: Iterable[Document], chunk_size: int, chunk_overlap: int) -> list[Document]:
    text_splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return text_splitter.split_documents(docs)


if __name__ == "__main__":
    embedding_model = create_embedding_model()
    vector_store = HanaVectorStore(embedding_model)
    # vector_store = ChromaVectoreStore(embedding_model)
    main(vector_store)
