from langchain.globals import set_debug
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnablePassthrough
from langchain_openai import ChatOpenAI

from chroma_vectore_store import ChromaVectoreStore
from embedding import create_embedding_model
from hana_vectore_store import HanaVectorStore


def main(vector_store: HanaVectorStore | ChromaVectoreStore) -> None:
    set_debug(True)

    query = "2024年の流行語大賞は？"

    # hub.pull("rlm/rag-prompt") のプロンプトテンプレートの内容
    template = """You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.
Question: {question} 
Context: {context} 
Answer:
"""
    prompt = ChatPromptTemplate.from_template(template)
    llm = ChatOpenAI(model="gpt-4o-mini")
    chain: Runnable = {"context": vector_store.as_retriever(), "question": RunnablePassthrough()} | prompt | llm
    response = chain.invoke(query)
    print(response.content)


def similarity_search(vector_store: HanaVectorStore | ChromaVectoreStore) -> None:
    query = "2024年の流行語大賞は？"
    docs = vector_store.similarity_search(query, 5)
    print(docs)


if __name__ == "__main__":
    embedding_model = create_embedding_model()
    with HanaVectorStore(embedding_model) as vector_store:
        main(vector_store)
        # similarity_search(vector_store)
    # vector_store = ChromaVectoreStore(embedding_model)
    # main(vector_store)
    # similarity_search(vector_store)
