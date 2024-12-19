# RAG example

## Getting started

Required Python >= 3.10.

### When using SAP HANA Cloud for vector store

```shell
export OPENAI_API_KEY="Your OpenAI API key"
export DB_HOST="Your HANA Cloud Host"
export DB_PORT="443"
export DB_USER="Your HANA Cloud DB User"
export DB_PASSWORD="Your HANA Cloud DB password"
export DB_SCHEMA="Your HANA Cloud Schema"

python add_docs.py
python query.py
```

### When using Chroma for vector store

Edit add_docs.py and query.py for using ChromaVectoreStore.

```shell
export OPENAI_API_KEY="Your OpenAI API key"

python add_docs.py
python query.py
```
