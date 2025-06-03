from langchain_huggingface import HuggingFaceEmbeddings
from langchain_postgres import PGEngine, PGVectorStore


def create_store():
  embeddings_model = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")
  engine = PGEngine.from_connection_string(url="postgresql+asyncpg://admin:admin@localhost:5432/yourdb")

  return PGVectorStore.create_sync(
      engine=engine,
      table_name="embeddings_table",
      embedding_service=embeddings_model,
  )