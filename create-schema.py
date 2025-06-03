from langchain_postgres import PGEngine

engine = PGEngine.from_connection_string(url="postgresql+asyncpg://admin:admin@localhost:5432/yourdb")

engine.init_vectorstore_table(
    table_name="embeddings_table",
    vector_size=1024,
)
