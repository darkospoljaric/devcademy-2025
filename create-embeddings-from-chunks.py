import json

from create_store import create_store
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from enum import Enum

CHUNK_SIZE = 50


class ChunkingStrategy(str, Enum):
  FIXED = "fixed"
  RECURSIVE = "recursive"
  SEMANTIC = "cluster_semantic"


store = create_store()
embeddingsModel = store.embeddings


def create_embeddings(string: str, strategy: ChunkingStrategy) -> list:
  """
  Create embeddings from text chunks based on the specified chunking strategy.

  Args:
      string: The input text to chunk
      strategy: The chunking strategy to apply

  Returns:
      list: The created embeddings
  """
  match strategy:
    case ChunkingStrategy.FIXED:
      return create_fixed_chunks(string)
    case ChunkingStrategy.RECURSIVE:
      return create_recursive_chunks(string)
    case ChunkingStrategy.SEMANTIC:
      return create_semantic_clusters(string)
    case _:
      raise ValueError(f"Unsupported chunking strategy: {strategy}")


def create_fixed_chunks(string: str) -> list:
  chunks = []

  for i in range(0, len(string), CHUNK_SIZE):
    # Extract chunk from position i to i+chunk_size
    chunk = string[i:i + CHUNK_SIZE]
    chunks.append(chunk)

  return chunks


def create_recursive_chunks(string: str) -> list:
  splitter = RecursiveCharacterTextSplitter(chunk_size=50, chunk_overlap=10)
  return splitter.split_text(text=string)


def create_semantic_clusters(string: str) -> list:
  spliter = SemanticChunker(embeddingsModel)
  return spliter.split_text(text=string)

with open('ragqa_arena_tech_corpus_reduced.jsonl', 'r') as json_file:
  json_list = list(json_file)

batch_size = 100

for text in json_list:
  data = json.loads(text)
  text = data["text"]

  fixedChunks = create_embeddings(text, ChunkingStrategy.FIXED)
  fixed_documents = [Document(chunk, metadata={"strategy": "FIXED"}) for chunk in fixedChunks]
  store.add_documents(fixed_documents)

  recursiveChunks = create_embeddings(text, ChunkingStrategy.RECURSIVE)
  recursive_documents = [Document(chunk, metadata={"strategy": "RECURSIVE"}) for chunk in recursiveChunks]
  store.add_documents(recursive_documents)

  semanticChunks = create_embeddings(text, ChunkingStrategy.SEMANTIC)
  semantic_documents = [Document(chunk, metadata={"strategy": "SEMANTIC"}) for chunk in semanticChunks]
  store.add_documents(semantic_documents)
