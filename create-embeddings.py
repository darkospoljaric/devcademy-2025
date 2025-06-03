import json

from langchain_core.documents import Document
from create_store import create_store

store = create_store()
with open('ragqa_arena_tech_corpus_reduced.jsonl', 'r') as json_file:
  json_list = list(json_file)

batch_size = 100
docs = [Document(page_content=json.loads(text)["text"]) for text in json_list]
store.add_documents(docs)

# for i in range(0, len(docs), batch_size):
#   batch = docs[i:i + batch_size]
#   documents = store.add_documents(batch)
#   print (f"Added documents {i} to {i + batch_size} to the store.")