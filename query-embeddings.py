
from create_store import create_store

store = create_store()
with open('ragqa_arena_tech_examples.jsonl', 'r') as json_file:
    examples = list(json_file)

    search1 = store.similarity_search(examples[0], k=2)
    print(f"Search results for example 0: {search1}")
    search2 = store.similarity_search(examples[1], k=2)
    print(f"Search results for example 0: {search2}")
