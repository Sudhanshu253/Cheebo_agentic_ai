# one-off script: src/make_index.py
from src.ingest import load_all_notes
from src.chunker import split_into_chunks
from src.indexer import build_index

docs = load_all_notes("data/raw_notes")
all_chunks = []
for fname, text in docs.items():
    chs = split_into_chunks(text)
    # optionally prefix chunk with filename/topic
    chs = [f"Source: {fname}\n\n{c}" for c in chs]
    all_chunks.extend(chs)

build_index(all_chunks)
