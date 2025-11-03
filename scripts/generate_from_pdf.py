# scripts/generate_from_pdf.py
import os
from src.ingest import pdf_to_text
from src.chunker import split_into_chunks
from src.indexer import build_index  # assumes you have a function to build index from chunks
from src.inference import generate_study_guide

pdf_path = "data/raw_notes/Lecture_16.pdf"  # change to your PDF
topic = "Short Line Model"                  # topic to generate for

# extract text
text = pdf_to_text(pdf_path)

# chunk & build a temporary index (index/temp.index, index/temp_meta.json)
chunks = split_into_chunks(text)
build_index(chunks, index_path="index/temp.index", meta_path="index/temp_meta.json")

# generate using the temp index (your query_index should read from temp index if implemented)
# if query_index defaults to index/faiss.index, either modify query_index to take index path,
# or replace the global index files with temp files temporarily.
out = generate_study_guide(topic, top_k=6)
print(out)
