from src.ingest import pdf_to_text
from src.chunker import split_into_chunks
txt = pdf_to_text("/Users/sudhanshusaroj/myvenv/cheebo/data/raw_notes/Begumpura_e34bd92a-b610-4df1-9dac-fab7361f2c9d.pdf", debug_write=True)
print("Extracted text length:", len(txt))
print("First 400 chars:\\n", txt[:400])
chunks = split_into_chunks(txt, max_chars=1200, overlap_chars=200)
print("Total chunks:", len(chunks))
for i,c in enumerate(chunks[:6],1):
    print("---- chunk", i, "len", len(c))
    print(c[:600].replace("\\n","\\n"))