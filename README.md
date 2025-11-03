## Cheebo — Study Guide Generator

Transform PDFs into structured, chunked text, index it, fine-tune a model, and generate **human-readable study materials** — all in one flow.

---

## Setup

Make sure you have **Python ≥ 3.9** and all dependencies installed:

```bash
pip install -r requirements.txt
```
How to Use
1. Ingest PDF
   python src/ingest.py {Extract and preview text/characters from your PDF files.}
2. Chunk and Clean Text
   python src/chunker.py {Chunk and clean the extracted text for downstream tasks.}
3. Build Index
   python src/indexer.py
   python src/make_index.py {Create and store vector indexes for efficient retrieval.}
4. Fine-tune Model
   python src/fine_tune.py {Fine-tune your model on the processed and indexed data.}
5. Generate Output
   python src/generate_from_gold.py {Generate model outputs based on the fine-tuned model.}
6. Convert to Human-readable Text
   python scripts/raw_to_readable.py "outputs/*.json" {Convert generated JSON outputs into readable .txt study materials.}

# Project Structure
src/
├── ingest.py
├── chunker.py
├── indexer.py
├── make_index.py
├── fine_tune.py
├── generate_from_gold.py
scripts/
└── raw_to_readable.py
outputs/
└── (generated files)

## Author

# Sudhanshu Saroj
IIT Kanpur | Electrical Engineering
Round 2 Task — Data Science Internship




