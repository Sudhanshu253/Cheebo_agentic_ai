# Cheebo — Study Guide Generator

![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)

Transforms PDFs into structured, chunked text, indexes it, fine-tunes a model, and generates **human-readable study materials** — all in one automated flow.

This project was developed for the **Round 2 Task — Data Science Internship**.

---

## Features

* **PDF Ingestion:** Extracts raw text from PDF files.
* **Text Processing:** Intelligently chunks and cleans extracted text for optimal processing.
* **Vector Indexing:** Creates and stores efficient vector indexes for fast data retrieval.
* **Model Fine-Tuning:** Fine-tunes a model on your specific, processed document data.
* **Study Guide Generation:** Converts the model's complex JSON output into clean, human-readable `.txt` study materials.

---

## Setup

Make sure you have **Python 3.9** or newer installed.

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/your-repository-name.git](https://github.com/your-username/your-repository-name.git)
    cd your-repository-name
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

---

## 💡 How to Use

The project runs as a sequential pipeline. Run the scripts from the root directory in the following order:

### 1. Ingest PDF
Extracts and previews text/characters from your PDF files.
```bash
python src/ingest.py
2. Chunk and Clean Text

Chunks and cleans the extracted text for downstream tasks.

Bash
python src/chunker.py
3. Build Index

Creates and stores vector indexes for efficient retrieval.

Bash
python src/indexer.py
python src/make_index.py
4. Fine-tune Model

Fine-tunes your model on the processed and indexed data.

Bash
python src/fine_tune.py
5. Generate Model Output

Generates model outputs (as .json files) based on the fine-tuned model.

Bash
python src/generate_from_gold.py
6. Convert to Human-readable Text

Converts the generated JSON outputs from the outputs/ directory into readable .txt study materials.

Bash
python scripts/raw_to_readable.py "outputs/*.json"
📁 Project Structure
.
├── src/
│   ├── ingest.py
│   ├── chunker.py
│   ├── indexer.py
│   ├── make_index.py
│   ├── fine_tune.py
│   └── generate_from_gold.py
├── scripts/
│   └── raw_to_readable.py
├── outputs/
│   └── (Generated .json and .txt files will appear here)
└── requirements.txt
👤 Author
Sudhanshu Saroj

IIT Kanpur | Electrical Engineering
