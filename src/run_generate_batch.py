# src/run_generate_batch.py
from src.inference import generate_study_guide

topics = [
    "Short Line Model",
    "Medium Line Model",
    "Long Line Model",
    "Surge Impedance Loading",
    "Reactive Power Compensation"
]

for t in topics:
    print("Generating:", t)
    out = generate_study_guide(t, top_k=6)   # top_k adjusts how many FAISS chunks are used
    print("Saved:", f"outputs/{t.replace(' ','_')}.json")

