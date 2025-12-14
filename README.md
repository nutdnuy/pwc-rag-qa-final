# Lightweight RAG Q&A 

## Approach
- Chunk policy.txt into overlapping word windows
- Retrieve top-k chunks using TF-IDF cosine similarity
- Select the most relevant sentence as an extractive answer
- Refuse to answer if similarity is below a threshold

## Trade-offs
- TF-IDF is fast and transparent but not semantic
- Extractive answers avoid hallucination but are less flexible

## Install
```bash
pip install -r requirements.txt
```

## Run (CLI)
```bash
python qa.py --question "What are the core working hours?"
```

## Run (API)
```bash
uvicorn main:app --reload
```



## Refusal policy & confidence
- The tool refuses if `best_chunk_similarity < refuse_threshold` (default: 0.12).
- Use `--show_confidence` to print confidence scores in human output.
- Use `--show_citations` to print chunk id + character offsets in human output.

Examples:
```bash
python qa.py --question "What are the core working hours?" --show_confidence --show_citations
python qa.py --question "What is the dress code?" --show_confidence --refuse_threshold 0.20
```
