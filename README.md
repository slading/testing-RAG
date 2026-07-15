# Simple RAG with ChromaDB

A minimal Retrieval-Augmented Generation (RAG) example in Python. It stores facts in a vector database, retrieves the most relevant ones by semantic similarity, and generates an answer.

## How it works

1. **Embed** — facts are converted to vectors using `sentence-transformers` (`all-MiniLM-L6-v2`)
2. **Store** — vectors are saved in ChromaDB
3. **Retrieve** — the query is embedded and matched against stored facts by cosine similarity
4. **Generate** — the top results are passed to Groq LLM (or a simple fallback if no API key is set)

## Requirements

- Python 3.11+
- ~120 MB model download on first run (via Hugging Face)

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

```bash
python main.py
```

### Optional: Groq LLM for natural answers

Without an API key, the script returns the most relevant fact directly. With Groq, it generates a natural-language answer from the retrieved context.

```bash
export GROQ_API_KEY="your-key-here"
python main.py
```

Get a free API key at [console.groq.com](https://console.groq.com).

## Example output

```
Query: Where is the headquarters of the fintech company?

Found facts (the smaller the distance, the closer in meaning):
  1. [0.496] Capital.com is a fintech company that provides online trading and investment services.
  2. [0.542] Capital.com headquarters is located in Limassol, Cyprus.

Answer: Capital.com headquarters is located in Limassol, Cyprus.
```

## Project structure

```
.
├── main.py            # RAG pipeline: embed → retrieve → generate
├── requirements.txt   # Python dependencies
└── README.md
```
