# RAG-Anything

A multimodal RAG pipeline built on [RAG-Anything](https://github.com/hkuds/rag-anything). Processes documents containing text, tables, charts, graphs, and formulas, and answers questions about them using OpenAI.

## Prerequisites

- Python 3.10–3.13 (Python 3.14+ is not yet supported by the `mineru` dependency)
- An [OpenAI API key](https://platform.openai.com/api-keys)

## Setup

**1. Clone the repo**

```bash
git clone <your-repo-url>
cd ragAnything
```

**2. Create a virtual environment**

```bash
python3 -m venv .venv
```

> If your default `python3` is 3.14 or later, specify a version explicitly:
> `python3.13 -m venv .venv` or `python3.12 -m venv .venv`

**3. Activate the virtual environment**

```bash
# macOS / Linux
source .venv/bin/activate

# Windows (Command Prompt)
.venv\Scripts\activate.bat

# Windows (PowerShell)
.venv\Scripts\Activate.ps1
```

**4. Install dependencies**

```bash
pip install -r requirements.txt
```

**5. Configure your API key**

```bash
cp .env.example .env
# Open .env and replace `your_openai_api_key_here` with your actual key
```

## Usage

```bash
python main.py <path-to-file.txt> "Your question here"
```

**Example using the included sample:**

```bash
python main.py sample.txt "What is RAG and how does it work?"
```

The first run downloads MinerU's pipeline models (~first run only) and indexes the document into `rag_storage/`. Subsequent queries on the same document skip parsing and answer directly from the index.

## Configuration

All settings live in `.env` and are read automatically by RAGAnythingConfig:

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | — | Required. Your OpenAI API key |
| `LLM_MODEL` | `gpt-4o-mini` | Chat model for answering queries |
| `EMBEDDING_MODEL` | `text-embedding-3-large` | Embedding model for indexing |
| `EMBEDDING_DIM` | `3072` | Must match the embedding model's output dimension |
| `WORKING_DIR` | `./rag_storage` | Knowledge graph and vector store location |
| `OUTPUT_DIR` | `./output` | Parsed document output location |
| `VISION_MODEL` | `gpt-4o` | Vision model for describing charts, graphs, and images |
| `PARSE_METHOD` | `auto` | `auto` = MinerU picks the best method per document |
| `MINERU_BACKEND` | `pipeline` | `pipeline` = lightweight, works without a GPU |

## Supported content types

| Content | Handled by |
|---------|-----------|
| Text paragraphs | MinerU text extraction |
| Tables | MinerU table parser → indexed as structured text |
| Charts / graphs | MinerU image extraction → described by `VISION_MODEL` |
| Formulas / equations | MinerU MFR model → LaTeX extracted and indexed |

## Project structure

```
ragAnything/
├── main.py          # Entry point
├── sample.txt       # Example document for testing
├── requirements.txt # Python dependencies
├── .env.example     # Configuration template (safe to commit)
├── .env             # Your actual keys and settings (NOT committed)
├── .gitignore
└── README.md
```

## Notes

- `rag_storage/` and `output/` are generated at runtime and excluded from git.
- On first run, MinerU downloads layout and formula detection models (~once, then cached).
- `MINERU_BACKEND=pipeline` works on any machine without a GPU. Switch to `hybrid-auto-engine` if you have a GPU for higher accuracy.
- Powered by [LightRAG](https://github.com/HKUDS/LightRAG) under the hood.
