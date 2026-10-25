# LocalRAG

A **private, fully local "chat with your documents" assistant.** Drop in your notes, PDFs, and files, then ask questions and get answers with citations pointing to the exact passages. A local model runs on your machine through [Ollama](https://ollama.com), so **nothing is sent to any cloud AI company**.

It is also a learning project: the code is small and readable, and it includes an evaluation harness so you can measure your system instead of just trusting it.

```
 your files ──► load ──► chunk ──► embed ──► ChromaDB (on disk)
                                                   │
 your question ──► embed ──► nearest chunks ◄──────┘
                                  │
                    prompt = rules + numbered excerpts + question
                                  │
                       local chat model (Ollama)
                                  │
              answer with [1][2] citations  (or "I couldn't find that")
```

## Features

- 100% local: models, vector database, and chat history stay on your disk
- Reads PDF, Word, PowerPoint, Excel, HTML and plain text (see the table below). Citations say `(p.3)`, `(slide 4)` or `(sheet Budget)`
- **Hybrid search:** meaning-based (embeddings) plus exact-keyword (BM25) matching, merged together, so names, codes and terms like `sta302` or `refund` are found reliably
- **Smart scoping:** mention a course/file code (`eco320`) or a file name in your question and search is limited to those files; say `slide 3` or `page 12` and it reads exactly that page
- Citations with similarity scores, so you can verify every answer
- Refuses to guess: if nothing relevant is found, it says so
- Smart re-indexing: files are hashed, so only new or edited files are re-embedded, and deleted files are pruned
- Follow-up questions work ("what about the second one?") via question rewriting
- Terminal chat **and** a web UI (Streamlit)
- Evaluation harness with retrieval and answer metrics
- Tests that run without any model installed

## Quickstart

### 1. Install Ollama and pull two models

Install Ollama from <https://ollama.com/download>, then:

```bash
ollama pull llama3.2            # the chat model (about 2 GB)
ollama pull nomic-embed-text    # the embedding model (about 270 MB)
```

> Low on RAM (8 GB)? `llama3.2` (3B) is a good fit. Have 16 GB+ or a GPU? Try `qwen2.5:7b` or `gemma3`, and set `chat_model` in `config.yaml`.

### 2. Install LocalRAG

Requires Python 3.10+.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

### 3. Check everything is wired up

```bash
localrag doctor
```

### 4. Add your documents and index them

Copy files into `data/docs/` (subfolders are fine). Two sample files are in `data/docs/example/` so you can try it right away.

```bash
localrag ingest
```

### 5. Ask questions

```bash
localrag ask "What is the default chunk size?"   # one-off question
localrag chat                                     # interactive terminal chat
localrag serve                                    # web UI at http://localhost:8501
```

(`python -m localrag <command>` works too if you skip the editable install.)

## Asking questions that point at specific files or pages

Search is by *meaning*, so vague references can't work on their own. LocalRAG understands these patterns:

| You type | What happens |
|---|---|
| `what percent is the midterm worth for sta302` | `sta302` is recognised as a course code that appears in your file paths, so only those files are searched |
| `summarize slide 3 of Dewees - SO2 in Sudbury` | the file name matches one document, and page/slide 3 of it is read directly |
| `summarize slide 3 of eco320` | if that matches more than 3 files with a slide 3, it lists them and asks you to name one |
| `compare eco320 and sta302` | both courses are searched |

Tips: put identifiers like course codes in your folder or file names, and for PDFs "slide N" means page N. Plain words like "midterm" don't narrow the search on purpose, because they'd hide the syllabus.

## Supported file types

| Type | Extensions | How it's read | Citation looks like |
|---|---|---|---|
| PDF | `.pdf` | text per page | `report.pdf (p.3)` |
| Word | `.docx` | paragraphs and tables in order | `policy.docx` |
| PowerPoint | `.pptx` | one section per slide, including speaker notes and tables | `deck.pptx (slide 4)` |
| Excel | `.xlsx`, `.xlsm` | one section per sheet; if row 1 is a header, each row becomes `Column: value` pairs | `sales.xlsx (sheet Q1)` |
| Web pages | `.html`, `.htm` | visible text only (scripts and styles removed), page title kept | `page.html` |
| Text / code / data | `.txt` `.md` `.rst` `.py` `.csv` `.json` | read as plain text | `notes.md` |

Limits to know about:

- **Old Office formats** (`.doc`, `.ppt`, `.xls`) are not readable. Ingest warns you; re-save them as `.docx`, `.pptx`, `.xlsx`.
- **Scanned PDFs and images** have no text layer, so they are skipped. OCR them first (e.g. `ocrmypdf scan.pdf scan_ocr.pdf`).
- **Excel formulas** are read as their last saved value (open and save the file in Excel once if values look empty). Charts, images, and comments are ignored.
- **Spreadsheets in general** are the weakest case for RAG: lookups like "what was Europe's revenue in Q1?" work, but "sum all of March" does not, because the model sees a few matching rows and cannot calculate over the whole sheet.
- **Password-protected files** fail with an error and are skipped; the rest of the batch still indexes.
- Word and PowerPoint temp files (`~$something.docx`) are ignored automatically.

To restrict or extend what gets indexed, edit `extensions` in `config.yaml`.

## Commands

| Command | What it does |
|---|---|
| `localrag doctor` | Checks that Ollama is running and the models are installed |
| `localrag ingest [--docs PATH] [--force]` | Indexes documents. `--force` re-embeds everything |
| `localrag ask "question" [--top-k N]` | Answers one question with sources |
| `localrag chat` | Interactive chat with memory of the conversation |
| `localrag serve` | Launches the web UI (bound to localhost, telemetry off) |
| `localrag stats` | Lists indexed documents and chunk counts |
| `localrag reset [--yes]` | Deletes the index (never touches your original files) |
| `localrag eval [--no-generate] [--questions FILE] [--top-k N]` | Runs the evaluation set |

## Configuration

Edit `config.yaml` (every key is optional). Point to another file with `--config path.yaml` or `LOCALRAG_CONFIG=path.yaml`.

| Key | Default | Meaning |
|---|---|---|
| `chat_model` | `llama3.2` | Ollama model used to write answers |
| `embed_model` | `nomic-embed-text` | Ollama model used to embed text |
| `query_prefix` / `doc_prefix` | nomic prefixes | Task prefixes some embedders expect. Set to `""` for other embedding models |
| `chunk_size` / `chunk_overlap` | `800` / `150` | Characters per chunk and overlap between neighbours |
| `top_k` | `8` | How many passages the model sees |
| `hybrid` | `true` | Combine embedding search with exact-keyword (BM25) search |
| `candidate_pool` | `30` | Candidates each search contributes before merging down to `top_k` |
| `min_similarity` | `0.30` | Below this, the assistant refuses instead of guessing |
| `condense_followups` | `true` | Rewrite follow-up questions into standalone ones |
| `temperature` | `0.2` | Lower is more literal |
| `num_ctx` | `4096` | Model context window. Raise it if you raise `top_k` or `chunk_size` |
| `docs_dir` / `db_dir` | `data/docs` / `data/chroma` | Where documents and the index live |

**Important:** if you change `embed_model`, `chunk_size`, `chunk_overlap`, or the prefixes, run `localrag reset --yes && localrag ingest` so the whole index is rebuilt consistently.

## Evaluating your system (the part that makes you a better engineer)

Edit `eval/questions.example.jsonl` (or copy it) with 20 to 30 questions about **your** documents. Each line is JSON:

```json
{"question": "How many hours do ferrets sleep?",
 "expected_sources": ["example/ferret_care.txt"],
 "expected_keywords": ["14", "18"],
 "should_refuse": false}
```

Then run:

```bash
localrag eval --no-generate     # fast: tests retrieval only
localrag eval                   # full: also generates answers and checks them
```

Metrics reported:

- **retrieval_hit_rate**: how often the right file appears in the top-k. If this is low, the model never had a chance. Fix retrieval first.
- **answer_accuracy**: how often the answer contains all expected keywords.
- **refusal_accuracy**: for questions your documents cannot answer, how often it correctly refuses.
- **false_refusal_rate**: how often it refuses a question it should have answered (usually `min_similarity` is too high).

Reports are saved to `eval/results/` with the config used, so you can compare runs.

### An experiment loop to try

Change **one thing at a time**, re-ingest if needed, re-run the eval, and keep a log:

1. Baseline with the defaults
2. `chunk_size` 400 vs 800 vs 1200
3. `top_k` 3 vs 5 vs 8
4. A different embedding model (`mxbai-embed-large`, `bge-m3`)
5. A bigger chat model
6. `min_similarity` 0.2 vs 0.3 vs 0.4

## Tuning and troubleshooting

| Symptom | Likely fix |
|---|---|
| Answers are wrong but sources look right | Try a bigger chat model, or lower `temperature` |
| Right file never retrieved | Mention its course code or file name in the question. Otherwise: smaller `chunk_size`, higher `top_k`, or a better embedding model |
| Answer exists but it says "couldn't find that" | Check the file is indexed: `localrag stats` (Windows: `localrag stats | findstr sta302`). Scanned/image PDFs and slides-as-images have no text and are skipped |
| "I couldn't find that" for things that are in your docs | Lower `min_similarity` (try 0.2) |
| It answers things that aren't in your docs | Raise `min_similarity` (try 0.4) |
| Answers cut off or ignore later sources | Raise `num_ctx` |
| `Cannot reach Ollama` | Start it with `ollama serve` (or open the Ollama app) |
| `Missing models` | Run the `ollama pull ...` command the error prints |
| A PDF shows "no extractable text" | It is probably a scan. OCR it first (e.g. `ocrmypdf`) |
| Slow on CPU | Use a smaller chat model, lower `top_k`, shorter chunks |

## Privacy notes

- Ollama, ChromaDB, and this app all run on `localhost`. The only network access is the one-time model download.
- `.gitignore` excludes your documents (`data/docs/*`), your index, and eval results, so you cannot accidentally commit private data.
- Chat history in the web UI lives in memory only and disappears when you close the tab. The index in `data/chroma/` is **not encrypted**; use full-disk encryption on shared machines.
- For airtight certainty, run it with Wi-Fi off after the models are downloaded.
- Be careful what you index: anything in `data/docs` can be surfaced in answers.

## Project structure

```
localrag/
  config.py      settings and validation
  ollama.py      tiny client for the local Ollama API (embeddings and streaming chat)
  loaders.py     read pdf/docx/pptx/xlsx/html/text into page-, slide- or sheet-aware sections
  chunking.py    boundary-aware overlapping chunker
  store.py       ChromaDB wrapper (cosine similarity)
  pipeline.py    ingest, hybrid retrieve, page lookup, condense follow-ups, prompt, answer
  scope.py       detects course codes, file names and 'slide N' inside a question
  bm25.py        dependency-free keyword index used for hybrid search
  evaluate.py    evaluation harness and report writer
  cli.py         command line interface
  webapp.py      Streamlit web UI
tests/           pytest suite (uses a fake client, no models required)
eval/            example evaluation questions
data/docs/       put your documents here
config.yaml      configuration
```

Run the tests with `pytest -q`.

## Where to go next

A suggested learning path, roughly in order of difficulty:

1. **Build your own eval set** from real questions you care about and push the scores up.
2. **Reranking:** retrieve 20 chunks, then re-score them with a cross-encoder (e.g. `bge-reranker`) and keep the best 5.
3. **Richer metadata filters:** filter by date or file type, not just course code and file name.
4. **More file types:** `.epub`, email exports (`.eml`/`.mbox`), Obsidian vaults, OCR for scanned PDFs and images.
5. **File watcher:** auto-ingest when files change (`watchdog`).
6. **Tool use / agents:** let the model call tools (calculator, calendar, local search) via Ollama's tool-calling support.
7. **Fine-tuning:** fine-tune a small model on your own data or writing style with [Unsloth](https://github.com/unslothai/unsloth), then serve it through Ollama.
8. **Package it:** a Dockerfile or a one-command installer so a friend can use it.

## License

MIT. See `LICENSE`.
