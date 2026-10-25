# About LocalRAG

LocalRAG is a learning project for retrieval-augmented generation (RAG). It lets you
ask questions about your own files without sending any data to a cloud service.

## Default settings

The default chat model is llama3.2, served by Ollama. The default embedding model is
nomic-embed-text. Documents are split into chunks of 800 characters, and neighbouring
chunks overlap by 150 characters so ideas cut at a boundary are not lost.

The vector database is ChromaDB and it is stored on disk in the folder data/chroma.
Retrieval returns the 5 most similar chunks by default.

## Refusing to guess

If the best matching chunks fall below the minimum similarity of 0.30, the assistant
replies that it could not find the answer instead of inventing one. This behaviour
reduces hallucination.

## Re-indexing

Each file is fingerprinted with a SHA-256 hash. When you run the ingest command again,
unchanged files are skipped and only new or edited files are re-embedded.
