# Palm Mind RAG

FastAPI backend for the Palm Mind hiring task. You can upload a PDF or a text file, ask questions about it, and book an interview in the same chat. There is no UI. Open [http://localhost:8000/docs](http://localhost:8000/docs) and try the routes from there.

The retrieval code is in `app/services/rag_service.py`. It embeds the question, searches Qdrant, builds the prompt, and calls the model. I did not use LangChain or RetrievalQA.

## What it does

`POST /api/v1/documents/upload` takes `multipart/form-data`:

- `file`: `.pdf` or `.txt`
- `chunking_strategy`: `recursive` or `fixed`

A PDF is read with PyMuPDF. A text file is read as UTF-8. Empty files, bad PDFs, and anything over the size limit are rejected. The text is split with the strategy you picked. `recursive` tries to break on paragraphs, lines, sentences, then words. `fixed` just cuts character windows. Both keep some overlap.

Chunks are embedded with `sentence-transformers/all-MiniLM-L6-v2` and stored in Qdrant (cosine). The first upload downloads that model, about 90 MB. Filename, file type, strategy, and the chunk text go into PostgreSQL. The vectors stay in Qdrant.

`POST /api/v1/chat`:

```json
{
  "session_id": "session_123",
  "message": "What machine learning experience does the candidate have?"
}
```

Redis stores the messages for that `session_id`, with a cap and a TTL. The next turn loads them back. If the new question says something like "he" or "that", the previous user question is added to the search text so the follow-up still finds the right chunks. The model is told to answer only from those chunks.

Interview booking uses the same endpoint. Say you want to book, then answer the questions for name, email, date, and time. The model suggests the fields, and Python checks them before anything is saved. A bad email or a past date is asked again and is not inserted. The half-finished draft sits in Redis until all four fields are valid, then a row goes into `bookings`.

## Layout

```text
app/
├── main.py                  # app startup, routers, db connections
├── api/routes/              # upload and chat
├── api/dependencies.py      # wires services together
├── schemas/                 # request and response models
├── services/                # chunking, ingest, rag, booking, memory
├── repositories/            # postgres reads and writes
├── models/                  # tables
├── db/                      # postgres, redis, qdrant clients
├── core/                    # settings, logging, errors
└── utils/file_parser.py
tests/
```

Routes only validate the request and call a service. `ChatService` checks booking first. If the message is not a booking, it goes to `RAGService`.

## Run it

You need Docker, and Python 3.12 or newer if you want to run the API on the host instead of in Compose.

```bash
cp .env.example .env
docker compose up --build
```

That starts:

- Postgres 16 on port 5432, user `rag`, password `rag`, database `rag`
- Redis 7 on port 6379
- Qdrant on port 6333, dashboard at [http://localhost:6333/dashboard](http://localhost:6333/dashboard)
- the API on port 8000

Tables are created when the API starts. The Qdrant collection is created on the first upload or question, once the embedding size is known.

If port 5432 is already taken by another Postgres container, stop that container or change the host port in `docker-compose.yml`, then run `docker compose up --force-recreate`.

To run the API on your machine and keep only the databases in Docker:

```bash
docker compose up -d postgres redis qdrant
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
uvicorn app.main:app --reload
```

`DATABASE_URL` has to start with `postgresql+asyncpg://`.

## LLM

Chat will not answer without a model. The client speaks the OpenAI Chat Completions API, and `LLM_API_KEY` cannot be empty. A random string is not an OpenAI key. Upload and `GET /health` still work with no key.

For a paid OpenAI key, leave `LLM_BASE_URL` empty and set:

```bash
LLM_API_KEY=sk-...
LLM_MODEL=gpt-4o-mini
```

For a free local model, install [Ollama](https://ollama.com), pull a model, and point the app at it. Ollama ignores the key, but this app still wants some non-empty value.

```bash
ollama pull llama3.2
```

In `.env`, when the API runs inside Docker:

```bash
LLM_API_KEY=ollama
LLM_BASE_URL=http://host.docker.internal:11434/v1
LLM_MODEL=llama3.2
```

If you started uvicorn on the host, use `http://localhost:11434/v1` instead of `host.docker.internal`. Restart Compose after changing `.env`. Smaller local models are fine for a smoke test. They are worse at returning the booking JSON, so a bad extraction just makes the assistant ask for the field again.

Any other Chat Completions server works the same way: set `LLM_BASE_URL`, `LLM_MODEL`, and whatever key that server gave you. Groq and Gemini's OpenAI-compatible endpoints are the usual free-tier ones. Both still give you a key, it just is not billed like OpenAI.

## Try it

```bash
curl -X POST http://localhost:8000/api/v1/documents/upload \
  -F "file=@resume.pdf" \
  -F "chunking_strategy=recursive"
```

```bash
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"session_id":"session_123","message":"What machine learning experience does the candidate have?"}'
```

Keep the same `session_id` when you book. Ask to book, then send name, email, date, and time as it asks.

Errors come back as `detail` and `code`. Stack traces stay in the server log.

| Status | Meaning |
| --- | --- |
| 400 | Bad file type, empty file, or a PDF that could not be read |
| 413 | File is bigger than `MAX_UPLOAD_BYTES` |
| 422 | Body or form is invalid |
| 503 | Embeddings, Qdrant, Postgres, Redis, or the LLM failed |

`GET /health` only means the process is up.

Settings are in `.env.example`: database, Redis, Qdrant, `LLM_API_KEY`, `LLM_MODEL`, `LLM_BASE_URL`, embedding model, chunk size, overlap, `RAG_TOP_K`, upload limit, and how long chat history is kept. `.env` is gitignored.

## Tests

The tests fake the embedding model, Qdrant, Postgres, Redis, and the LLM. They do not call a real model.

```bash
source .venv/bin/activate
pytest
```

They cover text upload, bad and empty files, both chunking strategies, a RAG answer, multi-turn history, booking, missing fields, and invalid email, date, and time.
