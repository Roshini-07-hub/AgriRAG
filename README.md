# 📚 RAG Application

A beginner-friendly **Retrieval-Augmented Generation (RAG)** app built with:

- **Node.js** — backend runtime
- **Google Gemini** — embeddings (`gemini-embedding-001`) + LLM (`gemini-2.0-flash`)
- **ChromaDB** — local vector database (no cloud account needed)
- **Keyword Search** — re-ranks ChromaDB results by keyword overlap for precise retrieval

The app answers questions **strictly from your own documents** — no outside knowledge.

---

## 📁 Project Structure

```
rag-app/
├── documents/          ← Put your .txt files here
│   └── sample.txt      ← Example document (AI/ML concepts)
├── src/
│   ├── config.js       ← Loads env vars, exports all settings
│   ├── embeddings.js   ← Calls Gemini to generate embeddings
│   ├── chromadb.js     ← ChromaDB client: upsert & query
│   ├── chunker.js      ← Splits documents into overlapping chunks
│   ├── ingest.js       ← One-time script: chunk → embed → store
│   ├── retriever.js    ← Keyword-enhanced retrieval from ChromaDB
│   └── rag.js          ← Full RAG pipeline: retrieve → prompt → Gemini LLM
├── index.js            ← Interactive CLI entry point
├── .env.example        ← Template for environment variables
├── .gitignore
├── package.json
└── README.md
```

---

## ⚙️ How It Works

```
Your .txt files
      │
      ▼
  [Chunker]   — splits text into 500-char overlapping chunks
      │
      ▼
  [Gemini]    — generates a 768-dim embedding per chunk
      │
      ▼
  [ChromaDB]  — stores vectors + raw text locally on disk
      │
      ▼  (at query time)
  [Retriever]
    ├─ embed the user question with Gemini
    ├─ fetch top-9 semantic candidates from ChromaDB
    └─ re-rank by keyword overlap → return top 3
      │
      ▼
  [Gemini LLM] — answers using ONLY the retrieved chunks
```

---

## 🚀 Setup Instructions

### 1. Clone / download the project

```bash
git clone <repo-url>
cd rag-app
```

### 2. Install dependencies

```bash
npm install
```

### 3. Install and start ChromaDB

ChromaDB runs as a local server. You need Python installed.

```bash
pip install chromadb
chroma run --path ./chroma-data
```

This starts ChromaDB at `http://localhost:8000` and stores data in `./chroma-data`.  
**Keep this terminal open** while using the app.

> On Windows you can also run: `python -m chromadb.cli.cli run --path ./chroma-data`

### 4. Get your Gemini API key

1. Go to [aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey)
2. Click **Create API Key**
3. Copy the key — it starts with `AIza...`

> Gemini API has a **free tier** — no billing required to get started.

### 5. Configure environment variables

```bash
# Windows
copy .env.example .env

# Mac/Linux
cp .env.example .env
```

Open `.env` and fill in:

```env
GEMINI_API_KEY=AIza...your-key-here...
CHROMA_HOST=http://localhost:8000
CHROMA_COLLECTION=rag-collection
```

### 6. Add your documents

Place `.txt` files in the `documents/` folder.  
A sample file (`sample.txt`) covering AI/ML concepts is already included.

### 7. Ingest documents (run once)

```bash
npm run ingest
```

Expected output:
```
🚀 Starting ingestion pipeline...

📄 Loaded 1 document(s):
   • sample.txt

✂️  "sample.txt" → 12 chunk(s)

📦 Total chunks to embed: 12

🔢 Generating embeddings via Gemini...
   ✅ Generated 12 embeddings

📤 Storing vectors in ChromaDB...
  ✅ Upserted 12 vectors into ChromaDB

🎉 Ingestion complete! You can now run: node index.js
```

> To re-ingest after changing documents, run `npm run ingest:reset` to wipe and reload.

### 8. Ask questions

```bash
npm run query
```

```
╔══════════════════════════════════════════╗
║        📚 RAG Application — CLI          ║
║  Ask questions about your documents.     ║
║  Type "exit" to quit.                    ║
╚══════════════════════════════════════════╝

❓ Your question: What is RAG?

🔍 Retrieving relevant chunks...
   [1] source="sample.txt" | keyword=3 | distance=0.082
   [2] source="sample.txt" | keyword=2 | distance=0.104
   [3] source="sample.txt" | keyword=1 | distance=0.143

🤖 Generating answer with Gemini...

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
💬 Answer:

Retrieval-Augmented Generation (RAG) is a technique that combines
information retrieval with text generation...

📎 Sources: sample.txt
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

---

## 💡 Retrieval Strategy: Keyword Search

1. **Semantic search** — ChromaDB finds the top 9 chunks closest to the query embedding (cosine similarity).
2. **Keyword re-ranking** — Each candidate is scored by how many query words appear in the chunk.
3. **Combined score** — `keyword_score × 10 + (1 - distance)` sorts the final results.
4. Top 3 chunks are passed to Gemini as context.

---

## 🛠️ Customisation

| What to change | Where |
|----------------|-------|
| Chunk size / overlap | `src/config.js` → `chunkSize`, `chunkOverlap` |
| Number of retrieved chunks | `src/config.js` → `topK` |
| Gemini model | `src/config.js` → `chatModel` |
| ChromaDB collection name | `.env` → `CHROMA_COLLECTION` |
| Add more documents | Drop `.txt` files in `documents/`, run `npm run ingest:reset` |

---

## 📦 Dependencies

| Package | Purpose |
|---------|---------|
| `@google/generative-ai` | Gemini embeddings + chat completions |
| `chromadb` | Local vector database client |
| `chromadb-default-embed` | Required peer package for chromadb |
| `dotenv` | Load `.env` variables |

---

## ❓ Troubleshooting

**"Failed to connect to ChromaDB"**  
→ Make sure ChromaDB is running: `chroma run --path ./chroma-data`

**"Missing required environment variable: GEMINI_API_KEY"**  
→ Make sure `.env` exists with your key filled in.

**"No .txt files found in /documents"**  
→ Add at least one `.txt` file to the `documents/` folder.

**Empty or irrelevant answers**  
→ Re-run `npm run ingest:reset` after adding new documents. Make sure your question uses words that appear in the documents.
