
# 📘 RAG Document Search – Node.js + Supabase + OpenAI

This project is a **local document-based RAG (Retrieval-Augmented Generation) system** built using:

- **Node.js** (Express server)
- **Supabase** (Postgres + pgvector)
- **OpenAI Embeddings + Chat Models**
- **pdf-parse** (PDF text extraction)

Instead of uploading documents, this app automatically **reads all PDF files in a local folder (`MyDocs/`)**, extracts text, chunks it, generates embeddings, and stores them in Supabase for vector search.

You can then ask questions about the document using semantic search + LLM reasoning.

---

# 🚀 Features

✔ Automatic PDF ingestion from `MyDocs/`  
✔ Text chunking with overlap (RAG-friendly)  
✔ Embeddings stored in Supabase using pgvector  
✔ Fast semantic search using an RPC function  
✔ Question-answering using retrieved document context  
✔ Clean and modular Node.js code  
✔ Local-only ingestion (no uploads)  

---

# 📂 Folder Structure

```
Document-RAG-App/
│
├── MyDocs/
│   ├── Policies.pdf           # Your local documents (auto-read)
│
├── index.js                   # Node server + RAG pipeline
├── package.json
├── package-lock.json
├── .env                       # Your API keys
└── README.md                  # This file
```

---

# 🛠️ Technologies Used

| Component      | Technology |
|----------------|------------|
| Server         | Node.js + Express |
| LLM Provider   | OpenAI API |
| Embeddings     | text-embedding-3-small |
| Database       | Supabase Postgres |
| Vector Index   | pgvector + ivfflat |
| Document Loader| pdf-parse |
| Query Transport| REST API |

---

# 🔧 Setup Instructions

## 1. Install dependencies

```
npm install
```

Required packages:

- express  
- cors  
- dotenv  
- openai  
- @supabase/supabase-js  
- pdf-parse  

---

## 2. Add your `.env` file

Create `.env` in the root:

```
OPENAI_API_KEY=your_openai_key
SUPABASE_URL=https://your-project-url.supabase.co
SUPABASE_ANON_KEY=your_supabase_anon_key
PORT=3000
```

---

## 3. Add your PDF files

Place your PDFs into:

```
MyDocs/
    Policies.pdf
```

You can add more PDFs anytime.

---

# 🗄️ Supabase Setup

Log in to Supabase → SQL Editor → Run these commands.

---

## 1. Enable pgvector

```
create extension if not exists vector;
```

---

## 2. Create the `MyPDFDocuments` table

```
create table if not exists "MyPDFDocuments" (
  id bigserial primary key,
  content text,
  embedding vector(1536),
  title text,
  source text,
  path text,
  created_at timestamptz default now()
);
```

---

## 3. Create vector index (recommended)

```
create index if not exists mypdfdocuments_embedding_idx
on "MyPDFDocuments"
using ivfflat (embedding vector_cosine_ops)
with (lists = 100);
```

---

## 4. RPC Function for vector search

```
create or replace function match_documents(
  query_embedding vector(1536),
  match_threshold float,
  match_count int
)
returns table (
  id bigint,
  content text,
  similarity float
)
language plpgsql
as $$
begin
  return query
  select
    d.id,
    d.content,
    1 - (d.embedding <=> query_embedding) as similarity
  from "MyPDFDocuments" d
  where 1 - (d.embedding <=> query_embedding) > match_threshold
  order by similarity desc
  limit match_count;
end;
$$;
```

---

# 📡 API Documentation

## ✔ 1. Index local PDFs  
```
POST /index-docs
```

This endpoint:

1. Scans the `MyDocs/` folder  
2. Extracts text from each PDF  
3. Splits into chunks  
4. Generates embeddings  
5. Writes to Supabase  

### Example (curl):

```
curl -X POST http://localhost:3000/index-docs
```

### Response:

```
{
  "message": "Indexed all PDFs from MyDocs folder into Supabase."
}
```

---

## ✔ 2. Ask a question  
```
POST /query
```

### Body:
```
{
  "query": "What is the leave policy?"
}
```

### Example curl:

```
curl -X POST http://localhost:3000/query   -H "Content-Type: application/json"   -d '{"query": "What is the leave policy?"}'
```

### Response:
```
{
  "answer": "According to the policy..."
}
```

---

# 🧠 How It Works (RAG Flow)

## 1. Extract
pdf-parse reads PDF contents.

## 2. Chunk
Chunking with overlap:

```
chunkSize = 1000 chars  
chunkOverlap = 200 chars  
```

## 3. Embed
Each chunk is converted to a 1536-dimension vector using:

```
text-embedding-3-small
```

## 4. Store
Stored in `MyPDFDocuments` table.

## 5. Search
`match_documents` performs semantic vector search.

## 6. Generate
OpenAI model produces the final answer using retrieved context.

---

# 🧪 Testing Your Pipeline

## Start the server

```
npm run dev
```

## Index documents

```
POST http://localhost:3000/index-docs
```

## Query

```
POST http://localhost:3000/query
```

---

# 🐛 Troubleshooting

| Issue | Cause | Fix |
|-------|--------|------|
| No text extracted | PDF is a scanned (image) PDF | Use OCR like tesseract |
| Embedding dimension error | Table vector size mismatch | Ensure `vector(1536)` |
| Empty results | Similarity threshold too high | Use threshold `0.2` |
| Slow search | Missing index | Add IVFFLAT index |
| OpenAI errors | Wrong API key | Check `.env` |


