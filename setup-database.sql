-- Enable the pgvector extension for vector similarity search
CREATE EXTENSION IF NOT EXISTS vector;

-- Create the MyPDFDocuments table
CREATE TABLE IF NOT EXISTS public."MyPDFDocuments" (
    id BIGSERIAL PRIMARY KEY,
    content TEXT NOT NULL,
    embedding vector(1536),  -- OpenAI text-embedding-3-small creates 1536-dimensional vectors
    title TEXT,
    source TEXT,
    path TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create an index on the embedding column for faster similarity search
CREATE INDEX IF NOT EXISTS "MyPDFDocuments_embedding_idx" 
ON public."MyPDFDocuments" 
USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);

-- Create the match_documents function for vector similarity search
CREATE OR REPLACE FUNCTION match_documents (
    query_embedding vector(1536),
    match_threshold float DEFAULT 0.5,
    match_count int DEFAULT 10
)
RETURNS TABLE (
    id bigint,
    content text,
    title text,
    source text,
    path text,
    similarity float
)
LANGUAGE plpgsql
AS $$
BEGIN
    RETURN QUERY
    SELECT
        "MyPDFDocuments".id,
        "MyPDFDocuments".content,
        "MyPDFDocuments".title,
        "MyPDFDocuments".source,
        "MyPDFDocuments".path,
        1 - ("MyPDFDocuments".embedding <=> query_embedding) AS similarity
    FROM public."MyPDFDocuments"
    WHERE 1 - ("MyPDFDocuments".embedding <=> query_embedding) > match_threshold
    ORDER BY "MyPDFDocuments".embedding <=> query_embedding
    LIMIT match_count;
END;
$$;
