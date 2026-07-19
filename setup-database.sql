-- Enable the pgvector extension for vector similarity search
CREATE EXTENSION IF NOT EXISTS vector;

-- Create the MyPDFDocuments table
CREATE TABLE IF NOT EXISTS public."MyPDFDocuments" (
    id BIGSERIAL PRIMARY KEY,
    content TEXT NOT NULL,
    -- Размерность должна совпадать с моделью эмбеддингов Ollama (см. OLLAMA_EMBED_MODEL в .env)
    -- nomic-embed-text → 768; mxbai-embed-large → 1024; при смене модели выполните ALTER COLUMN или пересоздайте таблицу
    embedding vector(768),

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
    query_embedding vector(768),
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

-- Поиск по RPC с анонимным ключом (иначе match_documents может быть недоступна)
GRANT EXECUTE ON FUNCTION public.match_documents TO anon, authenticated;

-- Если вставки из приложения падают с «row level security», выполните блок ниже
-- (или отключите RLS для таблицы только для разработки — не для продакшена с клиентским anon).

ALTER TABLE public."MyPDFDocuments" ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS "anon_rw_MyPDFDocuments" ON public."MyPDFDocuments";
CREATE POLICY "anon_rw_MyPDFDocuments"
ON public."MyPDFDocuments"
FOR ALL
TO anon
USING (true)
WITH CHECK (true);

DROP POLICY IF EXISTS "authenticated_rw_MyPDFDocuments" ON public."MyPDFDocuments";
CREATE POLICY "authenticated_rw_MyPDFDocuments"
ON public."MyPDFDocuments"
FOR ALL
TO authenticated
USING (true)
WITH CHECK (true);
