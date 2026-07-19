require('dotenv').config();

const dns = require('dns');
if (typeof dns.setDefaultResultOrder === 'function') {
    dns.setDefaultResultOrder('ipv4first');
}

const express = require('express');
const cors = require('cors');
const fs = require('fs');
const path = require('path');
const { PDFParse } = require('pdf-parse');
const { createClient } = require('@supabase/supabase-js');

const app = express();
const PORT = process.env.PORT || 3000;
// --- Ollama (local LLM + embeddings) ---
// Запустите Ollama и подтяните модели, например:
//   ollama pull nomic-embed-text
//   ollama pull llama3.2
const OLLAMA_BASE_URL = (process.env.OLLAMA_BASE_URL || 'http://127.0.0.1:11434').replace(/\/$/, '');
const OLLAMA_EMBED_MODEL = process.env.OLLAMA_EMBED_MODEL || 'nomic-embed-text';
const OLLAMA_CHAT_MODEL = process.env.OLLAMA_CHAT_MODEL || 'llama3.2';

/** Порог cosine-подобия в match_documents: ниже — больше отрывков (часто нужно 0.3–0.45). */
const MATCH_THRESHOLD = Math.min(
    0.95,
    Math.max(0.05, parseFloat(process.env.MATCH_THRESHOLD || '0.35', 10) || 0.35)
);
const MATCH_COUNT = Math.min(50, Math.max(1, parseInt(process.env.MATCH_COUNT || '12', 10) || 12));

/** Сколько чанков максимум в промпт после слияния вектора + ILIKE. */
const RAG_MAX_PASSAGES = Math.min(40, Math.max(5, parseInt(process.env.RAG_MAX_PASSAGES || '22', 10) || 22));

/** Максимум слов из вопроса для ILIKE (остальное отрезается — длинные URL фильтров). */
const RAG_ILIKE_MAX_TERMS = Math.min(18, Math.max(4, parseInt(process.env.RAG_ILIKE_MAX_TERMS || '12', 10) || 12));

/** Сообщение для TypeError: fetch failed (Node не всегда заполняет cause). */
const explainFetchFailure = (err, context) => {
    const cause = err && err.cause;
    const code = cause && typeof cause === 'object' && cause.code ? String(cause.code) : '';
    const tail = code ? ` (${code})` : '';
    return `${context}: ${err.message}${tail}`;
};

/** Полная цепочка сообщений для ответа API (fetch failed часто прячет причину в cause). */
const formatErrorChain = (err) => {
    const parts = [];
    let e = err;
    let depth = 0;
    while (e && depth < 8) {
        const msg = e.message || String(e);
        if (msg && !parts.includes(msg)) {
            parts.push(msg);
        }
        if (typeof e === 'object' && e.code && !parts.some((p) => p.includes(String(e.code)))) {
            parts.push(`code:${e.code}`);
        }
        e = e.cause;
        depth += 1;
    }
    return parts.join(' | ');
};

const ollamaEmbed = async (text) => {
    try {
        const res = await fetch(`${OLLAMA_BASE_URL}/api/embeddings`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                model: OLLAMA_EMBED_MODEL,
                prompt: text,
            }),
        });
        if (!res.ok) {
            const errText = await res.text();
            throw new Error(`Ollama embeddings HTTP ${res.status}: ${errText}`);
        }
        const data = await res.json();
        if (!data.embedding || !Array.isArray(data.embedding)) {
            throw new Error('Ollama embeddings response missing embedding array');
        }
        return data.embedding;
    } catch (err) {
        if (err.message && err.message.startsWith('Ollama embeddings')) {
            throw err;
        }
        throw new Error(
            `${explainFetchFailure(err, 'Ollama недоступна для эмбеддингов')} — проверьте, что запущена Ollama, адрес ${OLLAMA_BASE_URL} и модель \`${OLLAMA_EMBED_MODEL}\` (ollama pull).`
        );
    }
};

const ollamaChat = async (messages, options = {}) => {
    const res = await fetch(`${OLLAMA_BASE_URL}/api/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            model: OLLAMA_CHAT_MODEL,
            messages,
            stream: false,
            options: {
                temperature: options.temperature ?? 0.5,
                num_predict: options.maxTokens ?? 500,
            },
        }),
    });
    if (!res.ok) {
        const errText = await res.text();
        throw new Error(`Ollama chat failed (${res.status}): ${errText}`);
    }
    const data = await res.json();
    const content = data.message?.content;
    if (typeof content !== 'string') {
        throw new Error('Ollama chat response missing message.content');
    }
    return content;
};

const supabase = createClient(
    process.env.SUPABASE_URL,
    process.env.SUPABASE_ANON_KEY
);

// Folder where your docs live
const DOCS_DIR = path.join(__dirname, 'MyDocs');

// --- Middleware ---

app.use(cors());
app.use(express.json());
app.use(express.urlencoded({ extended: true }));

const PUBLIC_DIR = path.join(__dirname, 'public');

// --- RAG Helpers ---

// Split long text into overlapping chunks
const chunkText = (text, chunkSize = 1000, chunkOverlap = 200) => {
    const textChunks = [];
    const cleanText = text.replace(/\s+/g, ' ').trim();

    for (let i = 0; i < cleanText.length; i += chunkSize - chunkOverlap) {
        const chunk = cleanText.substring(i, i + chunkSize);
        if (chunk.trim().length > 0) {
            textChunks.push(chunk);
        }
    }

    return textChunks;
};

// Create embeddings for chunks and store in Supabase
const saveChunksAsEmbeddings = async (textChunks, metadata = {}) => {
    const runInsert = async (chunk, embedding) => {
        let result;
        try {
            result = await supabase.from('MyPDFDocuments').insert([
                {
                    content: chunk,
                    embedding: embedding,
                    title: metadata.title || null,
                    source: metadata.source || null,
                    path: metadata.path || null,
                },
            ]);
        } catch (e) {
            throw new Error(
                `${explainFetchFailure(e, 'Запрос к Supabase не дошёл')} — проверьте SUPABASE_URL, интернет и ключ.`
            );
        }

        const { error } = result;
        if (error) {
            console.error('Error inserting document into Supabase:', error);
            const detailStr =
                typeof error.details === 'string'
                    ? error.details.replace(/\s+/g, ' ').trim().slice(0, 500)
                    : '';
            const baseMsg = [error.message, detailStr].filter(Boolean).join(' — ');
            const hint =
                (error.message && error.message.includes('row-level security')) ||
                error.code === '42501'
                    ? ' Включите политику RLS для anon или используйте service_role на сервере. См. setup-database.sql.'
                    : '';
            const dimHint =
                error.message && /dimension|vector/i.test(error.message)
                    ? ' Размерность embedding в таблице должна совпадать с моделью (nomic-embed-text → 768).'
                    : '';
            const netHint = /ENOTFOUND|getaddrinfo|ETIMEDOUT|ECONNREFUSED/i.test(baseMsg)
                ? ' Нет доступа к хосту Supabase: проверьте интернет, DNS, VPN, антивирусный файрвол и SUPABASE_URL в .env.'
                : '';
            throw new Error(`${baseMsg}${hint}${dimHint}${netHint}`);
        }
    };

    const promises = textChunks.map(async (chunk) => {
        const embedding = await ollamaEmbed(chunk);
        await runInsert(chunk, embedding);
    });

    await Promise.all(promises);
};

// --- Index Local Docs (PDFs in MyDocs folder) ---
const indexLocalDocs = async () => {
    const stats = {
        docsFolder: DOCS_DIR,
        pdfFilesFound: 0,
        pdfsIndexed: 0,
        chunksInserted: 0,
        skippedNoText: [],
        note: null,
    };

    if (!fs.existsSync(DOCS_DIR)) {
        throw new Error(`Docs folder not found: ${DOCS_DIR}`);
    }

    const files = fs.readdirSync(DOCS_DIR);

    for (const file of files) {
        if (!file.toLowerCase().endsWith('.pdf')) {
            continue;
        }

        stats.pdfFilesFound += 1;
        const fullPath = path.join(DOCS_DIR, file);
        console.log(`Indexing PDF: ${fullPath}`);

        const dataBuffer = fs.readFileSync(fullPath);

        // Create PDFParse instance and parse
        const pdfParser = new PDFParse({ data: dataBuffer });
        const pdfData = await pdfParser.getText();
        const text = (pdfData.text || '').trim();

        if (!text) {
            console.warn(`No text extracted from ${file}, skipping.`);
            stats.skippedNoText.push(file);
            continue;
        }

        const textChunks = chunkText(text);

        await saveChunksAsEmbeddings(textChunks, {
            title: path.parse(file).name,
            source: 'pdf',
            path: fullPath,
        });
        stats.pdfsIndexed += 1;
        stats.chunksInserted += textChunks.length;
    }

    if (stats.pdfFilesFound === 0) {
        stats.note = `В папке нет файлов .pdf. Положите PDF в: ${DOCS_DIR}`;
    } else if (stats.chunksInserted === 0 && stats.skippedNoText.length > 0) {
        stats.note =
            'Текст из PDF не извлечён (часто скан без OCR). Нужен PDF с текстовым слоем или OCR.';
    }

    return stats;
};

// --- Endpoint to trigger indexing of local docs ---

app.post('/index-docs', async (req, res) => {
    try {
        const stats = await indexLocalDocs();
        res.status(200).json({
            message: 'Индексация завершена.',
            stats,
        });
    } catch (error) {
        console.error('Error indexing local docs:', error);
        console.error('Error details:', error.message);
        console.error('Stack trace:', error.stack);
        res.status(500).json({
            error: 'Failed to index local docs',
            details: formatErrorChain(error),
            hint:
                'Частые причины fetch failed: 1) не запущена Ollama (иконка в трее); 2) нет сети до Supabase; проверьте SUPABASE_URL в .env',
        });
    }
});


// --- Query Processing Helper ---

/** Опционально: `rag-term-aliases.json` — {"maksym":["maksim","khvatov"], "foo":["bar"]} для вариантов написания без правок кода. */
const loadTermAliases = () => {
    const fp = path.join(__dirname, 'rag-term-aliases.json');
    if (!fs.existsSync(fp)) {
        return {};
    }
    try {
        const raw = JSON.parse(fs.readFileSync(fp, 'utf8'));
        if (raw === null || typeof raw !== 'object') {
            return {};
        }
        const out = {};
        for (const [k, v] of Object.entries(raw)) {
            const key = String(k).toLowerCase();
            out[key] = Array.isArray(v) ? v.map((x) => String(x).toLowerCase()) : [String(v).toLowerCase()];
        }
        return out;
    } catch (e) {
        console.warn('[RAG] rag-term-aliases.json не прочитан:', e.message);
        return {};
    }
};

const TERM_ALIASES = loadTermAliases();

const dedupeRowsByContent = (rows) => {
    const seen = new Set();
    return rows.filter((row) => {
        const t = (row.content || '').trim();
        if (!t || seen.has(t)) {
            return false;
        }
        seen.add(t);
        return true;
    });
};

/** Слова из вопроса для ILIKE (любые доки: имена, продукты, аббревиатуры). Стоп-слова + алиасы из rag-term-aliases.json. */
const STOP_ILIKE = new Set([
    'who',
    'what',
    'when',
    'where',
    'which',
    'whom',
    'whose',
    'why',
    'how',
    'does',
    'did',
    'will',
    'would',
    'could',
    'should',
    'have',
    'has',
    'had',
    'been',
    'from',
    'with',
    'that',
    'this',
    'your',
    'they',
    'them',
    'their',
    'there',
    'about',
    'into',
    'such',
    'than',
    'then',
    'some',
    'very',
    'just',
    'like',
    'also',
    'only',
    'come',
    'make',
    'find',
    'tell',
    'give',
    'work',
    'help',
    'want',
    'need',
    'know',
    'take',
    'each',
    'many',
    'most',
    'other',
    'both',
    'same',
    'year',
    'time',
    'well',
    'much',
    'good',
    'here',
    'more',
    'back',
    'after',
    'before',
    'being',
    'using',
    'question',
    'answer',
    'the',
    'and',
    'for',
    'are',
    'but',
    'not',
    'you',
    'all',
    'can',
    'her',
    'was',
    'one',
    'our',
    'out',
    'day',
    'get',
    'his',
    'him',
    'its',
    'may',
    'new',
    'now',
    'old',
    'see',
    'two',
    'way',
    'she',
    'use',
]);

const extractContentSearchTerms = (query) => {
    const words = query.match(/[a-zA-Zа-яА-ЯёЁ0-9]{3,}/g) || [];
    const set = new Set();
    for (const w of words) {
        const lw = w.toLowerCase();
        if (STOP_ILIKE.has(lw)) {
            continue;
        }
        set.add(lw);
        const aliases = TERM_ALIASES[lw];
        if (aliases) {
            for (const a of aliases) {
                if (a.length >= 2) {
                    set.add(a);
                }
            }
        }
        // Латиница 5–24 буквы: типичное y↔im в транслитерации имён (Maksym / Maksim) без rag-term-aliases.json
        if (/^[a-z]{5,24}$/.test(lw)) {
            if (lw.endsWith('ym')) {
                set.add(`${lw.slice(0, -2)}im`);
            }
            if (lw.endsWith('im')) {
                set.add(`${lw.slice(0, -2)}ym`);
            }
        }
    }
    return [...set].slice(0, RAG_ILIKE_MAX_TERMS);
};

const sanitizeIlikeFragment = (s) => s.replace(/[^a-zA-Zа-яА-ЯёЁ0-9]/g, '').slice(0, 64);

/** Полнотекстовый поиск по словам вопроса (резюме, имена — вектор часто промахивается). */
const fetchPassagesByTermsIlike = async (terms) => {
    const parts = [];
    for (const raw of terms) {
        const safe = sanitizeIlikeFragment(raw);
        if (safe.length < 3) {
            continue;
        }
        parts.push(`content.ilike.%${safe}%`);
        parts.push(`title.ilike.%${safe}%`);
    }
    if (!parts.length) {
        return [];
    }

    const { data, error } = await supabase
        .from('MyPDFDocuments')
        .select('id, content, title, source, path')
        .or(parts.join(','))
        .limit(35);

    if (error) {
        console.warn('[RAG] ILIKE по словам вопроса:', error.message);
        return [];
    }
    return data || [];
};

const scoreRowByTerms = (row, terms) => {
    if (!terms.length) {
        return 0;
    }
    const b = `${row.content || ''} ${row.title || ''} ${row.path || ''}`.toLowerCase();
    let s = 0;
    for (const t of terms) {
        if (b.includes(t)) {
            s += 3;
        }
    }
    return s;
};

/** Один чанк — одна запись: при слиянии вектора и ILIKE оставляем больший similarity. */
const mergeVectorAndIlikeRows = (vectorRows, ilikeRows) => {
    const byKey = new Map();
    const keyOf = (row) => (row.content || '').trim();
    const sim = (row) => (typeof row.similarity === 'number' ? row.similarity : -1);

    const put = (row) => {
        const k = keyOf(row);
        if (!k) {
            return;
        }
        const prev = byKey.get(k);
        if (!prev || sim(row) > sim(prev)) {
            byKey.set(k, { ...row });
        }
    };

    (vectorRows || []).forEach(put);
    (ilikeRows || []).forEach((row) => put({ ...row, similarity: row.similarity ?? 0 }));
    return [...byKey.values()];
};

/**
 * Сортировка: отрывок, где реально есть слова из вопроса, выше любого «только похожего по вектору».
 * Иначе вопрос про человека может поднимать топ по Aspire — similarity ~0.5 перебивает ILIKE без бонуса.
 */
const scoreUniversal = (row, terms) => {
    const sim = typeof row.similarity === 'number' ? row.similarity : 0;
    if (!terms.length) {
        return sim;
    }
    const hits = scoreRowByTerms(row, terms);
    if (hits > 0) {
        return 1000 + hits + sim;
    }
    return sim * 1.08;
};

const resolvePassagesForQuery = async (vectorRows, query) => {
    const uniqueVector = dedupeRowsByContent(vectorRows || []);
    const terms = extractContentSearchTerms(query);

    const ilikeDisabled = String(process.env.RAG_DISABLE_ILIKE || '').toLowerCase() === '1';
    let termRows = [];
    if (!ilikeDisabled && terms.length > 0) {
        termRows = await fetchPassagesByTermsIlike(terms);
    }

    const merged = mergeVectorAndIlikeRows(uniqueVector, termRows);
    if (terms.length > 0) {
        merged.sort((a, b) => scoreUniversal(b, terms) - scoreUniversal(a, terms));
    } else {
        merged.sort((a, b) => (b.similarity || 0) - (a.similarity || 0));
    }

    const cap = merged.slice(0, RAG_MAX_PASSAGES);
    console.log(
        `[RAG] Контекст: вектор ${uniqueVector.length}, ILIKE ${termRows.length} по [${terms.join(', ') || '—'}] → ${cap.length}/${RAG_MAX_PASSAGES} отрывков`
    );
    return cap;
};

const processQuery = async (query) => {
    const input = query.replace(/\n/g, ' ');

    // 1. Embed the query
    const embedding = await ollamaEmbed(input);

    // 2. Vector search in Supabase
    const { data, error } = await supabase.rpc('match_documents', {
        query_embedding: embedding,
        match_threshold: MATCH_THRESHOLD,
        match_count: MATCH_COUNT,
    });

    if (error) {
        console.error('Error querying Supabase:', error);
        throw error;
    }

    console.log('Vector search results:', data?.length || 0, 'documents found');
    if (data && data.length > 0) {
        console.log('Top result similarity:', data[0].similarity);
    }

    // 3. Отрывки: вектор + ILIKE по словам вопроса (универсально для любых PDF)
    const uniqueRows = await resolvePassagesForQuery(data || [], query);

    let context = '';
    uniqueRows.forEach((row, idx) => {
        const src = row.title || row.path || 'unknown';
        context += `Passage ${idx + 1} (source: ${src}):\n${row.content.trim()}\n\n`;
    });

    if (!context.trim()) {
        return (
            'No matching passages were found in the index (similarity below threshold or empty database). ' +
            'Try MATCH_THRESHOLD in .env (e.g. 0.25), rephrase the question, or run indexing: POST /index-docs with PDFs in MyDocs.'
        );
    }

    console.log('Context length:', context.length);

    // 4. Ask the model (малые модели часто отвечают «don’t cover» — жёстко запрещаем отказ при наличии отрывков)
    const messages = [
        {
            role: 'system',
            content: `You are a technical documentation assistant. The user provides EXCERPTS (possibly from several PDFs; each block shows source filename) and a QUESTION.

Rules:
- Answer ONLY from excerpts that match the question topic. Ignore passages from a different subject than the question (e.g. policy manual vs product FAQ in the same index).
- If any excerpt clearly answers the question, state that answer directly.
- Forbidden when relevant excerpts exist: "I don't cover", "excerpts do not address", unless EVERY provided passage is clearly off-topic for the question.
- Do not mention RAG, search, or chunks.
- Write the answer as normal documentation prose. Do NOT cite "Passage 1", "According to Passage…", "source: filename", or similar — the user should not see excerpt numbers or PDF names unless they explicitly ask where it came from.`,
        },
        {
            role: 'user',
            content: `DOCUMENTATION EXCERPTS:\n\n${context}\n---\nQUESTION:\n${query}\n---\nAnswer in 2–6 sentences in the same language as the question. Use facts from excerpts that match the topic; do not label passages or sources in the answer text.`,
        },
    ];

    return ollamaChat(messages, { temperature: 0.2, maxTokens: 600 });
};

// --- Query Route ---

app.post('/query', async (req, res) => {
    try {
        const { query } = req.body;

        if (!query || !query.trim()) {
            return res.status(400).json({ error: 'Query is required' });
        }

        const answer = await processQuery(query);

        res.status(200).json({ answer });
    } catch (error) {
        console.error('Error handling user query:', error);
        res.status(500).json({
            error: 'Internal Server Error',
            details: error && error.message ? error.message : undefined,
        });
    }
});

// --- Health (JSON) — для проверки без браузера / curl ---

app.get('/api/health', (req, res) => {
    res.json({ message: 'Node RAG Document Search API is running 🚀' });
});

// --- Статика: веб-интерфейс (GET / → public/index.html) ---

app.use(express.static(PUBLIC_DIR));

// --- Start Server ---

app.listen(PORT, async () => {
    console.log(`Server is running on http://localhost:${PORT}`);

    if (!process.env.SUPABASE_URL || !process.env.SUPABASE_ANON_KEY) {
        console.warn('[warn] В .env должны быть SUPABASE_URL и SUPABASE_ANON_KEY');
    }

    try {
        const ctrl = new AbortController();
        const t = setTimeout(() => ctrl.abort(), 4000);
        const r = await fetch(`${OLLAMA_BASE_URL}/api/tags`, { signal: ctrl.signal });
        clearTimeout(t);
        if (r.ok) {
            console.log(`[ok] Ollama отвечает: ${OLLAMA_BASE_URL} (эмбеддинги: ${OLLAMA_EMBED_MODEL})`);
        } else {
            console.warn(`[warn] Ollama /api/tags HTTP ${r.status} — проверьте OLLAMA_BASE_URL`);
        }
    } catch (e) {
        console.warn(
            `[warn] Ollama недоступна (${OLLAMA_BASE_URL}): ${e.message}. Запустите приложение Ollama и команду: ollama pull ${OLLAMA_EMBED_MODEL}`
        );
    }
});

