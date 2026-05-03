require('dotenv').config();

const express = require('express');
const cors = require('cors');
const fs = require('fs');
const path = require('path');
const { PDFParse } = require('pdf-parse');
const OpenAI = require('openai');
const { createClient } = require('@supabase/supabase-js');

const app = express();
const PORT = process.env.PORT || 3000;

// --- Clients ---

const openai = new OpenAI({
    apiKey: process.env.OPENAI_API_KEY,
});

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
    const promises = textChunks.map(async (chunk) => {
        const embeddingResponse = await openai.embeddings.create({
            model: 'text-embedding-3-small',
            input: chunk,
        });

        const embedding = embeddingResponse.data[0].embedding;

        const { error } = await supabase.from('MyPDFDocuments').insert([
            {
                content: chunk,
                embedding: embedding,
                title: metadata.title || null,
                source: metadata.source || null,
                path: metadata.path || null,
            },
        ]);

        if (error) {
            console.error('Error inserting document into Supabase:', error);
            throw error;
        }
    });

    await Promise.all(promises);
};

// --- Index Local Docs (PDFs in MyDocs folder) ---
const indexLocalDocs = async () => {
    if (!fs.existsSync(DOCS_DIR)) {
        throw new Error(`Docs folder not found: ${DOCS_DIR}`);
    }

    const files = fs.readdirSync(DOCS_DIR);

    for (const file of files) {
        if (!file.toLowerCase().endsWith('.pdf')) {
            continue;
        }

        const fullPath = path.join(DOCS_DIR, file);
        console.log(`Indexing PDF: ${fullPath}`);

        const dataBuffer = fs.readFileSync(fullPath);

        // Create PDFParse instance and parse
        const pdfParser = new PDFParse({ data: dataBuffer });
        const pdfData = await pdfParser.getText();
        const text = (pdfData.text || '').trim();

        if (!text) {
            console.warn(`No text extracted from ${file}, skipping.`);
            continue;
        }

        const textChunks = chunkText(text);

        await saveChunksAsEmbeddings(textChunks, {
            title: path.parse(file).name,
            source: 'pdf',
            path: fullPath,
        });
    }
};

// --- Endpoint to trigger indexing of local docs ---

app.post('/index-docs', async (req, res) => {
    try {
        await indexLocalDocs();
        res.status(200).json({ message: 'Indexed all PDFs from MyDocs folder into Supabase.' });
    } catch (error) {
        console.error('Error indexing local docs:', error);
        console.error('Error details:', error.message);
        console.error('Stack trace:', error.stack);
        res.status(500).json({ error: 'Failed to index local docs', details: error.message });
    }
});


// --- Query Processing Helper ---

const processQuery = async (query) => {
    const input = query.replace(/\n/g, ' ');

    // 1. Embed the query
    const embeddingResponse = await openai.embeddings.create({
        model: 'text-embedding-3-small',
        input: [input],
    });

    const embedding = embeddingResponse.data[0].embedding;

    // 2. Vector search in Supabase
    const { data, error } = await supabase.rpc('match_documents', {
        query_embedding: embedding,
        match_threshold: 0.5,
        match_count: 10,
    });

    if (error) {
        console.error('Error querying Supabase:', error);
        throw error;
    }

    console.log('Vector search results:', data?.length || 0, 'documents found');
    if (data && data.length > 0) {
        console.log('Top result similarity:', data[0].similarity);
    }

    // 3. Build context
    let context = '';
    data.forEach((row, idx) => {
        context += `Chunk ${idx + 1}:\n${row.content.trim()}\n\n`;
    });

    if (!context) {
        context = 'No relevant context found.';
    }
    
    console.log('Context length:', context.length);

    // 4. Ask the model using the context
    const messages = [
        {
            role: 'system',
            content: `You are a representative that is very helpful, Only ever answer
    truthfully and be as helpful as you can!`,
        },
        {
            role: 'user',
            content: `Context:\n${context}\n\nQuestion: ${query}\n\nAnswer clearly and concisely.`,
        },
    ];

    const chatCompletion = await openai.chat.completions.create({
        model: 'gpt-4.1-mini', // or 'gpt-4o-mini', 'gpt-3.5-turbo' depending on what you use
        messages: messages,
        temperature: 0.5,
        max_tokens: 500,
    });

    return chatCompletion.choices[0].message.content;
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
        res.status(500).json({ error: 'Internal Server Error' });
    }
});

// --- Health Check Route ---

app.get('/', (req, res) => {
    res.json({ message: 'Node RAG Document Search API is running 🚀' });
});

// --- Start Server ---

app.listen(PORT, () => {
    console.log(`Server is running on http://localhost:${PORT}`);
});

