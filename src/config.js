import dotenv from "dotenv";
dotenv.config();

const required = [
  "GEMINI_API_KEY",
  "GROQ_API_KEY",
  "CHROMA_API_KEY",
  "CHROMA_TENANT",
  "CHROMA_DATABASE",
  "LANGSMITH_API_KEY",
];

for (const key of required) {
  if (!process.env[key]) {
    console.error(`❌ Missing required environment variable: ${key}`);
    console.error(`   Copy .env.example to .env and fill in your values.`);
    process.exit(1);
  }
}

export const config = {
  // Gemini -- embeddings only
  geminiApiKey:  process.env.GEMINI_API_KEY,

  // Groq -- LLM chat responses
  groqApiKey:    process.env.GROQ_API_KEY,

  // Chroma Cloud
  chromaHost:       process.env.CHROMA_HOST       || "api.trychroma.com",
  chromaApiKey:     process.env.CHROMA_API_KEY,
  chromaTenant:     process.env.CHROMA_TENANT,
  chromaDatabase:   process.env.CHROMA_DATABASE,
  chromaCollection: process.env.CHROMA_COLLECTION || "rag-collection",

  // LangSmith
  langsmithApiKey:   process.env.LANGSMITH_API_KEY,
  langsmithProject:  process.env.LANGSMITH_PROJECT  || "rag-basic",
  langsmithEndpoint: process.env.LANGSMITH_ENDPOINT || "https://api.smith.langchain.com",

  // Models
  embeddingModel: "gemini-embedding-001",  // 768-dim
  chatModel:      "llama-3.1-8b-instant",

  // Chunking (sentence-aware)
  chunkSize:    400,   // target chars per chunk
  chunkOverlap: 80,    // overlap chars between chunks
  minChunkSize: 100,   // discard chunks shorter than this

  // Retrieval
  topK:          5,    // final chunks sent to LLM (increased from 3 for better coverage)
  candidatePool: 30,   // how many candidates to fetch from ChromaDB before re-ranking

  // Reranking
  rrfK:          60,   // RRF constant (standard = 60)
  rerankTopN:    6,    // candidates passed to LLM reranker before final topK

  // Query optimisation
  useHyDE:       true, // generate hypothetical answer to improve embedding
  expandQuery:   true, // generate sub-questions for broader retrieval

  // Corrective RAG
  crag: {
    // Minimum weighted confidence score (0-10) to accept an answer without retry
    confidenceThreshold: 6.5,

    // Maximum number of retrieval+generation retries after the initial attempt
    // Total pipeline runs = 1 (initial) + maxRetries
    maxRetries: 2,

    // Cosine distance threshold for chunk filtering in retriever.
    // Chunks with distance > this value are too dissimilar and cause "I don't know" answers.
    // Range: 0 (identical) to 1 (orthogonal). 0.55 = ~45% minimum similarity.
    maxChunkDistance: 0.55,

    // Individual check pass thresholds (0-10)
    relevancePassThreshold:     6,
    groundingPassThreshold:     6,
    hallucinationPassThreshold: 7,
    consistencyPassThreshold:   6,
  },
};
