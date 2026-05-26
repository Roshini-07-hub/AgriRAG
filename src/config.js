import dotenv from "dotenv";
dotenv.config();

// Validate required environment variables
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
  // Gemini — embeddings only
  geminiApiKey: process.env.GEMINI_API_KEY,

  // Groq — LLM chat responses
  groqApiKey: process.env.GROQ_API_KEY,

  // Chroma Cloud
  chromaHost:       process.env.CHROMA_HOST       || "api.trychroma.com",
  chromaApiKey:     process.env.CHROMA_API_KEY,
  chromaTenant:     process.env.CHROMA_TENANT,
  chromaDatabase:   process.env.CHROMA_DATABASE,
  chromaCollection: process.env.CHROMA_COLLECTION || "rag-collection",

  // LangSmith — uses LANGSMITH_* env vars (auto-picked up by the SDK too)
  langsmithApiKey:  process.env.LANGSMITH_API_KEY,
  langsmithProject: process.env.LANGSMITH_PROJECT  || "rag-basic",
  langsmithEndpoint:process.env.LANGSMITH_ENDPOINT || "https://api.smith.langchain.com",

  // Models
  embeddingModel: "gemini-embedding-001",
  chatModel:      "llama-3.1-8b-instant",

  // Chunking
  chunkSize:    500,
  chunkOverlap: 50,

  // Retrieval
  topK: 3,
};
