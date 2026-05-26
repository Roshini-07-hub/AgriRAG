/**
 * rag.js
 * ------
 * Core RAG pipeline with LangSmith tracing.
 *
 * LangSmith auto-reads these env vars (set in .env):
 *   LANGSMITH_TRACING=true
 *   LANGSMITH_ENDPOINT=https://api.smith.langchain.com
 *   LANGSMITH_API_KEY=lsv2_pt_...
 *   LANGSMITH_PROJECT=rag-basic
 *
 * Trace tree visible in LangSmith:
 *   rag_pipeline          (chain)
 *     ├── retrieve_chunks (retriever)
 *     └── generate_answer (llm)
 */

import Groq from "groq-sdk";
import { traceable } from "langsmith/traceable";
import { wrapSDK } from "langsmith/wrappers";
import { config } from "./config.js";
import { retrieve } from "./retriever.js";

// Explicitly set LangSmith env vars before SDK uses them
// (dotenv loads them, but we set them on process.env to be safe)
process.env.LANGCHAIN_TRACING_V2   = "true";   // legacy key — some SDK versions need this
process.env.LANGSMITH_TRACING      = "true";
process.env.LANGCHAIN_API_KEY      = config.langsmithApiKey;   // legacy key
process.env.LANGSMITH_API_KEY      = config.langsmithApiKey;
process.env.LANGCHAIN_PROJECT      = config.langsmithProject;  // legacy key
process.env.LANGSMITH_PROJECT      = config.langsmithProject;
process.env.LANGCHAIN_ENDPOINT     = config.langsmithEndpoint; // legacy key
process.env.LANGSMITH_ENDPOINT     = config.langsmithEndpoint;

// Wrap Groq so every completion call is auto-traced as an LLM span
const groq = wrapSDK(new Groq({ apiKey: config.groqApiKey }));

// ── Traced step 1: retrieve relevant chunks ──────────────────────────────────
const retrieveChunks = traceable(
  async (question) => {
    const chunks = await retrieve(question);
    // Return in LangSmith document format so the retriever span shows docs
    return chunks.map((c) => ({
      pageContent: c.text,
      metadata: {
        source:       c.source,
        keywordScore: c.keywordScore,
        distance:     parseFloat(c.distance.toFixed(4)),
      },
    }));
  },
  {
    name:         "retrieve_chunks",
    run_type:     "retriever",
    project_name: config.langsmithProject,
  }
);

// ── Traced step 2: generate answer with Groq ─────────────────────────────────
const generateAnswer = traceable(
  async (question, context) => {
    const systemPrompt = `You are a helpful assistant that answers questions strictly based on the provided context.

Rules:
- Answer ONLY using information found in the context below.
- If the answer is not in the context, say: "I don't have enough information in the documents to answer that."
- Do NOT use any outside knowledge.
- Be concise and clear.`;

    const userPrompt = `Context:\n${context}\n\nQuestion: ${question}\n\nAnswer:`;

    const response = await groq.chat.completions.create({
      model: config.chatModel,
      messages: [
        { role: "system", content: systemPrompt },
        { role: "user",   content: userPrompt   },
      ],
      temperature: 0.2,
    });

    return response.choices[0].message.content.trim();
  },
  {
    name:         "generate_answer",
    run_type:     "llm",
    project_name: config.langsmithProject,
  }
);

// ── Root traced pipeline ─────────────────────────────────────────────────────
const ragPipeline = traceable(
  async (question) => {
    // Step 1 — retrieve
    console.log("\n🔍 Retrieving relevant chunks...");
    const docResults = await retrieveChunks(question);

    if (docResults.length === 0) {
      return {
        answer:  "I could not find any relevant information in the documents to answer your question.",
        sources: [],
        chunks:  [],
      };
    }

    // Unpack back to internal format for logging + response
    const chunks = docResults.map((d) => ({
      text:         d.pageContent,
      source:       d.metadata.source,
      keywordScore: d.metadata.keywordScore,
      distance:     d.metadata.distance,
    }));

    chunks.forEach((c, i) => {
      console.log(
        `   [${i + 1}] source="${c.source}" | keyword=${c.keywordScore} | distance=${c.distance}`
      );
    });

    // Step 2 — build context string
    const context = chunks
      .map((c, i) => `[Chunk ${i + 1} — ${c.source}]\n${c.text}`)
      .join("\n\n---\n\n");

    // Step 3 — generate
    console.log("\n🤖 Generating answer with Groq...");
    const answer = await generateAnswer(question, context);

    const sources = [...new Set(chunks.map((c) => c.source))];
    return { answer, sources, chunks };
  },
  {
    name:         "rag_pipeline",
    run_type:     "chain",
    project_name: config.langsmithProject,
    metadata: {
      embeddingModel: config.embeddingModel,
      chatModel:      config.chatModel,
      topK:           config.topK,
    },
  }
);

/**
 * Public entry point — called by server.js and index.js.
 */
export async function askRAG(question) {
  return ragPipeline(question);
}
