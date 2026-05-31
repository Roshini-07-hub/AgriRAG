/**
 * server.js
 * ---------
 * Express web server exposing the Corrective RAG pipeline via HTTP.
 *
 * Routes:
 *   GET  /        -> serves the chat UI (public/index.html)
 *   POST /ask     -> accepts { question } -> returns full CRAG result
 *   GET  /health  -> simple health check
 *
 * POST /ask response shape:
 * {
 *   answer            : string
 *   sources           : string[]
 *   chunks            : { text, source, keywordScore, distance }[]
 *   verdict           : "PASS" | "FAIL"
 *   overallConfidence : number   (0-10)
 *   totalAttempts     : number
 *   maxRetriesHit     : boolean
 *   queryIntent       : string
 *   queryEntities     : string[]
 *   attempts          : AttemptSummary[]
 *   criticReport      : CriticReport | null
 * }
 */

import express from "express";
import cors from "cors";
import path from "path";
import { fileURLToPath } from "url";
import { askRAG } from "./src/rag.js";

const __filename = fileURLToPath(import.meta.url);
const __dirname  = path.dirname(__filename);

const app  = express();
const PORT = process.env.PORT || 3000;

// Middleware
app.use(cors());
app.use(express.json());
app.use(express.static(path.join(__dirname, "public")));

// Health check
app.get("/health", (_req, res) => {
  res.json({ status: "ok", message: "Corrective RAG server is running" });
});

// Main CRAG endpoint
app.post("/ask", async (req, res) => {
  const { question } = req.body;

  if (!question || question.trim() === "") {
    return res.status(400).json({ error: "Question is required." });
  }

  try {
    console.log(`\n❓ Question: ${question}`);

    const result = await askRAG(question);

    // Guard: if result is missing critical fields, surface a clear error
    if (!result || result.answer === undefined) {
      return res.status(500).json({ error: "Pipeline returned an empty result. Check server logs." });
    }

    res.json({
      answer:            result.answer,
      sources:           result.sources,
      verdict:           result.verdict,
      overallConfidence: result.overallConfidence,
      totalAttempts:     result.totalAttempts,
      maxRetriesHit:     result.maxRetriesHit,
      queryIntent:       result.queryIntent,
      queryEntities:     result.queryEntities,

      // Normalise chunks for the UI
      chunks: (result.chunks || []).map((c) => ({
        text:          c.text,
        source:        c.source,
        keywordScore:  c.keywordScore,
        distance:      parseFloat(c.distance.toFixed(4)),
        combinedScore: parseFloat((c.combinedScore ?? 0).toFixed(4)),
      })),

      // Per-attempt summaries (query used, scores, verdict)
      attempts: result.attempts || [],

      // Full critic report for the final answer
      criticReport: result.criticReport
        ? {
            verdict:           result.criticReport.verdict,
            overallConfidence: result.criticReport.overallConfidence,
            failedChecks:      result.criticReport.failedChecks,
            relevance: {
              score:   result.criticReport.relevance.score,
              passed:  result.criticReport.relevance.passed,
              reason:  result.criticReport.relevance.reason,
            },
            grounding: {
              score:   result.criticReport.grounding.score,
              passed:  result.criticReport.grounding.passed,
              reason:  result.criticReport.grounding.reason,
            },
            hallucination: {
              score:   result.criticReport.hallucination.score,
              passed:  result.criticReport.hallucination.passed,
              reason:  result.criticReport.hallucination.reason,
            },
            consistency: {
              score:   result.criticReport.consistency.score,
              passed:  result.criticReport.consistency.passed,
              reason:  result.criticReport.consistency.reason,
            },
          }
        : null,
    });
  } catch (err) {
    console.error("❌ Error:", err.stack || err.message);

    // Surface specific known errors to the client
    let clientMessage = "Something went wrong. Please try again.";

    if (err.message?.includes("410") || err.message?.includes("Gone")) {
      clientMessage = "ChromaDB tenant not found (410 Gone). Please recreate your Chroma Cloud tenant and re-run ingestion.";
    } else if (err.message?.includes("401") || err.message?.includes("Unauthorized")) {
      clientMessage = "ChromaDB authentication failed. Check your CHROMA_API_KEY in .env.";
    } else if (err.message?.includes("tenant")) {
      clientMessage = "ChromaDB connection error. Check CHROMA_TENANT and CHROMA_DATABASE in .env.";
    } else if (err.message?.includes("ECONNREFUSED") || err.message?.includes("fetch")) {
      clientMessage = "Cannot reach ChromaDB. Check CHROMA_HOST in .env.";
    } else if (err.message?.includes("GROQ") || err.message?.includes("groq")) {
      clientMessage = "Groq API error. Check your GROQ_API_KEY in .env.";
    } else if (err.message?.includes("GEMINI") || err.message?.includes("generative")) {
      clientMessage = "Gemini API error. Check your GEMINI_API_KEY in .env.";
    }

    res.status(500).json({ error: clientMessage });
  }
});

// Start
app.listen(PORT, async () => {
  console.log(`\n🚀 Corrective RAG server running at http://localhost:${PORT}`);
  console.log(`   Open your browser and go to http://localhost:${PORT}\n`);

  // Startup connectivity check — warn early if ChromaDB is unreachable
  try {
    const { getCollection } = await import("./src/chromadb.js");
    await getCollection();
    console.log("✅ ChromaDB connection verified\n");
  } catch (err) {
    console.error("⚠️  ChromaDB connection FAILED on startup:", err.message);
    if (err.message?.includes("410") || err.message?.includes("Gone")) {
      console.error("   → Tenant not found (410 Gone).");
      console.error("   → Go to https://app.trychroma.com and recreate your tenant.");
      console.error("   → Update CHROMA_TENANT and CHROMA_DATABASE in .env");
      console.error("   → Then re-run: npm run ingest:reset\n");
    }
  }
});
