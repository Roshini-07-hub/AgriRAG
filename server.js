/**
 * server.js
 * ---------
 * Express web server exposing the RAG pipeline via HTTP.
 *
 * Routes:
 *   GET  /             → serves the chat UI (public/index.html)
 *   POST /ask          → accepts { question } → returns { answer, sources, chunks }
 *   GET  /health       → simple health check
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

// ── Middleware ───────────────────────────────────────────────────────────────
app.use(cors());
app.use(express.json());
app.use(express.static(path.join(__dirname, "public"))); // serve UI files

// ── Routes ───────────────────────────────────────────────────────────────────

// Health check
app.get("/health", (_req, res) => {
  res.json({ status: "ok", message: "RAG server is running" });
});

// Main RAG endpoint
app.post("/ask", async (req, res) => {
  const { question } = req.body;

  if (!question || question.trim() === "") {
    return res.status(400).json({ error: "Question is required." });
  }

  try {
    console.log(`\n❓ Question: ${question}`);
    const { answer, sources, chunks } = await askRAG(question);

    res.json({
      answer,
      sources,
      chunks: chunks.map((c) => ({
        text:         c.text,
        source:       c.source,
        keywordScore: c.keywordScore,
        distance:     parseFloat(c.distance.toFixed(4)),
      })),
    });
  } catch (err) {
    console.error("❌ Error:", err.message);
    res.status(500).json({ error: "Something went wrong. Please try again." });
  }
});

// ── Start ────────────────────────────────────────────────────────────────────
app.listen(PORT, () => {
  console.log(`\n🚀 RAG server running at http://localhost:${PORT}`);
  console.log(`   Open your browser and go to http://localhost:${PORT}\n`);
});
