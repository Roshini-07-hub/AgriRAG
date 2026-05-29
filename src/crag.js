/**
 * crag.js
 * -------
 * Corrective RAG Pipeline — full implementation matching the architecture diagram:
 *
 *  Step 1  User Query
 *  Step 2  Query Understanding / Optional Rewriting
 *  Step 3  Retriever
 *  Step 4  Vector Database Search
 *  Step 5  Retrieved Context
 *  Step 6  LLM Draft Generation
 *  Step 7  Draft Answer
 *  Step 8  Verification / Critic Layer
 *            ✔ Relevance check
 *            ✔ Evidence grounding
 *            ✔ Hallucination detection
 *            ✔ Answer consistency
 *  Step 9  PASS → Final Answer
 *  Step 10 FAIL → Correction & Refinement Loop
 *            • Refine query
 *            • Retrieve again
 *            • Revise answer
 *            (repeat until confidence threshold or max retries)
 *
 * Exports:
 *   runCRAG(question) → CRAGResult
 */

import Groq from "groq-sdk";
import { traceable } from "langsmith/traceable";
import { wrapSDK } from "langsmith/wrappers";
import { config } from "./config.js";
import { retrieve } from "./retriever.js";
import {
  understandQuery,
  runCriticLayer,
  refineQuery,
} from "./evaluator.js";

const groq = wrapSDK(new Groq({ apiKey: config.groqApiKey }));

// ─────────────────────────────────────────────────────────────────────────────
// Step 3+4+5 — Retrieval
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Retrieve chunks for a given search query and return them in a normalised shape.
 *
 * @param {string} searchQuery
 * @returns {Promise<Array<{text, source, keywordScore, distance, combinedScore}>>}
 */
const retrieveChunks = traceable(
  async (searchQuery) => {
    const raw = await retrieve(searchQuery);
    return raw.map((c) => ({
      text:          c.text,
      source:        c.source,
      keywordScore:  c.keywordScore,
      distance:      parseFloat(c.distance.toFixed(4)),
      combinedScore: parseFloat((c.combinedScore ?? 0).toFixed(4)),
    }));
  },
  { name: "retrieve_chunks", run_type: "retriever", project_name: config.langsmithProject }
);

// ─────────────────────────────────────────────────────────────────────────────
// Step 6+7 — Draft Generation
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Generate a draft answer from the retrieved context.
 * Called on every attempt (initial + retries).
 *
 * @param {string} question   original user question (always used for the prompt)
 * @param {Array}  chunks     retrieved chunks for this attempt
 * @param {number} attempt    0 = initial, 1+ = retry
 * @param {string[]} previousWeaknesses  critic feedback from prior attempt
 * @returns {Promise<string>}
 */
const generateDraft = traceable(
  async (question, chunks, attempt, previousWeaknesses) => {
    if (!chunks || chunks.length === 0) {
      return "I could not find any relevant information in the documents to answer your question.";
    }

    const context = chunks
      .map((c, i) => `[Chunk ${i + 1} — ${c.source}]\n${c.text}`)
      .join("\n\n---\n\n");

    // On retries, include critic feedback so the model can self-correct
    const wasEvasive = previousWeaknesses.some((w) =>
      w.toLowerCase().includes("sufficient") || w.toLowerCase().includes("did not contain")
    );

    const correctionHint = attempt > 0
      ? wasEvasive
        ? `\n\nIMPORTANT: Your previous answer said you didn't have enough information. The context above has been updated with better chunks. Extract and use whatever relevant information IS present in the context, even if partial. Do not say "I don't have enough information" unless the context is truly empty.`
        : `\n\nPrevious attempt issues to fix:\n${previousWeaknesses.slice(0, 4).map((w) => `- ${w}`).join("\n")}`
      : "";

    const systemPrompt = `You are a helpful assistant that answers questions strictly based on the provided context.

Rules:
- Answer ONLY using information found in the context below.
- Extract and use ALL relevant information present in the context, even if it is partial.
- Only say "I don't have enough information in the documents to answer that" if the context contains ZERO relevant information.
- Do NOT use any outside knowledge or make assumptions.
- Be concise, accurate, and well-structured.
- Every factual claim must be traceable to the context.`;

    const userPrompt = `Context:\n${context}${correctionHint}\n\nQuestion: ${question}\n\nAnswer:`;

    const response = await groq.chat.completions.create({
      model:    config.chatModel,
      messages: [
        { role: "system", content: systemPrompt },
        { role: "user",   content: userPrompt   },
      ],
      temperature: attempt === 0 ? 0.2 : 0.3,
    });

    return response.choices[0].message.content.trim();
  },
  { name: "generate_draft", run_type: "llm", project_name: config.langsmithProject }
);

// ─────────────────────────────────────────────────────────────────────────────
// Main CRAG Pipeline
// ─────────────────────────────────────────────────────────────────────────────

/**
 * @typedef {Object} AttemptRecord
 * @property {number}   attempt
 * @property {string}   searchQuery
 * @property {Array}    chunks
 * @property {string}   draft
 * @property {import('./evaluator.js').CriticReport} criticReport
 */

/**
 * @typedef {Object} CRAGResult
 * @property {string}          answer              final answer
 * @property {string[]}        sources             unique source filenames
 * @property {Array}           chunks              chunks used for the final answer
 * @property {string}          verdict             "PASS" | "FAIL" (best attempt used)
 * @property {number}          overallConfidence   0-10
 * @property {number}          totalAttempts       how many retrieval+generation cycles ran
 * @property {string}          queryIntent         extracted intent from step 2
 * @property {string[]}        queryEntities       key entities extracted
 * @property {AttemptRecord[]} attempts            full trace of every attempt
 * @property {import('./evaluator.js').CriticReport} criticReport  final critic report
 */

export const runCRAG = traceable(
  async (question) => {
    const { maxRetries, confidenceThreshold } = config.crag;

    // ── Step 2: Query Understanding ─────────────────────────────────────────
    console.log("\n🧠 Step 2 — Query understanding...");
    const { searchQuery: initialQuery, intent, entities } = await understandQuery(question);
    console.log(`   Intent : ${intent || "(none)"}`);
    console.log(`   Entities: ${entities.join(", ") || "(none)"}`);
    console.log(`   Search query: "${initialQuery}"`);

    /** @type {AttemptRecord[]} */
    const attempts = [];

    let currentQuery      = initialQuery;
    let bestResult        = null;   // best attempt so far (highest confidence)
    let previousWeaknesses = [];

    // ── Steps 3-10: Retrieval → Draft → Critic → Refine loop ───────────────
    for (let attempt = 0; attempt <= maxRetries; attempt++) {
      const isRetry = attempt > 0;
      console.log(`\n${ isRetry ? `🔄 Retry ${attempt}/${maxRetries}` : "🔍 Step 3-5 — Initial retrieval" } — query: "${currentQuery}"`);

      // Step 3+4+5 — Retrieve
      const chunks = await retrieveChunks(currentQuery);

      if (chunks.length === 0 && attempt === 0) {
        return buildEmptyResult(question, intent, entities, attempts);
      }

      if (chunks.length === 0) {
        console.log("   ⚠️  No chunks returned for this query — skipping to next retry");
        previousWeaknesses = ["No chunks retrieved for this query — try broader keywords"];
        currentQuery = await refineQuery(question, previousWeaknesses, attempt + 1);
        console.log(`   Refined query: "${currentQuery}"`);
        continue;
      }

      chunks.forEach((c, i) =>
        console.log(`   [${i + 1}] source="${c.source}" | keyword=${c.keywordScore} | distance=${c.distance} | combined=${c.combinedScore}`)
      );

      // Step 6+7 — Generate draft
      console.log(`\n✏️  Step 6-7 — Generating draft answer (attempt ${attempt + 1})...`);
      const draft = await generateDraft(question, chunks, attempt, previousWeaknesses);

      // Step 8 — Critic layer
      console.log(`\n🔬 Step 8 — Critic layer (attempt ${attempt + 1})...`);
      const criticReport = await runCriticLayer(question, draft, chunks);

      console.log(`   Relevance    : ${criticReport.relevance.score}/10 ${criticReport.relevance.passed ? "✅" : "❌"}`);
      console.log(`   Grounding    : ${criticReport.grounding.score}/10 ${criticReport.grounding.passed ? "✅" : "❌"}`);
      console.log(`   Hallucination: ${criticReport.hallucination.score}/10 ${criticReport.hallucination.passed ? "✅" : "❌"}`);
      console.log(`   Consistency  : ${criticReport.consistency.score}/10 ${criticReport.consistency.passed ? "✅" : "❌"}`);
      console.log(`   ─── Overall confidence: ${criticReport.overallConfidence}/10 → ${criticReport.verdict}`);

      /** @type {AttemptRecord} */
      const record = { attempt: attempt + 1, searchQuery: currentQuery, chunks, draft, criticReport };
      attempts.push(record);

      // Track best result by confidence
      if (!bestResult || criticReport.overallConfidence > bestResult.criticReport.overallConfidence) {
        bestResult = record;
      }

      // Step 9 — PASS: confidence threshold met
      if (criticReport.passed) {
        console.log(`\n✅ Step 9 — PASS (confidence ${criticReport.overallConfidence}/10 ≥ ${confidenceThreshold})`);
        return buildResult(question, record, intent, entities, attempts);
      }

      // Step 10 — FAIL: refine and retry (if retries remain)
      if (attempt < maxRetries) {
        console.log(`\n⚠️  Step 10 — FAIL (confidence ${criticReport.overallConfidence}/10 < ${confidenceThreshold})`);
        console.log(`   Failed checks: ${criticReport.failedChecks.join(", ")}`);

        previousWeaknesses = criticReport.weaknesses;
        currentQuery = await refineQuery(question, criticReport.weaknesses, attempt + 1);
        console.log(`   Refined query: "${currentQuery}"`);
      }
    }

    // Exhausted retries — use best attempt found
    console.log(`\n⚠️  Max retries (${maxRetries}) reached — using best attempt (confidence ${bestResult.criticReport.overallConfidence}/10)`);
    return buildResult(question, bestResult, intent, entities, attempts, true);
  },
  {
    name:         "crag_pipeline",
    run_type:     "chain",
    project_name: config.langsmithProject,
    metadata: {
      embeddingModel:      config.embeddingModel,
      chatModel:           config.chatModel,
      topK:                config.topK,
      maxRetries:          config.crag.maxRetries,
      confidenceThreshold: config.crag.confidenceThreshold,
    },
  }
);

// ─────────────────────────────────────────────────────────────────────────────
// Result builders
// ─────────────────────────────────────────────────────────────────────────────

function buildResult(question, record, intent, entities, attempts, maxRetriesHit = false) {
  const sources = [...new Set(record.chunks.map((c) => c.source))];
  return {
    answer:            record.draft,
    sources,
    chunks:            record.chunks,
    verdict:           record.criticReport.verdict,
    overallConfidence: record.criticReport.overallConfidence,
    totalAttempts:     attempts.length,
    maxRetriesHit,
    queryIntent:       intent,
    queryEntities:     entities,
    attempts:          attempts.map(summariseAttempt),
    criticReport:      record.criticReport,
  };
}

function buildEmptyResult(question, intent, entities, attempts) {
  return {
    answer:            "I could not find any relevant information in the documents to answer your question.",
    sources:           [],
    chunks:            [],
    verdict:           "FAIL",
    overallConfidence: 0,
    totalAttempts:     0,
    maxRetriesHit:     false,
    queryIntent:       intent,
    queryEntities:     entities,
    attempts:          [],
    criticReport:      null,
  };
}

/** Strip chunk text from attempt records to keep the API response lean. */
function summariseAttempt(r) {
  return {
    attempt:           r.attempt,
    searchQuery:       r.searchQuery,
    chunkCount:        r.chunks.length,
    verdict:           r.criticReport.verdict,
    overallConfidence: r.criticReport.overallConfidence,
    failedChecks:      r.criticReport.failedChecks,
    scores: {
      relevance:     r.criticReport.relevance.score,
      grounding:     r.criticReport.grounding.score,
      hallucination: r.criticReport.hallucination.score,
      consistency:   r.criticReport.consistency.score,
    },
  };
}
