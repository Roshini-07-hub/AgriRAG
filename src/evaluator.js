/**
 * evaluator.js
 * ------------
 * Corrective RAG — Verification / Critic Layer
 *
 * Runs four independent checks on every draft answer:
 *
 *   1. checkRelevance(question, chunks)
 *      → Are the retrieved chunks topically relevant to the question?
 *
 *   2. checkGrounding(question, answer, chunks)
 *      → Is every claim in the answer traceable to the retrieved context?
 *
 *   3. checkHallucination(question, answer, chunks)
 *      → Does the answer introduce facts NOT present in the context?
 *
 *   4. checkConsistency(question, answer)
 *      → Is the answer internally consistent and does it actually address the question?
 *
 * Each check returns a { score: 0-10, passed: boolean, reason, details[] }.
 *
 * runCriticLayer(question, answer, chunks) runs all four checks in parallel
 * and returns a combined CriticReport with an overall confidence score and
 * a PASS / FAIL verdict.
 */

import Groq from "groq-sdk";
import { traceable } from "langsmith/traceable";
import { wrapSDK } from "langsmith/wrappers";
import { config } from "./config.js";

const groq = wrapSDK(new Groq({ apiKey: config.groqApiKey }));

// ─────────────────────────────────────────────────────────────────────────────
// Shared helpers
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Call the LLM with a critic prompt and parse the JSON response.
 */
async function callCritic(prompt) {
  const response = await groq.chat.completions.create({
    model:       config.chatModel,
    messages:    [{ role: "user", content: prompt }],
    temperature: 0.1,
  });

  const raw     = response.choices[0].message.content.trim();
  const jsonStr = raw.replace(/^```(?:json)?\s*/i, "").replace(/\s*```$/, "").trim();
  return JSON.parse(jsonStr);
}

function buildContext(chunks) {
  return chunks
    .map((c, i) => `[Chunk ${i + 1} — ${c.source}]\n${c.text}`)
    .join("\n\n---\n\n");
}

/**
 * Detect whether the draft answer is an evasive "I don't know" response.
 * These answers are technically grounded but represent a retrieval failure —
 * the pipeline should retry with a better query rather than accepting them.
 *
 * @param {string} answer
 * @returns {boolean}
 */
function isEvasiveAnswer(answer) {
  const lower = answer.toLowerCase().trim();
  const evasivePatterns = [
    "i don't have enough information",
    "i do not have enough information",
    "i couldn't find",
    "i could not find",
    "not found in the documents",
    "not mentioned in the",
    "no information available",
    "the documents do not contain",
    "the context does not",
    "the provided context does not",
    "there is no information",
    "i cannot find",
    "i can't find",
    "not covered in",
    "not discussed in",
  ];
  return evasivePatterns.some((p) => lower.includes(p));
}

// ─────────────────────────────────────────────────────────────────────────────
// Check 1 — Retrieval Relevance
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Score how topically relevant the retrieved chunks are to the question.
 *
 * @param {string} question
 * @param {Array<{text:string, source:string}>} chunks
 * @returns {Promise<{score:number, passed:boolean, reason:string, details:string[]}>}
 */
export const checkRelevance = traceable(
  async (question, chunks) => {
    if (!chunks || chunks.length === 0) {
      return { score: 0, passed: false, reason: "No chunks retrieved.", details: ["Empty retrieval result"] };
    }

    const prompt = `You are a retrieval quality evaluator for a RAG system.

Task: Score how relevant the retrieved context chunks are to the question.

Question: "${question}"

Retrieved Context:
${buildContext(chunks)}

Respond with ONLY valid JSON:
{
  "score": <integer 0-10>,
  "passed": <true if score >= 6, else false>,
  "reason": "<one sentence verdict>",
  "details": ["<observation about chunk relevance>"]
}

Scoring:
- 8-10: chunks directly address the question topic
- 5-7:  chunks are related but only partially on-topic
- 0-4:  chunks are off-topic or too generic

JSON only. No markdown.`;

    try {
      const p = await callCritic(prompt);
      return {
        score:   p.score   ?? 0,
        passed:  p.passed  ?? (p.score >= 6),
        reason:  p.reason  || "",
        details: p.details || [],
      };
    } catch {
      return { score: 5, passed: true, reason: "Relevance check parse error — defaulting pass.", details: [] };
    }
  },
  { name: "check_relevance", run_type: "chain", project_name: config.langsmithProject }
);

// ─────────────────────────────────────────────────────────────────────────────
// Check 2 — Evidence Grounding
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Verify that the answer's claims are grounded in the retrieved context.
 *
 * @param {string} question
 * @param {string} answer
 * @param {Array<{text:string, source:string}>} chunks
 * @returns {Promise<{score:number, passed:boolean, reason:string, details:string[]}>}
 */
export const checkGrounding = traceable(
  async (question, answer, chunks) => {
    if (!chunks || chunks.length === 0) {
      return { score: 0, passed: false, reason: "No context to ground against.", details: [] };
    }

    // If the answer is evasive, the context failed to support a real answer —
    // score it low so the pipeline retries with a better query.
    if (isEvasiveAnswer(answer)) {
      return {
        score:   2,
        passed:  false,
        reason:  "Answer admits lack of information — context did not support a real answer.",
        details: ["Retrieved chunks did not contain sufficient information to answer the question"],
      };
    }

    const prompt = `You are an evidence grounding evaluator for a RAG system.

Task: Check whether every factual claim in the answer can be traced back to the provided context.

Question: "${question}"

Context:
${buildContext(chunks)}

Answer:
"${answer}"

For each claim in the answer, check if it appears in the context.
Respond with ONLY valid JSON:
{
  "score": <integer 0-10>,
  "passed": <true if score >= 6>,
  "reason": "<one sentence summary>",
  "details": ["<ungrounded claim or observation>"]
}

Scoring:
- 8-10: all claims are directly supported by the context
- 5-7:  most claims are supported; minor gaps
- 0-4:  significant claims are not found in the context

JSON only. No markdown.`;

    try {
      const p = await callCritic(prompt);
      return {
        score:   p.score   ?? 0,
        passed:  p.passed  ?? (p.score >= 6),
        reason:  p.reason  || "",
        details: p.details || [],
      };
    } catch {
      return { score: 5, passed: true, reason: "Grounding check parse error — defaulting pass.", details: [] };
    }
  },
  { name: "check_grounding", run_type: "chain", project_name: config.langsmithProject }
);

// ─────────────────────────────────────────────────────────────────────────────
// Check 3 — Hallucination Detection
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Detect whether the answer introduces facts that are NOT in the context.
 *
 * @param {string} question
 * @param {string} answer
 * @param {Array<{text:string, source:string}>} chunks
 * @returns {Promise<{score:number, passed:boolean, reason:string, details:string[]}>}
 */
export const checkHallucination = traceable(
  async (question, answer, chunks) => {
    if (!chunks || chunks.length === 0) {
      return { score: 0, passed: false, reason: "No context to check hallucinations against.", details: [] };
    }

    // Evasive answers mean retrieval failed — treat as low score to force retry
    if (isEvasiveAnswer(answer)) {
      return {
        score:   3,
        passed:  false,
        reason:  "Answer is evasive — context was insufficient to produce a real answer.",
        details: ["Answer did not use the retrieved context to answer the question"],
      };
    }

    const prompt = `You are a hallucination detector for a RAG system.

Task: Identify any statements in the answer that are NOT supported by the context and appear to be fabricated or assumed.

Question: "${question}"

Context:
${buildContext(chunks)}

Answer:
"${answer}"

Respond with ONLY valid JSON:
{
  "score": <integer 0-10>,
  "passed": <true if score >= 7>,
  "reason": "<one sentence summary>",
  "details": ["<hallucinated statement if any>"]
}

Scoring (inverse — higher = less hallucination):
- 9-10: no hallucinations detected; answer stays within context
- 6-8:  minor extrapolations but no fabricated facts
- 0-5:  clear hallucinations — facts stated that are absent from context

JSON only. No markdown.`;

    try {
      const p = await callCritic(prompt);
      return {
        score:   p.score   ?? 0,
        passed:  p.passed  ?? (p.score >= 7),
        reason:  p.reason  || "",
        details: p.details || [],
      };
    } catch {
      return { score: 7, passed: true, reason: "Hallucination check parse error — defaulting pass.", details: [] };
    }
  },
  { name: "check_hallucination", run_type: "chain", project_name: config.langsmithProject }
);

// ─────────────────────────────────────────────────────────────────────────────
// Check 4 — Answer Consistency
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Check that the answer is internally consistent and actually addresses the question.
 *
 * @param {string} question
 * @param {string} answer
 * @returns {Promise<{score:number, passed:boolean, reason:string, details:string[]}>}
 */
export const checkConsistency = traceable(
  async (question, answer) => {
    // Evasive answers are a retrieval failure, not a good answer
    if (isEvasiveAnswer(answer)) {
      return {
        score:   2,
        passed:  false,
        reason:  "Answer does not address the question — it only admits lack of information.",
        details: ["Answer failed to provide any substantive response to the question"],
      };
    }

    const prompt = `You are a consistency and coherence evaluator for a RAG system.

Task: Check whether the answer is internally consistent, logically coherent, and actually addresses the question asked.

Question: "${question}"

Answer:
"${answer}"

Respond with ONLY valid JSON:
{
  "score": <integer 0-10>,
  "passed": <true if score >= 6>,
  "reason": "<one sentence summary>",
  "details": ["<consistency issue if any>"]
}

Scoring:
- 8-10: answer directly and substantively addresses the question with clear, coherent information
- 5-7:  answer is mostly relevant but slightly off-topic or has minor inconsistencies
- 0-4:  answer contradicts itself, ignores the question, is incoherent, or gives no real information

JSON only. No markdown.`;

    try {
      const p = await callCritic(prompt);
      return {
        score:   p.score   ?? 0,
        passed:  p.passed  ?? (p.score >= 6),
        reason:  p.reason  || "",
        details: p.details || [],
      };
    } catch {
      return { score: 6, passed: true, reason: "Consistency check parse error — defaulting pass.", details: [] };
    }
  },
  { name: "check_consistency", run_type: "chain", project_name: config.langsmithProject }
);

// ─────────────────────────────────────────────────────────────────────────────
// Combined Critic Layer
// ─────────────────────────────────────────────────────────────────────────────

/**
 * @typedef {Object} CheckResult
 * @property {number}   score    0-10
 * @property {boolean}  passed
 * @property {string}   reason
 * @property {string[]} details
 */

/**
 * @typedef {Object} CriticReport
 * @property {boolean}     passed              true if all checks pass
 * @property {number}      overallConfidence   0-10 weighted average
 * @property {string}      verdict             "PASS" | "FAIL"
 * @property {CheckResult} relevance
 * @property {CheckResult} grounding
 * @property {CheckResult} hallucination
 * @property {CheckResult} consistency
 * @property {string[]}    failedChecks        names of checks that failed
 * @property {string[]}    weaknesses          aggregated detail strings for query refinement
 */

/**
 * Run all four critic checks in parallel and produce a combined report.
 *
 * Weights:
 *   relevance     25%
 *   grounding     30%
 *   hallucination 30%
 *   consistency   15%
 *
 * @param {string} question
 * @param {string} answer
 * @param {Array<{text:string, source:string}>} chunks
 * @returns {Promise<CriticReport>}
 */
export const runCriticLayer = traceable(
  async (question, answer, chunks) => {
    console.log("   🔬 Running critic layer (4 checks in parallel)...");

    // Fast-path: if the answer is evasive ("I don't know"), the retrieval failed.
    // Skip the LLM critic calls and immediately return FAIL so the pipeline retries.
    if (isEvasiveAnswer(answer)) {
      console.log("   ⚡ Fast-fail: answer is evasive — retrieval did not find relevant content");
      const evasiveCheck = {
        score:   2,
        passed:  false,
        reason:  "Answer admits lack of information — retrieval failed to find relevant content.",
        details: ["Retrieved chunks did not contain sufficient information to answer the question"],
      };
      return {
        passed:            false,
        overallConfidence: 2.0,
        verdict:           "FAIL",
        relevance:         { score: 2, passed: false, reason: "Chunks were not relevant enough to produce an answer.", details: ["Retrieval returned insufficient content"] },
        grounding:         evasiveCheck,
        hallucination:     evasiveCheck,
        consistency:       evasiveCheck,
        failedChecks:      ["relevance", "grounding", "hallucination", "consistency"],
        weaknesses:        ["Retrieved chunks did not contain sufficient information to answer the question"],
      };
    }

    const [relevance, grounding, hallucination, consistency] = await Promise.all([
      checkRelevance(question, chunks),
      checkGrounding(question, answer, chunks),
      checkHallucination(question, answer, chunks),
      checkConsistency(question, answer),
    ]);

    // Weighted confidence score
    const overallConfidence = parseFloat((
      relevance.score     * 0.25 +
      grounding.score     * 0.30 +
      hallucination.score * 0.30 +
      consistency.score   * 0.15
    ).toFixed(2));

    const failedChecks = [];
    if (!relevance.passed)     failedChecks.push("relevance");
    if (!grounding.passed)     failedChecks.push("grounding");
    if (!hallucination.passed) failedChecks.push("hallucination");
    if (!consistency.passed)   failedChecks.push("consistency");

    const passed = failedChecks.length === 0 && overallConfidence >= config.crag.confidenceThreshold;

    // Aggregate weakness details for query refinement
    const weaknesses = [
      ...relevance.details,
      ...grounding.details,
      ...hallucination.details,
      ...consistency.details,
    ].filter(Boolean);

    return {
      passed,
      overallConfidence,
      verdict:      passed ? "PASS" : "FAIL",
      relevance,
      grounding,
      hallucination,
      consistency,
      failedChecks,
      weaknesses,
    };
  },
  { name: "critic_layer", run_type: "chain", project_name: config.langsmithProject }
);

// ─────────────────────────────────────────────────────────────────────────────
// Query Refinement
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Rewrite the query to target the gaps identified by the critic layer.
 *
 * @param {string}   originalQuestion
 * @param {string[]} weaknesses        detail strings from the critic report
 * @param {number}   attempt           retry attempt number (1-based)
 * @returns {Promise<string>}          refined query
 */
export const refineQuery = traceable(
  async (originalQuestion, weaknesses, attempt) => {
    const insufficientRetrieval = weaknesses.some((w) =>
      w.toLowerCase().includes("sufficient") ||
      w.toLowerCase().includes("relevant content") ||
      w.toLowerCase().includes("did not contain")
    );

    const gapList = weaknesses.length > 0
      ? weaknesses.slice(0, 6).map((w) => `- ${w}`).join("\n")
      : "- General lack of relevant or grounded information";

    const prompt = `You are a search query optimizer for a Corrective RAG system operating over Indian agricultural documents covering crop cultivation, diseases, pests, fertilizers, irrigation, and government schemes.

The previous retrieval attempt failed verification. Rewrite the query to improve retrieval.

Original question: "${originalQuestion}"
Retry attempt: ${attempt}
${insufficientRetrieval ? "Note: The previous query returned chunks that did NOT contain enough information to answer the question. Use broader or alternative agricultural keywords." : ""}

Identified gaps / issues:
${gapList}

Write a single refined search query that:
1. Uses different agricultural keywords or synonyms from the original
2. Is broader if the original was too specific, or more specific if too broad
3. Stays within the agricultural/farming domain
4. Is a single concise query — not multiple questions

Respond with ONLY the refined query string. No quotes, no explanation, no punctuation at the end.`;

    try {
      const response = await groq.chat.completions.create({
        model:       config.chatModel,
        messages:    [{ role: "user", content: prompt }],
        temperature: 0.45,
      });
      const refined = response.choices[0].message.content.trim();
      return refined || originalQuestion;
    } catch {
      return originalQuestion;
    }
  },
  { name: "refine_query", run_type: "chain", project_name: config.langsmithProject }
);

// ─────────────────────────────────────────────────────────────────────────────
// Query Understanding / Rewriting (Step 2 in the pipeline)
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Analyse and optionally rewrite the user's raw question before first retrieval.
 * Extracts intent, key entities, and produces an optimised search query.
 *
 * @param {string} question  raw user question
 * @returns {Promise<{searchQuery:string, intent:string, entities:string[]}>}
 */
export const understandQuery = traceable(
  async (question) => {
    const prompt = `You are a query understanding module for an Agricultural Knowledge RAG system.

Users ask questions about farming, crop cultivation, crop diseases, pest management, fertilizers, soil health, irrigation, and government agricultural schemes in India.

Analyse the user question and produce an optimised search query for vector retrieval over these agricultural documents.

User question: "${question}"

Respond with ONLY valid JSON:
{
  "searchQuery": "<optimised search query — may differ from the original>",
  "intent": "<one sentence describing what the farmer wants to know>",
  "entities": ["<key crop/pest/disease/scheme entity 1>", "<key entity 2>"]
}

Rules:
- This is an AGRICULTURAL domain system covering Indian farming practices.
- Interpret terms in agricultural context: "blast" = rice blast disease, "BPH" = brown planthopper, "MSP" = minimum support price, "IPM" = integrated pest management, "FYM" = farmyard manure, "DAP" = di-ammonium phosphate.
- searchQuery should be concise and keyword-rich for embedding search
- Keep crop names, disease names, pest names, and scheme names intact
- If the question is already well-formed, searchQuery can equal the original

JSON only. No markdown.`;

    try {
      const response = await groq.chat.completions.create({
        model:       config.chatModel,
        messages:    [{ role: "user", content: prompt }],
        temperature: 0.1,
      });
      const raw     = response.choices[0].message.content.trim();
      const jsonStr = raw.replace(/^```(?:json)?\s*/i, "").replace(/\s*```$/, "").trim();
      const parsed  = JSON.parse(jsonStr);
      return {
        searchQuery: parsed.searchQuery || question,
        intent:      parsed.intent      || "",
        entities:    parsed.entities    || [],
      };
    } catch {
      return { searchQuery: question, intent: "", entities: [] };
    }
  },
  { name: "understand_query", run_type: "chain", project_name: config.langsmithProject }
);
