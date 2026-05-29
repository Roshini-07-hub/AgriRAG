/**
 * rag.js
 * ------
 * Public entry point for the RAG pipeline.
 *
 * Delegates entirely to the Corrective RAG (CRAG) pipeline in crag.js.
 * All tracing, verification, and retry logic lives there.
 *
 * LangSmith trace tree:
 *   crag_pipeline
 *     understand_query
 *     retrieve_chunks          (attempt 1)
 *     generate_draft           (attempt 1)
 *     critic_layer             (attempt 1)
 *       check_relevance
 *       check_grounding
 *       check_hallucination
 *       check_consistency
 *     refine_query             (if retry needed)
 *     retrieve_chunks          (attempt 2)
 *     generate_draft           (attempt 2)
 *     critic_layer             (attempt 2)
 *       ...
 */

import { config } from "./config.js";
import { runCRAG } from "./crag.js";

// Ensure LangSmith env vars are set before any SDK initialises
process.env.LANGCHAIN_TRACING_V2   = "true";
process.env.LANGSMITH_TRACING      = "true";
process.env.LANGCHAIN_API_KEY      = config.langsmithApiKey;
process.env.LANGSMITH_API_KEY      = config.langsmithApiKey;
process.env.LANGCHAIN_PROJECT      = config.langsmithProject;
process.env.LANGSMITH_PROJECT      = config.langsmithProject;
process.env.LANGCHAIN_ENDPOINT     = config.langsmithEndpoint;
process.env.LANGSMITH_ENDPOINT     = config.langsmithEndpoint;

/**
 * Ask a question through the Corrective RAG pipeline.
 *
 * Returns everything the server and CLI need:
 *   answer            — final answer string
 *   sources           — unique source filenames
 *   chunks            — retrieved chunks used for the final answer
 *   verdict           — "PASS" | "FAIL"
 *   overallConfidence — 0-10 weighted confidence score
 *   totalAttempts     — how many retrieval+generation cycles ran
 *   maxRetriesHit     — true if we exhausted retries without passing
 *   queryIntent       — extracted intent from query understanding step
 *   queryEntities     — key entities extracted
 *   attempts          — summary of every attempt (query, scores, verdict)
 *   criticReport      — full critic report for the final answer
 *
 * @param {string} question
 * @returns {Promise<import('./crag.js').CRAGResult>}
 */
export async function askRAG(question) {
  return runCRAG(question);
}
