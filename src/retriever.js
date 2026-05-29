/**
 * retriever.js
 * ------------
 * Hybrid retrieval: semantic (ChromaDB cosine) + keyword re-ranking.
 *
 * Strategy:
 *   1. Fetch candidatePool chunks from ChromaDB via embedding similarity.
 *   2. Filter out chunks whose cosine distance exceeds maxDistance
 *      (too dissimilar to be useful — sending them causes "I don't know" answers).
 *   3. Re-rank survivors by combined semantic + keyword score.
 *   4. Return the best topK chunks as context for the LLM.
 */

import { generateEmbedding } from "./embeddings.js";
import { queryDocuments } from "./chromadb.js";
import { config } from "./config.js";

/**
 * Keyword overlap score between a chunk and a query.
 * Stop-word filtered, normalised by query length.
 */
function keywordScore(text, query) {
  const normalize = (s) =>
    s.toLowerCase().replace(/[^a-z0-9\s]/g, "").split(/\s+/).filter(Boolean);

  const stopWords = new Set([
    "a","an","the","is","are","was","were","be","been","being",
    "have","has","had","do","does","did","will","would","could",
    "should","may","might","shall","can","need","dare","ought",
    "and","but","or","nor","for","yet","so","in","on","at","to",
    "of","with","by","from","as","into","through","about","what",
    "who","which","how","when","where","why","i","you","he","she",
    "it","we","they","me","him","her","us","them","my","your",
    "his","its","our","their","this","that","these","those",
  ]);

  const queryWords = normalize(query).filter((w) => !stopWords.has(w));
  if (queryWords.length === 0) return 0;

  const chunkWords = normalize(text);
  const querySet   = new Set(queryWords);

  let score = 0;
  for (const word of chunkWords) {
    if (querySet.has(word)) score++;
  }

  return score / queryWords.length;
}

/**
 * Retrieve the most relevant chunks for a user query.
 *
 * @param {string} query
 * @returns {Promise<Array<{text, source, keywordScore, distance, combinedScore}>>}
 */
export async function retrieve(query) {
  const queryEmbedding = await generateEmbedding(query);

  // Use candidatePool from config (falls back to topK * 6 if not set)
  const poolSize = config.candidatePool || config.topK * 6;
  const candidates = await queryDocuments(queryEmbedding, poolSize);

  if (candidates.length === 0) return [];

  // Filter out chunks that are too dissimilar — these cause evasive "I don't know" answers.
  // Cosine distance: 0 = identical, 1 = orthogonal, 2 = opposite.
  // Threshold of 0.55 keeps chunks with at least ~45% cosine similarity.
  const maxDistance = config.crag?.maxChunkDistance ?? 0.55;
  const relevant = candidates.filter((c) => c.distance <= maxDistance);

  // If filtering removed everything, fall back to the closest chunk so the
  // pipeline can still attempt an answer and the critic can judge it.
  const pool = relevant.length > 0 ? relevant : candidates.slice(0, 1);

  const scored = pool.map((c) => {
    const kw       = keywordScore(c.text, query);
    const semantic = 1 - c.distance; // higher = more similar

    return {
      ...c,
      keywordScore:  parseFloat(kw.toFixed(3)),
      combinedScore: parseFloat((semantic * 0.6 + kw * 0.4).toFixed(4)),
    };
  });

  scored.sort((a, b) => b.combinedScore - a.combinedScore);

  const topK = config.topK || 3;
  return scored.slice(0, topK);
}
