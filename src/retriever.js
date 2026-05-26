/**
 * retriever.js
 * ------------
 * Keyword-based retrieval on top of ChromaDB results.
 *
 * Strategy:
 *   1. Fetch top (topK * 4) semantically similar chunks from ChromaDB.
 *   2. Re-rank by keyword overlap with the query — using TF-IDF-style scoring
 *      (keyword matches are weighted by rarity, not just count).
 *   3. Return the best topK chunks as context for the LLM.
 */

import { generateEmbedding } from "./embeddings.js";
import { queryDocuments } from "./chromadb.js";
import { config } from "./config.js";

/**
 * Score a chunk against a query.
 * Uses a balanced combined score: semantic similarity gets equal weight
 * to keyword overlap so domain-specific docs aren't drowned out.
 *
 * @param {string} text  - Chunk text.
 * @param {string} query - User query.
 * @returns {number}
 */
function keywordScore(text, query) {
  const normalize = (s) =>
    s.toLowerCase().replace(/[^a-z0-9\s]/g, "").split(/\s+/).filter(Boolean);

  // Common stop words to ignore so they don't inflate scores
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

  // Normalise by query length so longer queries don't dominate
  return score / queryWords.length;
}

/**
 * Retrieve the most relevant chunks for a user query.
 *
 * @param {string} query - The user's question.
 * @returns {Promise<Array<{text, source, keywordScore, distance}>>}
 */
export async function retrieve(query) {
  const queryEmbedding = await generateEmbedding(query);

  // Fetch a wider pool — more candidates = better chance of finding the right doc
  const candidates = await queryDocuments(queryEmbedding, config.topK * 4);

  if (candidates.length === 0) return [];

  const scored = candidates.map((c) => {
    const kw = keywordScore(c.text, query);
    const semantic = 1 - c.distance; // convert distance → similarity (higher = better)

    return {
      ...c,
      keywordScore: parseFloat(kw.toFixed(3)),
      // Equal weighting: semantic similarity + normalised keyword score
      combinedScore: semantic * 0.6 + kw * 0.4,
    };
  });

  scored.sort((a, b) => b.combinedScore - a.combinedScore);
  return scored.slice(0, config.topK);
}
