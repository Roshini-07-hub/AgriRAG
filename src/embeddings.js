/**
 * embeddings.js
 * -------------
 * Generates text embeddings using Google Gemini's embedding model.
 * Model: gemini-embedding-001 → 768-dimensional vectors
 *
 * Key improvements for large-scale ingestion:
 *   - Concurrent batching: processes BATCH_SIZE chunks in parallel
 *   - Automatic retry with exponential backoff on rate-limit errors (429)
 *   - Progress logging every batch
 */

import { GoogleGenerativeAI } from "@google/generative-ai";
import { config } from "./config.js";

const genAI = new GoogleGenerativeAI(config.geminiApiKey);
const embeddingModel = genAI.getGenerativeModel({ model: config.embeddingModel });

// Number of concurrent embedding requests per batch.
// Gemini free tier allows ~1500 RPM; 5 concurrent is safe.
const BATCH_SIZE = 5;

// Delay between batches in ms (avoids sustained rate-limit pressure)
const BATCH_DELAY_MS = 500;

// Max retries per individual embedding on transient errors
const MAX_RETRIES = 4;

/**
 * Sleep for a given number of milliseconds.
 */
function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

/**
 * Generate an embedding for a single text with retry + backoff.
 *
 * @param {string} text
 * @param {number} attempt  current attempt number (0-based)
 * @returns {Promise<number[]>}
 */
async function embedWithRetry(text, attempt = 0) {
  try {
    const result = await embeddingModel.embedContent(text.trim());
    return result.embedding.values;
  } catch (err) {
    const isRateLimit = err?.status === 429 ||
      (err?.message || "").toLowerCase().includes("quota") ||
      (err?.message || "").toLowerCase().includes("rate");

    if (attempt < MAX_RETRIES) {
      // Exponential backoff: 2s, 4s, 8s, 16s
      const waitMs = isRateLimit
        ? Math.pow(2, attempt + 1) * 2000   // longer wait for rate limits
        : Math.pow(2, attempt) * 1000;       // shorter wait for other errors

      console.warn(`   ⚠️  Embedding error (attempt ${attempt + 1}/${MAX_RETRIES + 1}): ${err.message}`);
      console.warn(`   ⏳ Retrying in ${waitMs / 1000}s...`);
      await sleep(waitMs);
      return embedWithRetry(text, attempt + 1);
    }

    throw new Error(`Embedding failed after ${MAX_RETRIES + 1} attempts: ${err.message}`);
  }
}

/**
 * Generate an embedding vector for a single piece of text.
 *
 * @param {string} text
 * @returns {Promise<number[]>}
 */
export async function generateEmbedding(text) {
  return embedWithRetry(text);
}

/**
 * Generate embeddings for multiple texts using concurrent batching.
 * Processes BATCH_SIZE texts in parallel, then waits BATCH_DELAY_MS
 * before the next batch to stay within rate limits.
 *
 * @param {string[]} texts
 * @returns {Promise<number[][]>}
 */
export async function generateEmbeddings(texts) {
  const embeddings = new Array(texts.length);
  const total = texts.length;

  for (let i = 0; i < total; i += BATCH_SIZE) {
    const batchEnd   = Math.min(i + BATCH_SIZE, total);
    const batchTexts = texts.slice(i, batchEnd);
    const batchNums  = `${i + 1}–${batchEnd}`;

    process.stdout.write(`   Embedding chunks ${batchNums}/${total}...`);

    // Run this batch concurrently
    const batchResults = await Promise.all(
      batchTexts.map((text) => embedWithRetry(text))
    );

    // Store results in correct positions
    for (let j = 0; j < batchResults.length; j++) {
      embeddings[i + j] = batchResults[j];
    }

    process.stdout.write(` ✓\n`);

    // Pause between batches (skip after last batch)
    if (batchEnd < total) {
      await sleep(BATCH_DELAY_MS);
    }
  }

  return embeddings;
}
