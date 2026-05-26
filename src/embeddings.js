/**
 * embeddings.js
 * -------------
 * Generates text embeddings using Google Gemini's embedding model.
 * Model: text-embedding-004  →  768-dimensional vectors
 */

import { GoogleGenerativeAI } from "@google/generative-ai";
import { config } from "./config.js";

const genAI = new GoogleGenerativeAI(config.geminiApiKey);
const embeddingModel = genAI.getGenerativeModel({ model: config.embeddingModel });

/**
 * Generate an embedding vector for a single piece of text.
 *
 * @param {string} text - The text to embed.
 * @returns {Promise<number[]>} - A 768-dimensional float array.
 */
export async function generateEmbedding(text) {
  const result = await embeddingModel.embedContent(text.trim());
  return result.embedding.values;
}

/**
 * Generate embeddings for multiple texts.
 * Gemini's JS SDK doesn't support batch embedding in one call,
 * so we run them sequentially (fast enough for typical doc sizes).
 *
 * @param {string[]} texts - Array of strings to embed.
 * @returns {Promise<number[][]>} - Array of embedding vectors.
 */
export async function generateEmbeddings(texts) {
  const embeddings = [];

  for (const text of texts) {
    const result = await embeddingModel.embedContent(text.trim());
    embeddings.push(result.embedding.values);
  }

  return embeddings;
}
