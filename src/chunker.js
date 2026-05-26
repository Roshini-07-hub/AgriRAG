import { config } from "./config.js";

/**
 * Split a long text into overlapping chunks.
 *
 * Why overlap? When a sentence is split across two chunks, the overlap
 * ensures neither chunk loses important context from the boundary.
 *
 * @param {string} text         - The full document text.
 * @param {number} chunkSize    - Max characters per chunk.
 * @param {number} chunkOverlap - Characters shared between consecutive chunks.
 * @returns {string[]}          - Array of text chunks.
 */
export function chunkText(
  text,
  chunkSize = config.chunkSize,
  chunkOverlap = config.chunkOverlap
) {
  const chunks = [];
  let start = 0;

  while (start < text.length) {
    const end = Math.min(start + chunkSize, text.length);
    const chunk = text.slice(start, end).trim();

    if (chunk.length > 0) {
      chunks.push(chunk);
    }

    // Move forward by (chunkSize - overlap) so the next chunk shares
    // `chunkOverlap` characters with the current one
    start += chunkSize - chunkOverlap;
  }

  return chunks;
}
