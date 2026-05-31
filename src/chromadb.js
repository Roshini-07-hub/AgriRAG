/**
 * chromadb.js
 * -----------
 * Chroma Cloud client wrapper — compatible with chromadb v3 (v2 API).
 *
 * chromadb v3 changed the constructor:
 *   - 'path' → replaced by 'ssl', 'host', 'port'
 *   - 'auth' → replaced by 'headers'
 */

import { ChromaClient } from "chromadb";
import { config } from "./config.js";

// chromadb v3 constructor
const client = new ChromaClient({
  ssl:      true,
  host:     config.chromaHost,          // api.trychroma.com
  port:     443,
  headers:  { "X-Chroma-Token": config.chromaApiKey },
  tenant:   config.chromaTenant,
  database: config.chromaDatabase,
});

/**
 * Get (or create) the ChromaDB collection.
 * embeddings are provided externally so no default embedding function needed.
 *
 * @returns {Promise<Collection>}
 */
export async function getCollection() {
  return await client.getOrCreateCollection({
    name:     config.chromaCollection,
    metadata: { "hnsw:space": "cosine" },
    // No embeddingFunction — we supply our own vectors from Gemini
    embeddingFunction: null,
  });
}

/**
 * Upsert chunks into ChromaDB.
 *
 * @param {Array<{id: string, embedding: number[], text: string, source: string}>} items
 */
export async function upsertDocuments(items) {
  const collection = await getCollection();

  await collection.upsert({
    ids:        items.map((i) => i.id),
    embeddings: items.map((i) => i.embedding),
    documents:  items.map((i) => i.text),
    metadatas:  items.map((i) => ({ source: i.source })),
  });

  console.log(`  ✅ Upserted ${items.length} vectors into Chroma Cloud`);
}

/**
 * Query ChromaDB for the most similar chunks to a given embedding.
 *
 * @param {number[]} queryEmbedding
 * @param {number}   topK
 * @returns {Promise<Array<{text: string, source: string, distance: number}>>}
 */
export async function queryDocuments(queryEmbedding, topK = config.topK) {
  const collection = await getCollection();

  const results = await collection.query({
    queryEmbeddings: [queryEmbedding],
    nResults:        topK,
    include:         ["documents", "metadatas", "distances"],
  });

  const docs      = results.documents[0] || [];
  const metas     = results.metadatas[0]  || [];
  const distances = results.distances[0]  || [];

  return docs.map((text, i) => ({
    text,
    source:   metas[i]?.source || "unknown",
    distance: distances[i],
  }));
}

/**
 * Delete and recreate the collection (clean re-ingest).
 */
export async function resetCollection() {
  try {
    await client.deleteCollection({ name: config.chromaCollection });
    console.log(`🗑️  Deleted existing collection "${config.chromaCollection}"`);
  } catch {
    // Collection didn't exist — that's fine
  }
  await getCollection();
  console.log(`✅ Created fresh collection "${config.chromaCollection}"`);
}
