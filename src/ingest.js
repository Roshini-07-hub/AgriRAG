/**
 * ingest.js
 * ---------
 * Run this script ONCE to:
 *   1. Read all .txt files from /documents
 *   2. Split them into overlapping chunks
 *   3. Generate embeddings via Gemini
 *   4. Store everything in ChromaDB
 *
 * Usage:  node src/ingest.js
 *
 * Pass --reset to wipe the collection before ingesting:
 *         node src/ingest.js --reset
 */

import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";
import { chunkText } from "./chunker.js";
import { generateEmbeddings } from "./embeddings.js";
import { upsertDocuments, resetCollection } from "./chromadb.js";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const DOCUMENTS_DIR = path.join(__dirname, "..", "documents");

function loadDocuments() {
  if (!fs.existsSync(DOCUMENTS_DIR)) {
    console.error(`❌ Documents folder not found: ${DOCUMENTS_DIR}`);
    process.exit(1);
  }

  const files = fs.readdirSync(DOCUMENTS_DIR).filter((f) => f.endsWith(".txt"));

  if (files.length === 0) {
    console.error("❌ No .txt files found in /documents folder.");
    process.exit(1);
  }

  return files.map((filename) => ({
    filename,
    content: fs.readFileSync(path.join(DOCUMENTS_DIR, filename), "utf-8"),
  }));
}

async function main() {
  console.log("🚀 Starting ingestion pipeline...\n");

  // Optional: wipe existing collection for a clean slate
  if (process.argv.includes("--reset")) {
    await resetCollection();
    console.log();
  }

  // ── Step 1: Load documents ───────────────────────────────────────────────
  const documents = loadDocuments();
  console.log(`📄 Loaded ${documents.length} document(s):`);
  documents.forEach((d) => console.log(`   • ${d.filename}`));
  console.log();

  // ── Step 2: Chunk each document ──────────────────────────────────────────
  const allChunks = [];

  for (const doc of documents) {
    const chunks = chunkText(doc.content);
    console.log(`✂️  "${doc.filename}" → ${chunks.length} chunk(s)`);

    chunks.forEach((text, i) => {
      allChunks.push({
        id: `${doc.filename}-chunk-${i}`,
        text,
        source: doc.filename,
      });
    });
  }

  console.log(`\n📦 Total chunks to embed: ${allChunks.length}\n`);

  // ── Step 3: Generate embeddings ──────────────────────────────────────────
  console.log("🔢 Generating embeddings via Gemini...");
  const embeddings = await generateEmbeddings(allChunks.map((c) => c.text));
  console.log(`   ✅ Generated ${embeddings.length} embeddings\n`);

  // ── Step 4: Upsert into ChromaDB ─────────────────────────────────────────
  console.log("📤 Storing vectors in ChromaDB...");
  const items = allChunks.map((chunk, i) => ({
    id: chunk.id,
    embedding: embeddings[i],
    text: chunk.text,
    source: chunk.source,
  }));

  await upsertDocuments(items);

  console.log("\n🎉 Ingestion complete! You can now run: node index.js");
}

main().catch((err) => {
  console.error("❌ Ingestion failed:", err.message);
  process.exit(1);
});
