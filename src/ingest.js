/**
 * ingest.js
 * ---------
 * Reads all .txt files from /documents, chunks them, generates embeddings
 * via Gemini, and stores vectors in ChromaDB.
 *
 * Usage:
 *   node src/ingest.js           — incremental upsert (safe to re-run)
 *   node src/ingest.js --reset   — wipe collection first, then ingest
 *
 * Improvements over original:
 *   - Embeds and upserts one document at a time (progress saved per doc)
 *   - If a document fails, others are not affected
 *   - Clear progress reporting
 */

import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";
import { chunkText } from "./chunker.js";
import { generateEmbeddings } from "./embeddings.js";
import { upsertDocuments, resetCollection } from "./chromadb.js";

const __filename = fileURLToPath(import.meta.url);
const __dirname  = path.dirname(__filename);
const DOCUMENTS_DIR = path.join(__dirname, "..", "documents");

function loadDocuments() {
  if (!fs.existsSync(DOCUMENTS_DIR)) {
    console.error(`❌ Documents folder not found: ${DOCUMENTS_DIR}`);
    process.exit(1);
  }

  const files = fs
    .readdirSync(DOCUMENTS_DIR)
    .filter((f) => f.endsWith(".txt"))
    .sort(); // consistent ordering

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

  if (process.argv.includes("--reset")) {
    await resetCollection();
    console.log();
  }

  // ── Step 1: Load documents ───────────────────────────────────────────────
  const documents = loadDocuments();
  console.log(`📄 Loaded ${documents.length} document(s)\n`);

  let totalChunks  = 0;
  let totalFailed  = 0;
  const failedDocs = [];

  // ── Step 2–4: Process one document at a time ─────────────────────────────
  // Processing per-document means a rate-limit error on doc 30 doesn't
  // lose the embeddings already generated for docs 1–29.

  for (let d = 0; d < documents.length; d++) {
    const doc = documents[d];
    console.log(`\n[${d + 1}/${documents.length}] 📄 ${doc.filename}`);

    try {
      // Chunk
      const chunks = chunkText(doc.content);
      console.log(`   ✂️  ${chunks.length} chunk(s)`);

      // Build items array
      const items = chunks.map((text, i) => ({
        id:     `${doc.filename}-chunk-${i}`,
        text,
        source: doc.filename,
      }));

      // Embed
      console.log(`   🔢 Generating ${chunks.length} embeddings...`);
      const embeddings = await generateEmbeddings(items.map((it) => it.text));

      // Upsert
      console.log(`   📤 Upserting to ChromaDB...`);
      await upsertDocuments(
        items.map((item, i) => ({
          id:        item.id,
          embedding: embeddings[i],
          text:      item.text,
          source:    item.source,
        }))
      );

      totalChunks += chunks.length;
      console.log(`   ✅ Done`);

    } catch (err) {
      console.error(`   ❌ Failed: ${err.message}`);
      failedDocs.push(doc.filename);
      totalFailed++;
    }
  }

  // ── Summary ───────────────────────────────────────────────────────────────
  console.log("\n" + "═".repeat(50));
  console.log(`✅ Ingestion complete`);
  console.log(`   Documents processed : ${documents.length - totalFailed}/${documents.length}`);
  console.log(`   Total chunks stored : ${totalChunks}`);

  if (failedDocs.length > 0) {
    console.log(`\n⚠️  Failed documents (${failedDocs.length}):`);
    failedDocs.forEach((f) => console.log(`   • ${f}`));
    console.log(`\n   Re-run without --reset to retry failed documents.`);
  } else {
    console.log(`\n🎉 All documents ingested successfully!`);
    console.log(`   Start the server: npm start`);
    console.log(`   Or use the CLI:   npm run query`);
  }
}

main().catch((err) => {
  console.error("\n❌ Ingestion failed:", err.message);
  process.exit(1);
});
