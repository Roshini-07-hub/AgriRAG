/**
 * index.js -- CLI entry point
 * ---------------------------
 * Interactive question-answering loop using the Corrective RAG pipeline.
 * Type a question and press Enter. Type "exit" or Ctrl+C to quit.
 *
 * Usage:  node index.js
 */

import readline from "readline";
import { askRAG } from "./src/rag.js";

const rl = readline.createInterface({
  input:  process.stdin,
  output: process.stdout,
});

function prompt(question) {
  return new Promise((resolve) => rl.question(question, resolve));
}

function bar(label, score, width = 20) {
  const filled = Math.round((score / 10) * width);
  const empty  = width - filled;
  return `${label.padEnd(14)} [${"█".repeat(filled)}${"░".repeat(empty)}] ${score}/10`;
}

async function main() {
  console.log("╔══════════════════════════════════════════════════╗");
  console.log("║     📚 Corrective RAG Application -- CLI         ║");
  console.log("║  Ask questions about your documents.             ║");
  console.log('║  Type "exit" to quit.                            ║');
  console.log("╚══════════════════════════════════════════════════╝\n");

  while (true) {
    const question = (await prompt("❓ Your question: ")).trim();

    if (!question) {
      console.log("   (Please enter a question)\n");
      continue;
    }

    if (question.toLowerCase() === "exit") {
      console.log("\n👋 Goodbye!");
      rl.close();
      break;
    }

    try {
      const result = await askRAG(question);

      const verdictIcon = result.verdict === "PASS" ? "✅" : "⚠️ ";

      console.log("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
      console.log("💬 Answer:\n");
      console.log(result.answer);

      console.log("\n📊 Verification Report:");
      console.log(`   ${verdictIcon} Verdict: ${result.verdict}  |  Confidence: ${result.overallConfidence}/10  |  Attempts: ${result.totalAttempts}`);

      if (result.criticReport) {
        const r = result.criticReport;
        console.log(`   ${bar("Relevance",     r.relevance.score)}`);
        console.log(`   ${bar("Grounding",     r.grounding.score)}`);
        console.log(`   ${bar("Hallucination", r.hallucination.score)}`);
        console.log(`   ${bar("Consistency",   r.consistency.score)}`);
      }

      if (result.totalAttempts > 1) {
        console.log("\n🔄 Retry History:");
        result.attempts.forEach((a) => {
          console.log(`   Attempt ${a.attempt}: "${a.searchQuery}" → ${a.verdict} (${a.overallConfidence}/10)`);
        });
      }

      console.log("\n📎 Sources:", result.sources.join(", ") || "(none)");
      console.log("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");
    } catch (err) {
      console.error("\n❌ Error:", err.message, "\n");
    }
  }
}

main();
