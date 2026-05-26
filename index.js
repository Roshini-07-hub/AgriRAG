/**
 * index.js — CLI entry point
 * --------------------------
 * Starts an interactive question-answering loop in the terminal.
 * Type a question and press Enter to get an answer from your documents.
 * Type "exit" or press Ctrl+C to quit.
 *
 * Usage:  node index.js
 */

import readline from "readline";
import { askRAG } from "./src/rag.js";

// Create a readline interface for interactive CLI input
const rl = readline.createInterface({
  input: process.stdin,
  output: process.stdout,
});

function prompt(question) {
  return new Promise((resolve) => rl.question(question, resolve));
}

async function main() {
  console.log("╔══════════════════════════════════════════╗");
  console.log("║        📚 RAG Application — CLI          ║");
  console.log("║  Ask questions about your documents.     ║");
  console.log('║  Type "exit" to quit.                    ║');
  console.log("╚══════════════════════════════════════════╝\n");

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
      const { answer, sources } = await askRAG(question);

      console.log("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
      console.log("💬 Answer:\n");
      console.log(answer);
      console.log("\n📎 Sources:", sources.join(", "));
      console.log("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");
    } catch (err) {
      console.error("\n❌ Error:", err.message, "\n");
    }
  }
}

main();
