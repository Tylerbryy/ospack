#!/usr/bin/env node

import { Command } from "commander";
import { packCommand, interactiveCommand } from "./commands/pack";
import chalk from "chalk";
import * as fs from "node:fs";
import * as path from "node:path";

const packageJson = JSON.parse(
  fs.readFileSync(path.join(__dirname, "../package.json"), "utf-8")
);

const program = new Command()
  .name("ospack")
  .description("Semantic Context Packer - Build perfect AI prompts from your codebase")
  .version(packageJson.version)
  .addCommand(packCommand)
  .addCommand(interactiveCommand);

// Default action when no command is specified
program
  .option("-f, --focus <file>", "Focus file to start from")
  .option("-q, --query <text>", "Semantic search query for related code")
  .option("-d, --depth <number>", "Import resolution depth", "2")
  .option("-m, --max-files <number>", "Maximum total files to include", "20")
  .option("--format <type>", "Output format: xml, json, markdown, compact", "xml")
  .option("-o, --output <file>", "Output to file instead of stdout")
  .action((options) => {
    // If options are provided, run pack command
    if (options.focus || options.query) {
      packCommand.parse(process.argv);
    } else {
      // Show help if no options provided
      program.outputHelp();
    }
  });

// Add helpful examples
program.on("--help", () => {
  console.log("");
  console.log(chalk.bold("Examples:"));
  console.log("");
  console.log("  # Pack context for a specific file");
  console.log("  $ ospack --focus src/auth/login.ts --depth 2");
  console.log("");
  console.log("  # Search for related code and pack it");
  console.log('  $ ospack --query "password hashing" --max-files 10');
  console.log("");
  console.log("  # Combine focus file with semantic search");
  console.log('  $ ospack --focus src/api/users.ts --query "user validation"');
  console.log("");
  console.log("  # Output to a file");
  console.log("  $ ospack --focus src/index.ts -o context.xml");
  console.log("");
  console.log("  # Interactive mode");
  console.log("  $ ospack interactive");
  console.log("");
  console.log(chalk.dim("Powered by osgrep's semantic search"));
});

// Error handling
process.on("uncaughtException", (error) => {
  console.error(chalk.red("Unexpected error:"), error.message);
  process.exit(1);
});

process.on("unhandledRejection", (reason, promise) => {
  console.error(chalk.red("Unhandled promise rejection:"), reason);
  process.exit(1);
});

// Parse arguments
program.parse(process.argv);

// Show help if no arguments provided
if (process.argv.length === 2) {
  program.outputHelp();
}