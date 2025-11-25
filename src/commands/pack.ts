import * as path from "node:path";
import * as fs from "node:fs";
import { Command } from "commander";
import * as prompts from "@clack/prompts";
import chalk from "chalk";
import { ContextPacker } from "../lib/packer";

export const packCommand = new Command("pack")
  .description("Pack context from your codebase for AI prompts")
  .option("-f, --focus <file>", "Focus file to start from")
  .option("-q, --query <text>", "Semantic search query for related code")
  .option("-d, --depth <number>", "Import resolution depth", "2")
  .option("-m, --max-files <number>", "Maximum total files to include", "20")
  .option("--max-semantic <number>", "Maximum semantic search results", "5")
  .option("--include-tests", "Include test files in results", false)
  .option("--format <type>", "Output format: xml, json, markdown, compact", "xml")
  .option("-o, --output <file>", "Output to file instead of stdout")
  .option("--root <dir>", "Project root directory")
  .action(async (options) => {
    const {
      focus,
      query,
      depth,
      maxFiles,
      maxSemantic,
      includeTests,
      format,
      output,
      root,
    } = options;

    if (!focus && !query) {
      console.error(chalk.red("Error: You must provide either --focus or --query"));
      process.exit(1);
    }

    const rootDir = root || process.cwd();
    const packer = new ContextPacker();

    const spinner = prompts.spinner();
    spinner.start("Packing context...");

    try {
      const result = await packer.pack({
        focus,
        query,
        depth: parseInt(depth, 10),
        maxFiles: parseInt(maxFiles, 10),
        maxSemanticFiles: parseInt(maxSemantic, 10),
        includeTests,
        format: format as "xml" | "json" | "markdown" | "compact",
        rootDir,
      });

      spinner.stop("Context packed successfully!");

      if (output) {
        fs.writeFileSync(output, result, "utf-8");
        console.log(chalk.green(`✓ Context written to ${output}`));
        
        // Show summary
        const lineCount = result.split("\n").length;
        const fileCount = (result.match(/<file /g) || []).length;
        console.log(chalk.dim(`  ${fileCount} files, ${lineCount} lines`));
      } else {
        console.log(result);
      }
    } catch (error) {
      spinner.stop("Failed to pack context");
      console.error(chalk.red("Error:"), error instanceof Error ? error.message : error);
      process.exit(1);
    }
  });

export const interactiveCommand = new Command("interactive")
  .alias("i")
  .description("Interactive mode for packing context")
  .action(async () => {
    prompts.intro(chalk.bgBlue(" ospack - Semantic Context Packer "));

    const focusFile = await prompts.text({
      message: "Focus file (optional):",
      placeholder: "src/components/Button.tsx",
    });

    if (typeof focusFile === "symbol") {
      prompts.cancel("Cancelled");
      process.exit(1);
    }

    const query = await prompts.text({
      message: "Semantic search query (optional):",
      placeholder: "authentication and user management",
    });

    if (typeof query === "symbol") {
      prompts.cancel("Cancelled");
      process.exit(1);
    }

    if (!focusFile && !query) {
      prompts.cancel("You must provide either a focus file or a search query");
      process.exit(1);
    }

    const depth = await prompts.select({
      message: "Import resolution depth:",
      options: [
        { value: 1, label: "1 - Direct imports only" },
        { value: 2, label: "2 - Imports and their imports (recommended)" },
        { value: 3, label: "3 - Three levels deep" },
      ],
      initialValue: 2,
    });

    if (typeof depth === "symbol") {
      prompts.cancel("Cancelled");
      process.exit(1);
    }

    const format = await prompts.select({
      message: "Output format:",
      options: [
        { value: "xml", label: "XML - Optimized for Claude" },
        { value: "json", label: "JSON - Machine readable" },
        { value: "markdown", label: "Markdown - Human readable" },
        { value: "compact", label: "Compact - File list only" },
      ],
      initialValue: "xml",
    });

    if (typeof format === "symbol") {
      prompts.cancel("Cancelled");
      process.exit(1);
    }

    const outputToFile = await prompts.confirm({
      message: "Save to file?",
      initialValue: false,
    });

    if (typeof outputToFile === "symbol") {
      prompts.cancel("Cancelled");
      process.exit(1);
    }

    let outputPath: string | undefined;
    if (outputToFile) {
      const filename = await prompts.text({
        message: "Output filename:",
        placeholder: "context.xml",
        initialValue: `context.${format}`,
      });
      if (typeof filename === "symbol") {
        prompts.cancel("Cancelled");
        process.exit(1);
      }
      outputPath = filename as string;
    }

    const spinner = prompts.spinner();
    spinner.start("Packing context...");

    try {
      const packer = new ContextPacker();
      const result = await packer.pack({
        focus: focusFile as string | undefined,
        query: query as string | undefined,
        depth: depth as number,
        format: format as "xml" | "json" | "markdown" | "compact",
        rootDir: process.cwd(),
      });

      spinner.stop("Context packed successfully!");

      if (outputPath) {
        fs.writeFileSync(outputPath, result, "utf-8");
        prompts.outro(chalk.green(`✓ Context written to ${outputPath}`));
      } else {
        console.log("\n" + result);
      }
    } catch (error) {
      spinner.stop("Failed to pack context");
      prompts.cancel(error instanceof Error ? error.message : "Unknown error");
      process.exit(1);
    }
  });