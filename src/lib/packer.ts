import * as fs from "node:fs";
import * as path from "node:path";
import { ImportResolver } from "./import-resolver";
import { SemanticLinker } from "./semantic-linker";
import { ContextFormatter, FileContext } from "./formatter";

export interface PackOptions {
  focus?: string;          // Target file to focus on
  query?: string;          // Semantic search query
  depth?: number;          // Import resolution depth
  maxFiles?: number;       // Maximum total files to include
  maxSemanticFiles?: number; // Maximum semantic search results
  includeTests?: boolean;  // Include test files
  format?: "xml" | "json" | "markdown" | "compact";
  rootDir?: string;        // Project root directory
}

export class ContextPacker {
  private importResolver: ImportResolver;
  private semanticLinker: SemanticLinker;
  private formatter: ContextFormatter;
  private processedFiles: Set<string>;
  private fileContexts: Map<string, FileContext>;

  constructor(osgrepPath?: string) {
    this.importResolver = new ImportResolver();
    this.semanticLinker = new SemanticLinker(osgrepPath);
    this.formatter = new ContextFormatter();
    this.processedFiles = new Set();
    this.fileContexts = new Map();
  }

  async pack(options: PackOptions): Promise<string> {
    const {
      focus,
      query,
      depth = 2,
      maxFiles = 20,
      maxSemanticFiles = 5,
      includeTests = false,
      format = "xml",
      rootDir = process.cwd(),
    } = options;

    // Reset state
    this.processedFiles.clear();
    this.fileContexts.clear();

    // Ensure osgrep index is up to date
    await this.semanticLinker.ensureIndexed(rootDir);

    // Step 1: Add focus file if specified
    if (focus) {
      const focusPath = path.isAbsolute(focus) 
        ? focus 
        : path.join(rootDir, focus);
      
      if (fs.existsSync(focusPath) && fs.statSync(focusPath).isFile()) {
        await this.addFile(focusPath, "focus");
        
        // Step 2: Resolve imports (hard links)
        await this.resolveImports(focusPath, depth);
      } else {
        throw new Error(`Focus file not found: ${focus}`);
      }
    }

    // Step 3: Semantic search (soft links)
    if (query || focus) {
      const searchQuery = query || (focus ? `code related to ${path.basename(focus)}` : "");
      const semanticResults = await this.semanticLinker.search(
        searchQuery,
        rootDir,
        maxSemanticFiles
      );

      for (const result of semanticResults) {
        if (!this.processedFiles.has(result.path) && this.fileContexts.size < maxFiles) {
          if (!includeTests && this.isTestFile(result.path)) {
            continue;
          }
          await this.addFile(result.path, "semantic", result.score);
        }
      }
    }

    // Step 4: Sort and format results
    const sortedContexts = this.getSortedContexts();
    
    switch (format) {
      case "json":
        return this.formatter.formatJSON(sortedContexts);
      case "markdown":
        return this.formatter.formatMarkdown(sortedContexts);
      case "compact":
        return this.formatter.formatCompact(sortedContexts);
      case "xml":
      default:
        return this.formatter.formatXML(sortedContexts);
    }
  }

  private async resolveImports(filePath: string, depth: number): Promise<void> {
    if (depth <= 0) return;

    // Don't pass processedFiles as visited set - let the import resolver 
    // track its own visited files to avoid the focus file blocking traversal
    const imports = await this.importResolver.resolveImportsRecursively(
      filePath,
      depth
    );

    for (const [importPath, fileInfo] of imports) {
      if (!this.processedFiles.has(importPath)) {
        await this.addFile(importPath, "import");
      }
    }
  }

  private async addFile(
    filePath: string,
    reason: FileContext["reason"],
    score?: number
  ): Promise<void> {
    if (this.processedFiles.has(filePath)) {
      return;
    }

    this.processedFiles.add(filePath);

    try {
      const content = fs.readFileSync(filePath, "utf-8");
      this.fileContexts.set(filePath, {
        path: filePath,
        content,
        reason,
        score,
      });
    } catch (error) {
      console.warn(`Failed to read file ${filePath}:`, error);
    }
  }

  private getSortedContexts(): FileContext[] {
    const contexts = Array.from(this.fileContexts.values());
    
    // Sort by: focus first, then imports, then semantic by score
    return contexts.sort((a, b) => {
      const reasonOrder = { focus: 0, import: 1, semantic: 2, context: 3 };
      const aOrder = reasonOrder[a.reason];
      const bOrder = reasonOrder[b.reason];
      
      if (aOrder !== bOrder) {
        return aOrder - bOrder;
      }
      
      // Within same reason, sort by score if available
      if (a.score !== undefined && b.score !== undefined) {
        return b.score - a.score;
      }
      
      return 0;
    });
  }

  private isTestFile(filePath: string): boolean {
    const testPatterns = [
      /\.test\.[^.]+$/,
      /\.spec\.[^.]+$/,
      /_test\.[^.]+$/,
      /_spec\.[^.]+$/,
      /\/tests?\//,
      /\/__tests__\//,
      /\/test_/,
    ];

    return testPatterns.some(pattern => pattern.test(filePath));
  }

  async findRelatedContext(
    focusFile: string,
    additionalQuery?: string
  ): Promise<FileContext[]> {
    const rootDir = process.cwd();
    
    // Get semantically related files
    const related = await this.semanticLinker.findRelatedFiles(
      focusFile,
      rootDir,
      additionalQuery,
      10
    );

    const contexts: FileContext[] = [];
    
    for (const match of related) {
      try {
        const content = fs.readFileSync(match.path, "utf-8");
        contexts.push({
          path: match.path,
          content,
          reason: "semantic",
          score: match.score,
        });
      } catch {
        // Skip files that can't be read
      }
    }

    return contexts;
  }
}