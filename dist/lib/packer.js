"use strict";
var __createBinding = (this && this.__createBinding) || (Object.create ? (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    var desc = Object.getOwnPropertyDescriptor(m, k);
    if (!desc || ("get" in desc ? !m.__esModule : desc.writable || desc.configurable)) {
      desc = { enumerable: true, get: function() { return m[k]; } };
    }
    Object.defineProperty(o, k2, desc);
}) : (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    o[k2] = m[k];
}));
var __setModuleDefault = (this && this.__setModuleDefault) || (Object.create ? (function(o, v) {
    Object.defineProperty(o, "default", { enumerable: true, value: v });
}) : function(o, v) {
    o["default"] = v;
});
var __importStar = (this && this.__importStar) || (function () {
    var ownKeys = function(o) {
        ownKeys = Object.getOwnPropertyNames || function (o) {
            var ar = [];
            for (var k in o) if (Object.prototype.hasOwnProperty.call(o, k)) ar[ar.length] = k;
            return ar;
        };
        return ownKeys(o);
    };
    return function (mod) {
        if (mod && mod.__esModule) return mod;
        var result = {};
        if (mod != null) for (var k = ownKeys(mod), i = 0; i < k.length; i++) if (k[i] !== "default") __createBinding(result, mod, k[i]);
        __setModuleDefault(result, mod);
        return result;
    };
})();
Object.defineProperty(exports, "__esModule", { value: true });
exports.ContextPacker = void 0;
const fs = __importStar(require("node:fs"));
const path = __importStar(require("node:path"));
const import_resolver_1 = require("./import-resolver");
const semantic_linker_1 = require("./semantic-linker");
const formatter_1 = require("./formatter");
class ContextPacker {
    importResolver;
    semanticLinker;
    formatter;
    processedFiles;
    fileContexts;
    constructor(osgrepPath) {
        this.importResolver = new import_resolver_1.ImportResolver();
        this.semanticLinker = new semantic_linker_1.SemanticLinker(osgrepPath);
        this.formatter = new formatter_1.ContextFormatter();
        this.processedFiles = new Set();
        this.fileContexts = new Map();
    }
    async pack(options) {
        const { focus, query, depth = 2, maxFiles = 20, maxSemanticFiles = 5, includeTests = false, format = "xml", rootDir = process.cwd(), } = options;
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
            }
            else {
                throw new Error(`Focus file not found: ${focus}`);
            }
        }
        // Step 3: Semantic search (soft links)
        if (query || focus) {
            const searchQuery = query || (focus ? `code related to ${path.basename(focus)}` : "");
            const semanticResults = await this.semanticLinker.search(searchQuery, rootDir, maxSemanticFiles);
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
    async resolveImports(filePath, depth) {
        if (depth <= 0)
            return;
        // Don't pass processedFiles as visited set - let the import resolver 
        // track its own visited files to avoid the focus file blocking traversal
        const imports = await this.importResolver.resolveImportsRecursively(filePath, depth);
        for (const [importPath, fileInfo] of imports) {
            if (!this.processedFiles.has(importPath)) {
                await this.addFile(importPath, "import");
            }
        }
    }
    async addFile(filePath, reason, score) {
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
        }
        catch (error) {
            console.warn(`Failed to read file ${filePath}:`, error);
        }
    }
    getSortedContexts() {
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
    isTestFile(filePath) {
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
    async findRelatedContext(focusFile, additionalQuery) {
        const rootDir = process.cwd();
        // Get semantically related files
        const related = await this.semanticLinker.findRelatedFiles(focusFile, rootDir, additionalQuery, 10);
        const contexts = [];
        for (const match of related) {
            try {
                const content = fs.readFileSync(match.path, "utf-8");
                contexts.push({
                    path: match.path,
                    content,
                    reason: "semantic",
                    score: match.score,
                });
            }
            catch {
                // Skip files that can't be read
            }
        }
        return contexts;
    }
}
exports.ContextPacker = ContextPacker;
//# sourceMappingURL=packer.js.map