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
exports.ImportResolver = void 0;
const fs = __importStar(require("node:fs"));
const path = __importStar(require("node:path"));
const os = __importStar(require("node:os"));
const TreeSitter = require("web-tree-sitter");
const Parser = TreeSitter.Parser;
const Language = TreeSitter.Language;
const GRAMMARS_DIR = path.join(os.homedir(), ".osgrep", "grammars");
const GRAMMAR_URLS = {
    typescript: "https://github.com/tree-sitter/tree-sitter-typescript/releases/latest/download/tree-sitter-typescript.wasm",
    tsx: "https://github.com/tree-sitter/tree-sitter-typescript/releases/latest/download/tree-sitter-tsx.wasm",
    javascript: "https://github.com/tree-sitter/tree-sitter-javascript/releases/latest/download/tree-sitter-javascript.wasm",
    python: "https://github.com/tree-sitter/tree-sitter-python/releases/latest/download/tree-sitter-python.wasm",
    go: "https://github.com/tree-sitter/tree-sitter-go/releases/latest/download/tree-sitter-go.wasm",
};
class ImportResolver {
    parser = null;
    languages = new Map();
    initialized = false;
    visitedPaths = new Set();
    async init() {
        if (this.initialized)
            return;
        try {
            await Parser.init({
                locator: require.resolve("web-tree-sitter/tree-sitter.wasm"),
            });
            this.parser = new Parser();
        }
        catch (err) {
            console.warn("Failed to initialize TreeSitter:", err);
            this.parser = null;
        }
        if (!fs.existsSync(GRAMMARS_DIR)) {
            fs.mkdirSync(GRAMMARS_DIR, { recursive: true });
        }
        this.initialized = true;
    }
    async downloadFile(url, dest) {
        const response = await fetch(url);
        if (!response.ok) {
            throw new Error(`Failed to download: ${response.statusText}`);
        }
        const buffer = await response.arrayBuffer();
        fs.writeFileSync(dest, Buffer.from(buffer));
    }
    async getLanguage(lang) {
        const cached = this.languages.get(lang);
        if (cached !== undefined)
            return cached;
        const wasmPath = path.join(GRAMMARS_DIR, `tree-sitter-${lang}.wasm`);
        if (!fs.existsSync(wasmPath)) {
            const url = GRAMMAR_URLS[lang];
            if (!url) {
                this.languages.set(lang, null);
                return null;
            }
            try {
                await this.downloadFile(url, wasmPath);
            }
            catch (err) {
                console.warn(`Could not download ${lang} grammar:`, err);
                this.languages.set(lang, null);
                return null;
            }
        }
        try {
            const language = await Language.load(wasmPath);
            this.languages.set(lang, language);
            return language;
        }
        catch (err) {
            console.warn(`Could not load ${lang} grammar:`, err);
            this.languages.set(lang, null);
            return null;
        }
    }
    detectLanguage(filePath) {
        const ext = path.extname(filePath).toLowerCase();
        switch (ext) {
            case ".ts":
                return "typescript";
            case ".tsx":
                return "tsx";
            case ".js":
            case ".mjs":
            case ".cjs":
                return "javascript";
            case ".jsx":
                return "javascript";
            case ".py":
                return "python";
            case ".go":
                return "go";
            default:
                return null;
        }
    }
    extractImportsFromAST(node, imports, filePath) {
        // TypeScript/JavaScript imports
        if (node.type === "import_statement") {
            const sourceNode = node.childForFieldName?.("source");
            if (sourceNode) {
                const source = sourceNode.text.slice(1, -1); // Remove quotes
                imports.push({
                    source,
                    type: "import",
                    line: node.startPosition.row + 1,
                });
            }
        }
        // CommonJS require
        if (node.type === "call_expression") {
            const functionNode = node.childForFieldName?.("function");
            const argumentsNode = node.childForFieldName?.("arguments");
            if (functionNode?.text === "require" &&
                argumentsNode?.namedChildren?.[0]) {
                const arg = argumentsNode.namedChildren[0];
                if (arg.type === "string") {
                    const source = arg.text.slice(1, -1); // Remove quotes
                    imports.push({
                        source,
                        type: "require",
                        line: node.startPosition.row + 1,
                    });
                }
            }
        }
        // Python imports
        if (node.type === "import_statement" || node.type === "import_from_statement") {
            const moduleNode = node.childForFieldName?.("module_name") ||
                node.namedChildren?.find(n => n.type === "dotted_name");
            if (moduleNode) {
                imports.push({
                    source: moduleNode.text,
                    type: "import",
                    line: node.startPosition.row + 1,
                });
            }
        }
        // Go imports
        if (node.type === "import_declaration") {
            const specs = node.namedChildren?.filter(n => n.type === "import_spec") || [];
            for (const spec of specs) {
                const pathNode = spec.childForFieldName?.("path");
                if (pathNode) {
                    const source = pathNode.text.slice(1, -1); // Remove quotes
                    imports.push({
                        source,
                        type: "import",
                        line: node.startPosition.row + 1,
                    });
                }
            }
        }
        // Recursively process children
        if (node.children) {
            for (const child of node.children) {
                this.extractImportsFromAST(child, imports, filePath);
            }
        }
    }
    resolveImportPath(source, fromFile) {
        const dir = path.dirname(fromFile);
        // Handle relative imports
        if (source.startsWith(".")) {
            const possibleExtensions = [".ts", ".tsx", ".js", ".jsx", ".mjs", ".cjs", ""];
            const possiblePaths = [
                source,
                path.join(source, "index"),
            ];
            for (const basePath of possiblePaths) {
                for (const ext of possibleExtensions) {
                    const fullPath = path.resolve(dir, basePath + ext);
                    if (fs.existsSync(fullPath) && fs.statSync(fullPath).isFile()) {
                        return fullPath;
                    }
                }
            }
        }
        // Handle node_modules or absolute imports (simplified)
        // In a real implementation, you'd need to walk up directories looking for node_modules
        const nodeModulesPath = path.join(dir, "node_modules", source);
        if (fs.existsSync(nodeModulesPath)) {
            const pkgJsonPath = path.join(nodeModulesPath, "package.json");
            if (fs.existsSync(pkgJsonPath)) {
                try {
                    const pkg = JSON.parse(fs.readFileSync(pkgJsonPath, "utf-8"));
                    const main = pkg.main || "index.js";
                    const mainPath = path.join(nodeModulesPath, main);
                    if (fs.existsSync(mainPath)) {
                        return mainPath;
                    }
                }
                catch {
                    // Ignore parse errors
                }
            }
        }
        return undefined;
    }
    async resolveImports(filePath) {
        await this.init();
        if (!fs.existsSync(filePath)) {
            return { path: filePath, imports: [] };
        }
        const content = fs.readFileSync(filePath, "utf-8");
        const lang = this.detectLanguage(filePath);
        const imports = [];
        if (!lang || !this.parser) {
            // Fallback: simple regex-based import detection
            const importRegex = /(?:import|require)\s*\(?["']([^"']+)["']\)?/g;
            let match;
            const lines = content.split("\n");
            while ((match = importRegex.exec(content)) !== null) {
                const lineNum = content.substring(0, match.index).split("\n").length;
                imports.push({
                    source: match[1],
                    type: match[0].includes("import") ? "import" : "require",
                    line: lineNum,
                });
            }
        }
        else {
            const language = await this.getLanguage(lang);
            if (language) {
                this.parser.setLanguage(language);
                const tree = this.parser.parse(content);
                this.extractImportsFromAST(tree.rootNode, imports, filePath);
            }
        }
        // Resolve import paths
        for (const imp of imports) {
            imp.resolvedPath = this.resolveImportPath(imp.source, filePath);
        }
        return { path: filePath, imports };
    }
    async resolveImportsRecursively(filePath, depth = 2, visited = new Set()) {
        const results = new Map();
        if (visited.has(filePath)) {
            return results;
        }
        visited.add(filePath);
        const fileInfo = await this.resolveImports(filePath);
        results.set(filePath, fileInfo);
        // Only recurse to imports if we have depth remaining
        if (depth > 0) {
            for (const imp of fileInfo.imports) {
                if (imp.resolvedPath && !imp.source.includes("node_modules")) {
                    const childResults = await this.resolveImportsRecursively(imp.resolvedPath, depth - 1, visited);
                    for (const [path, info] of childResults) {
                        results.set(path, info);
                    }
                }
            }
        }
        return results;
    }
}
exports.ImportResolver = ImportResolver;
//# sourceMappingURL=import-resolver.js.map