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
exports.SemanticLinker = void 0;
const node_child_process_1 = require("node:child_process");
const node_util_1 = require("node:util");
const path = __importStar(require("node:path"));
const execAsync = (0, node_util_1.promisify)(node_child_process_1.exec);
class SemanticLinker {
    osgrepPath;
    constructor(osgrepPath) {
        // Try to use locally installed osgrep first, then global
        this.osgrepPath = osgrepPath || "osgrep";
    }
    async search(query, rootDir, limit = 10) {
        try {
            // Use osgrep's JSON output mode for machine consumption
            const command = `${this.osgrepPath} search --json -m ${limit} "${query}"`;
            const { stdout, stderr } = await execAsync(command, {
                cwd: rootDir,
                maxBuffer: 10 * 1024 * 1024, // 10MB buffer for large results
            });
            if (stderr && !stderr.includes("warning")) {
                console.warn("osgrep stderr:", stderr);
            }
            // Parse osgrep's JSON output
            try {
                // Filter out any non-JSON lines (like Worker: messages)
                const lines = stdout.split('\n');
                const jsonLine = lines.find(line => line.trim().startsWith('{'));
                if (!jsonLine) {
                    console.warn("No JSON output from osgrep");
                    return [];
                }
                const result = JSON.parse(jsonLine);
                if (!result.results || !Array.isArray(result.results)) {
                    return [];
                }
                // Transform osgrep results to our format
                return result.results.map((item) => ({
                    path: path.isAbsolute(item.path)
                        ? item.path
                        : path.join(rootDir, item.path),
                    score: item.score || 0,
                    content: item.content || "",
                }));
            }
            catch (parseError) {
                console.error("Failed to parse osgrep output:", parseError);
                return [];
            }
        }
        catch (error) {
            // If osgrep is not installed or fails, return empty results
            console.warn("osgrep search failed:", error);
            return [];
        }
    }
    async findRelatedFiles(filePath, rootDir, query, limit = 5) {
        // Build a contextual query based on the file
        const fileName = path.basename(filePath, path.extname(filePath));
        // If no explicit query, generate one based on the filename
        const searchQuery = query || `files related to ${fileName} functionality`;
        const results = await this.search(searchQuery, rootDir, limit * 2);
        // Filter out the original file and deduplicate
        const filtered = results.filter(r => {
            const normalizedPath = path.normalize(r.path);
            const normalizedTarget = path.normalize(filePath);
            return normalizedPath !== normalizedTarget;
        });
        // Sort by score and limit results
        return filtered
            .sort((a, b) => b.score - a.score)
            .slice(0, limit);
    }
    async searchByPattern(pattern, rootDir, targetPath) {
        // If a target path is specified, search within that directory
        const searchPath = targetPath
            ? path.relative(rootDir, targetPath)
            : "";
        try {
            const command = targetPath
                ? `${this.osgrepPath} search --json -m 25 "${pattern}" "${searchPath}"`
                : `${this.osgrepPath} search --json -m 25 "${pattern}"`;
            const { stdout } = await execAsync(command, {
                cwd: rootDir,
                maxBuffer: 10 * 1024 * 1024,
            });
            // Filter out any non-JSON lines
            const lines = stdout.split('\n');
            const jsonLine = lines.find(line => line.trim().startsWith('{'));
            if (!jsonLine) {
                return [];
            }
            const result = JSON.parse(jsonLine);
            if (!result.results) {
                return [];
            }
            return result.results.map((item) => ({
                path: path.isAbsolute(item.path)
                    ? item.path
                    : path.join(rootDir, item.path),
                score: item.score || 0,
                content: item.content || "",
            }));
        }
        catch (error) {
            console.warn("Pattern search failed:", error);
            return [];
        }
    }
    async ensureIndexed(rootDir) {
        try {
            // Run osgrep index to ensure the repository is indexed
            const { stderr } = await execAsync(`${this.osgrepPath} index`, {
                cwd: rootDir,
            });
            if (stderr && !stderr.includes("Indexing complete")) {
                console.warn("Index warning:", stderr);
            }
        }
        catch (error) {
            console.warn("Failed to ensure index:", error);
        }
    }
}
exports.SemanticLinker = SemanticLinker;
//# sourceMappingURL=semantic-linker.js.map