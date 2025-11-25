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
exports.ContextFormatter = void 0;
const path = __importStar(require("node:path"));
class ContextFormatter {
    formatXML(files) {
        const xmlParts = ['<context>'];
        for (const file of files) {
            const attributes = [`path="${this.escapeXML(file.path)}"`, `reason="${file.reason}"`];
            if (file.score !== undefined) {
                attributes.push(`score="${file.score.toFixed(3)}"`);
            }
            if (file.line !== undefined) {
                attributes.push(`line="${file.line}"`);
            }
            xmlParts.push(`  <file ${attributes.join(" ")}>`);
            xmlParts.push(this.escapeXML(file.content));
            xmlParts.push('  </file>');
        }
        xmlParts.push('</context>');
        return xmlParts.join('\n');
    }
    formatJSON(files) {
        return JSON.stringify({
            context: files.map(f => ({
                path: f.path,
                reason: f.reason,
                score: f.score,
                line: f.line,
                content: f.content,
            })),
        }, null, 2);
    }
    formatMarkdown(files) {
        const mdParts = ['# Context Pack\n'];
        // Group files by reason
        const byReason = new Map();
        for (const file of files) {
            if (!byReason.has(file.reason)) {
                byReason.set(file.reason, []);
            }
            byReason.get(file.reason).push(file);
        }
        // Format each group
        for (const [reason, group] of byReason) {
            mdParts.push(`## ${this.capitalizeReason(reason)} Files\n`);
            for (const file of group) {
                mdParts.push(`### ${file.path}`);
                if (file.score !== undefined) {
                    mdParts.push(`*Relevance Score: ${file.score.toFixed(3)}*`);
                }
                const ext = path.extname(file.path).slice(1) || 'text';
                mdParts.push(`\n\`\`\`${this.getLanguageForExt(ext)}`);
                mdParts.push(file.content);
                mdParts.push('```\n');
            }
        }
        return mdParts.join('\n');
    }
    formatCompact(files) {
        const lines = [];
        for (const file of files) {
            const score = file.score !== undefined ? ` (${file.score.toFixed(3)})` : '';
            const reason = ` [${file.reason}]`;
            lines.push(`${file.path}${reason}${score}`);
        }
        return lines.join('\n');
    }
    escapeXML(str) {
        return str
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;')
            .replace(/"/g, '&quot;')
            .replace(/'/g, '&apos;');
    }
    capitalizeReason(reason) {
        switch (reason) {
            case 'focus':
                return 'Focus';
            case 'import':
                return 'Imported';
            case 'semantic':
                return 'Semantically Related';
            case 'context':
                return 'Additional Context';
            default:
                return reason.charAt(0).toUpperCase() + reason.slice(1);
        }
    }
    getLanguageForExt(ext) {
        const langMap = {
            ts: 'typescript',
            tsx: 'typescript',
            js: 'javascript',
            jsx: 'javascript',
            py: 'python',
            go: 'go',
            rs: 'rust',
            java: 'java',
            cpp: 'cpp',
            c: 'c',
            h: 'c',
            hpp: 'cpp',
            css: 'css',
            html: 'html',
            json: 'json',
            yaml: 'yaml',
            yml: 'yaml',
            md: 'markdown',
            sh: 'bash',
            sql: 'sql',
        };
        return langMap[ext] || ext;
    }
}
exports.ContextFormatter = ContextFormatter;
//# sourceMappingURL=formatter.js.map