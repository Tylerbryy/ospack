import * as fs from "node:fs";
import * as path from "node:path";

export interface FileContext {
  path: string;
  content: string;
  reason: "focus" | "import" | "semantic" | "context";
  score?: number;
  line?: number;
}

export class ContextFormatter {
  formatXML(files: FileContext[]): string {
    const xmlParts: string[] = ['<context>'];
    
    for (const file of files) {
      const attributes: string[] = [`path="${this.escapeXML(file.path)}"`, `reason="${file.reason}"`];
      
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

  formatJSON(files: FileContext[]): string {
    return JSON.stringify(
      {
        context: files.map(f => ({
          path: f.path,
          reason: f.reason,
          score: f.score,
          line: f.line,
          content: f.content,
        })),
      },
      null,
      2
    );
  }

  formatMarkdown(files: FileContext[]): string {
    const mdParts: string[] = ['# Context Pack\n'];
    
    // Group files by reason
    const byReason = new Map<string, FileContext[]>();
    for (const file of files) {
      if (!byReason.has(file.reason)) {
        byReason.set(file.reason, []);
      }
      byReason.get(file.reason)!.push(file);
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

  formatCompact(files: FileContext[]): string {
    const lines: string[] = [];
    
    for (const file of files) {
      const score = file.score !== undefined ? ` (${file.score.toFixed(3)})` : '';
      const reason = ` [${file.reason}]`;
      lines.push(`${file.path}${reason}${score}`);
    }
    
    return lines.join('\n');
  }

  private escapeXML(str: string): string {
    return str
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;')
      .replace(/'/g, '&apos;');
  }

  private capitalizeReason(reason: string): string {
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

  private getLanguageForExt(ext: string): string {
    const langMap: Record<string, string> = {
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