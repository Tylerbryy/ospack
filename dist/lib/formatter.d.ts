export interface FileContext {
    path: string;
    content: string;
    reason: "focus" | "import" | "semantic" | "context";
    score?: number;
    line?: number;
}
export declare class ContextFormatter {
    formatXML(files: FileContext[]): string;
    formatJSON(files: FileContext[]): string;
    formatMarkdown(files: FileContext[]): string;
    formatCompact(files: FileContext[]): string;
    private escapeXML;
    private capitalizeReason;
    private getLanguageForExt;
}
//# sourceMappingURL=formatter.d.ts.map