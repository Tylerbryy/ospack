export interface ImportInfo {
    source: string;
    resolvedPath?: string;
    type: "import" | "require" | "dynamic";
    line: number;
}
export interface ResolvedFile {
    path: string;
    imports: ImportInfo[];
}
export declare class ImportResolver {
    private parser;
    private languages;
    private initialized;
    private visitedPaths;
    init(): Promise<void>;
    private downloadFile;
    private getLanguage;
    private detectLanguage;
    private extractImportsFromAST;
    private resolveImportPath;
    resolveImports(filePath: string): Promise<ResolvedFile>;
    resolveImportsRecursively(filePath: string, depth?: number, visited?: Set<string>): Promise<Map<string, ResolvedFile>>;
}
//# sourceMappingURL=import-resolver.d.ts.map