export interface SemanticMatch {
    path: string;
    score: number;
    content: string;
}
export declare class SemanticLinker {
    private osgrepPath;
    constructor(osgrepPath?: string);
    search(query: string, rootDir: string, limit?: number): Promise<SemanticMatch[]>;
    findRelatedFiles(filePath: string, rootDir: string, query?: string, limit?: number): Promise<SemanticMatch[]>;
    searchByPattern(pattern: string, rootDir: string, targetPath?: string): Promise<SemanticMatch[]>;
    ensureIndexed(rootDir: string): Promise<void>;
}
//# sourceMappingURL=semantic-linker.d.ts.map