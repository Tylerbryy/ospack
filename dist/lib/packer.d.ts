import { FileContext } from "./formatter";
export interface PackOptions {
    focus?: string;
    query?: string;
    depth?: number;
    maxFiles?: number;
    maxSemanticFiles?: number;
    includeTests?: boolean;
    format?: "xml" | "json" | "markdown" | "compact";
    rootDir?: string;
}
export declare class ContextPacker {
    private importResolver;
    private semanticLinker;
    private formatter;
    private processedFiles;
    private fileContexts;
    constructor(osgrepPath?: string);
    pack(options: PackOptions): Promise<string>;
    private resolveImports;
    private addFile;
    private getSortedContexts;
    private isTestFile;
    findRelatedContext(focusFile: string, additionalQuery?: string): Promise<FileContext[]>;
}
//# sourceMappingURL=packer.d.ts.map