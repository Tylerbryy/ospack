import { exec } from "node:child_process";
import { promisify } from "node:util";
import * as path from "node:path";

const execAsync = promisify(exec);

export interface SemanticMatch {
  path: string;
  score: number;
  content: string;
}

export class SemanticLinker {
  private osgrepPath: string;

  constructor(osgrepPath?: string) {
    // Try to use locally installed osgrep first, then global
    this.osgrepPath = osgrepPath || "osgrep";
  }

  async search(
    query: string,
    rootDir: string,
    limit: number = 10
  ): Promise<SemanticMatch[]> {
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
        return result.results.map((item: any) => ({
          path: path.isAbsolute(item.path) 
            ? item.path 
            : path.join(rootDir, item.path),
          score: item.score || 0,
          content: item.content || "",
        }));
      } catch (parseError) {
        console.error("Failed to parse osgrep output:", parseError);
        return [];
      }
    } catch (error) {
      // If osgrep is not installed or fails, return empty results
      console.warn("osgrep search failed:", error);
      return [];
    }
  }

  async findRelatedFiles(
    filePath: string,
    rootDir: string,
    query?: string,
    limit: number = 5
  ): Promise<SemanticMatch[]> {
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

  async searchByPattern(
    pattern: string,
    rootDir: string,
    targetPath?: string
  ): Promise<SemanticMatch[]> {
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

      return result.results.map((item: any) => ({
        path: path.isAbsolute(item.path) 
          ? item.path 
          : path.join(rootDir, item.path),
        score: item.score || 0,
        content: item.content || "",
      }));
    } catch (error) {
      console.warn("Pattern search failed:", error);
      return [];
    }
  }

  async ensureIndexed(rootDir: string): Promise<void> {
    try {
      // Run osgrep index to ensure the repository is indexed
      const { stderr } = await execAsync(`${this.osgrepPath} index`, {
        cwd: rootDir,
      });

      if (stderr && !stderr.includes("Indexing complete")) {
        console.warn("Index warning:", stderr);
      }
    } catch (error) {
      console.warn("Failed to ensure index:", error);
    }
  }
}