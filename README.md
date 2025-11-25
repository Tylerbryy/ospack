<div align="center">
  <h1>ospack</h1>
  <p><em>Semantic context packer for AI coding assistants.</em></p>
  <a href="https://opensource.org/licenses/Apache-2.0"><img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg" alt="License: Apache 2.0" /></a>
</div>

<br>

Combines import resolution with semantic search to build optimal code context.

## Install

```bash
pip install ospack
```

## Usage

```bash
# Pack context from a file + its imports
ospack pack --focus src/auth.py

# Search for code semantically
ospack search "user authentication"

# Combine both
ospack pack --focus src/api.py --query "error handling"
```

## Commands

| Command | Description |
|---------|-------------|
| `ospack pack` | Pack context from focus file and/or semantic query |
| `ospack search` | Quick semantic search |
| `ospack map` | Generate repo structure map with signatures |
| `ospack index` | Build/rebuild the search index |
| `ospack info` | Show device and index info |

## Options

### pack
```
-f, --focus FILE      Entry point for import resolution
-q, --query TEXT      Semantic search query
-m, --max-files N     Max files to include (default: 10)
-d, --import-depth N  Import traversal depth (default: 2)
-o, --format FORMAT   Output: xml, compact, or chunks
-p, --pagerank        Use PageRank ranking for smarter dependency ordering
-S, --skeleton        Collapse non-focus function bodies to save tokens
```

### map
```
-m, --max-sigs N      Max signatures per file (default: unlimited)
--no-signatures       Show only file tree, no code signatures
```

## Agent Integration

### Claude Code Plugin

```bash
# Add the ospack marketplace
/plugin marketplace add ospack/ospack

# Install the plugin
/plugin install ospack@ospack
```

The plugin provides:
- `/pack` - Pack context command
- `/search` - Semantic search command
- Auto-invoked skill for codebase exploration

### MCP Server

```bash
# Start MCP server for AI agents
ospack mcp
```

Or add to your MCP config:

```json
{
  "mcpServers": {
    "ospack": {
      "command": "ospack",
      "args": ["mcp"]
    }
  }
}
```

**Available MCP Tools:**

| Tool | Description |
|------|-------------|
| `ospack_pack` | Pack context with imports + semantic search |
| `ospack_search` | Semantic code search |
| `ospack_map` | Generate repo structure map |
| `ospack_index` | Build/rebuild search index |
| `ospack_probe` | Detect missing symbols and suggest follow-up queries |

## Advanced Usage

### Smart Packing with PageRank + Skeletonization

```bash
# Get optimized context: important files first, non-focus code collapsed
ospack pack --focus src/api.py --pagerank --skeleton --max-files 5
```

PageRank identifies "hub" symbols that are referenced by many files. Skeletonization collapses function bodies to signatures in non-focus files, saving tokens while preserving structure.

### Repo Mapping

```bash
# Get a bird's-eye view of the codebase
ospack map --max-sigs 10

# Output shows hierarchy with methods under classes:
# ├── src/
# │   ├── auth.py
# │   │     class AuthService:
# │   │         def login(self, user, password):
# │   │         def logout(self):
# │   │     ... (5 more)
```

### Iterative Context Building (MCP)

Use `ospack_probe` to iteratively build complete context:

1. `ospack_pack(focus="auth.py")` → get initial context
2. `ospack_probe(content=...)` → find missing `UserModel`, `TokenService`
3. `ospack_pack(query="UserModel definition")` → fetch missing pieces
4. Repeat until context is complete

## How It Works

1. **Hard links**: Follow imports from focus file
2. **Soft links**: Semantic search for related code
3. **Hybrid ranking**: BM25 + embeddings + reranking
4. **PageRank**: Graph-based importance scoring for dependencies
5. **Skeletonization**: AST-based body collapsing for token efficiency

## Tech

- **Chunking**: tree-sitter AST parsing (20+ languages)
- **Embeddings**: Jina code embeddings
- **Vector store**: LanceDB
- **Search**: BM25 + semantic + cross-encoder reranking

## ospack vs osgrep

ospack is inspired by [osgrep](https://github.com/Ryandonofrio3/osgrep). Both use tree-sitter, LanceDB, and hybrid search, but serve different purposes:

| | **ospack** | **osgrep** |
|---|---|---|
| **Purpose** | Context packing for AI prompts | Semantic grep replacement |
| **Key feature** | Import resolution + semantic search | Live server with file watching |
| **Output** | Packed context (XML/markdown) | Search results |
| **Language** | Python | TypeScript |

**When to use osgrep**: "Find code about X" — fast semantic search tool.

**When to use ospack**: "Give me everything I need to understand/modify X" — builds complete context by following imports then augmenting with semantic matches.

## License

Apache-2.0
