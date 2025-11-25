# ospack

Semantic context packer for AI coding assistants. Combines import resolution with semantic search to build optimal code context.

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
| `ospack index` | Build/rebuild the search index |
| `ospack info` | Show device and index info |
| `ospack mcp` | Start MCP server for AI agents |

## Options

```
-f, --focus FILE      Entry point for import resolution
-q, --query TEXT      Semantic search query
-m, --max-files N     Max files to include (default: 10)
-d, --import-depth N  Import traversal depth (default: 2)
-o, --format FORMAT   Output: xml, compact, or chunks
```

## How It Works

1. **Hard links**: Follow imports from focus file
2. **Soft links**: Semantic search for related code
3. **Hybrid ranking**: BM25 + embeddings + reranking

## Tech

- **Chunking**: tree-sitter AST parsing (20+ languages)
- **Embeddings**: Jina code embeddings
- **Vector store**: LanceDB
- **Search**: BM25 + semantic + cross-encoder reranking

## License

Apache-2.0
