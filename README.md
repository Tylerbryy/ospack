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

## Options

```
-f, --focus FILE      Entry point for import resolution
-q, --query TEXT      Semantic search query
-m, --max-files N     Max files to include (default: 10)
-d, --import-depth N  Import traversal depth (default: 2)
-o, --format FORMAT   Output: xml, compact, or chunks
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

## How It Works

1. **Hard links**: Follow imports from focus file
2. **Soft links**: Semantic search for related code
3. **Hybrid ranking**: BM25 + embeddings + reranking

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
