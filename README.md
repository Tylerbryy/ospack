# ospack - Semantic Context Packer

Build perfect AI prompts from your codebase with intelligent context discovery.

ospack combines **hard links** (import resolution) with **soft links** (semantic search) to automatically discover and package the most relevant code context for AI coding assistants.

## Installation

```bash
pip install ospack
```

**Requirements:** Python 3.10+

## Quick Start

```bash
# Build the search index (runs automatically on first use)
ospack index

# Pack context for a specific file + its imports
ospack pack --focus src/auth.py

# Search for code semantically
ospack search "user authentication"

# Combine both: file context + related code
ospack pack --focus src/api.py --query "error handling"
```

## Commands

### `ospack pack`
Main context packing command. Combines import resolution with semantic search.

```bash
# Focus on a file and follow its imports
ospack pack --focus src/main.py --import-depth 2

# Semantic search for related code
ospack pack --query "database connection pooling" --max-files 5

# Both together (recommended)
ospack pack --focus src/api.py --query "validation" --format compact

# Output formats: xml (default), compact, chunks
ospack pack --query "auth" --format chunks --max-chunks 10
```

**Key options:**
- `-f, --focus` - Entry point file for import resolution
- `-q, --query` - Natural language semantic search
- `-m, --max-files` - Maximum files to include (default: 10)
- `-d, --import-depth` - Import traversal depth (default: 2)
- `-o, --format` - Output: `xml`, `compact`, or `chunks`

### `ospack search`
Quick semantic search without full context packing.

```bash
ospack search "password hashing"
ospack search "API rate limiting" --limit 5
```

### `ospack index`
Build or rebuild the semantic search index.

```bash
ospack index           # Incremental update
ospack index --force   # Full rebuild
```

### Analysis Commands

```bash
# Find implementations of a concept
ospack find "caching layer"

# Explain code with full context
ospack explain --file src/auth.py --function login

# Discover related code from an entry point
ospack discover --entry src/api.py

# Analyze impact of changes to a file
ospack impact --file src/models.py
```

### `ospack mcp`
Start MCP server for AI agent integration.

```bash
ospack mcp  # Starts stdio MCP server
```

## How It Works

### 1. Hard Links (Import Resolution)
Starting from a focus file, ospack follows import statements to build a dependency graph:

```
src/auth/login.py
├── src/auth/utils.py (import)
├── src/database/user.py (import)
└── src/types/auth.py (import)
```

### 2. Soft Links (Semantic Search)
AI embeddings find conceptually related code beyond direct imports:

```
Query: "user authentication"
├── src/middleware/auth.py (0.89 similarity)
├── src/routes/login.py (0.84 similarity)
└── src/utils/jwt.py (0.78 similarity)
```

### 3. Hybrid Ranking
Results are ranked using:
- BM25 keyword matching
- Semantic embedding similarity
- Cross-encoder reranking for precision

## Output Formats

### XML (Default - Optimized for Claude)
```xml
<context>
  <file path="src/auth/login.py" reason="focus" lines="1-45">
    def login(email: str, password: str):
        ...
  </file>
  <file path="src/auth/utils.py" reason="import" lines="1-30">
    ...
  </file>
</context>
```

### Compact (Human Readable)
```markdown
## src/auth/login.py
```python
def login(email: str, password: str):
    ...
```

### Chunks (Function-Level)
Best for large codebases - returns individual functions/classes:
```
src/auth/login.py:login (lines 10-45) [score: 0.92]
src/auth/utils.py:hash_password (lines 5-15) [score: 0.87]
```

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `OSPACK_DEVICE` | Compute device: `cpu`, `cuda`, `mps` | Auto-detect |
| `OSPACK_LOG_LEVEL` | Logging level: `DEBUG`, `INFO`, `WARNING` | `INFO` |
| `OSPACK_MAX_FILE_SIZE` | Skip files larger than this (bytes) | 1048576 (1MB) |

### GPU Acceleration

ospack auto-detects the best available device:
- **macOS Apple Silicon**: MPS (Metal Performance Shaders)
- **NVIDIA GPU**: CUDA
- **No GPU**: CPU fallback

### Cache Directories

- `~/.ospack/index/{repo-hash}/` - Per-repository vector indexes
- `~/.cache/huggingface/` - Embedding models (shared)

## Tech Stack

- **Chunking**: langchain-text-splitters (26+ languages via regex patterns)
- **Embeddings**: sentence-transformers with Jina embeddings
- **Vector Store**: LanceDB
- **Hybrid Search**: BM25 (rank-bm25) + semantic + cross-encoder reranking
- **CLI**: Click + Rich

## Supported Languages

Python, JavaScript, TypeScript, Go, Rust, Java, C, C++, Ruby, PHP, Swift, Kotlin, Scala, C#, Lua, Haskell, Perl, Elixir, Markdown, HTML, LaTeX, and more.

## License

Apache-2.0
