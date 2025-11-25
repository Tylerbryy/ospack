# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build and Development Commands

```bash
pip install -e .         # Install in development mode
ospack --help            # Show CLI help
ospack pack --help       # Show pack command help
ospack info              # Show device (GPU) and index info
ospack index             # Build/rebuild the semantic index
ospack search "query"    # Search the index
```

## Architecture Overview

ospack is a semantic context packer CLI written in Python. It combines **hard links** (import resolution) with **soft links** (AI-powered semantic search using embeddings) to build context packages for AI coding assistants.

### GPU Acceleration

ospack auto-detects the best available device for embeddings:
- **macOS Apple Silicon**: Uses MPS (Metal Performance Shaders)
- **NVIDIA GPU**: Uses CUDA
- **No GPU**: Falls back to CPU

Override with `OSPACK_DEVICE=cpu` environment variable if needed.

### Core Components

**CLI** (`ospack/cli.py`)
- Click-based CLI with commands: `pack`, `index`, `search`, `info`
- Uses Rich for pretty console output

**Embedder** (`ospack/embedder.py`)
- `Embedder` class using sentence-transformers with GPU auto-detection
- Default model: `all-MiniLM-L6-v2` (384 dimensions)
- Lazy model loading for faster startup when not needed

**Chunker** (`ospack/chunker.py`)
- `Chunker` uses tree-sitter-languages for AST-based code chunking
- Extracts functions, classes, methods as semantic units
- Supports: Python, TypeScript, JavaScript, Rust, Go, Java, C/C++

**Indexer** (`ospack/indexer.py`)
- `Indexer` class using LanceDB for vector storage
- Automatic index updates when files change
- Per-repository indexes in `~/.ospack/lancedb/{repo-hash}/`

**Resolver** (`ospack/resolver.py`)
- `ImportResolver` extracts and resolves imports via regex patterns
- Builds dependency graphs for import-based context

**Packer** (`ospack/packer.py`)
- `Packer` orchestrates the packing process:
  1. Focus file + import resolution (hard links)
  2. Semantic search for query (soft links)
- Output formats: XML (Claude-optimized), Compact (Markdown)

### Data Flow

```
User Input (focus file + query)
    ↓
ImportResolver → Hard-linked files (imports)
    ↓
Indexer → Soft-linked files (embedding similarity via LanceDB)
    ↓
Packer → Combined, deduplicated files
    ↓
format_output() → XML/Compact output
```

### Cache Directories

- `~/.ospack/lancedb/` - LanceDB vector indexes (per repository)
- `~/.cache/torch/sentence_transformers/` - Embedding models (shared)

## Code Style

- Python 3.10+
- Type hints throughout
- Dataclasses for structured data
- Global singletons for expensive resources (embedder, chunker)
