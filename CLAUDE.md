# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build and Development Commands

```bash
pip install -e .         # Install in development mode
ospack --help            # Show CLI help
ospack pack --help       # Show pack command help
ospack info              # Show index info and status
ospack index             # Build/rebuild the semantic index
ospack search "query"    # Search the index
```

## Architecture Overview

ospack is a semantic context packer CLI written in Python. It combines **hard links** (import resolution) with **soft links** (BM25+ keyword search) to build context packages for AI coding assistants.

### Core Components

**CLI** (`ospack/cli.py`)
- Click-based CLI with commands: `pack`, `index`, `search`, `info`, `map`
- Uses Rich for pretty console output

**Chunker** (`ospack/chunker.py`)
- Tree-sitter based code chunking
- Extracts functions, classes, methods as individual chunks
- Supports 15+ languages: Python, JS/TS, Go, Rust, Java, C/C++, Ruby, etc.

**Indexer** (`ospack/indexer.py`)
- BM25+ search using `bm25s` library with numba backend
- PyStemmer for better recall (stemming)
- Code-aware tokenization (camelCase, snake_case splitting)
- Memory-mapped loading for reduced RAM usage
- Per-repository indexes in `~/.ospack/index/{repo-hash}/`

**Resolver** (`ospack/resolver.py`)
- `ImportResolver` extracts and resolves imports via regex patterns
- Builds dependency graphs for import-based context

**Packer** (`ospack/packer.py`)
- `Packer` orchestrates the packing process:
  1. Focus file + import resolution (hard links)
  2. BM25+ search for query (soft links)
- Skeletonization: collapse function bodies to signatures (saves tokens)
- Output formats: XML (Claude-optimized), Compact (Markdown)

### Data Flow

```
User Input (focus file + query)
    ↓
ImportResolver → Hard-linked files (imports)
    ↓
Indexer → Soft-linked files (BM25+ keyword match)
    ↓
Packer → Combined, deduplicated files
    ↓
format_output() → XML/Compact output
```

### Cache Directories

- `~/.ospack/index/` - BM25+ indexes (per repository)

## Code Style

- Python 3.10+
- Type hints throughout
- Dataclasses for structured data
- Global singletons for expensive resources (chunker, indexer)
