---
description: Pack context from a focus file and/or semantic query
allowed-tools: Bash
---

# Pack Context

Use ospack to gather relevant code context for the current task.

## Usage

Run `ospack pack` with the appropriate flags:

```bash
# Pack from a specific file + its imports
ospack pack --focus <file_path> --root $(pwd)

# Semantic search for related code
ospack pack --query "<search_terms>" --root $(pwd)

# Combine both (recommended)
ospack pack --focus <file_path> --query "<search_terms>" --root $(pwd)
```

## Options

- `--focus, -f`: Entry point file for import resolution
- `--query, -q`: Natural language semantic search
- `--max-files, -m`: Max files to include (default: 10)
- `--import-depth, -d`: Import traversal depth (default: 2)
- `--format, -o`: Output format: xml, compact, or chunks

## When to use

Use this command when you need to:
- Understand a file and its dependencies
- Find related code for a concept
- Build context for refactoring or debugging
