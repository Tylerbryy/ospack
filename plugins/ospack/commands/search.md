---
description: Semantic search for code in the current repository
allowed-tools: Bash
---

# Semantic Search

Use ospack to find code semantically by concept, not just keywords.

## Usage

```bash
ospack search "<query>" --root $(pwd)
```

## Examples

```bash
# Find authentication logic
ospack search "user authentication" --root $(pwd)

# Find error handling patterns
ospack search "error handling middleware" --root $(pwd)

# Find database queries
ospack search "database connection pooling" --root $(pwd)
```

## Options

- `--limit, -l`: Max results to return (default: 10)
- `--root`: Repository root directory

## When to use

Use this for quick semantic searches when you need to find:
- Where a concept is implemented
- Related code across the codebase
- Patterns and examples
