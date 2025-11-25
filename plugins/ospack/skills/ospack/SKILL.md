# ospack - Semantic Context Packer

Use ospack to gather relevant code context through import resolution and semantic search.

## When to use this skill

Use ospack when you need to:
- Understand a file and all its dependencies
- Find code related to a concept across the codebase
- Build comprehensive context for refactoring, debugging, or feature work
- Explore unfamiliar codebases

## Commands

### Pack context (recommended)

Combines import resolution with semantic search:

```bash
# From a focus file
ospack pack --focus src/auth.py --root $(pwd)

# With semantic query
ospack pack --query "error handling" --root $(pwd)

# Both together (best results)
ospack pack --focus src/api.py --query "validation" --root $(pwd)
```

### Quick search

For fast semantic searches:

```bash
ospack search "database connection" --root $(pwd)
```

## Output formats

- `--format xml`: Structured XML (default, best for context)
- `--format compact`: Human-readable markdown
- `--format chunks`: Function-level results with scores

## Tips

- Use `--focus` when you have a specific entry point
- Use `--query` when searching by concept
- Combine both for comprehensive context
- Increase `--max-files` for larger explorations
