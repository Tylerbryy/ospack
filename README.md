# ospack - Semantic Context Packer

> Build perfect AI prompts from your codebase with intelligent context discovery

ospack is a production-ready semantic context packer that combines static analysis (hard links) with AI-powered semantic search (soft links) to automatically discover and package the most relevant code context for AI coding assistance.

## 🚀 Features

### Hard Links (Import Resolution)
- **Tree-sitter AST analysis** for precise import tracking
- **Multi-language support** (TypeScript, JavaScript, Python, Go)
- **Recursive dependency resolution** with configurable depth
- **Smart filtering** to avoid circular dependencies

### Soft Links (Semantic Search)
- **AI-powered semantic search** using mixedbread-ai embeddings
- **Worker-based processing** for Node.js compatibility
- **Vector similarity matching** for contextually relevant code
- **LRU caching** for performance optimization

### Production Ready
- **Repository isolation** - each project gets its own semantic index
- **Incremental indexing** - only reprocess changed files
- **Multiple output formats** - XML, JSON, Markdown, Compact
- **Interactive CLI** - user-friendly interface with prompts
- **Worker pool management** - efficient multi-threaded processing

## 📦 Installation

```bash
npm install -g ospack
```

## 🔧 Quick Start

### Focus on a specific file
```bash
ospack pack --focus src/auth/login.ts --depth 2
```

### Search for related code semantically
```bash
ospack pack --query "password hashing and validation"
```

### Combine focus with semantic search
```bash
ospack pack --focus src/api/users.ts --query "user authentication"
```

### Interactive mode
```bash
ospack interactive
```

## 📋 Command Reference

### `ospack pack`

Pack context from your codebase for AI prompts.

**Options:**
- `-f, --focus <file>` - Focus file to start from
- `-q, --query <text>` - Semantic search query for related code
- `-d, --depth <number>` - Import resolution depth (default: 2)
- `-m, --max-files <number>` - Maximum total files to include (default: 20)
- `--format <type>` - Output format: xml, json, markdown, compact (default: xml)
- `-o, --output <file>` - Output to file instead of stdout
- `--include-tests` - Include test files in results
- `--root <dir>` - Project root directory
- `--reindex` - Force re-indexing of repository

**Examples:**
```bash
# Pack context for a specific file
ospack pack --focus src/auth/login.ts --depth 2

# Search for related code and pack it
ospack pack --query "password hashing" --max-files 10

# Combine focus file with semantic search
ospack pack --focus src/api/users.ts --query "user validation"

# Output to a file
ospack pack --focus src/index.ts -o context.xml

# Include test files
ospack pack --query "authentication" --include-tests
```

### `ospack interactive`

Interactive mode for packing context with guided prompts.

**Alias:** `ospack i`

## 🏗️ How It Works

### 1. Hard Links (Import Resolution)
ospack uses tree-sitter to parse your code and build a dependency graph:

```
src/auth/login.ts
├── src/auth/utils.ts (import)
├── src/database/user.ts (import)
└── src/types/auth.ts (import)
```

### 2. Soft Links (Semantic Search)
AI embeddings find contextually related code:

```
Query: "user authentication"
Results:
├── src/middleware/auth.ts (0.89 similarity)
├── src/routes/login.ts (0.84 similarity)
└── src/utils/jwt.ts (0.78 similarity)
```

### 3. Context Assembly
Combines both sources with intelligent deduplication:

```xml
<context>
  <file path="src/auth/login.ts" reason="focus">
    <!-- Your focused file -->
  </file>
  <file path="src/auth/utils.ts" reason="import">
    <!-- Imported dependency -->
  </file>
  <file path="src/middleware/auth.ts" reason="semantic">
    <!-- Semantically related -->
  </file>
</context>
```

## 🎯 Output Formats

### XML (Default - Optimized for Claude)
```xml
<context>
  <file path="src/auth/login.ts" reason="focus">
    export async function login(email: string, password: string) {
      // Implementation
    }
  </file>
</context>
```

### JSON (Machine Readable)
```json
{
  "context": [
    {
      "path": "src/auth/login.ts",
      "reason": "focus",
      "content": "export async function login..."
    }
  ]
}
```

### Markdown (Human Readable)
```markdown
# Context Pack

## Focus Files
### src/auth/login.ts
```typescript
export async function login(email: string, password: string) {
  // Implementation
}
```

### Compact (File List Only)
```
src/auth/login.ts [focus]
src/auth/utils.ts [import]
src/middleware/auth.ts [semantic]
```

## ⚙️ Configuration

### Environment Variables

- `OSPACK_WORKER_COUNT` - Number of worker threads (default: 1)
- `OSPACK_SINGLE_WORKER=1` - Force single worker mode
- `OSPACK_VECTOR_CACHE_MAX` - Maximum vector cache size (default: 10000)
- `OSPACK_WORKER_TIMEOUT_MS` - Worker timeout in milliseconds (default: 60000)

### Cache Directories

ospack creates cache directories in your home folder:
- `~/.ospack/models/` - AI embedding models
- `~/.ospack/grammars/` - Tree-sitter language grammars  
- `~/.ospack/indexes/` - Repository-specific semantic indexes

## 🔍 Advanced Usage

### Repository Isolation
Each project gets its own semantic index based on directory hash:
```bash
# Project A
cd /path/to/project-a
ospack pack --query "authentication"  # Uses index for project-a

# Project B  
cd /path/to/project-b
ospack pack --query "authentication"  # Uses index for project-b
```

### Performance Optimization
```bash
# Re-index after major changes
ospack pack --reindex --query "new feature"

# Use compact format for faster processing
ospack pack --focus src/main.ts --format compact

# Limit scope for faster results
ospack pack --query "search term" --max-files 5 --depth 1
```

### Integration with AI Tools

ospack is optimized for Claude and other AI coding assistants:

```bash
# Generate context and pipe to Claude
ospack pack --focus src/bug.ts --query "error handling" -o context.xml
# Then upload context.xml to Claude

# Quick compact overview
ospack pack --query "feature implementation" --format compact
```

## 🛠️ Troubleshooting

### Common Issues

**Model Download Issues**
```bash
# Clear model cache and retry
rm -rf ~/.ospack/models/
ospack pack --query "test"
```

**Tree-sitter Grammar Issues**
```bash
# Clear grammar cache
rm -rf ~/.ospack/grammars/
```

**Performance Issues**
```bash
# Use single worker
OSPACK_SINGLE_WORKER=1 ospack pack --query "search"

# Reduce worker count
OSPACK_WORKER_COUNT=2 ospack pack --query "search"
```

**Index Corruption**
```bash
# Force re-indexing
ospack pack --reindex --focus src/main.ts
```

### Debug Mode
```bash
# Enable verbose logging
DEBUG=ospack* ospack pack --query "debug info"
```

## 🆚 ospack vs osgrep

### osgrep: Semantic Search Engine
**osgrep** is a semantic search engine for codebases that replaces traditional grep/ripgrep:

```bash
# osgrep searches for semantically similar code
osgrep "authentication logic"  # Returns relevant auth-related code
osgrep "password validation"   # Finds validation functions
```

**Use osgrep when you want to:**
- Find code snippets related to a concept
- Explore unfamiliar codebases
- Search by functionality rather than exact text

### ospack: Context Assembler  
**ospack** is a context assembler that builds complete file contexts for AI assistants:

```bash
# ospack builds comprehensive context packages
ospack pack --focus login.ts --query "auth middleware"
# Returns: login.ts + its imports + related auth files
```

**Use ospack when you want to:**
- Give AI assistants complete context about a feature
- Understand all dependencies around a specific file
- Build prompts for code modification/debugging

### Key Differences

| Feature | osgrep | ospack |
|---------|--------|--------|
| **Purpose** | Search for code snippets | Assemble complete contexts |
| **Output** | Code snippets with scores | Complete files with relationships |
| **Use Case** | "Find where auth happens" | "Give me everything about auth" |
| **AI Integration** | Search results for analysis | Ready-to-use prompt contexts |
| **Import Analysis** | ❌ No | ✅ Yes (hard links) |
| **File Relationships** | ❌ No | ✅ Yes (dependency tracking) |

### Example Comparison

**osgrep search:**
```bash
osgrep "user authentication"
# Returns: 15 code snippets across 8 files
# Good for: Understanding auth patterns in codebase
```

**ospack context:**
```bash
ospack pack --focus auth/login.ts --query "user authentication"
# Returns: Complete files with full context
# Good for: AI assistant to modify/debug auth system
```

### When to Use Which

**Use osgrep for:**
- 🔍 Code exploration and discovery
- 📚 Learning how things work in a codebase
- 🎯 Finding specific implementation patterns
- 🔎 Quick semantic searches

**Use ospack for:**
- 🤖 AI assistant interactions
- 🛠️ Code modification tasks  
- 🐛 Bug fixing with full context
- 📝 Documentation generation
- 🔧 Refactoring projects

**Use both together:**
```bash
# 1. Discover with osgrep
osgrep "payment processing" | head -10

# 2. Build context with ospack
ospack pack --focus src/payments/stripe.ts --query "payment processing"
```

## 🏆 Why ospack?

### vs. Manual File Selection
- **Problem**: Manually selecting relevant files is time-consuming and error-prone
- **Solution**: Automatic discovery of both imported dependencies and semantically related code

### vs. Simple grep/ripgrep
- **Problem**: Text search misses semantic relationships and can't understand code structure
- **Solution**: AI embeddings understand code meaning and relationships beyond text matching

### vs. IDE "Find References"
- **Problem**: Only finds direct references, misses related functionality
- **Solution**: Combines static analysis with semantic search for comprehensive context

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Make your changes and add tests
4. Run the build: `npm run build`
5. Run tests: `npm test`
6. Commit changes: `git commit -m "Add amazing feature"`
7. Push to branch: `git push origin feature/amazing-feature`
8. Open a Pull Request

## 📄 License

Apache-2.0 License - see [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built on [tree-sitter](https://tree-sitter.github.io/) for code parsing
- Uses [Hugging Face Transformers](https://huggingface.co/transformers/) for embeddings
- Inspired by [osgrep](https://github.com/ryandonofrio3/osgrep) architecture
- Embedding models by [mixedbread-ai](https://huggingface.co/mixedbread-ai)

## 📊 Performance

**Typical Performance (on MacBook Pro M1)**
- **Indexing**: ~1000 files/minute
- **Search**: <2 seconds for most queries  
- **Memory**: ~200MB during indexing, ~50MB at rest
- **Storage**: ~1MB per 1000 files indexed

**Scalability**
- **Tested up to**: 10,000+ file codebases
- **Recommended max**: 50,000 files per repository
- **Worker threads**: 1-8 workers depending on system

---

**Transform your AI coding workflow with intelligent context discovery.** 🚀