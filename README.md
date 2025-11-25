# ospack - The Semantic Context Packer

> Build perfect AI prompts from your codebase by intelligently selecting relevant context.

## The Problem

Current AI coding assistants often fail because they either have **too little context** (missing files) or **too much context** (distracting the model with irrelevant code). When a developer asks an agent to "fix the bug in the payment handler," the agent usually needs more than just `payment.ts`. It needs the interface definitions, utility functions, and database schema it interacts with.

## The Solution

ospack acts as a smart pre-processor that builds the perfect prompt context by combining:

- **Hard Links (Static Analysis)**: Uses tree-sitter to parse AST and follow import paths  
- **Soft Links (Semantic Search)**: Leverages osgrep's embedding engine to find conceptually related code

## Quick Example

```bash
ospack --focus src/checkout.ts --query "payment processing" --depth 2
```

Output:
```xml
<context>
  <file path="src/checkout.ts" reason="focus">...</file>
  <file path="src/utils/payment.ts" reason="import">...</file>
  <file path="src/stripe-webhooks.ts" reason="semantic" score="0.850">...</file>
</context>
```

The tool found:
1. **Focus file**: Your target file  
2. **Import**: Direct dependency via static analysis
3. **Semantic**: Logically related file found via embeddings

## How It Works

ospack combines two approaches:

1. **Hard Links (Static Analysis)**: Uses tree-sitter to parse your code's AST and follow import/require statements
2. **Soft Links (Semantic Search)**: Leverages osgrep's embedding search to find conceptually related code

## Installation

```bash
# Clone and setup
git clone <repository-url>
cd ospack
npm install

# Build the project
npm run build

# Optional: Link globally for CLI usage
npm link
```

## Usage

### Basic Examples

```bash
# Pack context for a specific file
ospack --focus src/auth/login.ts --depth 2

# Search for related code and pack it
ospack --query "password hashing" --max-files 10

# Combine focus file with semantic search
ospack --focus src/api/users.ts --query "user validation"

# Output to a file
ospack --focus src/index.ts -o context.xml

# Interactive mode
ospack interactive
```

### Options

- `--focus <file>`: Target file to start from
- `--query <text>`: Semantic search query for related code
- `--depth <n>`: Import resolution depth (default: 2)
- `--max-files <n>`: Maximum total files to include (default: 20)
- `--max-semantic <n>`: Maximum semantic search results (default: 5)
- `--format <type>`: Output format: xml, json, markdown, compact (default: xml)
- `--output <file>`: Output to file instead of stdout
- `--include-tests`: Include test files in results
- `--root <dir>`: Project root directory

## Output Formats

### XML (Default - Optimized for Claude)
```xml
<context>
  <file path="src/auth/login.ts" reason="focus">
    // file content
  </file>
  <file path="src/utils/hash.ts" reason="import">
    // file content
  </file>
  <file path="src/config/security.ts" reason="semantic" score="0.850">
    // file content
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
      "content": "// file content"
    }
  ]
}
```

### Markdown (Human Readable)
Beautiful formatted markdown with syntax highlighting and organization by file type.

### Compact (File List)
Simple list of files with their reason for inclusion and relevance scores.

## Integration with AI Assistants

ospack is designed to work seamlessly with AI coding assistants:

```bash
# Generate context and pipe to Claude
ospack --focus src/checkout.ts --query "payment processing" | pbcopy

# Generate context for a bug fix
ospack --focus src/api/error.ts --query "error handling middleware" -o bug-context.xml
```

## Requirements

- Node.js 18+
- osgrep installed and accessible in PATH
- Repository must be indexed by osgrep (`osgrep index`)

## Architecture

ospack is built on top of osgrep's powerful semantic search capabilities:

1. **Import Resolver**: Parses TypeScript, JavaScript, Python, and Go files to extract import statements
2. **Semantic Linker**: Interfaces with osgrep to find conceptually related files
3. **Context Packer**: Intelligently combines hard and soft links while respecting depth and size limits
4. **Formatter**: Outputs in multiple formats optimized for different use cases

## License

Apache 2.0