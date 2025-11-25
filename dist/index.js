#!/usr/bin/env node
"use strict";
var __createBinding = (this && this.__createBinding) || (Object.create ? (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    var desc = Object.getOwnPropertyDescriptor(m, k);
    if (!desc || ("get" in desc ? !m.__esModule : desc.writable || desc.configurable)) {
      desc = { enumerable: true, get: function() { return m[k]; } };
    }
    Object.defineProperty(o, k2, desc);
}) : (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    o[k2] = m[k];
}));
var __setModuleDefault = (this && this.__setModuleDefault) || (Object.create ? (function(o, v) {
    Object.defineProperty(o, "default", { enumerable: true, value: v });
}) : function(o, v) {
    o["default"] = v;
});
var __importStar = (this && this.__importStar) || (function () {
    var ownKeys = function(o) {
        ownKeys = Object.getOwnPropertyNames || function (o) {
            var ar = [];
            for (var k in o) if (Object.prototype.hasOwnProperty.call(o, k)) ar[ar.length] = k;
            return ar;
        };
        return ownKeys(o);
    };
    return function (mod) {
        if (mod && mod.__esModule) return mod;
        var result = {};
        if (mod != null) for (var k = ownKeys(mod), i = 0; i < k.length; i++) if (k[i] !== "default") __createBinding(result, mod, k[i]);
        __setModuleDefault(result, mod);
        return result;
    };
})();
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
const commander_1 = require("commander");
const pack_1 = require("./commands/pack");
const chalk_1 = __importDefault(require("chalk"));
const fs = __importStar(require("node:fs"));
const path = __importStar(require("node:path"));
const packageJson = JSON.parse(fs.readFileSync(path.join(__dirname, "../package.json"), "utf-8"));
const program = new commander_1.Command()
    .name("ospack")
    .description("Semantic Context Packer - Build perfect AI prompts from your codebase")
    .version(packageJson.version)
    .addCommand(pack_1.packCommand)
    .addCommand(pack_1.interactiveCommand);
// Default action when no command is specified
program
    .option("-f, --focus <file>", "Focus file to start from")
    .option("-q, --query <text>", "Semantic search query for related code")
    .option("-d, --depth <number>", "Import resolution depth", "2")
    .option("-m, --max-files <number>", "Maximum total files to include", "20")
    .option("--format <type>", "Output format: xml, json, markdown, compact", "xml")
    .option("-o, --output <file>", "Output to file instead of stdout")
    .action((options) => {
    // If options are provided, run pack command
    if (options.focus || options.query) {
        pack_1.packCommand.parse(process.argv);
    }
    else {
        // Show help if no options provided
        program.outputHelp();
    }
});
// Add helpful examples
program.on("--help", () => {
    console.log("");
    console.log(chalk_1.default.bold("Examples:"));
    console.log("");
    console.log("  # Pack context for a specific file");
    console.log("  $ ospack --focus src/auth/login.ts --depth 2");
    console.log("");
    console.log("  # Search for related code and pack it");
    console.log('  $ ospack --query "password hashing" --max-files 10');
    console.log("");
    console.log("  # Combine focus file with semantic search");
    console.log('  $ ospack --focus src/api/users.ts --query "user validation"');
    console.log("");
    console.log("  # Output to a file");
    console.log("  $ ospack --focus src/index.ts -o context.xml");
    console.log("");
    console.log("  # Interactive mode");
    console.log("  $ ospack interactive");
    console.log("");
    console.log(chalk_1.default.dim("Powered by osgrep's semantic search"));
});
// Error handling
process.on("uncaughtException", (error) => {
    console.error(chalk_1.default.red("Unexpected error:"), error.message);
    process.exit(1);
});
process.on("unhandledRejection", (reason, promise) => {
    console.error(chalk_1.default.red("Unhandled promise rejection:"), reason);
    process.exit(1);
});
// Parse arguments
program.parse(process.argv);
// Show help if no arguments provided
if (process.argv.length === 2) {
    program.outputHelp();
}
//# sourceMappingURL=index.js.map