"""Structured error types for agent-friendly error reporting."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ErrorCode(str, Enum):
    """Standard error codes for ospack operations."""

    # Input errors
    INVALID_PATH = "INVALID_PATH"
    INVALID_QUERY = "INVALID_QUERY"
    MISSING_REQUIRED = "MISSING_REQUIRED"

    # Index errors
    NO_INDEX = "NO_INDEX"
    INDEX_STALE = "INDEX_STALE"
    INDEX_BUILD_FAILED = "INDEX_BUILD_FAILED"

    # Search errors
    NO_RESULTS = "NO_RESULTS"
    SEARCH_FAILED = "SEARCH_FAILED"

    # File errors
    FILE_NOT_FOUND = "FILE_NOT_FOUND"
    FILE_READ_ERROR = "FILE_READ_ERROR"
    UNSUPPORTED_FILE_TYPE = "UNSUPPORTED_FILE_TYPE"

    # System errors
    MODEL_LOAD_FAILED = "MODEL_LOAD_FAILED"
    OUT_OF_MEMORY = "OUT_OF_MEMORY"
    INTERNAL_ERROR = "INTERNAL_ERROR"


# Suggestions for each error code
ERROR_SUGGESTIONS = {
    ErrorCode.INVALID_PATH: "Check that the path exists and is accessible",
    ErrorCode.INVALID_QUERY: "Provide a non-empty search query",
    ErrorCode.MISSING_REQUIRED: "Provide either --focus or --query (or both)",
    ErrorCode.NO_INDEX: "Run 'ospack index' to build the search index first",
    ErrorCode.INDEX_STALE: "Run 'ospack index --force' to rebuild the index",
    ErrorCode.INDEX_BUILD_FAILED: "Check file permissions and disk space",
    ErrorCode.NO_RESULTS: "Try a broader query or lower --min-score threshold",
    ErrorCode.SEARCH_FAILED: "Try rebuilding the index with 'ospack index --force'",
    ErrorCode.FILE_NOT_FOUND: "Verify the file path is correct",
    ErrorCode.FILE_READ_ERROR: "Check file permissions and encoding",
    ErrorCode.UNSUPPORTED_FILE_TYPE: (
        "ospack supports: .py, .ts, .tsx, .js, .jsx, .rs, .go, .java, .c, .h, .cpp, .hpp"
    ),
    ErrorCode.MODEL_LOAD_FAILED: (
        "Check internet connection for model download, or set OSPACK_DEVICE=cpu"
    ),
    ErrorCode.OUT_OF_MEMORY: "Try reducing --max-files or --max-chunks, or use --format chunks",
    ErrorCode.INTERNAL_ERROR: "Please report this issue at github.com/ospack/ospack",
}


@dataclass
class ErrorResponse:
    """Structured error response for agent consumption."""

    error: str  # Human-readable error message
    code: ErrorCode  # Machine-readable error code
    suggestion: str  # Actionable suggestion to resolve
    context: dict[str, Any] = field(default_factory=dict)  # Additional debug info

    @classmethod
    def create(
        cls,
        code: ErrorCode,
        error: str | None = None,
        context: dict[str, Any] | None = None,
    ) -> ErrorResponse:
        """Create an error response with automatic suggestion lookup."""
        return cls(
            error=error or code.value.replace("_", " ").title(),
            code=code,
            suggestion=ERROR_SUGGESTIONS.get(code, "Check the error details"),
            context=context or {},
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "error": self.error,
            "code": self.code.value,
            "suggestion": self.suggestion,
            "context": self.context,
        }

    def format_xml(self) -> str:
        """Format as XML for agent output."""
        context_attrs = " ".join(f'{k}="{v}"' for k, v in self.context.items())
        context_str = f" {context_attrs}" if context_attrs else ""
        return (
            f'<error code="{self.code.value}"{context_str}>\n'
            f"  <message>{self.error}</message>\n"
            f"  <suggestion>{self.suggestion}</suggestion>\n"
            f"</error>"
        )

    def format_compact(self) -> str:
        """Format as compact text."""
        lines = [
            f"Error [{self.code.value}]: {self.error}",
            f"Suggestion: {self.suggestion}",
        ]
        if self.context:
            lines.append(f"Context: {self.context}")
        return "\n".join(lines)


class OspackError(Exception):
    """Base exception for ospack with structured error support."""

    def __init__(
        self,
        code: ErrorCode,
        message: str | None = None,
        context: dict[str, Any] | None = None,
    ):
        self.code = code
        self.context = context or {}
        self.message = message or code.value.replace("_", " ").title()
        super().__init__(self.message)

    def to_response(self) -> ErrorResponse:
        """Convert exception to ErrorResponse."""
        return ErrorResponse.create(
            code=self.code,
            error=self.message,
            context=self.context,
        )


class InvalidPathError(OspackError):
    """Raised when a path is invalid or doesn't exist."""

    def __init__(self, path: str, message: str | None = None):
        super().__init__(
            code=ErrorCode.INVALID_PATH,
            message=message or f"Path not found: {path}",
            context={"path": path},
        )


class NoIndexError(OspackError):
    """Raised when the search index doesn't exist."""

    def __init__(self, root_dir: str):
        super().__init__(
            code=ErrorCode.NO_INDEX,
            message="Search index not found. Run 'ospack index' first.",
            context={"root_dir": root_dir},
        )


class NoResultsError(OspackError):
    """Raised when search returns no results."""

    def __init__(self, query: str, filters: dict[str, Any] | None = None):
        super().__init__(
            code=ErrorCode.NO_RESULTS,
            message=f"No results found for query: {query}",
            context={"query": query, **(filters or {})},
        )


class MissingRequiredError(OspackError):
    """Raised when required parameters are missing."""

    def __init__(self, missing: list[str]):
        super().__init__(
            code=ErrorCode.MISSING_REQUIRED,
            message=f"Missing required parameter(s): {', '.join(missing)}",
            context={"missing": missing},
        )
