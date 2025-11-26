"""Background file watcher for automatic index updates.

Watches for file changes and triggers incremental indexing with debouncing
to avoid excessive updates during rapid file modifications.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import TYPE_CHECKING

from pathspec import PathSpec
from pathspec.patterns import GitWildMatchPattern
from watchdog.events import FileSystemEvent, FileSystemEventHandler
from watchdog.observers import Observer

from .log import get_logger

if TYPE_CHECKING:
    from .indexer import Indexer

logger = get_logger(__name__)

# Reuse exclude patterns from indexer
EXCLUDE_PATTERNS = [
    ".git/**", ".svn/**", ".hg/**",
    "node_modules/**", ".venv/**", "venv/**", "vendor/**",
    "dist/**", "build/**", "target/**", "__pycache__/**",
    "*.pyc", "*.pyo",
    ".idea/**", ".vscode/**", "*.swp", "*.swo",
    ".next/**", ".turbo/**", "coverage/**",
    "package-lock.json", "yarn.lock", "pnpm-lock.yaml",
    "Cargo.lock", "poetry.lock", "Pipfile.lock",
    "*.min.js", "*.min.css",
    "~/.ospack/**",  # Don't watch our own index files
]

# Debounce delay in seconds - wait this long after last change before indexing
DEBOUNCE_DELAY = 2.0


class IndexEventHandler(FileSystemEventHandler):
    """Handles file system events and queues index updates."""

    def __init__(self, root_dir: Path, indexer: Indexer):
        super().__init__()
        self.root_dir = root_dir
        self.indexer = indexer
        self.exclude_spec = PathSpec.from_lines(GitWildMatchPattern, EXCLUDE_PATTERNS)

        # Debouncing state
        self._timer: threading.Timer | None = None
        self._timer_lock = threading.Lock()

        # Execution state - prevents overlapping build_index calls
        self._execution_lock = threading.Lock()
        self._update_pending_during_execution = False

    def _should_process(self, path: str, is_directory: bool) -> bool:
        """Check if file should trigger an index update.

        Args:
            path: The filesystem path
            is_directory: Whether the path is a directory (from event.is_directory)
        """
        # Use the event's is_directory flag, not disk state (file may be deleted)
        if is_directory:
            return False

        try:
            rel_path = str(Path(path).relative_to(self.root_dir))
        except ValueError:
            # Path is outside root_dir (possible during moves)
            return False

        # Skip excluded patterns
        if self.exclude_spec.match_file(rel_path):
            return False

        return True

    def _schedule_update(self):
        """Schedule a debounced index update."""
        with self._timer_lock:
            # Cancel existing timer
            if self._timer is not None:
                self._timer.cancel()

            # Schedule new timer
            self._timer = threading.Timer(DEBOUNCE_DELAY, self._trigger_update)
            self._timer.daemon = True
            self._timer.start()

    def _trigger_update(self):
        """Attempt to run the update, preventing overlapping execution."""
        # Try to acquire execution lock without blocking
        if self._execution_lock.acquire(blocking=False):
            try:
                self._do_update()
            finally:
                self._execution_lock.release()

                # Check if a change happened while we were indexing
                with self._timer_lock:
                    if self._update_pending_during_execution:
                        self._update_pending_during_execution = False
                        # Reschedule to catch changes that happened during indexing
                        self._schedule_update()
        else:
            # Indexing is currently busy - mark that we need another run
            with self._timer_lock:
                self._update_pending_during_execution = True

    def _do_update(self):
        """Perform the actual index update (heavy lifting)."""
        try:
            logger.info("Auto-indexing triggered by file changes...")
            chunks = self.indexer.build_index(force=False)
            if chunks > 0:
                logger.info("Auto-indexed %d new chunks", chunks)
            else:
                logger.debug("Index already up to date")
        except Exception as e:
            logger.warning("Auto-index failed: %s", e)

    def stop(self):
        """Clean up pending timers on stop."""
        with self._timer_lock:
            if self._timer is not None:
                self._timer.cancel()
                self._timer = None

    def on_created(self, event: FileSystemEvent):
        if self._should_process(event.src_path, event.is_directory):
            logger.debug("File created: %s", event.src_path)
            self._schedule_update()

    def on_modified(self, event: FileSystemEvent):
        if self._should_process(event.src_path, event.is_directory):
            logger.debug("File modified: %s", event.src_path)
            self._schedule_update()

    def on_deleted(self, event: FileSystemEvent):
        if self._should_process(event.src_path, event.is_directory):
            logger.debug("File deleted: %s", event.src_path)
            self._schedule_update()

    def on_moved(self, event: FileSystemEvent):
        # Check source path
        if self._should_process(event.src_path, event.is_directory):
            logger.debug("File moved from: %s", event.src_path)
            self._schedule_update()
            return  # Avoid double scheduling

        # Check destination (only if source didn't trigger)
        if hasattr(event, 'dest_path'):
            if self._should_process(event.dest_path, event.is_directory):
                logger.debug("File moved to: %s", event.dest_path)
                self._schedule_update()


class IndexWatcher:
    """Watches a directory for changes and auto-updates the index."""

    def __init__(self, root_dir: str, indexer: Indexer):
        self.root_dir = Path(root_dir).resolve()
        self.indexer = indexer
        self._observer: Observer | None = None
        self._handler: IndexEventHandler | None = None
        self._started = False
        self._lock = threading.Lock()

    def start(self):
        """Start watching for file changes."""
        with self._lock:
            if self._started:
                return

            self._handler = IndexEventHandler(self.root_dir, self.indexer)
            self._observer = Observer()
            self._observer.schedule(self._handler, str(self.root_dir), recursive=True)
            self._observer.daemon = True  # Don't block process exit
            self._observer.start()
            self._started = True
            logger.info("Started watching %s for changes", self.root_dir)

    def stop(self):
        """Stop watching for file changes."""
        with self._lock:
            if not self._started:
                return

            # Cancel pending timers in handler
            if self._handler is not None:
                self._handler.stop()

            if self._observer is not None:
                self._observer.stop()
                self._observer.join(timeout=2.0)
                self._observer = None

            self._handler = None
            self._started = False
            logger.info("Stopped watching %s", self.root_dir)

    @property
    def is_running(self) -> bool:
        """Check if watcher is running."""
        with self._lock:
            return self._started and self._observer is not None and self._observer.is_alive()


# Global watcher registry
_watchers: dict[str, IndexWatcher] = {}
_watchers_lock = threading.Lock()


def get_watcher(root_dir: str, indexer: Indexer) -> IndexWatcher:
    """Get or create a watcher for the given directory."""
    root_dir = str(Path(root_dir).resolve())

    with _watchers_lock:
        if root_dir not in _watchers:
            _watchers[root_dir] = IndexWatcher(root_dir, indexer)
        return _watchers[root_dir]


def start_watching(root_dir: str, indexer: Indexer) -> IndexWatcher:
    """Start watching a directory for changes."""
    watcher = get_watcher(root_dir, indexer)
    watcher.start()
    return watcher


def stop_watching(root_dir: str):
    """Stop watching a directory."""
    root_dir = str(Path(root_dir).resolve())

    with _watchers_lock:
        if root_dir in _watchers:
            _watchers[root_dir].stop()
            del _watchers[root_dir]


def stop_all_watchers():
    """Stop all active watchers."""
    with _watchers_lock:
        for watcher in _watchers.values():
            watcher.stop()
        _watchers.clear()
