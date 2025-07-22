"""Async utilities for better UI responsiveness during long operations."""

import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, List, Optional

from logger_config import logger


class AsyncProcessor:
    """Handles async processing of long-running operations."""

    def __init__(self, max_workers: int = 4):
        """
        Initialize the async processor.

        Args:
            max_workers: Maximum number of worker threads
        """
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.logger = logger

    async def process_batch_async(
        self,
        items: List[Any],
        process_func: Callable[[Any], Any],
        batch_size: int = 10,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> List[Any]:
        """
        Process items in batches asynchronously.

        Args:
            items: List of items to process
            process_func: Function to process each item
            batch_size: Number of items to process in each batch
            progress_callback: Optional callback for progress updates

        Returns:
            List of processed results
        """
        results = []
        total_items = len(items)

        for i in range(0, total_items, batch_size):
            batch = items[i : i + batch_size]

            # Process batch in parallel
            loop = asyncio.get_event_loop()
            batch_results = await asyncio.gather(*[loop.run_in_executor(self.executor, process_func, item) for item in batch])

            results.extend(batch_results)

            # Call progress callback if provided
            if progress_callback:
                progress_callback(min(i + batch_size, total_items), total_items)

            # Small delay to allow UI updates
            await asyncio.sleep(0.01)

        return results

    def __del__(self):
        """Clean up the thread pool executor."""
        if hasattr(self, "executor"):
            self.executor.shutdown(wait=False)


def run_with_progress(
    func: Callable[[], Any],
    progress_callback: Optional[Callable[[str], None]] = None,
    error_callback: Optional[Callable[[Exception], None]] = None,
) -> Any:
    """
    Run a function with progress reporting.

    Args:
        func: Function to run
        progress_callback: Optional callback for progress messages
        error_callback: Optional callback for error handling

    Returns:
        Result of the function
    """
    try:
        start_time = time.time()

        if progress_callback:
            progress_callback("Starting operation...")

        result = func()

        end_time = time.time()
        duration = end_time - start_time

        if progress_callback:
            progress_callback(f"Operation completed in {duration:.2f} seconds")

        logger.info(f"Operation completed successfully in {duration:.2f} seconds")
        return result

    except Exception as e:
        logger.error(f"Operation failed: {e}")
        if error_callback:
            error_callback(e)
        raise
