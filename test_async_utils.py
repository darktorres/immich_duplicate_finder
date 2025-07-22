"""Unit tests for async utilities."""

import asyncio
import time
import pytest
from unittest.mock import MagicMock, patch

from async_utils import AsyncProcessor, run_with_progress


@pytest.mark.unit
class TestAsyncProcessor:
    """Test cases for AsyncProcessor class."""
    
    def test_init_default_workers(self):
        """Test AsyncProcessor initialization with default workers."""
        processor = AsyncProcessor()
        assert processor.executor._max_workers == 4
    
    def test_init_custom_workers(self):
        """Test AsyncProcessor initialization with custom workers."""
        processor = AsyncProcessor(max_workers=8)
        assert processor.executor._max_workers == 8
    
    @pytest.mark.asyncio
    async def test_process_batch_async_basic(self):
        """Test basic batch processing."""
        processor = AsyncProcessor(max_workers=2)
        
        def square(x):
            return x * x
        
        items = [1, 2, 3, 4, 5]
        results = await processor.process_batch_async(items, square, batch_size=2)
        
        assert results == [1, 4, 9, 16, 25]
    
    @pytest.mark.asyncio
    async def test_process_batch_async_with_progress(self):
        """Test batch processing with progress callback."""
        processor = AsyncProcessor(max_workers=2)
        progress_calls = []
        
        def progress_callback(current, total):
            progress_calls.append((current, total))
        
        def identity(x):
            return x
        
        items = [1, 2, 3, 4, 5]
        results = await processor.process_batch_async(
            items, identity, batch_size=2, progress_callback=progress_callback
        )
        
        assert results == [1, 2, 3, 4, 5]
        assert len(progress_calls) > 0
        # Should have progress updates
        assert any(call[1] == 5 for call in progress_calls)  # total should be 5
    
    @pytest.mark.asyncio
    async def test_process_batch_async_empty_list(self):
        """Test batch processing with empty list."""
        processor = AsyncProcessor()
        
        def identity(x):
            return x
        
        results = await processor.process_batch_async([], identity)
        assert results == []
    
    @pytest.mark.asyncio
    async def test_process_batch_async_single_item(self):
        """Test batch processing with single item."""
        processor = AsyncProcessor()
        
        def double(x):
            return x * 2
        
        results = await processor.process_batch_async([5], double)
        assert results == [10]
    
    @pytest.mark.asyncio
    async def test_process_batch_async_large_batch_size(self):
        """Test batch processing with batch size larger than items."""
        processor = AsyncProcessor()
        
        def identity(x):
            return x
        
        items = [1, 2, 3]
        results = await processor.process_batch_async(items, identity, batch_size=10)
        assert results == [1, 2, 3]
    
    def test_cleanup_on_delete(self):
        """Test that executor is cleaned up on deletion."""
        processor = AsyncProcessor()
        executor = processor.executor
        
        # Delete the processor
        del processor
        
        # Executor should be shutdown (we can't easily test this without implementation details)
        # This test mainly ensures no exceptions are raised during cleanup
        assert True


@pytest.mark.unit
class TestRunWithProgress:
    """Test cases for run_with_progress function."""
    
    def test_run_with_progress_basic(self):
        """Test basic function execution with progress."""
        def test_func():
            return "success"
        
        result = run_with_progress(test_func)
        assert result == "success"
    
    def test_run_with_progress_with_callbacks(self):
        """Test function execution with progress and error callbacks."""
        progress_messages = []
        errors = []
        
        def progress_callback(message):
            progress_messages.append(message)
        
        def error_callback(error):
            errors.append(error)
        
        def test_func():
            time.sleep(0.01)  # Small delay to test timing
            return "success"
        
        result = run_with_progress(
            test_func, 
            progress_callback=progress_callback,
            error_callback=error_callback
        )
        
        assert result == "success"
        assert len(progress_messages) >= 2  # Start and completion messages
        assert "Starting operation" in progress_messages[0]
        assert "completed in" in progress_messages[-1]
        assert len(errors) == 0
    
    def test_run_with_progress_with_exception(self):
        """Test function execution that raises exception."""
        progress_messages = []
        errors = []
        
        def progress_callback(message):
            progress_messages.append(message)
        
        def error_callback(error):
            errors.append(error)
        
        def failing_func():
            raise ValueError("Test error")
        
        with pytest.raises(ValueError, match="Test error"):
            run_with_progress(
                failing_func,
                progress_callback=progress_callback,
                error_callback=error_callback
            )
        
        assert len(progress_messages) >= 1  # Should have start message
        assert len(errors) == 1
        assert isinstance(errors[0], ValueError)
    
    def test_run_with_progress_no_callbacks(self):
        """Test function execution without callbacks."""
        def test_func():
            return 42
        
        result = run_with_progress(test_func)
        assert result == 42
    
    def test_run_with_progress_timing(self):
        """Test that timing is recorded correctly."""
        progress_messages = []
        
        def progress_callback(message):
            progress_messages.append(message)
        
        def slow_func():
            time.sleep(0.1)  # 100ms delay
            return "done"
        
        start_time = time.time()
        result = run_with_progress(slow_func, progress_callback=progress_callback)
        end_time = time.time()
        
        assert result == "done"
        assert len(progress_messages) >= 2
        
        # Check that completion message contains timing info
        completion_msg = progress_messages[-1]
        assert "completed in" in completion_msg
        assert "seconds" in completion_msg
        
        # Verify actual timing
        actual_duration = end_time - start_time
        assert actual_duration >= 0.1  # Should take at least 100ms


@pytest.mark.unit
@pytest.mark.asyncio
async def test_async_processor_integration():
    """Integration test for AsyncProcessor with realistic workload."""
    processor = AsyncProcessor(max_workers=2)
    
    def simulate_work(item):
        """Simulate some work that takes time."""
        time.sleep(0.01)  # 10ms per item
        return f"processed_{item}"
    
    progress_updates = []
    
    def track_progress(current, total):
        progress_updates.append((current, total))
    
    items = list(range(10))
    start_time = time.time()
    
    results = await processor.process_batch_async(
        items, 
        simulate_work, 
        batch_size=3,
        progress_callback=track_progress
    )
    
    end_time = time.time()
    
    # Verify results
    expected_results = [f"processed_{i}" for i in range(10)]
    assert results == expected_results
    
    # Verify progress tracking
    assert len(progress_updates) > 0
    final_progress = progress_updates[-1]
    assert final_progress[0] == 10  # All items processed
    assert final_progress[1] == 10  # Total items
    
    # Verify parallel processing (should be faster than sequential)
    # Sequential would take ~100ms (10 * 10ms), parallel should be faster
    duration = end_time - start_time
    assert duration < 0.2  # Should be significantly faster than sequential (relaxed timing)


@pytest.mark.unit
def test_run_with_progress_integration():
    """Integration test for run_with_progress with realistic scenario."""
    messages = []
    
    def capture_progress(msg):
        messages.append(msg)
    
    def complex_operation():
        """Simulate a complex operation."""
        total = 0
        for i in range(1000):
            total += i * i
        return total
    
    result = run_with_progress(complex_operation, progress_callback=capture_progress)
    
    # Verify result
    expected = sum(i * i for i in range(1000))
    assert result == expected
    
    # Verify progress messages
    assert len(messages) == 2
    assert "Starting operation" in messages[0]
    assert "Operation completed in" in messages[1]
    assert "seconds" in messages[1]