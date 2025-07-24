# Batch Size Optimization Guide

This guide explains how to find and use the optimal batch size for your image duplicate finder application.

## Overview

Batch size optimization is crucial for balancing processing speed and memory usage. The right batch size can significantly improve performance while preventing out-of-memory errors.

## Quick Start

### 1. Find Your Optimal Batch Size

Run the optimization script:

```bash
python find_optimal_batch_size.py
```

This will:
- Analyze your system specifications
- Test different batch sizes
- Recommend the optimal configuration
- Optionally update your settings

### 2. Manual Testing

For more control, run the test files directly:

```bash
# Quick optimization (recommended)
python test_quick_batch_optimization.py

# Comprehensive testing
python -m pytest test_batch_size_optimization.py -v

# FAISS-specific optimization
python test_faiss_batch_optimization.py
```

## Test Files Explained

### 1. `test_quick_batch_optimization.py`
- **Purpose**: Fast batch size testing with immediate results
- **Runtime**: ~10-30 seconds
- **Use case**: Regular optimization checks
- **Features**:
  - System-aware batch size recommendations
  - Memory usage analysis
  - Performance scoring
  - Current configuration benchmarking

### 2. `test_batch_size_optimization.py`
- **Purpose**: Comprehensive batch size analysis
- **Runtime**: 2-10 minutes
- **Use case**: Detailed performance analysis
- **Features**:
  - Real image processing simulation
  - Multiple test iterations
  - Memory regression detection
  - System configuration testing

### 3. `test_faiss_batch_optimization.py`
- **Purpose**: FAISS-specific optimization
- **Runtime**: 1-5 minutes
- **Use case**: Optimizing vector operations
- **Features**:
  - Feature extraction testing
  - Search operation optimization
  - Combined workload analysis
  - Memory scaling analysis

### 4. `find_optimal_batch_size.py`
- **Purpose**: User-friendly optimization tool
- **Runtime**: 30 seconds - 5 minutes
- **Use case**: Easy batch size optimization
- **Features**:
  - Interactive configuration updates
  - System recommendations
  - Performance comparisons

## Understanding Results

### Performance Metrics

- **Images per second**: Processing throughput
- **Memory increase**: Additional RAM usage during processing
- **Memory per image**: RAM efficiency
- **Composite score**: Overall performance rating (0-1 scale)

### Batch Size Categories

| System Memory | Recommended Range | Optimal Typically |
|---------------|-------------------|-------------------|
| < 4 GB        | 1-5              | 2-3               |
| 4-8 GB        | 2-15             | 5-10              |
| 8-16 GB       | 5-30             | 10-20             |
| > 16 GB       | 10-100           | 20-50             |

### Interpreting Scores

- **Speed Score**: Higher = faster processing
- **Memory Score**: Higher = more memory efficient
- **Stability Score**: Higher = more consistent performance
- **Composite Score**: Weighted combination of all factors

## Configuration Options

### Memory Configuration Classes

```python
# Low memory systems (< 8GB RAM)
config = MemoryConfig.get_low_memory_config()
# batch_size = 5, aggressive_gc = True, max_image_size = (512, 512)

# Balanced systems (8-16GB RAM)
config = MemoryConfig.get_balanced_config()
# batch_size = 10, max_image_size = (1024, 1024)

# High performance systems (> 16GB RAM)
config = MemoryConfig.get_high_performance_config()
# batch_size = 20, aggressive_gc = False, no image resizing
```

### Manual Configuration

Edit `memory_config.py` or update programmatically:

```python
from memory_config import MemoryConfig, update_config

# Create custom configuration
custom_config = MemoryConfig(
    batch_size=25,              # Your optimal batch size
    use_cpu_only=False,         # Enable GPU if available
    aggressive_gc=True,         # Memory cleanup frequency
    max_image_size=(1024, 1024) # Image size limit
)

# Apply configuration
update_config(custom_config)
```

## Running Tests

### Unit Tests

```bash
# Run all batch optimization tests
python -m pytest test_*batch*.py -v

# Run only quick tests
python -m pytest test_quick_batch_optimization.py -v

# Run only unit tests (fast)
python -m pytest test_quick_batch_optimization.py -m unit -v
```

### Integration Tests

```bash
# Run comprehensive tests (slower)
python -m pytest test_batch_size_optimization.py -m integration -v

# Run FAISS-specific tests
python -m pytest test_faiss_batch_optimization.py -v
```

### Performance Tests

```bash
# Run slow/comprehensive tests
python -m pytest -m slow -v

# Skip slow tests
python -m pytest -m "not slow" -v
```

## Troubleshooting

### Common Issues

1. **Out of Memory Errors**
   - Reduce batch size
   - Enable aggressive garbage collection
   - Reduce max image size
   - Enable CPU-only mode

2. **Slow Processing**
   - Increase batch size (if memory allows)
   - Disable aggressive garbage collection
   - Enable GPU processing
   - Increase max image size limit

3. **Inconsistent Performance**
   - Run tests multiple times
   - Check system load during testing
   - Verify memory availability
   - Consider thermal throttling

### Memory Optimization Tips

1. **For Low Memory Systems**:
   ```python
   config = MemoryConfig(
       batch_size=2,
       use_cpu_only=True,
       aggressive_gc=True,
       max_image_size=(512, 512)
   )
   ```

2. **For High Memory Systems**:
   ```python
   config = MemoryConfig(
       batch_size=50,
       use_cpu_only=False,
       aggressive_gc=False,
       max_image_size=None  # No limit
   )
   ```

3. **For Balanced Performance**:
   ```python
   config = MemoryConfig(
       batch_size=15,
       aggressive_gc=True,
       clear_cache_after_batch=True,
       max_image_size=(1024, 1024)
   )
   ```

## Advanced Usage

### Custom Test Parameters

```python
from test_quick_batch_optimization import QuickBatchOptimizer

optimizer = QuickBatchOptimizer()

# Test specific batch sizes
results = optimizer.test_batch_sizes([5, 10, 15, 20], num_images=50)

# Find optimal
optimal = optimizer.find_optimal_batch_size(results)
print(f"Optimal batch size: {optimal['optimal_batch_size']}")
```

### Automated Optimization

```python
from find_optimal_batch_size import run_quick_optimization
from memory_config import update_config, MemoryConfig

# Run optimization
optimal_result, _ = run_quick_optimization()
optimal_batch_size = optimal_result['optimal_batch_size']

# Auto-apply optimal configuration
new_config = MemoryConfig(batch_size=optimal_batch_size)
update_config(new_config)
```

### Continuous Monitoring

```python
# Add to your application startup
from find_optimal_batch_size import benchmark_current_config

# Benchmark current configuration
result = benchmark_current_config()
if result['images_per_second'] < expected_performance:
    print("Consider running batch size optimization")
```

## Best Practices

1. **Regular Testing**: Run optimization tests when:
   - System configuration changes
   - Application updates
   - Performance issues occur
   - New hardware is installed

2. **Environment-Specific**: Different environments may need different batch sizes:
   - Development vs. production
   - Different hardware configurations
   - Varying workload sizes

3. **Monitoring**: Track performance metrics:
   - Processing speed
   - Memory usage
   - Error rates
   - System resource utilization

4. **Documentation**: Record optimal settings for:
   - Different system configurations
   - Various workload types
   - Performance baselines

## Example Workflows

### Development Workflow

```bash
# 1. Quick check during development
python find_optimal_batch_size.py --quick --no-apply

# 2. Comprehensive testing before deployment
python find_optimal_batch_size.py --comprehensive

# 3. Validate with unit tests
python -m pytest test_quick_batch_optimization.py -v
```

### Production Deployment

```bash
# 1. System analysis
python find_optimal_batch_size.py --quick

# 2. Apply optimal configuration
# (follow prompts to update configuration)

# 3. Validate performance
python -c "from find_optimal_batch_size import benchmark_current_config; benchmark_current_config()"
```

### Performance Tuning

```bash
# 1. Baseline measurement
python test_quick_batch_optimization.py

# 2. Test different configurations
python test_batch_size_optimization.py

# 3. FAISS-specific optimization
python test_faiss_batch_optimization.py

# 4. Apply best configuration
python find_optimal_batch_size.py
```

## Conclusion

Batch size optimization is essential for optimal performance. Use these tools to:

- Find the best batch size for your system
- Monitor performance over time
- Adapt to changing requirements
- Ensure efficient resource utilization

Start with the quick optimization script and use the comprehensive tests for detailed analysis when needed.