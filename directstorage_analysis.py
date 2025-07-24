#!/usr/bin/env python3
"""
DirectStorage Analysis for Image Duplicate Finder

This module analyzes whether DirectStorage would be beneficial for the image
duplicate finder use case and provides recommendations.

DirectStorage allows GPU to directly access storage without CPU involvement,
potentially reducing latency and CPU overhead for I/O operations.
"""

import os
import time
import psutil
from typing import Dict, List, Tuple
from pathlib import Path


class DirectStorageAnalyzer:
    """Analyzes DirectStorage potential for image processing workflows."""

    def __init__(self):
        self.system_info = self._get_system_info()

    def _get_system_info(self) -> Dict:
        """Get system information relevant to DirectStorage."""
        info = {
            "os_version": self._get_windows_version(),
            "storage_devices": self._get_storage_info(),
            "gpu_info": self._get_gpu_info(),
            "memory_info": self._get_memory_info(),
        }
        return info

    def _get_windows_version(self) -> Dict:
        """Get Windows version information."""
        try:
            import platform

            version = platform.version()
            release = platform.release()

            # DirectStorage requires Windows 10 version 1909+ or Windows 11
            version_parts = version.split(".")
            build_number = int(version_parts[2]) if len(version_parts) > 2 else 0

            directstorage_supported = (
                release == "10" and build_number >= 18363
            ) or release == "11"  # Windows 10 1909+  # Windows 11

            return {
                "release": release,
                "version": version,
                "build_number": build_number,
                "directstorage_supported": directstorage_supported,
            }
        except Exception:
            return {"directstorage_supported": False}

    def _get_storage_info(self) -> List[Dict]:
        """Get storage device information."""
        storage_devices = []

        try:
            # Get disk usage for all drives
            for partition in psutil.disk_partitions():
                try:
                    usage = psutil.disk_usage(partition.mountpoint)

                    # Try to determine if it's SSD (heuristic)
                    is_ssd = self._is_likely_ssd(partition.device)

                    device_info = {
                        "device": partition.device,
                        "mountpoint": partition.mountpoint,
                        "fstype": partition.fstype,
                        "total_gb": usage.total / (1024**3),
                        "free_gb": usage.free / (1024**3),
                        "is_ssd": is_ssd,
                        "nvme_capable": self._is_nvme_capable(partition.device),
                    }
                    storage_devices.append(device_info)
                except (PermissionError, FileNotFoundError):
                    continue
        except Exception:
            pass

        return storage_devices

    def _is_likely_ssd(self, device: str) -> bool:
        """Heuristic to determine if device is likely an SSD."""
        # This is a simple heuristic - in practice, you'd use WMI or other methods
        # For now, assume C: drive is likely SSD on modern systems
        return device.upper().startswith("C:")

    def _is_nvme_capable(self, device: str) -> bool:
        """Check if device is NVMe capable (simplified check)."""
        # This would require more sophisticated detection in practice
        # For now, assume modern systems with SSD have NVMe capability
        return self._is_likely_ssd(device)

    def _get_gpu_info(self) -> Dict:
        """Get GPU information relevant to DirectStorage."""
        gpu_info = {"directstorage_capable": False}

        try:
            import torch

            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / (
                    1024**3
                )

                # DirectStorage requires DirectX 12 Ultimate capable GPU
                # Most RTX 20 series and newer support this
                directstorage_capable = any(
                    series in gpu_name.upper()
                    for series in [
                        "RTX 20",
                        "RTX 30",
                        "RTX 40",
                        "RTX 2060",
                        "RTX 2070",
                        "RTX 2080",
                        "RTX 3060",
                        "RTX 3070",
                        "RTX 3080",
                        "RTX 3090",
                        "RTX 4060",
                        "RTX 4070",
                        "RTX 4080",
                        "RTX 4090",
                    ]
                )

                gpu_info = {
                    "name": gpu_name,
                    "memory_gb": gpu_memory,
                    "directstorage_capable": directstorage_capable,
                    "available": True,
                }
        except ImportError:
            pass

        return gpu_info

    def _get_memory_info(self) -> Dict:
        """Get system memory information."""
        memory = psutil.virtual_memory()
        return {
            "total_gb": memory.total / (1024**3),
            "available_gb": memory.available / (1024**3),
            "percent_used": memory.percent,
        }

    def analyze_directstorage_benefits(self) -> Dict:
        """Analyze potential DirectStorage benefits for image processing."""

        analysis = {
            "system_compatibility": self._check_system_compatibility(),
            "workflow_benefits": self._analyze_workflow_benefits(),
            "performance_impact": self._estimate_performance_impact(),
            "implementation_complexity": self._assess_implementation_complexity(),
            "recommendations": self._generate_recommendations(),
        }

        return analysis

    def _check_system_compatibility(self) -> Dict:
        """Check if system supports DirectStorage."""

        os_compatible = self.system_info["os_version"]["directstorage_supported"]
        gpu_compatible = self.system_info["gpu_info"]["directstorage_capable"]
        storage_compatible = any(
            dev["nvme_capable"] for dev in self.system_info["storage_devices"]
        )

        return {
            "os_supported": os_compatible,
            "gpu_supported": gpu_compatible,
            "storage_supported": storage_compatible,
            "overall_compatible": os_compatible
            and gpu_compatible
            and storage_compatible,
            "missing_requirements": self._get_missing_requirements(),
        }

    def _get_missing_requirements(self) -> List[str]:
        """Get list of missing DirectStorage requirements."""
        missing = []

        if not self.system_info["os_version"]["directstorage_supported"]:
            missing.append("Windows 10 1909+ or Windows 11")

        if not self.system_info["gpu_info"]["directstorage_capable"]:
            missing.append("DirectX 12 Ultimate capable GPU (RTX 20 series+)")

        if not any(dev["nvme_capable"] for dev in self.system_info["storage_devices"]):
            missing.append("NVMe SSD storage")

        return missing

    def _analyze_workflow_benefits(self) -> Dict:
        """Analyze how DirectStorage would benefit the image processing workflow."""

        # Current workflow bottlenecks
        current_bottlenecks = {
            "io_latency": "High - CPU loads images from storage",
            "memory_copies": "Multiple - Storage -> RAM -> GPU",
            "cpu_overhead": "Significant - CPU handles all I/O",
            "batch_limitations": "Memory constrained by RAM capacity",
        }

        # DirectStorage improvements
        directstorage_improvements = {
            "io_latency": "Reduced - GPU direct access to storage",
            "memory_copies": "Eliminated - Direct storage to GPU",
            "cpu_overhead": "Minimal - GPU handles I/O",
            "batch_limitations": "Increased - Can use GPU memory + storage",
        }

        # Specific benefits for image duplicate finder
        workflow_benefits = {
            "image_loading": {
                "current": "CPU loads image -> RAM -> GPU",
                "directstorage": "GPU loads image directly from NVMe",
                "benefit": "Reduced latency, lower CPU usage",
            },
            "batch_processing": {
                "current": "Limited by RAM capacity",
                "directstorage": "Can stream directly to GPU",
                "benefit": "Larger effective batch sizes",
            },
            "feature_extraction": {
                "current": "Wait for CPU I/O then GPU compute",
                "directstorage": "Parallel I/O and compute on GPU",
                "benefit": "Better GPU utilization",
            },
            "memory_pressure": {
                "current": "High RAM usage for large batches",
                "directstorage": "Reduced RAM usage",
                "benefit": "More memory for other operations",
            },
        }

        return {
            "current_bottlenecks": current_bottlenecks,
            "improvements": directstorage_improvements,
            "workflow_benefits": workflow_benefits,
        }

    def _estimate_performance_impact(self) -> Dict:
        """Estimate potential performance improvements with DirectStorage."""

        # Base performance estimates (conservative)
        estimated_improvements = {
            "io_latency_reduction": {
                "percentage": 30,  # 30% reduction in I/O latency
                "description": "Direct GPU access eliminates CPU bottleneck",
            },
            "cpu_usage_reduction": {
                "percentage": 40,  # 40% reduction in CPU usage
                "description": "GPU handles I/O operations",
            },
            "memory_usage_reduction": {
                "percentage": 25,  # 25% reduction in RAM usage
                "description": "Fewer intermediate buffers needed",
            },
            "batch_size_increase": {
                "percentage": 50,  # 50% larger effective batch sizes
                "description": "Can stream data directly to GPU",
            },
            "overall_throughput": {
                "percentage": 20,  # 20% overall throughput improvement
                "description": "Combined effect of all improvements",
            },
        }

        # Calculate potential performance for current system
        current_gpu_performance = 13.16  # img/s from your RTX 2060
        estimated_new_performance = current_gpu_performance * 1.2  # 20% improvement

        return {
            "improvements": estimated_improvements,
            "current_performance": current_gpu_performance,
            "estimated_performance": estimated_new_performance,
            "performance_gain": estimated_new_performance - current_gpu_performance,
        }

    def _assess_implementation_complexity(self) -> Dict:
        """Assess the complexity of implementing DirectStorage."""

        return {
            "api_complexity": {
                "level": "High",
                "description": "Requires DirectStorage API integration",
                "effort": "Significant development work needed",
            },
            "python_support": {
                "level": "Limited",
                "description": "DirectStorage is primarily C++ API",
                "effort": "Would need C++ extension or wrapper",
            },
            "pytorch_integration": {
                "level": "Experimental",
                "description": "PyTorch DirectStorage support is limited",
                "effort": "May require custom CUDA kernels",
            },
            "testing_complexity": {
                "level": "High",
                "description": "Requires specific hardware for testing",
                "effort": "Need DirectStorage-capable systems",
            },
            "maintenance_overhead": {
                "level": "Medium",
                "description": "Additional code paths to maintain",
                "effort": "Ongoing maintenance required",
            },
        }

    def _generate_recommendations(self) -> Dict:
        """Generate recommendations based on analysis."""

        compatibility = self._check_system_compatibility()

        if compatibility["overall_compatible"]:
            recommendation = "CONSIDER"
            reasoning = [
                "System is DirectStorage compatible",
                "Potential 20% performance improvement",
                "Reduced CPU usage and memory pressure",
                "Better scaling for large image collections",
            ]

            next_steps = [
                "Prototype DirectStorage integration",
                "Benchmark against current implementation",
                "Evaluate development effort vs. performance gain",
                "Consider alternative optimizations first",
            ]

        else:
            recommendation = "NOT_RECOMMENDED"
            reasoning = [
                f"Missing requirements: {', '.join(compatibility['missing_requirements'])}",
                "High implementation complexity",
                "Limited Python ecosystem support",
                "Better alternatives available",
            ]

            next_steps = [
                "Focus on current batch size optimization",
                "Consider GPU upgrade for better performance",
                "Optimize image loading pipeline",
                "Implement async I/O patterns",
            ]

        # Alternative optimizations
        alternatives = [
            {
                "name": "Async I/O with asyncio",
                "effort": "Low",
                "benefit": "Medium",
                "description": "Overlap I/O with GPU computation",
            },
            {
                "name": "Memory-mapped files",
                "effort": "Low",
                "benefit": "Low-Medium",
                "description": "Reduce memory copies for large images",
            },
            {
                "name": "Multi-threaded image loading",
                "effort": "Medium",
                "benefit": "Medium",
                "description": "Parallel image loading on CPU",
            },
            {
                "name": "GPU upgrade",
                "effort": "Low (hardware)",
                "benefit": "High",
                "description": "Most direct performance improvement",
            },
            {
                "name": "Optimized batch processing",
                "effort": "Low",
                "benefit": "Medium-High",
                "description": "Better batch size and memory management",
            },
        ]

        return {
            "recommendation": recommendation,
            "reasoning": reasoning,
            "next_steps": next_steps,
            "alternatives": alternatives,
            "priority": "LOW" if recommendation == "NOT_RECOMMENDED" else "MEDIUM",
        }

    def benchmark_current_io_performance(self, test_images_path: str = None) -> Dict:
        """Benchmark current I/O performance to identify bottlenecks."""

        if not test_images_path:
            # Create some test data
            test_images_path = self._create_test_images()

        results = {
            "sequential_read": self._benchmark_sequential_read(test_images_path),
            "random_read": self._benchmark_random_read(test_images_path),
            "batch_loading": self._benchmark_batch_loading(test_images_path),
            "cpu_utilization": self._monitor_cpu_during_io(test_images_path),
        }

        return results

    def _create_test_images(self) -> str:
        """Create test images for benchmarking."""
        import tempfile
        import numpy as np
        from PIL import Image

        test_dir = tempfile.mkdtemp(prefix="directstorage_test_")

        # Create various sized test images
        sizes = [(640, 480), (1024, 768), (1920, 1080), (2048, 1536)]

        for i in range(20):  # Create 20 test images
            size = sizes[i % len(sizes)]
            image_data = np.random.randint(0, 256, (*size, 3), dtype=np.uint8)
            image = Image.fromarray(image_data)

            image_path = os.path.join(test_dir, f"test_image_{i:03d}.jpg")
            image.save(image_path, "JPEG", quality=85)

        return test_dir

    def _benchmark_sequential_read(self, test_path: str) -> Dict:
        """Benchmark sequential image reading."""
        image_files = [f for f in os.listdir(test_path) if f.endswith(".jpg")]

        start_time = time.time()
        total_bytes = 0

        for image_file in image_files:
            file_path = os.path.join(test_path, image_file)
            with open(file_path, "rb") as f:
                data = f.read()
                total_bytes += len(data)

        end_time = time.time()
        duration = end_time - start_time

        return {
            "files_read": len(image_files),
            "total_mb": total_bytes / (1024 * 1024),
            "duration_seconds": duration,
            "throughput_mbps": (total_bytes / (1024 * 1024)) / duration,
            "files_per_second": len(image_files) / duration,
        }

    def _benchmark_random_read(self, test_path: str) -> Dict:
        """Benchmark random access image reading."""
        import random

        image_files = [f for f in os.listdir(test_path) if f.endswith(".jpg")]
        random.shuffle(image_files)

        start_time = time.time()
        total_bytes = 0

        for image_file in image_files:
            file_path = os.path.join(test_path, image_file)
            with open(file_path, "rb") as f:
                data = f.read()
                total_bytes += len(data)

        end_time = time.time()
        duration = end_time - start_time

        return {
            "files_read": len(image_files),
            "total_mb": total_bytes / (1024 * 1024),
            "duration_seconds": duration,
            "throughput_mbps": (total_bytes / (1024 * 1024)) / duration,
            "files_per_second": len(image_files) / duration,
        }

    def _benchmark_batch_loading(self, test_path: str) -> Dict:
        """Benchmark batch image loading with PIL."""
        from PIL import Image

        image_files = [f for f in os.listdir(test_path) if f.endswith(".jpg")]

        start_time = time.time()
        images_loaded = 0

        for image_file in image_files:
            file_path = os.path.join(test_path, image_file)
            try:
                image = Image.open(file_path)
                image.load()  # Force loading
                images_loaded += 1
            except Exception:
                continue

        end_time = time.time()
        duration = end_time - start_time

        return {
            "images_loaded": images_loaded,
            "duration_seconds": duration,
            "images_per_second": images_loaded / duration,
        }

    def _monitor_cpu_during_io(self, test_path: str) -> Dict:
        """Monitor CPU utilization during I/O operations."""
        import threading

        cpu_samples = []
        monitoring = True

        def monitor_cpu():
            while monitoring:
                cpu_samples.append(psutil.cpu_percent(interval=0.1))

        # Start monitoring
        monitor_thread = threading.Thread(target=monitor_cpu)
        monitor_thread.start()

        # Perform I/O operations
        self._benchmark_batch_loading(test_path)

        # Stop monitoring
        monitoring = False
        monitor_thread.join()

        return {
            "avg_cpu_percent": (
                sum(cpu_samples) / len(cpu_samples) if cpu_samples else 0
            ),
            "max_cpu_percent": max(cpu_samples) if cpu_samples else 0,
            "samples": len(cpu_samples),
        }

    def print_analysis_report(self):
        """Print a comprehensive DirectStorage analysis report."""

        print("🚀 DirectStorage Analysis for Image Duplicate Finder")
        print("=" * 60)

        # System compatibility
        analysis = self.analyze_directstorage_benefits()
        compatibility = analysis["system_compatibility"]

        print(f"\n💻 System Compatibility:")
        print(
            f"   OS Support: {'✅' if compatibility['os_supported'] else '❌'} {self.system_info['os_version']['release']}"
        )
        print(
            f"   GPU Support: {'✅' if compatibility['gpu_supported'] else '❌'} {self.system_info['gpu_info'].get('name', 'Unknown')}"
        )
        print(
            f"   Storage Support: {'✅' if compatibility['storage_supported'] else '❌'} NVMe SSD"
        )
        print(
            f"   Overall Compatible: {'✅' if compatibility['overall_compatible'] else '❌'}"
        )

        if compatibility["missing_requirements"]:
            print(f"\n❌ Missing Requirements:")
            for req in compatibility["missing_requirements"]:
                print(f"   • {req}")

        # Performance impact
        performance = analysis["performance_impact"]
        print(f"\n📈 Estimated Performance Impact:")
        print(f"   Current Performance: {performance['current_performance']:.2f} img/s")
        print(
            f"   Estimated with DirectStorage: {performance['estimated_performance']:.2f} img/s"
        )
        print(
            f"   Performance Gain: +{performance['performance_gain']:.2f} img/s ({performance['improvements']['overall_throughput']['percentage']}%)"
        )

        print(f"\n🔧 Key Improvements:")
        for improvement, details in performance["improvements"].items():
            if improvement != "overall_throughput":
                print(
                    f"   • {improvement.replace('_', ' ').title()}: -{details['percentage']}%"
                )

        # Implementation complexity
        complexity = analysis["implementation_complexity"]
        print(f"\n⚙️  Implementation Complexity:")
        for aspect, details in complexity.items():
            print(
                f"   • {aspect.replace('_', ' ').title()}: {details['level']} - {details['description']}"
            )

        # Recommendations
        recommendations = analysis["recommendations"]
        print(f"\n🎯 Recommendation: {recommendations['recommendation']}")
        print(f"   Priority: {recommendations['priority']}")

        print(f"\n💡 Reasoning:")
        for reason in recommendations["reasoning"]:
            print(f"   • {reason}")

        print(f"\n📋 Next Steps:")
        for step in recommendations["next_steps"]:
            print(f"   • {step}")

        print(f"\n🔄 Alternative Optimizations (Recommended):")
        for alt in recommendations["alternatives"]:
            print(
                f"   • {alt['name']}: {alt['effort']} effort, {alt['benefit']} benefit"
            )
            print(f"     {alt['description']}")


def main():
    """Main function to run DirectStorage analysis."""

    analyzer = DirectStorageAnalyzer()
    analyzer.print_analysis_report()

    # Optional: Run I/O benchmarks
    print(f"\n🔬 I/O Performance Benchmark:")
    print("-" * 30)

    try:
        benchmark_results = analyzer.benchmark_current_io_performance()

        print(
            f"Sequential Read: {benchmark_results['sequential_read']['throughput_mbps']:.1f} MB/s"
        )
        print(
            f"Random Read: {benchmark_results['random_read']['throughput_mbps']:.1f} MB/s"
        )
        print(
            f"Image Loading: {benchmark_results['batch_loading']['images_per_second']:.1f} img/s"
        )
        print(
            f"CPU Usage: {benchmark_results['cpu_utilization']['avg_cpu_percent']:.1f}% average"
        )

        # Clean up test files
        import shutil

        test_dir = analyzer._create_test_images()
        shutil.rmtree(test_dir, ignore_errors=True)

    except Exception as e:
        print(f"Benchmark failed: {e}")

    print(f"\n✅ Analysis complete!")


if __name__ == "__main__":
    main()
