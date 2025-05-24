"""Performance monitoring and benchmarking utilities."""

import time
import asyncio
import statistics
from typing import Dict, List, Any, Callable, Optional
from functools import wraps
from dataclasses import dataclass
import psutil
import threading
from contextlib import contextmanager

@dataclass
class PerformanceMetrics:
    """Container for performance metrics."""
    execution_time: float
    memory_usage: float
    cpu_usage: float
    success: bool
    error_message: Optional[str] = None
    additional_metrics: Dict[str, Any] = None

class PerformanceMonitor:
    """Monitor performance metrics for classification operations."""
    
    def __init__(self):
        self.metrics_history: List[PerformanceMetrics] = []
        self._monitoring = False
        self._monitor_thread = None
        
    def start_monitoring(self, interval: float = 1.0):
        """Start continuous performance monitoring."""
        if self._monitoring:
            return
            
        self._monitoring = True
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop,
            args=(interval,),
            daemon=True
        )
        self._monitor_thread.start()
    
    def stop_monitoring(self):
        """Stop continuous monitoring."""
        self._monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=2.0)
    
    def _monitor_loop(self, interval: float):
        """Continuous monitoring loop."""
        while self._monitoring:
            try:
                # Collect system metrics
                cpu_percent = psutil.cpu_percent(interval=0.1)
                memory_info = psutil.virtual_memory()
                
                # Store metrics (simplified for continuous monitoring)
                metrics = PerformanceMetrics(
                    execution_time=0,  # Not applicable for continuous monitoring
                    memory_usage=memory_info.percent,
                    cpu_usage=cpu_percent,
                    success=True
                )
                
                self.metrics_history.append(metrics)
                
                # Keep only recent metrics (last 1000 entries)
                if len(self.metrics_history) > 1000:
                    self.metrics_history = self.metrics_history[-1000:]
                
                time.sleep(interval)
                
            except Exception:
                # Silently continue monitoring
                time.sleep(interval)
    
    @contextmanager
    def measure_performance(self):
        """Context manager to measure performance of a code block."""
        start_time = time.time()
        start_memory = psutil.virtual_memory().percent
        start_cpu = psutil.cpu_percent()
        
        success = True
        error_message = None
        
        try:
            yield
        except Exception as e:
            success = False
            error_message = str(e)
            raise
        finally:
            end_time = time.time()
            end_memory = psutil.virtual_memory().percent
            end_cpu = psutil.cpu_percent()
            
            metrics = PerformanceMetrics(
                execution_time=end_time - start_time,
                memory_usage=max(end_memory, start_memory),
                cpu_usage=max(end_cpu, start_cpu),
                success=success,
                error_message=error_message
            )
            
            self.metrics_history.append(metrics)
    
    def get_summary_stats(self, last_n: Optional[int] = None) -> Dict[str, Any]:
        """Get summary statistics from collected metrics."""
        if not self.metrics_history:
            return {}
        
        metrics = self.metrics_history[-last_n:] if last_n else self.metrics_history
        successful_metrics = [m for m in metrics if m.success]
        
        if not successful_metrics:
            return {"error": "No successful operations recorded"}
        
        execution_times = [m.execution_time for m in successful_metrics if m.execution_time > 0]
        memory_usages = [m.memory_usage for m in successful_metrics]
        cpu_usages = [m.cpu_usage for m in successful_metrics]
        
        stats = {
            "total_operations": len(metrics),
            "successful_operations": len(successful_metrics),
            "success_rate": len(successful_metrics) / len(metrics),
            "error_rate": 1 - (len(successful_metrics) / len(metrics))
        }
        
        if execution_times:
            stats.update({
                "avg_execution_time": statistics.mean(execution_times),
                "median_execution_time": statistics.median(execution_times),
                "min_execution_time": min(execution_times),
                "max_execution_time": max(execution_times),
                "std_execution_time": statistics.stdev(execution_times) if len(execution_times) > 1 else 0
            })
        
        if memory_usages:
            stats.update({
                "avg_memory_usage": statistics.mean(memory_usages),
                "max_memory_usage": max(memory_usages),
                "min_memory_usage": min(memory_usages)
            })
        
        if cpu_usages:
            stats.update({
                "avg_cpu_usage": statistics.mean(cpu_usages),
                "max_cpu_usage": max(cpu_usages),
                "min_cpu_usage": min(cpu_usages)
            })
        
        return stats
    
    def clear_history(self):
        """Clear metrics history."""
        self.metrics_history.clear()

def profile_function(func: Callable) -> Callable:
    """Decorator to profile function performance."""
    monitor = PerformanceMonitor()
    
    @wraps(func)
    def sync_wrapper(*args, **kwargs):
        with monitor.measure_performance():
            result = func(*args, **kwargs)
        return result
    
    @wraps(func)
    async def async_wrapper(*args, **kwargs):
        with monitor.measure_performance():
            result = await func(*args, **kwargs)
        return result
    
    if asyncio.iscoroutinefunction(func):
        wrapper = async_wrapper
    else:
        wrapper = sync_wrapper
    
    # Attach monitor to wrapper for access to metrics
    wrapper._performance_monitor = monitor
    return wrapper

async def benchmark_classifier(classifier_service, test_texts: List[str], 
                             num_iterations: int = 1,
                             concurrent: bool = False) -> Dict[str, Any]:
    """Benchmark classifier performance with test texts."""
    monitor = PerformanceMonitor()
    results = []
    
    async def classify_single(text: str) -> Dict[str, Any]:
        with monitor.measure_performance():
            try:
                result = await classifier_service.classify(text)
                return {"success": True, "result": result}
            except Exception as e:
                return {"success": False, "error": str(e)}
    
    start_time = time.time()
    
    for iteration in range(num_iterations):
        if concurrent:
            # Run all texts concurrently
            tasks = [classify_single(text) for text in test_texts]
            iteration_results = await asyncio.gather(*tasks, return_exceptions=True)
        else:
            # Run texts sequentially
            iteration_results = []
            for text in test_texts:
                result = await classify_single(text)
                iteration_results.append(result)
        
        results.extend(iteration_results)
    
    total_time = time.time() - start_time
    
    # Analyze results
    successful_results = [r for r in results if isinstance(r, dict) and r.get("success")]
    failed_results = [r for r in results if not (isinstance(r, dict) and r.get("success"))]
    
    # Get performance statistics
    perf_stats = monitor.get_summary_stats()
    
    benchmark_results = {
        "total_time": total_time,
        "total_operations": len(results),
        "successful_operations": len(successful_results),
        "failed_operations": len(failed_results),
        "success_rate": len(successful_results) / len(results) if results else 0,
        "operations_per_second": len(results) / total_time if total_time > 0 else 0,
        "concurrent_mode": concurrent,
        "iterations": num_iterations,
        "texts_per_iteration": len(test_texts),
        "performance_stats": perf_stats
    }
    
    # Add confidence statistics if available
    confidences = []
    for result in successful_results:
        if "result" in result and "confidence" in result["result"]:
            confidences.append(result["result"]["confidence"])
    
    if confidences:
        benchmark_results["confidence_stats"] = {
            "avg_confidence": statistics.mean(confidences),
            "median_confidence": statistics.median(confidences),
            "min_confidence": min(confidences),
            "max_confidence": max(confidences),
            "std_confidence": statistics.stdev(confidences) if len(confidences) > 1 else 0
        }
    
    return benchmark_results

class LoadTester:
    """Load testing utility for classifier service."""
    
    def __init__(self, classifier_service):
        self.classifier_service = classifier_service
        self.monitor = PerformanceMonitor()
    
    async def run_load_test(self, 
                           test_texts: List[str],
                           concurrent_users: int = 10,
                           requests_per_user: int = 10,
                           ramp_up_time: float = 0) -> Dict[str, Any]:
        """Run a load test with multiple concurrent users."""
        
        async def user_session(user_id: int, delay: float = 0):
            """Simulate a user session."""
            if delay > 0:
                await asyncio.sleep(delay)
            
            user_results = []
            for i in range(requests_per_user):
                text = test_texts[i % len(test_texts)]
                
                with self.monitor.measure_performance():
                    try:
                        result = await self.classifier_service.classify(text)
                        user_results.append({
                            "user_id": user_id,
                            "request_id": i,
                            "success": True,
                            "response_time": result.get("processing_time", 0)
                        })
                    except Exception as e:
                        user_results.append({
                            "user_id": user_id,
                            "request_id": i,
                            "success": False,
                            "error": str(e)
                        })
            
            return user_results
        
        # Calculate ramp-up delays
        ramp_up_delay = ramp_up_time / concurrent_users if concurrent_users > 1 else 0
        
        # Start load test
        start_time = time.time()
        
        tasks = [
            user_session(user_id, user_id * ramp_up_delay)
            for user_id in range(concurrent_users)
        ]
        
        all_results = await asyncio.gather(*tasks)
        
        total_time = time.time() - start_time
        
        # Flatten results
        flat_results = [result for user_results in all_results for result in user_results]
        
        # Analyze results
        successful_requests = [r for r in flat_results if r["success"]]
        failed_requests = [r for r in flat_results if not r["success"]]
        
        response_times = [r["response_time"] for r in successful_requests if "response_time" in r]
        
        load_test_results = {
            "total_time": total_time,
            "concurrent_users": concurrent_users,
            "requests_per_user": requests_per_user,
            "total_requests": len(flat_results),
            "successful_requests": len(successful_requests),
            "failed_requests": len(failed_requests),
            "success_rate": len(successful_requests) / len(flat_results) if flat_results else 0,
            "requests_per_second": len(flat_results) / total_time if total_time > 0 else 0,
            "ramp_up_time": ramp_up_time
        }
        
        if response_times:
            load_test_results["response_time_stats"] = {
                "avg_response_time": statistics.mean(response_times),
                "median_response_time": statistics.median(response_times),
                "min_response_time": min(response_times),
                "max_response_time": max(response_times),
                "p95_response_time": sorted(response_times)[int(0.95 * len(response_times))],
                "p99_response_time": sorted(response_times)[int(0.99 * len(response_times))]
            }
        
        # Add system performance stats
        load_test_results["system_performance"] = self.monitor.get_summary_stats()
        
        return load_test_results 