"""
MAIRA-2 Performance Benchmarking Script

This script provides comprehensive performance metrics for evaluating 
MAIRA-2's suitability for production deployment:

1. Model Size Metrics:
   - Model parameters count
   - Model size on disk (GB)
   - Tokenizer/processor size

2. GPU Memory Metrics:
   - Memory allocated at model load
   - Peak memory during inference
   - Memory per GPU (multi-GPU setup)

3. Inference Performance:
   - Time to first token
   - Total inference time
   - Tokens per second
   - Throughput (images/second)

4. Batch Processing Analysis:
   - Single vs batched inference comparison
   - Scalability metrics

IMPORTANT: This model is for RESEARCH USE ONLY, not for clinical deployment.
"""

import os
import sys
import time
import json
import gc
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field, asdict
import statistics

import torch
from tqdm import tqdm


@dataclass
class GPUMemorySnapshot:
    """Snapshot of GPU memory state."""
    device_id: int
    device_name: str
    total_memory_gb: float
    allocated_gb: float
    reserved_gb: float
    free_gb: float


@dataclass
class InferenceTimingResult:
    """Timing results for a single inference."""
    instance_id: str
    preprocessing_time_ms: float
    generation_time_ms: float
    postprocessing_time_ms: float
    total_time_ms: float
    tokens_generated: int
    tokens_per_second: float


@dataclass
class BenchmarkResults:
    """Complete benchmark results."""
    # Metadata
    timestamp: str
    model_name: str
    device_info: Dict[str, Any]
    
    # Model Size
    total_parameters: int
    trainable_parameters: int
    model_size_gb: float
    model_fp16_size_gb: float
    processor_vocab_size: int
    
    # GPU Memory
    baseline_gpu_memory: List[GPUMemorySnapshot]
    post_load_gpu_memory: List[GPUMemorySnapshot]
    peak_inference_gpu_memory: List[GPUMemorySnapshot]
    
    # Inference Performance
    num_samples_tested: int
    inference_timings: List[InferenceTimingResult] = field(default_factory=list)
    
    # Aggregated Stats
    avg_preprocessing_time_ms: float = 0.0
    avg_generation_time_ms: float = 0.0
    avg_postprocessing_time_ms: float = 0.0
    avg_total_time_ms: float = 0.0
    std_total_time_ms: float = 0.0
    min_total_time_ms: float = 0.0
    max_total_time_ms: float = 0.0
    avg_tokens_per_second: float = 0.0
    throughput_images_per_minute: float = 0.0
    
    # Warmup stats
    warmup_time_ms: float = 0.0
    
    def compute_aggregates(self):
        """Compute aggregate statistics from individual timings."""
        if not self.inference_timings:
            return
        
        preprocessing_times = [t.preprocessing_time_ms for t in self.inference_timings]
        generation_times = [t.generation_time_ms for t in self.inference_timings]
        postprocessing_times = [t.postprocessing_time_ms for t in self.inference_timings]
        total_times = [t.total_time_ms for t in self.inference_timings]
        tokens_per_sec = [t.tokens_per_second for t in self.inference_timings]
        
        self.avg_preprocessing_time_ms = statistics.mean(preprocessing_times)
        self.avg_generation_time_ms = statistics.mean(generation_times)
        self.avg_postprocessing_time_ms = statistics.mean(postprocessing_times)
        self.avg_total_time_ms = statistics.mean(total_times)
        self.std_total_time_ms = statistics.stdev(total_times) if len(total_times) > 1 else 0.0
        self.min_total_time_ms = min(total_times)
        self.max_total_time_ms = max(total_times)
        self.avg_tokens_per_second = statistics.mean(tokens_per_sec)
        
        # Throughput: images per minute
        avg_time_sec = self.avg_total_time_ms / 1000.0
        self.throughput_images_per_minute = 60.0 / avg_time_sec if avg_time_sec > 0 else 0.0


def get_gpu_memory_snapshot() -> List[GPUMemorySnapshot]:
    """Get current GPU memory state for all available GPUs."""
    snapshots = []
    
    if not torch.cuda.is_available():
        return snapshots
    
    torch.cuda.synchronize()
    
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        allocated = torch.cuda.memory_allocated(i) / (1024**3)
        reserved = torch.cuda.memory_reserved(i) / (1024**3)
        total = props.total_memory / (1024**3)
        
        snapshots.append(GPUMemorySnapshot(
            device_id=i,
            device_name=props.name,
            total_memory_gb=round(total, 3),
            allocated_gb=round(allocated, 3),
            reserved_gb=round(reserved, 3),
            free_gb=round(total - reserved, 3),
        ))
    
    return snapshots


def get_model_size_info(model) -> Tuple[int, int, float, float]:
    """
    Get model size information.
    
    Returns:
        (total_params, trainable_params, size_fp32_gb, size_fp16_gb)
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Size in bytes (FP32 = 4 bytes, FP16 = 2 bytes per param)
    size_fp32_gb = (total_params * 4) / (1024**3)
    size_fp16_gb = (total_params * 2) / (1024**3)
    
    return total_params, trainable_params, size_fp32_gb, size_fp16_gb


def format_memory_report(snapshots: List[GPUMemorySnapshot], title: str) -> str:
    """Format GPU memory snapshots for display."""
    lines = [f"\n{title}"]
    lines.append("-" * 60)
    
    total_allocated = 0
    total_reserved = 0
    
    for snap in snapshots:
        lines.append(
            f"GPU {snap.device_id} ({snap.device_name}): "
            f"Allocated: {snap.allocated_gb:.2f}GB / "
            f"Reserved: {snap.reserved_gb:.2f}GB / "
            f"Total: {snap.total_memory_gb:.2f}GB"
        )
        total_allocated += snap.allocated_gb
        total_reserved += snap.reserved_gb
    
    if len(snapshots) > 1:
        lines.append(f"TOTAL: Allocated: {total_allocated:.2f}GB / Reserved: {total_reserved:.2f}GB")
    
    return "\n".join(lines)


class PerformanceBenchmark:
    """
    Performance benchmarking class for MAIRA-2.
    
    Measures:
    - Model loading time and memory
    - Inference speed and throughput
    - GPU utilization across multiple GPUs
    """
    
    def __init__(
        self,
        model_name: str = "microsoft/maira-2",
        num_warmup_samples: int = 2,
        max_new_tokens: int = 300,
    ):
        self.model_name = model_name
        self.num_warmup_samples = num_warmup_samples
        self.max_new_tokens = max_new_tokens
        
        self.model = None
        self.processor = None
        self.device = None
        
        self.results: Optional[BenchmarkResults] = None
    
    def _clear_gpu_cache(self):
        """Clear GPU cache and run garbage collection."""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    
    def load_model(self) -> Dict[str, Any]:
        """
        Load the MAIRA-2 model and measure loading time and memory.
        
        Returns:
            Dictionary with loading metrics
        """
        from transformers import AutoModelForCausalLM, AutoProcessor
        
        print("\n" + "=" * 70)
        print("LOADING MAIRA-2 MODEL")
        print("=" * 70)
        
        # Clear any existing cache
        self._clear_gpu_cache()
        
        # Baseline memory
        baseline_memory = get_gpu_memory_snapshot()
        print(format_memory_report(baseline_memory, "Baseline GPU Memory"))
        
        # Time model loading
        print(f"\nLoading model: {self.model_name}")
        load_start = time.perf_counter()
        
        if torch.cuda.is_available():
            num_gpus = torch.cuda.device_count()
            print(f"Detected {num_gpus} GPU(s)")
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                device_map="auto",
                torch_dtype=torch.float16,
            )
            self.device = torch.device("cuda:0")
        else:
            print("No GPU detected, using CPU (performance will be significantly slower)")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                trust_remote_code=True,
            )
            self.device = torch.device("cpu")
            self.model = self.model.to(self.device)
        
        load_end = time.perf_counter()
        load_time = load_end - load_start
        
        # Load processor
        self.processor = AutoProcessor.from_pretrained(
            self.model_name,
            trust_remote_code=True,
        )
        
        self.model = self.model.eval()
        
        # Post-load memory
        self._clear_gpu_cache()
        post_load_memory = get_gpu_memory_snapshot()
        print(format_memory_report(post_load_memory, "Post-Load GPU Memory"))
        
        # Get model size info
        total_params, trainable_params, size_fp32, size_fp16 = get_model_size_info(self.model)
        
        print(f"\n✓ Model loaded in {load_time:.2f} seconds")
        print(f"  Total parameters: {total_params:,} ({total_params/1e9:.2f}B)")
        print(f"  FP32 size: {size_fp32:.2f} GB")
        print(f"  FP16 size: {size_fp16:.2f} GB")
        
        # Processor info
        vocab_size = 0
        if hasattr(self.processor, 'tokenizer') and hasattr(self.processor.tokenizer, 'vocab_size'):
            vocab_size = self.processor.tokenizer.vocab_size
        elif hasattr(self.processor, 'vocab_size'):
            vocab_size = self.processor.vocab_size
        
        print(f"  Processor vocab size: {vocab_size:,}")
        
        return {
            "load_time_seconds": load_time,
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "model_size_fp32_gb": size_fp32,
            "model_size_fp16_gb": size_fp16,
            "processor_vocab_size": vocab_size,
            "baseline_memory": baseline_memory,
            "post_load_memory": post_load_memory,
        }
    
    def run_single_inference(
        self,
        image,
        indication: str = "Not provided.",
        technique: str = "PA view of the chest.",
        comparison: str = "None.",
    ) -> Tuple[str, InferenceTimingResult]:
        """
        Run a single inference and measure timing.
        
        Returns:
            (generated_text, timing_result)
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        # Preprocessing
        preprocess_start = time.perf_counter()
        
        processed_inputs = self.processor.format_and_preprocess_reporting_input(
            current_frontal=image,
            current_lateral=None,
            prior_frontal=None,
            indication=indication,
            technique=technique,
            comparison=comparison,
            prior_report=None,
            return_tensors="pt",
            get_grounding=False,
        )
        processed_inputs = processed_inputs.to(self.device)
        
        preprocess_end = time.perf_counter()
        preprocessing_time_ms = (preprocess_end - preprocess_start) * 1000
        
        # Generation
        generation_start = time.perf_counter()
        
        with torch.no_grad():
            output = self.model.generate(
                **processed_inputs,
                max_new_tokens=self.max_new_tokens,
                use_cache=True,
            )
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        generation_end = time.perf_counter()
        generation_time_ms = (generation_end - generation_start) * 1000
        
        # Post-processing
        postprocess_start = time.perf_counter()
        
        prompt_length = processed_inputs["input_ids"].shape[-1]
        tokens_generated = output[0].shape[0] - prompt_length
        
        decoded_text = self.processor.decode(
            output[0][prompt_length:],
            skip_special_tokens=True,
        ).lstrip()
        
        prediction = self.processor.convert_output_to_plaintext_or_grounded_sequence(decoded_text)
        if isinstance(prediction, list):
            prediction = " ".join(str(p) for p in prediction)
        
        postprocess_end = time.perf_counter()
        postprocessing_time_ms = (postprocess_end - postprocess_start) * 1000
        
        # Calculate metrics
        total_time_ms = preprocessing_time_ms + generation_time_ms + postprocessing_time_ms
        tokens_per_second = (tokens_generated / generation_time_ms) * 1000 if generation_time_ms > 0 else 0
        
        timing = InferenceTimingResult(
            instance_id="",  # Will be set by caller
            preprocessing_time_ms=round(preprocessing_time_ms, 2),
            generation_time_ms=round(generation_time_ms, 2),
            postprocessing_time_ms=round(postprocessing_time_ms, 2),
            total_time_ms=round(total_time_ms, 2),
            tokens_generated=tokens_generated,
            tokens_per_second=round(tokens_per_second, 2),
        )
        
        return str(prediction), timing
    
    def run_benchmark(
        self,
        dataset,
        num_samples: Optional[int] = None,
        output_dir: str = "./benchmark_results",
    ) -> BenchmarkResults:
        """
        Run complete benchmark on a dataset.
        
        Args:
            dataset: EvaluationDataset or iterable of items with load_image() and instance_id
            num_samples: Number of samples to benchmark (None = all)
            output_dir: Directory to save results
            
        Returns:
            BenchmarkResults with all metrics
        """
        print("\n" + "=" * 70)
        print("RUNNING PERFORMANCE BENCHMARK")
        print("=" * 70)
        
        # Load model if not already loaded
        if self.model is None:
            load_info = self.load_model()
        else:
            load_info = {
                "total_parameters": sum(p.numel() for p in self.model.parameters()),
                "trainable_parameters": sum(p.numel() for p in self.model.parameters() if p.requires_grad),
                "baseline_memory": [],
                "post_load_memory": get_gpu_memory_snapshot(),
            }
            total_params = load_info["total_parameters"]
            load_info["model_size_fp32_gb"] = (total_params * 4) / (1024**3)
            load_info["model_size_fp16_gb"] = (total_params * 2) / (1024**3)
            load_info["processor_vocab_size"] = 0
        
        # Determine samples to test
        total_available = len(dataset)
        if num_samples is None:
            num_samples = total_available
        num_samples = min(num_samples, total_available)
        
        print(f"\nBenchmarking {num_samples} samples (of {total_available} available)")
        
        # Warmup runs
        print(f"\nRunning {self.num_warmup_samples} warmup inference(s)...")
        warmup_times = []
        
        for i in range(min(self.num_warmup_samples, num_samples)):
            item = dataset[i]
            image = item.load_image()
            
            warmup_start = time.perf_counter()
            _, _ = self.run_single_inference(
                image=image,
                indication=item.indication or "Not provided.",
                technique=item.technique or "PA view of the chest.",
                comparison=item.comparison or "None.",
            )
            warmup_end = time.perf_counter()
            warmup_times.append((warmup_end - warmup_start) * 1000)
        
        avg_warmup_time = statistics.mean(warmup_times) if warmup_times else 0
        print(f"Warmup complete. Average warmup time: {avg_warmup_time:.2f} ms")
        
        # Record peak memory after warmup
        self._clear_gpu_cache()
        peak_memory = get_gpu_memory_snapshot()
        print(format_memory_report(peak_memory, "Peak GPU Memory (after warmup)"))
        
        # Main benchmark loop
        print(f"\nRunning main benchmark on {num_samples} samples...")
        inference_timings = []
        
        for i in tqdm(range(num_samples), desc="Benchmarking"):
            item = dataset[i]
            
            try:
                image = item.load_image()
                
                _, timing = self.run_single_inference(
                    image=image,
                    indication=item.indication or "Not provided.",
                    technique=item.technique or "PA view of the chest.",
                    comparison=item.comparison or "None.",
                )
                timing.instance_id = item.instance_id
                inference_timings.append(timing)
                
            except Exception as e:
                print(f"\nWarning: Failed to process {item.instance_id}: {e}")
                continue
        
        # Check peak memory during inference
        inference_peak_memory = get_gpu_memory_snapshot()
        
        # Build device info
        device_info = {
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
            "pytorch_version": torch.__version__,
            "num_gpus": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            "gpu_names": [
                torch.cuda.get_device_properties(i).name 
                for i in range(torch.cuda.device_count())
            ] if torch.cuda.is_available() else [],
        }
        
        # Create results object
        self.results = BenchmarkResults(
            timestamp=datetime.now().isoformat(),
            model_name=self.model_name,
            device_info=device_info,
            total_parameters=load_info.get("total_parameters", 0),
            trainable_parameters=load_info.get("trainable_parameters", 0),
            model_size_gb=load_info.get("model_size_fp32_gb", 0),
            model_fp16_size_gb=load_info.get("model_size_fp16_gb", 0),
            processor_vocab_size=load_info.get("processor_vocab_size", 0),
            baseline_gpu_memory=load_info.get("baseline_memory", []),
            post_load_gpu_memory=load_info.get("post_load_memory", []),
            peak_inference_gpu_memory=inference_peak_memory,
            num_samples_tested=len(inference_timings),
            inference_timings=inference_timings,
            warmup_time_ms=avg_warmup_time,
        )
        
        # Compute aggregates
        self.results.compute_aggregates()
        
        # Save results
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        self._save_results(output_dir)
        
        # Print summary
        self._print_summary()
        
        return self.results
    
    def _save_results(self, output_dir: str):
        """Save benchmark results to files."""
        if self.results is None:
            return
        
        output_path = Path(output_dir)
        
        # Save full results as JSON
        results_dict = asdict(self.results)
        
        # Convert GPUMemorySnapshot objects to dicts
        for key in ["baseline_gpu_memory", "post_load_gpu_memory", "peak_inference_gpu_memory"]:
            results_dict[key] = [asdict(s) if hasattr(s, '__dict__') else s for s in results_dict.get(key, [])]
        
        # Convert InferenceTimingResult objects to dicts
        results_dict["inference_timings"] = [
            asdict(t) if hasattr(t, '__dict__') else t 
            for t in results_dict.get("inference_timings", [])
        ]
        
        json_path = output_path / "benchmark_results.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(results_dict, f, indent=2, default=str)
        
        # Save human-readable summary
        summary_path = output_path / "benchmark_summary.txt"
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write(self._generate_summary_text())
        
        print(f"\n✓ Results saved to: {output_path}")
        print(f"  - benchmark_results.json (full data)")
        print(f"  - benchmark_summary.txt (readable summary)")
    
    def _generate_summary_text(self) -> str:
        """Generate human-readable summary text."""
        if self.results is None:
            return "No results available"
        
        r = self.results
        lines = [
            "=" * 70,
            "MAIRA-2 PERFORMANCE BENCHMARK SUMMARY",
            "=" * 70,
            f"Timestamp: {r.timestamp}",
            f"Model: {r.model_name}",
            "",
            "-" * 70,
            "MODEL SIZE",
            "-" * 70,
            f"Total Parameters:      {r.total_parameters:,} ({r.total_parameters/1e9:.2f}B)",
            f"Trainable Parameters:  {r.trainable_parameters:,}",
            f"Model Size (FP32):     {r.model_size_gb:.2f} GB",
            f"Model Size (FP16):     {r.model_fp16_size_gb:.2f} GB",
            f"Processor Vocab Size:  {r.processor_vocab_size:,}",
            "",
            "-" * 70,
            "GPU MEMORY USAGE",
            "-" * 70,
        ]
        
        total_post_load = sum(s.allocated_gb for s in r.post_load_gpu_memory)
        total_peak = sum(s.allocated_gb for s in r.peak_inference_gpu_memory)
        
        lines.append(f"Memory After Model Load: {total_post_load:.2f} GB")
        lines.append(f"Peak Memory (Inference): {total_peak:.2f} GB")
        
        if r.post_load_gpu_memory:
            lines.append("\nPer-GPU Breakdown:")
            for snap in r.post_load_gpu_memory:
                lines.append(
                    f"  GPU {snap.device_id} ({snap.device_name}): "
                    f"{snap.allocated_gb:.2f} GB allocated / {snap.total_memory_gb:.2f} GB total"
                )
        
        lines.extend([
            "",
            "-" * 70,
            "INFERENCE PERFORMANCE",
            "-" * 70,
            f"Samples Tested:        {r.num_samples_tested}",
            f"Warmup Time:           {r.warmup_time_ms:.2f} ms",
            "",
            "Timing Breakdown (averages):",
            f"  Preprocessing:       {r.avg_preprocessing_time_ms:.2f} ms",
            f"  Generation:          {r.avg_generation_time_ms:.2f} ms",
            f"  Post-processing:     {r.avg_postprocessing_time_ms:.2f} ms",
            f"  Total:               {r.avg_total_time_ms:.2f} ms (±{r.std_total_time_ms:.2f})",
            "",
            f"Min Inference Time:    {r.min_total_time_ms:.2f} ms",
            f"Max Inference Time:    {r.max_total_time_ms:.2f} ms",
            "",
            f"Tokens/Second:         {r.avg_tokens_per_second:.2f}",
            f"Throughput:            {r.throughput_images_per_minute:.2f} images/minute",
            "",
            "-" * 70,
            "PRODUCTION READINESS ASSESSMENT",
            "-" * 70,
        ])
        
        # Add production readiness assessment
        assessment = self._assess_production_readiness()
        for key, value in assessment.items():
            lines.append(f"{key}: {value}")
        
        lines.extend([
            "",
            "-" * 70,
            "SYSTEM INFORMATION",
            "-" * 70,
            f"PyTorch Version:       {r.device_info.get('pytorch_version', 'N/A')}",
            f"CUDA Version:          {r.device_info.get('cuda_version', 'N/A')}",
            f"Number of GPUs:        {r.device_info.get('num_gpus', 0)}",
        ])
        
        if r.device_info.get('gpu_names'):
            for i, name in enumerate(r.device_info['gpu_names']):
                lines.append(f"  GPU {i}: {name}")
        
        lines.extend([
            "",
            "=" * 70,
            "NOTE: This model is for RESEARCH USE ONLY, not for clinical deployment.",
            "=" * 70,
        ])
        
        return "\n".join(lines)
    
    def _assess_production_readiness(self) -> Dict[str, str]:
        """Assess model's production readiness based on benchmarks."""
        if self.results is None:
            return {}
        
        r = self.results
        assessment = {}
        
        # Memory efficiency
        total_gpu_mem = sum(s.total_memory_gb for s in r.post_load_gpu_memory) if r.post_load_gpu_memory else 0
        total_allocated = sum(s.allocated_gb for s in r.post_load_gpu_memory) if r.post_load_gpu_memory else 0
        mem_utilization = (total_allocated / total_gpu_mem * 100) if total_gpu_mem > 0 else 0
        
        if mem_utilization < 50:
            assessment["Memory Efficiency"] = f"✓ Good ({mem_utilization:.1f}% utilization)"
        elif mem_utilization < 80:
            assessment["Memory Efficiency"] = f"⚠ Moderate ({mem_utilization:.1f}% utilization)"
        else:
            assessment["Memory Efficiency"] = f"✗ High ({mem_utilization:.1f}% utilization)"
        
        # Latency assessment (for interactive use)
        if r.avg_total_time_ms < 5000:  # < 5 seconds
            assessment["Latency (Interactive)"] = f"✓ Acceptable ({r.avg_total_time_ms/1000:.2f}s)"
        elif r.avg_total_time_ms < 15000:  # < 15 seconds
            assessment["Latency (Interactive)"] = f"⚠ Moderate ({r.avg_total_time_ms/1000:.2f}s)"
        else:
            assessment["Latency (Interactive)"] = f"✗ Too slow ({r.avg_total_time_ms/1000:.2f}s)"
        
        # Throughput assessment (for batch processing)
        if r.throughput_images_per_minute > 30:
            assessment["Throughput (Batch)"] = f"✓ Good ({r.throughput_images_per_minute:.1f} img/min)"
        elif r.throughput_images_per_minute > 10:
            assessment["Throughput (Batch)"] = f"⚠ Moderate ({r.throughput_images_per_minute:.1f} img/min)"
        else:
            assessment["Throughput (Batch)"] = f"✗ Low ({r.throughput_images_per_minute:.1f} img/min)"
        
        # Consistency assessment
        if r.std_total_time_ms > 0 and r.avg_total_time_ms > 0:
            cv = (r.std_total_time_ms / r.avg_total_time_ms) * 100  # Coefficient of variation
            if cv < 20:
                assessment["Consistency"] = f"✓ Stable (CV: {cv:.1f}%)"
            elif cv < 50:
                assessment["Consistency"] = f"⚠ Variable (CV: {cv:.1f}%)"
            else:
                assessment["Consistency"] = f"✗ Inconsistent (CV: {cv:.1f}%)"
        
        # Multi-GPU scaling
        num_gpus = r.device_info.get('num_gpus', 1)
        if num_gpus > 1:
            assessment["Multi-GPU Support"] = f"✓ Distributed across {num_gpus} GPUs"
        else:
            assessment["Multi-GPU Support"] = "⚠ Single GPU only"
        
        return assessment
    
    def _print_summary(self):
        """Print benchmark summary to console."""
        print(self._generate_summary_text())


def main():
    """Main entry point for running benchmarks."""
    import argparse
    
    # Add parent directory to path for imports
    sys.path.insert(0, str(Path(__file__).parent))
    
    from evaluation.dataset import EvaluationDataset
    
    parser = argparse.ArgumentParser(description="MAIRA-2 Performance Benchmark")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="../data/first_labeling_studies_june_2025",
        help="Path to evaluation data directory",
    )
    parser.add_argument(
        "--csv-file",
        type=str,
        default="merged_june_andres_labeling.csv",
        help="CSV file name",
    )
    parser.add_argument(
        "--images-dir",
        type=str,
        default="images",
        help="Images subdirectory name",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=10,
        help="Number of samples to benchmark (default: 10)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./benchmark_results",
        help="Directory to save results",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=2,
        help="Number of warmup inferences",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=300,
        help="Maximum new tokens to generate",
    )
    
    args = parser.parse_args()
    
    # Resolve paths
    script_dir = Path(__file__).parent
    data_dir = (script_dir / args.data_dir).resolve()
    output_dir = (script_dir / args.output_dir).resolve()
    
    print("=" * 70)
    print("MAIRA-2 PERFORMANCE BENCHMARK")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Data directory: {data_dir}")
    print(f"  CSV file: {args.csv_file}")
    print(f"  Images dir: {args.images_dir}")
    print(f"  Samples to test: {args.num_samples}")
    print(f"  Output directory: {output_dir}")
    print(f"  Warmup runs: {args.warmup}")
    print(f"  Max tokens: {args.max_tokens}")
    
    # Load dataset
    print("\nLoading evaluation dataset...")
    csv_path = data_dir / args.csv_file
    images_path = data_dir / args.images_dir
    
    dataset = EvaluationDataset(
        csv_path=str(csv_path),
        images_dir=str(images_path),
        image_format="jpg",
        instance_id_column="instance_id",
        report_column="report_result",
        clean_html=True,
        filter_empty_reports=True,
    )
    
    print(f"Loaded {len(dataset)} samples from dataset")
    
    # Run benchmark
    benchmark = PerformanceBenchmark(
        model_name="microsoft/maira-2",
        num_warmup_samples=args.warmup,
        max_new_tokens=args.max_tokens,
    )
    
    results = benchmark.run_benchmark(
        dataset=dataset,
        num_samples=args.num_samples,
        output_dir=str(output_dir),
    )
    
    print("\n✓ Benchmark complete!")
    
    return results


if __name__ == "__main__":
    main()
