# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "acquire-zarr>=0.2.4",
#     "zarr",
#     "rich",
#     "tensorstore",
# ]
# ///
# !/usr/bin/env python3
"""Compare write performance of TensorStore vs. acquire-zarr for a Zarr v3 store.

Runs multiple iterations of the comparison to generate distribution data and visualizes results.

Thanks to Talley Lambert @tlambert03 for the original version of this script:
https://gist.github.com/tlambert03/f8c1b069c2947b411ce24ea05aa370b1
"""

from pathlib import Path
import sys
import time
import shutil
import os
from typing import Tuple, Dict, List

import acquire_zarr as aqz
import numpy as np
import tensorstore
import zarr
from rich import print
import matplotlib.pyplot as plt
import pandas as pd


class CyclicArray:
    def __init__(self, data: np.ndarray, n_frames: int):
        self.data = data
        self.t = self.data.shape[0]  # Size of first dimension
        self.shape = (n_frames,) + self.data.shape[1:]

    def __getitem__(self, idx):
        if isinstance(idx, tuple) and len(idx) > 0:
            if isinstance(idx[0], int):
                return self.data[(idx[0] % self.t,) + idx[1:]]
        elif isinstance(idx, int):
            return self.data[idx % self.t]

        return self.data[idx]

    def compare_array(self, arr: np.ndarray) -> None:
        """Compare an array with a CyclicArray."""
        assert self.shape == arr.shape

        for i in range(0, arr.shape[0], self.t):
            start = i
            stop = min(i + self.t, arr.shape[0])
            np.testing.assert_array_equal(
                self.data[0: (stop - start)], arr[start:stop]
            )


def run_tensorstore_test(data: CyclicArray, path: str, metadata: dict) -> Tuple[float, np.ndarray]:
    """Write data using TensorStore and print per-plane and total write times."""
    # Define a TensorStore spec for a Zarr v3 store.
    spec = {
        "driver": "zarr3",
        "kvstore": {"driver": "file", "path": path},
        "metadata": metadata,
        "delete_existing": True,
        "create": True,
    }
    # Open (or create) the store.
    ts = tensorstore.open(spec).result()
    print(ts)
    total_start = time.perf_counter_ns()
    futures = []
    elapsed_times = []

    # cache data until we've reached a write-chunk-aligned block
    chunk_length = ts.schema.chunk_layout.write_chunk.shape[0]
    write_chunk_shape = (chunk_length, *ts.domain.shape[1:])
    chunk = np.empty(write_chunk_shape, dtype=np.uint16)
    for i in range(data.shape[0]):
        start_plane = time.perf_counter_ns()
        chunk_idx = i % chunk_length
        chunk[chunk_idx] = data[i]
        if chunk_idx == chunk_length - 1:
            slc = slice(i - chunk_length + 1, i + 1)
            futures.append(ts[slc].write(chunk))
            chunk = np.empty(write_chunk_shape, dtype=np.uint16)
        elapsed = time.perf_counter_ns() - start_plane
        elapsed_times.append(elapsed)
        print(f"TensorStore: Plane {i} written in {elapsed / 1e6:.3f} ms")

    start_futures = time.perf_counter_ns()
    # Wait for all writes to finish.
    for future in futures:
        future.result()
    elapsed = time.perf_counter_ns() - start_futures
    elapsed_times.append(elapsed)
    print(f"TensorStore: Final futures took {elapsed / 1e6:.3f} ms")

    total_elapsed = time.perf_counter_ns() - total_start
    tot_ms = total_elapsed / 1e6
    print(f"TensorStore: Total write time: {tot_ms:.3f} ms")

    return tot_ms, np.array(elapsed_times) / 1e6


def run_acquire_zarr_test(
        data: CyclicArray,
        path: str,
        tchunk_size: int = 1,
        xy_chunk_size: int = 2048,
        xy_shard_size: int = 1,
) -> Tuple[float, np.ndarray]:
    """Write data using acquire-zarr and print per-plane and total write times."""
    settings = aqz.StreamSettings(
        store_path=path,
        data_type=aqz.DataType.UINT16,
        version=aqz.ZarrVersion.V3,
        compression=aqz.CompressionSettings(
            codec=aqz.CompressionCodec.BLOSC_ZSTD,
            compressor=aqz.Compressor.BLOSC1,
            compression_level=3,
            shuffle=0
        )
    )
    settings.dimensions.extend(
        [
            aqz.Dimension(
                name="t",
                type=aqz.DimensionType.TIME,
                array_size_px=0,
                chunk_size_px=tchunk_size,
                shard_size_chunks=1,
            ),
            aqz.Dimension(
                name="y",
                type=aqz.DimensionType.SPACE,
                array_size_px=2048,
                chunk_size_px=xy_chunk_size,
                shard_size_chunks=xy_shard_size,
            ),
            aqz.Dimension(
                name="x",
                type=aqz.DimensionType.SPACE,
                array_size_px=2048,
                chunk_size_px=xy_chunk_size,
                shard_size_chunks=xy_shard_size,
            ),
        ],
    )

    # Create a ZarrStream for appending frames.
    stream = aqz.ZarrStream(settings)

    elapsed_times = []

    total_start = time.perf_counter_ns()
    for i in range(data.shape[0]):
        start_plane = time.perf_counter_ns()
        stream.append(data[i])
        elapsed = time.perf_counter_ns() - start_plane
        elapsed_times.append(elapsed)
        print(f"Acquire-zarr: Plane {i} written in {elapsed / 1e6:.3f} ms")

    # Close (or flush) the stream to finalize writes.
    del stream
    total_elapsed = time.perf_counter_ns() - total_start
    tot_ms = total_elapsed / 1e6
    print(f"Acquire-zarr: Total write time: {tot_ms:.3f} ms")

    return tot_ms, np.array(elapsed_times) / 1e6


def cleanup_test_directories():
    """Remove test directories to clean up between runs."""
    dirs_to_remove = ["acquire_zarr_test.zarr", "tensorstore_test.zarr"]
    for dir_path in dirs_to_remove:
        if os.path.exists(dir_path):
            try:
                if os.path.isdir(dir_path):
                    shutil.rmtree(dir_path)
                else:
                    os.remove(dir_path)
                print(f"Removed {dir_path}")
            except Exception as e:
                print(f"Error removing {dir_path}: {e}")


def run_single_comparison(
        t_chunk_size: int, xy_chunk_size: int, xy_shard_size: int, frame_count: int
) -> dict:
    """Run a single comparison and return the results as a dictionary."""
    print("tchunk_size:", t_chunk_size)
    print("xy_chunk_size:", xy_chunk_size)
    print("xy_shard_size:", xy_shard_size)
    print("frame_count:", frame_count)

    # Pre-generate the data (timing excluded)
    data = CyclicArray(
        np.random.randint(0, 2 ** 16 - 1, (128, 2048, 2048), dtype=np.uint16), frame_count
    )

    # Run acquire-zarr test
    print("\nRunning acquire-zarr test:")
    az_path = "acquire_zarr_test.zarr"
    print("Saving to", Path(az_path).absolute())
    time_az_ms, frame_write_times_az = run_acquire_zarr_test(
        data, az_path, t_chunk_size, xy_chunk_size, xy_shard_size
    )

    # Use the same metadata for TensorStore
    az = zarr.open(az_path)["0"]

    # Run TensorStore test
    print("\nRunning TensorStore test:")
    ts_path = "tensorstore_test.zarr"
    time_ts_ms, frame_write_times_ts = run_tensorstore_test(
        data,
        ts_path,
        {**az.metadata.to_dict(), "data_type": "uint16"},
    )

    # Verify metadata matches
    ts = zarr.open(ts_path)
    assert ts.metadata == az.metadata
    print("Metadata matches")

    # Calculate throughput
    data_size_gib = (2048 * 2048 * 2 * frame_count) / (1 << 30)
    az_throughput = 1000 * data_size_gib / time_az_ms
    ts_throughput = 1000 * data_size_gib / time_ts_ms
    ts_az_ratio = time_ts_ms / time_az_ms

    # Print performance comparison
    print("\nPerformance comparison:")
    print(
        f"  acquire-zarr: {time_az_ms:.3f} ms, {az_throughput:.3f} GiB/s, "
        f"50th percentile frame write time: {np.percentile(frame_write_times_az, 50):.3f} ms, "
        f"99th percentile: {np.percentile(frame_write_times_az, 99):.3f} ms"
    )
    print(
        f"  TensorStore: {time_ts_ms:.3f} ms, {ts_throughput:.3f} GiB/s, "
        f"50th percentile frame write time: {np.percentile(frame_write_times_ts, 50):.3f} ms, "
        f"99th percentile: {np.percentile(frame_write_times_ts, 99):.3f} ms"
    )
    print(f"  TS/AZ Ratio: {ts_az_ratio:.3f}")

    # Return results
    return {
        "acquire_zarr_time_ms": time_az_ms,
        "tensorstore_time_ms": time_ts_ms,
        "acquire_zarr_throughput_gibs": az_throughput,
        "tensorstore_throughput_gibs": ts_throughput,
        "ts_az_ratio": ts_az_ratio,
        "acquire_zarr_frame_times": frame_write_times_az,
        "tensorstore_frame_times": frame_write_times_ts,
        "acquire_zarr_p50_ms": np.percentile(frame_write_times_az, 50),
        "acquire_zarr_p99_ms": np.percentile(frame_write_times_az, 99),
        "tensorstore_p50_ms": np.percentile(frame_write_times_ts, 50),
        "tensorstore_p99_ms": np.percentile(frame_write_times_ts, 99),
    }


def visualize_results(all_results: List[dict]):
    """Visualize the results of multiple benchmark runs."""
    # Convert results to DataFrame
    df = pd.DataFrame(all_results)

    # Create figure with multiple subplots
    fig, axs = plt.subplots(2, 2, figsize=(15, 12))

    # Plot 1: Total execution time comparison
    axs[0, 0].boxplot([df['acquire_zarr_time_ms'], df['tensorstore_time_ms']])
    axs[0, 0].set_title('Total Execution Time (ms)')
    axs[0, 0].set_xticklabels(['acquire-zarr', 'TensorStore'])
    axs[0, 0].grid(True)

    # Plot 2: Throughput comparison
    axs[0, 1].boxplot([df['acquire_zarr_throughput_gibs'], df['tensorstore_throughput_gibs']])
    axs[0, 1].set_title('Throughput (GiB/s)')
    axs[0, 1].set_xticklabels(['acquire-zarr', 'TensorStore'])
    axs[0, 1].grid(True)

    # Plot 3: Frame write time percentiles
    p50_p99_data = [
        df['acquire_zarr_p50_ms'],
        df['acquire_zarr_p99_ms'],
        df['tensorstore_p50_ms'],
        df['tensorstore_p99_ms']
    ]

    # Create boxplot
    bp = axs[1, 0].boxplot(p50_p99_data, patch_artist=True)

    # Set colors for boxplots
    colors = ['lightblue', 'blue', 'lightgreen', 'green']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)

    axs[1, 0].set_title('Frame Write Time Percentiles (ms)')
    axs[1, 0].set_xticklabels(['AZ p50', 'AZ p99', 'TS p50', 'TS p99'])
    axs[1, 0].grid(True, linestyle='--', alpha=0.7)

    # Plot 4: TS/AZ ratio
    axs[1, 1].boxplot(df['ts_az_ratio'])
    axs[1, 1].set_title('TensorStore/acquire-zarr Time Ratio')
    axs[1, 1].set_xticklabels(['TS/AZ Ratio'])
    axs[1, 1].axhline(y=1.0, color='r', linestyle='--', label='Equal performance')
    axs[1, 1].grid(True)
    axs[1, 1].legend()

    plt.tight_layout()
    plt.savefig('benchmark_results.png')
    print("Results visualization saved to 'benchmark_results.png'")

    # Also save raw data
    df.to_csv('benchmark_results.csv', index=False)
    print("Raw results saved to 'benchmark_results.csv'")

    # Print summary statistics
    print("\nSUMMARY STATISTICS:")
    print(f"Number of benchmark runs: {len(all_results)}")
    print("\nAcquire-zarr:")
    print(f"  Mean total time: {df['acquire_zarr_time_ms'].mean():.2f} ms")
    print(f"  Mean throughput: {df['acquire_zarr_throughput_gibs'].mean():.2f} GiB/s")
    print(f"  Mean p50 frame time: {df['acquire_zarr_p50_ms'].mean():.2f} ms")
    print(f"  Mean p99 frame time: {df['acquire_zarr_p99_ms'].mean():.2f} ms")

    print("\nTensorStore:")
    print(f"  Mean total time: {df['tensorstore_time_ms'].mean():.2f} ms")
    print(f"  Mean throughput: {df['tensorstore_throughput_gibs'].mean():.2f} GiB/s")
    print(f"  Mean p50 frame time: {df['tensorstore_p50_ms'].mean():.2f} ms")
    print(f"  Mean p99 frame time: {df['tensorstore_p99_ms'].mean():.2f} ms")

    print(f"\nMean TS/AZ Ratio: {df['ts_az_ratio'].mean():.2f}")

    return fig


def main():
    """Run multiple benchmark comparisons and visualize the results."""
    # Parse command line arguments
    T_CHUNK_SIZE = int(sys.argv[1]) if len(sys.argv) > 1 else 64
    XY_CHUNK_SIZE = int(sys.argv[2]) if len(sys.argv) > 2 else 64
    XY_SHARD_SIZE = int(sys.argv[3]) if len(sys.argv) > 3 else 16
    FRAME_COUNT = int(sys.argv[4]) if len(sys.argv) > 4 else 1024
    NUM_RUNS = int(sys.argv[5]) if len(sys.argv) > 5 else 20

    print(f"Running {NUM_RUNS} benchmark iterations with parameters:")
    print(f"  T_CHUNK_SIZE: {T_CHUNK_SIZE}")
    print(f"  XY_CHUNK_SIZE: {XY_CHUNK_SIZE}")
    print(f"  XY_SHARD_SIZE: {XY_SHARD_SIZE}")
    print(f"  FRAME_COUNT: {FRAME_COUNT}")

    # Collect results from multiple runs
    all_results = []

    for run_idx in range(NUM_RUNS):
        print(f"\n\n--- BENCHMARK RUN {run_idx + 1}/{NUM_RUNS} ---\n")

        # Clean up from previous run
        cleanup_test_directories()

        # Run comparison and collect results
        run_results = run_single_comparison(T_CHUNK_SIZE, XY_CHUNK_SIZE, XY_SHARD_SIZE, FRAME_COUNT)
        all_results.append(run_results)

        # Add run index to results
        run_results['run_idx'] = run_idx

    # Clean up after all runs
    cleanup_test_directories()

    # Visualize results
    visualize_results(all_results)


if __name__ == "__main__":
    main()
