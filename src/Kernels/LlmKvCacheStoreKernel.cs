using System;

namespace LevelZero.Kernels;

/// <summary>
/// Copies one token's key and value rows into the KV caches at a given position, on the GPU, so the
/// decode loop never has to touch the caches from the host.
/// </summary>
public sealed class LlmKvCacheStoreKernel : IDisposable
{
    private const uint LocalSize = 64;

    private readonly ComputeKernel _kernel;
    private readonly ComputeKernel _rowsKernel;

    internal LlmKvCacheStoreKernel(ComputeKernel kernel, ComputeKernel rowsKernel)
    {
        _kernel = kernel;
        _rowsKernel = rowsKernel;
    }

    /// <summary>Stores <paramref name="k"/> and <paramref name="v"/> at slot <paramref name="position"/>.</summary>
    /// <param name="device">The L0 compute device.</param>
    /// <param name="k">This token's keys [kvDim] (numKvHeads * headDim), already rotated by RoPE.</param>
    /// <param name="v">This token's values [kvDim].</param>
    /// <param name="kCache">Key cache [maxSeqLen, kvDim].</param>
    /// <param name="vCache">Value cache [maxSeqLen, kvDim].</param>
    /// <param name="position">Cache slot to write, from 0 to maxSeqLen - 1.</param>
    /// <param name="kvDim">Width of one cache row.</param>
    public void Execute(
        ComputeDevice device,
        SharedBuffer<float> k,
        SharedBuffer<float> v,
        SharedBuffer<float> kCache,
        SharedBuffer<float> vCache,
        int position,
        int kvDim)
    {
        ArgumentNullException.ThrowIfNull(device);
        ArgumentNullException.ThrowIfNull(k);
        ArgumentNullException.ThrowIfNull(v);
        ArgumentNullException.ThrowIfNull(kCache);
        ArgumentNullException.ThrowIfNull(vCache);
        ArgumentOutOfRangeException.ThrowIfNegative(position);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(kvDim);
        long end = ((long)position + 1) * kvDim;
        if (end > kCache.Count || end > vCache.Count)
        {
            throw new ArgumentException(
                $"Position {position} with kvDim {kvDim} needs {end} cache elements; caches hold {kCache.Count} and {vCache.Count}.");
        }

        if (k.Count < kvDim || v.Count < kvDim)
        {
            throw new ArgumentException($"k and v must hold at least {kvDim} elements.");
        }

        _ = _kernel
            .WithArg(0, k)
            .WithArg(1, v)
            .WithArg(2, kCache)
            .WithArg(3, vCache)
            .WithArg(4, position)
            .WithArg(5, kvDim)
            .WithGroupSize(LocalSize);
        device.Launch(_kernel, ComputeDevice.GroupCount(kvDim, LocalSize));
    }
    /// <summary>Stores a chunk of <paramref name="rows"/> tokens: row r of k and v goes to slot <paramref name="position"/> + r.</summary>
    public void ExecuteRows(
        ComputeDevice device,
        SharedBuffer<float> k,
        SharedBuffer<float> v,
        SharedBuffer<float> kCache,
        SharedBuffer<float> vCache,
        int position,
        int kvDim,
        int rows)
    {
        ArgumentNullException.ThrowIfNull(device);
        ArgumentNullException.ThrowIfNull(k);
        ArgumentNullException.ThrowIfNull(v);
        ArgumentNullException.ThrowIfNull(kCache);
        ArgumentNullException.ThrowIfNull(vCache);
        ArgumentOutOfRangeException.ThrowIfNegative(position);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(kvDim);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(rows);
        long end = ((long)position + rows) * kvDim;
        if (end > kCache.Count || end > vCache.Count)
        {
            throw new ArgumentException(
                $"Positions {position}..{position + rows - 1} with kvDim {kvDim} need {end} cache elements; caches hold {kCache.Count} and {vCache.Count}.");
        }
        long source = (long)rows * kvDim;
        if (k.Count < source || v.Count < source)
        {
            throw new ArgumentException($"k and v must hold at least {source} elements.");
        }
        _ = _rowsKernel
            .WithArg(0, k)
            .WithArg(1, v)
            .WithArg(2, kCache)
            .WithArg(3, vCache)
            .WithArg(4, position)
            .WithArg(5, kvDim)
            .WithArg(6, rows)
            .WithGroupSize(LocalSize);
        device.Launch(_rowsKernel, ComputeDevice.GroupCount((int)source, LocalSize));
    }
    /// <summary>Disposes the underlying kernel handles.</summary>
    public void Dispose()
    {
        _kernel.Dispose();
        _rowsKernel.Dispose();
    }
}
