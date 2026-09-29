using System;

namespace LevelZero.Kernels;

/// <summary>
/// GPU-accelerated Fused Self-Attention (SDPA) kernel wrapper.
/// </summary>
public sealed class LlmAttentionKernel : IDisposable
{
    private readonly ComputeKernel _kernel;
    private readonly ComputeKernel _prefillKernel;

    internal LlmAttentionKernel(ComputeKernel kernel, ComputeKernel prefillKernel)
    {
        _kernel = kernel;
        _prefillKernel = prefillKernel;
    }

    /// <summary>
    /// Executes the fused attention calculation for a single query token.
    /// </summary>
    /// <param name="device">The L0 compute device.</param>
    /// <param name="q">Query states [numHeads, headDim].</param>
    /// <param name="kCache">Key Cache states [maxSeqLen, numKvHeads, headDim].</param>
    /// <param name="vCache">Value Cache states [maxSeqLen, numKvHeads, headDim].</param>
    /// <param name="output">Output states [numHeads, headDim].</param>
    /// <param name="headDim">Dimensionality of each attention head.</param>
    /// <param name="numHeads">Number of Query heads.</param>
    /// <param name="numKvHeads">Number of Key/Value heads (for Grouped Query Attention).</param>
    /// <param name="seqLen">Active sequence length (context size).</param>
    /// <param name="maxSeqLen">Maximum allocated sequence length of the KV Cache.</param>
    public void Execute(
        ComputeDevice device,
        SharedBuffer<float> q,
        SharedBuffer<float> kCache,
        SharedBuffer<float> vCache,
        SharedBuffer<float> output,
        int headDim,
        int numHeads,
        int numKvHeads,
        int seqLen,
        int maxSeqLen)
    {
        ArgumentNullException.ThrowIfNull(device);
        ArgumentNullException.ThrowIfNull(q);
        ArgumentNullException.ThrowIfNull(kCache);
        ArgumentNullException.ThrowIfNull(vCache);
        ArgumentNullException.ThrowIfNull(output);

        const uint localSize = 128;
        // Local scores scratchpad size is: active seqLen elements + localSize elements for group reductions + 2 floats for group max/sum
        int localScoresCount = seqLen + (int)localSize + 2;
        int localMemSizeBytes = localScoresCount * sizeof(float);

        _ = _kernel
            .WithArg(0, q)
            .WithArg(1, kCache)
            .WithArg(2, vCache)
            .WithArg(3, output)
            .WithArg(4, headDim)
            .WithArg(5, numHeads)
            .WithArg(6, numKvHeads)
            .WithArg(7, seqLen)
            .WithArg(8, maxSeqLen)
            .WithLocalArg(9, localMemSizeBytes)
            .WithGroupSize(localSize);

        device.Launch(_kernel, (uint)numHeads);
    }
    /// <summary>
    /// Causal attention for a chunk of <paramref name="numTokens"/> prompt tokens. Token t sits at position
    /// <paramref name="pos0"/> + t and attends to cache rows 0..pos0 + t. The chunk's keys and values must already be stored.
    /// </summary>
    /// <param name="device">The L0 compute device.</param>
    /// <param name="q">Query states [numTokens, numHeads, headDim].</param>
    /// <param name="kCache">Key cache [maxSeqLen, numKvHeads, headDim].</param>
    /// <param name="vCache">Value cache [maxSeqLen, numKvHeads, headDim].</param>
    /// <param name="output">Output states [numTokens, numHeads, headDim].</param>
    /// <param name="headDim">Dimensionality of each attention head.</param>
    /// <param name="numHeads">Number of query heads.</param>
    /// <param name="numKvHeads">Number of key/value heads.</param>
    /// <param name="pos0">Position of the first token of the chunk.</param>
    /// <param name="numTokens">Tokens in the chunk.</param>
    /// <param name="maxSeqLen">Allocated cache length.</param>
    public void ExecutePrefill(
        ComputeDevice device,
        SharedBuffer<float> q,
        SharedBuffer<float> kCache,
        SharedBuffer<float> vCache,
        SharedBuffer<float> output,
        int headDim,
        int numHeads,
        int numKvHeads,
        int pos0,
        int numTokens,
        int maxSeqLen)
    {
        ArgumentNullException.ThrowIfNull(device);
        ArgumentNullException.ThrowIfNull(q);
        ArgumentNullException.ThrowIfNull(kCache);
        ArgumentNullException.ThrowIfNull(vCache);
        ArgumentNullException.ThrowIfNull(output);
        ArgumentOutOfRangeException.ThrowIfNegative(pos0);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(numTokens);
        if (pos0 + numTokens > maxSeqLen)
        {
            throw new ArgumentException($"Positions up to {pos0 + numTokens - 1} exceed the {maxSeqLen}-slot cache.");
        }
        long rowsNeeded = (long)numTokens * numHeads * headDim;
        if (q.Count < rowsNeeded || output.Count < rowsNeeded)
        {
            throw new ArgumentException($"q and output must hold at least {rowsNeeded} elements.");
        }
        const uint localSize = 128;
        // Scores scratchpad for the longest token of the chunk, plus the reduction area and max/sum slots.
        int localScoresCount = pos0 + numTokens + (int)localSize + 2;
        _ = _prefillKernel
            .WithArg(0, q)
            .WithArg(1, kCache)
            .WithArg(2, vCache)
            .WithArg(3, output)
            .WithArg(4, headDim)
            .WithArg(5, numHeads)
            .WithArg(6, numKvHeads)
            .WithArg(7, pos0)
            .WithArg(8, maxSeqLen)
            .WithLocalArg(9, localScoresCount * sizeof(float))
            .WithGroupSize(localSize);
        device.Launch(_prefillKernel, (uint)numHeads, (uint)numTokens);
    }
    /// <summary>Disposes the underlying kernel handles.</summary>
    public void Dispose()
    {
        _kernel.Dispose();
        _prefillKernel.Dispose();
    }
}
