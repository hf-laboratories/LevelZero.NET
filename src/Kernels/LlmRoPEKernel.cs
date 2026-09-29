using System;

namespace LevelZero.Kernels;

/// <summary>
/// GPU-accelerated Fused Rotary Position Embedding (RoPE) kernel wrapper.
/// </summary>
public sealed class LlmRoPEKernel : IDisposable
{
    private readonly ComputeKernel _kernel;
    private readonly ComputeKernel? _frequencyKernel;

    internal LlmRoPEKernel(ComputeKernel kernel, ComputeKernel? frequencyKernel = null)
    {
        _kernel = kernel;
        _frequencyKernel = frequencyKernel;
    }

    /// <summary>
    /// Executes the fused in-place RoPE kernel on Query and Key tensors.
    /// </summary>
    /// <param name="device">The L0 compute device.</param>
    /// <param name="q">Query states [numTokens * numHeads * headDim]. Updated in-place.</param>
    /// <param name="k">Key states [numTokens * numKvHeads * headDim]. Updated in-place.</param>
    /// <param name="positionIds">Token sequence positions [numTokens].</param>
    /// <param name="headDim">Dimensionality of each attention head.</param>
    /// <param name="numHeads">Number of Query heads.</param>
    /// <param name="numKvHeads">Number of Key/Value heads.</param>
    /// <param name="ropeBase">Rotary embedding base constant (typically 10000.0 or 500000.0).</param>
    /// <param name="numTokens">Number of tokens to process.</param>
    public void Execute(
        ComputeDevice device,
        SharedBuffer<float> q,
        SharedBuffer<float> k,
        SharedBuffer<int> positionIds,
        int headDim,
        int numHeads,
        int numKvHeads,
        float ropeBase,
        int numTokens)
    {
        ArgumentNullException.ThrowIfNull(device);
        ArgumentNullException.ThrowIfNull(q);
        ArgumentNullException.ThrowIfNull(k);
        ArgumentNullException.ThrowIfNull(positionIds);

        // 3D dispatch grid:
        // x: numTokens
        // y: max(numHeads, numKvHeads)
        // z: headDim / 2
        int maxHeads = Math.Max(numHeads, numKvHeads);
        int halfDim = headDim / 2;

        _ = _kernel
            .WithArg(0, q)
            .WithArg(1, k)
            .WithArg(2, positionIds)
            .WithArg(3, headDim)
            .WithArg(4, numHeads)
            .WithArg(5, numKvHeads)
            .WithArg(6, ropeBase)
            .WithArg(7, numTokens)
            .WithGroupSize(1, 1, (uint)halfDim); // All half dims inside same workgroup for fast memory access

        device.Launch(_kernel, (uint)numTokens, (uint)maxHeads, 1);
    }

    /// <summary>
    /// Like <see cref="Execute"/> but takes a precomputed inverse-frequency table, which is how scaled
    /// RoPE (linear, YaRN) is expressed.
    /// </summary>
    /// <param name="device">The L0 compute device.</param>
    /// <param name="q">Query states [numTokens * numHeads * headDim]. Updated in-place.</param>
    /// <param name="k">Key states [numTokens * numKvHeads * headDim]. Updated in-place.</param>
    /// <param name="positionIds">Token sequence positions [numTokens].</param>
    /// <param name="inverseFrequencies">One inverse frequency per rotated pair [headDim / 2].</param>
    /// <param name="headDim">Dimensionality of each attention head.</param>
    /// <param name="numHeads">Number of Query heads.</param>
    /// <param name="numKvHeads">Number of Key/Value heads.</param>
    /// <param name="attentionScale">Factor applied to both cos and sin (1 when unscaled).</param>
    /// <param name="rotaryDim">Elements per head that rotate (partial rotary); 0 means the whole head.</param>
    /// <param name="numTokens">Number of tokens to process.</param>
    public void ExecuteWithFrequencies(
        ComputeDevice device,
        SharedBuffer<float> q,
        SharedBuffer<float> k,
        SharedBuffer<int> positionIds,
        SharedBuffer<float> inverseFrequencies,
        int headDim,
        int numHeads,
        int numKvHeads,
        float attentionScale,
        int numTokens,
        int rotaryDim = 0)
    {
        ArgumentNullException.ThrowIfNull(device);
        ArgumentNullException.ThrowIfNull(q);
        ArgumentNullException.ThrowIfNull(k);
        ArgumentNullException.ThrowIfNull(positionIds);
        ArgumentNullException.ThrowIfNull(inverseFrequencies);
        if (_frequencyKernel is null)
        {
            throw new NotSupportedException("The kernel module has no rope_fused_freq entry point; rebuild the kernels.");
        }
        int rotary = rotaryDim == 0 ? headDim : rotaryDim;
        if (rotary <= 0 || rotary > headDim || rotary % 2 != 0)
        {
            throw new ArgumentOutOfRangeException(nameof(rotaryDim), rotaryDim, "rotaryDim must be even and at most headDim.");
        }

        int halfDim = rotary / 2;
        if (inverseFrequencies.Count < halfDim)
        {
            throw new ArgumentException($"Need {halfDim} inverse frequencies, got {inverseFrequencies.Count}.", nameof(inverseFrequencies));
        }
        int maxHeads = Math.Max(numHeads, numKvHeads);
        _ = _frequencyKernel
            .WithArg(0, q)
            .WithArg(1, k)
            .WithArg(2, positionIds)
            .WithArg(3, inverseFrequencies)
            .WithArg(4, headDim)
            .WithArg(5, numHeads)
            .WithArg(6, numKvHeads)
            .WithArg(7, attentionScale)
            .WithArg(8, numTokens)
            .WithArg(9, rotary)
            .WithGroupSize(1, 1, (uint)halfDim);
        device.Launch(_frequencyKernel, (uint)numTokens, (uint)maxHeads, 1);
    }
    /// <summary>Disposes the underlying kernel handle.</summary>
    public void Dispose()
    {
        _kernel.Dispose();
        _frequencyKernel?.Dispose();
    }
}
