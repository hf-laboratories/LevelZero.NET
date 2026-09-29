using System;

namespace LevelZero.Kernels;

/// <summary>
/// GPU-accelerated Embedding Lookup kernel wrapper.
/// </summary>
public sealed class LlmEmbeddingKernel : IDisposable
{
    private readonly ComputeKernel _kernel;

    internal LlmEmbeddingKernel(ComputeKernel kernel)
    {
        _kernel = kernel;
    }

    /// <summary>
    /// Executes the embedding lookup kernel on the specified device.
    /// </summary>
    /// <param name="device">The L0 compute device.</param>
    /// <param name="tokenIds">USM buffer containing token indices [sequence_length].</param>
    /// <param name="embedWeights">USM buffer containing the flat vocab embedding weights [vocab_size * hidden_dim].</param>
    /// <param name="output">USM buffer to write the looked up embeddings to [sequence_length * hidden_dim].</param>
    /// <param name="hiddenDim">Dimensionality of the hidden state.</param>
    /// <param name="totalElements">Total elements (sequence_length * hidden_dim) to copy.</param>
    public void Execute(
        ComputeDevice device,
        SharedBuffer<int> tokenIds,
        SharedBuffer<float> embedWeights,
        SharedBuffer<float> output,
        int hiddenDim,
        int totalElements)
    {
        ArgumentNullException.ThrowIfNull(device);
        ArgumentNullException.ThrowIfNull(tokenIds);
        ArgumentNullException.ThrowIfNull(embedWeights);
        ArgumentNullException.ThrowIfNull(output);

        const uint localSize = 64;

        _ = _kernel
            .WithArg(0, tokenIds)
            .WithArg(1, embedWeights)
            .WithArg(2, output)
            .WithArg(3, hiddenDim)
            .WithArg(4, totalElements)
            .WithGroupSize(localSize);

        device.Launch(_kernel, ComputeDevice.GroupCount(totalElements, localSize));
    }

    /// <summary>Disposes the underlying kernel handle.</summary>
    public void Dispose() => _kernel.Dispose();
}
