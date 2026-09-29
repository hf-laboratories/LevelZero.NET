using System;

namespace LevelZero.Kernels;

/// <summary>
/// GPU-accelerated Logits Sampling kernel wrapper.
/// </summary>
public sealed class LlmSamplingKernel : IDisposable
{
    private readonly ComputeKernel _kernel;

    internal LlmSamplingKernel(ComputeKernel kernel)
    {
        _kernel = kernel;
    }

    /// <summary>
    /// Computes Softmax and samples a token ID from the final vocabulary logits.
    /// </summary>
    /// <param name="device">The L0 compute device.</param>
    /// <param name="logits">Vocabulary logits [vocabSize].</param>
    /// <param name="outputTokenId">Single-element buffer containing the sampled token index.</param>
    /// <param name="temperature">Temperature scale. If <= 0, greedy selection (argmax) is used.</param>
    /// <param name="randomVal">Random uniform float [0, 1) used for roulette wheel selection when temperature > 0.</param>
    /// <param name="vocabSize">Vocabulary size.</param>
    public void Execute(
        ComputeDevice device,
        SharedBuffer<float> logits,
        SharedBuffer<int> outputTokenId,
        float temperature,
        float randomVal,
        int vocabSize)
    {
        ArgumentNullException.ThrowIfNull(device);
        ArgumentNullException.ThrowIfNull(logits);
        ArgumentNullException.ThrowIfNull(outputTokenId);

        const uint localSize = 256;
        int localMaxValsSize = (int)(localSize * sizeof(float));
        int localMaxIdxsSize = (int)(localSize * sizeof(int));

        _ = _kernel
            .WithArg(0, logits)
            .WithArg(1, outputTokenId)
            .WithArg(2, temperature)
            .WithArg(3, randomVal)
            .WithArg(4, vocabSize)
            .WithLocalArg(5, localMaxValsSize)
            .WithLocalArg(6, localMaxIdxsSize)
            .WithGroupSize(localSize);

        device.Launch(_kernel, 1);
    }

    /// <summary>Disposes the underlying kernel handle.</summary>
    public void Dispose() => _kernel.Dispose();
}
