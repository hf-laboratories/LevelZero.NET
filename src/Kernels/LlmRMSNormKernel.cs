using System;

namespace LevelZero.Kernels;

/// <summary>
/// GPU-accelerated Fused RMSNorm (with optional residual addition) kernel wrapper.
/// </summary>
public sealed class LlmRMSNormKernel : IDisposable
{
    private readonly ComputeKernel _kernel;

    internal LlmRMSNormKernel(ComputeKernel kernel)
    {
        _kernel = kernel;
    }

    /// <summary>
    /// Executes the fused RMSNorm kernel on the specified device.
    /// </summary>
    /// <param name="device">The L0 compute device.</param>
    /// <param name="x">The hidden state buffer [batchSize * hiddenDim]. Updated in-place with residual if addResidual is true.</param>
    /// <param name="residual">Optional residual buffer [batchSize * hiddenDim]. Can be null.</param>
    /// <param name="weight">RMSNorm scaling weights [hiddenDim].</param>
    /// <param name="output">Buffer to write normalized output hidden states [batchSize * hiddenDim].</param>
    /// <param name="epsilon">Epsilon small float value for numerical stability.</param>
    /// <param name="hiddenDim">Dimensionality of the hidden state.</param>
    /// <param name="batchSize">Number of parallel sequences or tokens.</param>
    /// <param name="addResidual">If true, adds residual to x in-place before normalizing.</param>
    public void Execute(
        ComputeDevice device,
        SharedBuffer<float> x,
        SharedBuffer<float>? residual,
        SharedBuffer<float> weight,
        SharedBuffer<float> output,
        float epsilon,
        int hiddenDim,
        int batchSize,
        bool addResidual)
    {
        ArgumentNullException.ThrowIfNull(device);
        ArgumentNullException.ThrowIfNull(x);
        ArgumentNullException.ThrowIfNull(weight);
        ArgumentNullException.ThrowIfNull(output);

        if (addResidual && residual == null)
        {
            throw new ArgumentException("Residual buffer must be provided when addResidual is true.", nameof(residual));
        }

        const uint localSize = 256;
        int localMemSize = (int)(localSize * sizeof(float));

        IntPtr residualPtr = residual?.Pointer ?? IntPtr.Zero;

        _ = _kernel
            .WithArg(0, x)
            .WithArg(1, residualPtr)
            .WithArg(2, weight)
            .WithArg(3, output)
            .WithArg(4, epsilon)
            .WithArg(5, hiddenDim)
            .WithArg(6, addResidual ? 1 : 0)
            .WithLocalArg(7, localMemSize)
            .WithGroupSize(localSize);

        device.Launch(_kernel, (uint)batchSize);
    }

    /// <summary>Disposes the underlying kernel handle.</summary>
    public void Dispose() => _kernel.Dispose();
}
