using System;

namespace LevelZero.Kernels;

/// <summary>
/// GPU-accelerated Fused SwiGLU Gated Activation kernel wrapper.
/// </summary>
public sealed class LlmSwiGLUKernel : IDisposable
{
    private readonly ComputeKernel _kernel;

    internal LlmSwiGLUKernel(ComputeKernel kernel)
    {
        _kernel = kernel;
    }

    /// <summary>
    /// Executes the SwiGLU activation: gate = SiLU(gate) * up in-place on the gate buffer.
    /// </summary>
    /// <param name="device">The L0 compute device.</param>
    /// <param name="gate">The Gate states [numTokens * intermediateDim]. Updated in-place with the activation result.</param>
    /// <param name="up">The Up projection states [numTokens * intermediateDim].</param>
    /// <param name="totalElements">Total elements (numTokens * intermediateDim) in the buffers.</param>
    public void Execute(
        ComputeDevice device,
        SharedBuffer<float> gate,
        SharedBuffer<float> up,
        int totalElements)
    {
        ArgumentNullException.ThrowIfNull(device);
        ArgumentNullException.ThrowIfNull(gate);
        ArgumentNullException.ThrowIfNull(up);

        const uint localSize = 64;

        _ = _kernel
            .WithArg(0, gate)
            .WithArg(1, up)
            .WithArg(2, totalElements)
            .WithGroupSize(localSize);

        device.Launch(_kernel, ComputeDevice.GroupCount(totalElements, localSize));
    }

    /// <summary>Disposes the underlying kernel handle.</summary>
    public void Dispose() => _kernel.Dispose();
}
