using System;

namespace LevelZero.Kernels;

/// <summary>GPU in-place elementwise add: <c>x[i] += y[i]</c>, used for residual connections.</summary>
public sealed class LlmAddKernel : IDisposable
{
    private const uint LocalSize = 64;

    private readonly ComputeKernel _kernel;

    internal LlmAddKernel(ComputeKernel kernel)
    {
        _kernel = kernel;
    }

    /// <summary>Adds the first <paramref name="count"/> elements of <paramref name="y"/> into <paramref name="x"/>.</summary>
    /// <param name="device">The L0 compute device.</param>
    /// <param name="x">Buffer updated in place.</param>
    /// <param name="y">Values to add.</param>
    /// <param name="count">Number of elements to add.</param>
    public void Execute(ComputeDevice device, SharedBuffer<float> x, SharedBuffer<float> y, int count)
    {
        ArgumentNullException.ThrowIfNull(device);
        ArgumentNullException.ThrowIfNull(x);
        ArgumentNullException.ThrowIfNull(y);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(count);
        if (x.Count < count || y.Count < count)
        {
            throw new ArgumentException($"Buffers hold {x.Count} and {y.Count} elements but count is {count}.");
        }

        _ = _kernel
            .WithArg(0, x)
            .WithArg(1, y)
            .WithArg(2, count)
            .WithGroupSize(LocalSize);
        device.Launch(_kernel, ComputeDevice.GroupCount(count, LocalSize));
    }

    /// <summary>Disposes the underlying kernel handle.</summary>
    public void Dispose() => _kernel.Dispose();
}
