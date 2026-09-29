using System;

namespace LevelZero.Kernels;

/// <summary>
/// GPU broadcast bias add: <c>x[row * n + i] += bias[i]</c>. Used for the Q, K and V projection
/// biases of Qwen2-style models.
/// </summary>
public sealed class LlmBiasAddKernel : IDisposable
{
    private const uint LocalSize = 64;

    private readonly ComputeKernel _kernel;

    internal LlmBiasAddKernel(ComputeKernel kernel)
    {
        _kernel = kernel;
    }

    /// <summary>Adds <paramref name="bias"/> to every row of <paramref name="x"/> in place.</summary>
    /// <param name="device">The L0 compute device.</param>
    /// <param name="x">Row-major matrix [rows * n], updated in place.</param>
    /// <param name="bias">Bias vector [n].</param>
    /// <param name="n">Row width (length of <paramref name="bias"/>).</param>
    /// <param name="rows">Number of rows in <paramref name="x"/>.</param>
    public void Execute(ComputeDevice device, SharedBuffer<float> x, SharedBuffer<float> bias, int n, int rows)
    {
        ArgumentNullException.ThrowIfNull(device);
        ArgumentNullException.ThrowIfNull(x);
        ArgumentNullException.ThrowIfNull(bias);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(n);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(rows);
        if (bias.Count < n)
        {
            throw new ArgumentException($"Bias holds {bias.Count} elements but n is {n}.", nameof(bias));
        }

        int total = checked(n * rows);
        if (x.Count < total)
        {
            throw new ArgumentException($"x holds {x.Count} elements but rows * n is {total}.", nameof(x));
        }

        _ = _kernel
            .WithArg(0, x)
            .WithArg(1, bias)
            .WithArg(2, n)
            .WithArg(3, total)
            .WithGroupSize(LocalSize);
        device.Launch(_kernel, ComputeDevice.GroupCount(total, LocalSize));
    }

    /// <summary>Disposes the underlying kernel handle.</summary>
    public void Dispose() => _kernel.Dispose();
}
