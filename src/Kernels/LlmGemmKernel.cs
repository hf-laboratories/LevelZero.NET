using System;

namespace LevelZero.Kernels;

/// <summary>
/// GPU-accelerated Matrix Multiplication (GEMM/GEMV) kernel wrapper.
/// </summary>
public sealed class LlmGemmKernel : IDisposable
{
    private readonly ComputeKernel _kernel;

    internal LlmGemmKernel(ComputeKernel kernel)
    {
        _kernel = kernel;
    }

    /// <summary>
    /// Executes the matrix multiplication kernel C = A * B.
    /// </summary>
    /// <param name="device">The L0 compute device.</param>
    /// <param name="m">Height of matrix A and C.</param>
    /// <param name="n">Width of matrix B and C.</param>
    /// <param name="k">Width of matrix A, height of matrix B.</param>
    /// <param name="a">Flat matrix A [m * k].</param>
    /// <param name="b">Flat matrix B [k * n].</param>
    /// <param name="c">Flat matrix C [m * n].</param>
    public void Execute(
        ComputeDevice device,
        int m, int n, int k,
        SharedBuffer<float> a,
        SharedBuffer<float> b,
        SharedBuffer<float> c)
    {
        ArgumentNullException.ThrowIfNull(device);
        ArgumentNullException.ThrowIfNull(a);
        ArgumentNullException.ThrowIfNull(b);
        ArgumentNullException.ThrowIfNull(c);

        _ = _kernel
            .WithArg(0, m)
            .WithArg(1, n)
            .WithArg(2, k)
            .WithArg(3, a)
            .WithArg(4, b)
            .WithArg(5, c);

        // Grid sizing for simple 2D execution
        const uint tileDimX = 16;
        const uint tileDimY = 16;

        _ = _kernel.WithGroupSize(tileDimX, tileDimY);

        uint groupCountX = ComputeDevice.GroupCount(n, tileDimX);
        uint groupCountY = ComputeDevice.GroupCount(m, tileDimY);

        device.Launch(_kernel, groupCountX, groupCountY);
    }

    /// <summary>Disposes the underlying kernel handle.</summary>
    public void Dispose() => _kernel.Dispose();
}
