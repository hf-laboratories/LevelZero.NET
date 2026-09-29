using System;
namespace LevelZero.Kernels;
/// <summary>
/// Single-row matrix-vector product for decode: <c>y[n] = x[k] * W[k x n]</c> with <c>W</c> stored row-major as
/// input x output (the layout <see cref="LlmGemmKernel"/> uses). Work-items split the k dimension so narrow
/// outputs still fill the GPU, and the weights can be stored as IEEE half to halve memory traffic.
/// </summary>
public sealed class LlmGemvKernel : IDisposable
{
    /// <summary>Output columns handled by one work-group.</summary>
    private const int ColumnsPerGroup = 32;
    /// <summary>Rows of X that the multi-row kernels multiply per weight pass (the kernel's GEMM_ROWS).</summary>
    public const int MaxRows = 8;
    private const uint GroupLanes = 8;
    private const uint GroupKSlices = 32;
    private readonly ComputeKernel _kernelF32;
    private readonly ComputeKernel _kernelF16;
    private readonly ComputeKernel _rowsF32;
    private readonly ComputeKernel _rowsF16;
    internal LlmGemvKernel(ComputeKernel kernelF32, ComputeKernel kernelF16, ComputeKernel rowsF32, ComputeKernel rowsF16)
    {
        _kernelF32 = kernelF32;
        _kernelF16 = kernelF16;
        _rowsF32 = rowsF32;
        _rowsF16 = rowsF16;
    }
    /// <summary>True when <paramref name="n"/> outputs can use the GEMV kernels (a multiple of 4).</summary>
    public static bool Supports(int n) => n > 0 && n % 4 == 0;
    /// <summary><c>y = x * W</c> with float32 weights.</summary>
    /// <param name="device">The L0 compute device.</param>
    /// <param name="n">Number of outputs; a multiple of 4.</param>
    /// <param name="k">Number of inputs.</param>
    /// <param name="x">Input vector [k].</param>
    /// <param name="weights">Weight matrix [k * n], row-major.</param>
    /// <param name="y">Output vector [n].</param>
    public void Execute(ComputeDevice device, int n, int k, SharedBuffer<float> x, SharedBuffer<float> weights, SharedBuffer<float> y)
    {
        Validate(device, n, k, x, weights?.Count ?? 0, y);
        Launch(device, _kernelF32, n, k, x, weights!, y);
    }
    /// <summary><c>y = x * W</c> with half-precision weights (accumulated in float32).</summary>
    public void Execute(ComputeDevice device, int n, int k, SharedBuffer<float> x, SharedBuffer<Half> weights, SharedBuffer<float> y)
    {
        Validate(device, n, k, x, weights?.Count ?? 0, y);
        Launch(device, _kernelF16, n, k, x, weights!, y);
    }
    /// <summary>
    /// <c>Y = X * W</c> for <paramref name="m"/> rows (prefill), float32 weights. Each weight is loaded once per
    /// block of <see cref="MaxRows"/> rows, so a chunk of tokens streams the matrix once instead of once per token.
    /// </summary>
    /// <param name="device">The L0 compute device.</param>
    /// <param name="m">Number of rows of X and Y.</param>
    /// <param name="n">Number of outputs per row; a multiple of 4.</param>
    /// <param name="k">Number of inputs per row.</param>
    /// <param name="x">Input rows [m * k].</param>
    /// <param name="weights">Weight matrix [k * n], row-major.</param>
    /// <param name="y">Output rows [m * n].</param>
    public void ExecuteRows(ComputeDevice device, int m, int n, int k, SharedBuffer<float> x, SharedBuffer<float> weights, SharedBuffer<float> y)
    {
        ValidateRows(device, m, n, k, x, weights?.Count ?? 0, y);
        LaunchRows(device, _rowsF32, m, n, k, x, weights!, y);
    }
    /// <summary>Multi-row product with half-precision weights.</summary>
    public void ExecuteRows(ComputeDevice device, int m, int n, int k, SharedBuffer<float> x, SharedBuffer<Half> weights, SharedBuffer<float> y)
    {
        ValidateRows(device, m, n, k, x, weights?.Count ?? 0, y);
        LaunchRows(device, _rowsF16, m, n, k, x, weights!, y);
    }
    /// <summary>Disposes the underlying kernel handles.</summary>
    public void Dispose()
    {
        _kernelF32.Dispose();
        _kernelF16.Dispose();
        _rowsF32.Dispose();
        _rowsF16.Dispose();
    }
    private static void ValidateRows(ComputeDevice device, int m, int n, int k, SharedBuffer<float> x, int weightCount, SharedBuffer<float> y)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(m);
        Validate(device, n, k, x, weightCount, y);
        if ((long)m * k > x.Count)
        {
            throw new ArgumentException($"x holds {x.Count} elements but m * k is {(long)m * k}.", nameof(x));
        }
        if ((long)m * n > y.Count)
        {
            throw new ArgumentException($"y holds {y.Count} elements but m * n is {(long)m * n}.", nameof(y));
        }
    }
    private static void LaunchRows<T>(ComputeDevice device, ComputeKernel kernel, int m, int n, int k, SharedBuffer<float> x, SharedBuffer<T> w, SharedBuffer<float> y)
        where T : unmanaged
    {
        _ = kernel
            .WithArg(0, m)
            .WithArg(1, n)
            .WithArg(2, k)
            .WithArg(3, x)
            .WithArg(4, w)
            .WithArg(5, y)
            .WithGroupSize(GroupLanes, GroupKSlices);
        device.Launch(kernel, ComputeDevice.GroupCount(n, ColumnsPerGroup), ComputeDevice.GroupCount(m, MaxRows));
    }
    private static void Validate(ComputeDevice device, int n, int k, SharedBuffer<float> x, int weightCount, SharedBuffer<float> y)
    {
        ArgumentNullException.ThrowIfNull(device);
        ArgumentNullException.ThrowIfNull(x);
        ArgumentNullException.ThrowIfNull(y);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(k);
        if (!Supports(n))
        {
            throw new NotSupportedException($"GEMV needs an output width that is a multiple of 4, got {n}.");
        }
        if (x.Count < k)
        {
            throw new ArgumentException($"x holds {x.Count} elements but k is {k}.", nameof(x));
        }
        if (y.Count < n)
        {
            throw new ArgumentException($"y holds {y.Count} elements but n is {n}.", nameof(y));
        }
        if ((long)n * k > weightCount)
        {
            throw new ArgumentException($"Weights hold {weightCount} elements but k * n is {(long)n * k}.");
        }
    }
    private static void Launch<T>(ComputeDevice device, ComputeKernel kernel, int n, int k, SharedBuffer<float> x, SharedBuffer<T> w, SharedBuffer<float> y)
        where T : unmanaged
    {
        _ = kernel
            .WithArg(0, n)
            .WithArg(1, k)
            .WithArg(2, x)
            .WithArg(3, w)
            .WithArg(4, y)
            .WithGroupSize(GroupLanes, GroupKSlices);
        device.Launch(kernel, ComputeDevice.GroupCount(n, ColumnsPerGroup));
    }
}