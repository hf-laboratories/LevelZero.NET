using System;
namespace LevelZero.Kernels;

/// <summary>
/// Int8 weight-only GEMV/GEMM using DP4A (4-way int8 dot product, native on Xe-LP and newer Intel GPUs).
/// Weights are quantized per output column in groups of <see cref="GroupSize"/> consecutive inputs: <c>wq</c> is
/// <c>[k / 4][n]</c> ints (four int8 weights of one output packed, lowest k in the low byte) and <c>ws</c> is
/// <c>[k / GroupSize][n]</c> half scales. Activations are quantized on the device with the same group size, so each
/// group's dot product is exact integer arithmetic and one float multiply-add applies both scales.
/// </summary>
public sealed class LlmGemvQ8Kernel : IDisposable
{
    /// <summary>Consecutive inputs sharing one quantization scale.</summary>
    public const int GroupSize = 32;

    /// <summary>Rows of activations processed per weight pass by the multi-row kernel.</summary>
    public const int MaxRows = LlmGemvKernel.MaxRows;

    private const uint GroupLanes = 8;
    private const uint GroupKSlices = 16;
    private const int ColumnsPerGroup = 32;
    private const uint QuantizeLocalSize = 32;

    private readonly ComputeKernel _quantize;
    private readonly ComputeKernel _gemv;
    private readonly ComputeKernel _rows;

    internal LlmGemvQ8Kernel(ComputeKernel quantize, ComputeKernel gemv, ComputeKernel rows)
    {
        _quantize = quantize;
        _gemv = gemv;
        _rows = rows;
    }

    /// <summary>True when a matrix of this shape can use the int8 kernels.</summary>
    public static bool Supports(int n, int k) => n > 0 && n % 4 == 0 && k > 0 && k % GroupSize == 0;

    /// <summary>
    /// Quantizes <paramref name="m"/> rows of <paramref name="x"/> ([m * k] floats) into <paramref name="activations"/>.
    /// </summary>
    public void Quantize(ComputeDevice device, int m, int k, SharedBuffer<float> x, LlmQ8Activations activations)
    {
        ArgumentNullException.ThrowIfNull(device);
        ArgumentNullException.ThrowIfNull(x);
        ArgumentNullException.ThrowIfNull(activations);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(m);
        if (k <= 0 || k % GroupSize != 0)
        {
            throw new ArgumentException($"k must be a positive multiple of {GroupSize}, got {k}.", nameof(k));
        }

        if ((long)m * k > x.Count)
        {
            throw new ArgumentException($"x holds {x.Count} elements but m * k is {(long)m * k}.", nameof(x));
        }

        activations.EnsureCapacity(m, k);
        int items = m * (k / GroupSize);
        _ = _quantize
            .WithArg(0, m)
            .WithArg(1, k)
            .WithArg(2, x)
            .WithArg(3, activations.Quants)
            .WithArg(4, activations.Scales)
            .WithGroupSize(QuantizeLocalSize);
        device.Launch(_quantize, ComputeDevice.GroupCount(items, QuantizeLocalSize));
    }

    /// <summary>
    /// <c>y = x * W</c> for one row already quantized into <paramref name="activations"/> (row 0).
    /// </summary>
    public void Execute(ComputeDevice device, int n, int k, LlmQ8Activations activations, SharedBuffer<int> wq, SharedBuffer<Half> ws, SharedBuffer<float> y)
    {
        Validate(device, 1, n, k, activations, wq, ws, y);
        _ = _gemv
            .WithArg(0, n)
            .WithArg(1, k)
            .WithArg(2, activations.Quants)
            .WithArg(3, activations.Scales)
            .WithArg(4, wq)
            .WithArg(5, ws)
            .WithArg(6, y)
            .WithGroupSize(GroupLanes, GroupKSlices);
        device.Launch(_gemv, ComputeDevice.GroupCount(n, ColumnsPerGroup));
    }

    /// <summary>
    /// <c>Y = X * W</c> for <paramref name="m"/> rows already quantized into <paramref name="activations"/> (prefill).
    /// </summary>
    public void ExecuteRows(ComputeDevice device, int m, int n, int k, LlmQ8Activations activations, SharedBuffer<int> wq, SharedBuffer<Half> ws, SharedBuffer<float> y)
    {
        Validate(device, m, n, k, activations, wq, ws, y);
        _ = _rows
            .WithArg(0, m)
            .WithArg(1, n)
            .WithArg(2, k)
            .WithArg(3, activations.Quants)
            .WithArg(4, activations.Scales)
            .WithArg(5, wq)
            .WithArg(6, ws)
            .WithArg(7, y)
            .WithGroupSize(GroupLanes, GroupKSlices);
        device.Launch(_rows, ComputeDevice.GroupCount(n, ColumnsPerGroup), ComputeDevice.GroupCount(m, MaxRows));
    }

    /// <summary>Disposes the underlying kernel handles.</summary>
    public void Dispose()
    {
        _quantize.Dispose();
        _gemv.Dispose();
        _rows.Dispose();
    }

    private static void Validate(ComputeDevice device, int m, int n, int k, LlmQ8Activations activations, SharedBuffer<int> wq, SharedBuffer<Half> ws, SharedBuffer<float> y)
    {
        ArgumentNullException.ThrowIfNull(device);
        ArgumentNullException.ThrowIfNull(activations);
        ArgumentNullException.ThrowIfNull(wq);
        ArgumentNullException.ThrowIfNull(ws);
        ArgumentNullException.ThrowIfNull(y);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(m);
        if (!Supports(n, k))
        {
            throw new NotSupportedException($"Int8 kernels need n % 4 == 0 and k % {GroupSize} == 0, got n={n}, k={k}.");
        }

        if (activations.RowCapacity < m || activations.KCapacity < k)
        {
            throw new ArgumentException("The activation scratch is smaller than the projection input.", nameof(activations));
        }

        if ((long)n * k / 4 > wq.Count || (long)n * (k / GroupSize) > ws.Count)
        {
            throw new ArgumentException("The quantized weight buffers are smaller than the matrix shape.");
        }

        if ((long)m * n > y.Count)
        {
            throw new ArgumentException($"y holds {y.Count} elements but m * n is {(long)m * n}.", nameof(y));
        }
    }
}

/// <summary>
/// Device scratch for dynamically quantized activations: packed int8 quads and one float scale per group. Sized once
/// for the largest projection input; it must not be reallocated while command lists that reference it are recorded.
/// </summary>
public sealed class LlmQ8Activations : IDisposable
{
    /// <summary>Allocates scratch for up to <paramref name="rows"/> rows of up to <paramref name="k"/> inputs.</summary>
    public LlmQ8Activations(ComputeDevice device, int rows, int k)
    {
        ArgumentNullException.ThrowIfNull(device);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(rows);
        if (k <= 0 || k % LlmGemvQ8Kernel.GroupSize != 0)
        {
            throw new ArgumentException($"k must be a positive multiple of {LlmGemvQ8Kernel.GroupSize}.", nameof(k));
        }

        RowCapacity = rows;
        KCapacity = k;
        Quants = device.AllocShared<int>(rows * (k / 4));
        Scales = device.AllocShared<float>(rows * (k / LlmGemvQ8Kernel.GroupSize));
    }

    /// <summary>Maximum number of rows.</summary>
    public int RowCapacity { get; }

    /// <summary>Maximum number of inputs per row.</summary>
    public int KCapacity { get; }

    /// <summary>Packed int8 activations, <c>[rows][k / 4]</c>.</summary>
    public SharedBuffer<int> Quants { get; }

    /// <summary>Per-group scales, <c>[rows][k / GroupSize]</c>.</summary>
    public SharedBuffer<float> Scales { get; }

    internal void EnsureCapacity(int rows, int k)
    {
        if (rows > RowCapacity || k > KCapacity)
        {
            throw new InvalidOperationException($"Activation scratch holds {RowCapacity} rows x {KCapacity} inputs but {rows} x {k} was requested.");
        }
    }

    /// <inheritdoc />
    public void Dispose()
    {
        Quants.Dispose();
        Scales.Dispose();
    }
}
