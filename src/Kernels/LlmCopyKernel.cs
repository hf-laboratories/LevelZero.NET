using System;
namespace LevelZero.Kernels;
/// <summary>Copies a slice of one device buffer into another: <c>dst[i] = src[srcOffset + i]</c>.</summary>
public sealed class LlmCopyKernel : IDisposable
{
    private const uint LocalSize = 64;
    private readonly ComputeKernel _kernel;
    internal LlmCopyKernel(ComputeKernel kernel)
    {
        _kernel = kernel;
    }
    /// <summary>Copies <paramref name="count"/> floats starting at <paramref name="srcOffset"/> into <paramref name="dst"/>.</summary>
    public void Execute(ComputeDevice device, SharedBuffer<float> src, SharedBuffer<float> dst, int srcOffset, int count)
    {
        ArgumentNullException.ThrowIfNull(device);
        ArgumentNullException.ThrowIfNull(src);
        ArgumentNullException.ThrowIfNull(dst);
        ArgumentOutOfRangeException.ThrowIfNegative(srcOffset);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(count);
        if ((long)srcOffset + count > src.Count)
        {
            throw new ArgumentException($"Slice {srcOffset}..{srcOffset + count} is outside src ({src.Count} elements).", nameof(src));
        }
        if (count > dst.Count)
        {
            throw new ArgumentException($"dst holds {dst.Count} elements but count is {count}.", nameof(dst));
        }
        _ = _kernel
            .WithArg(0, src)
            .WithArg(1, dst)
            .WithArg(2, srcOffset)
            .WithArg(3, count)
            .WithGroupSize(LocalSize);
        device.Launch(_kernel, ComputeDevice.GroupCount(count, LocalSize));
    }
    /// <summary>Disposes the underlying kernel handle.</summary>
    public void Dispose() => _kernel.Dispose();
}