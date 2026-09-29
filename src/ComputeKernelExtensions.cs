using System;

namespace LevelZero;

/// <summary>
/// Fluent extension methods for setting arguments and configuring ComputeKernel parameters cleanly.
/// </summary>
public static class ComputeKernelExtensions
{
    /// <summary>Sets a scalar int argument and returns the kernel instance.</summary>
    public static ComputeKernel WithArg(this ComputeKernel kernel, uint index, int value)
    {
        ArgumentNullException.ThrowIfNull(kernel);
        kernel.SetArgInt(index, value);
        return kernel;
    }

    /// <summary>Sets a scalar float argument and returns the kernel instance.</summary>
    public static ComputeKernel WithArg(this ComputeKernel kernel, uint index, float value)
    {
        ArgumentNullException.ThrowIfNull(kernel);
        kernel.SetArgFloat(index, value);
        return kernel;
    }

    /// <summary>Sets a scalar uint argument and returns the kernel instance.</summary>
    public static ComputeKernel WithArg(this ComputeKernel kernel, uint index, uint value)
    {
        ArgumentNullException.ThrowIfNull(kernel);
        kernel.SetArgUInt(index, value);
        return kernel;
    }

    /// <summary>Sets a SharedBuffer argument and returns the kernel instance.</summary>
    public static ComputeKernel WithArg<T>(this ComputeKernel kernel, uint index, SharedBuffer<T> buffer) where T : unmanaged
    {
        ArgumentNullException.ThrowIfNull(kernel);
        ArgumentNullException.ThrowIfNull(buffer);
        kernel.SetArgBuffer(index, buffer);
        return kernel;
    }

    /// <summary>Sets a raw memory pointer argument and returns the kernel instance.</summary>
    public static ComputeKernel WithArg(this ComputeKernel kernel, uint index, IntPtr ptr)
    {
        ArgumentNullException.ThrowIfNull(kernel);
        kernel.SetArgBuffer(index, ptr);
        return kernel;
    }

    /// <summary>Sets a local/shared memory argument of a specific byte size and returns the kernel instance.</summary>
    public static ComputeKernel WithLocalArg(this ComputeKernel kernel, uint index, int sizeInBytes)
    {
        ArgumentNullException.ThrowIfNull(kernel);
        LevelZero.Native.LevelZeroNative.EnsureSuccess(
            LevelZero.Native.LevelZeroNative.lz_kernel_set_arg_value(kernel.Handle, index, (UIntPtr)sizeInBytes, IntPtr.Zero));
        return kernel;
    }

    /// <summary>Sets local work group size and returns the kernel instance.</summary>
    public static ComputeKernel WithGroupSize(this ComputeKernel kernel, uint x, uint y = 1, uint z = 1)
    {
        ArgumentNullException.ThrowIfNull(kernel);
        kernel.SetGroupSize(x, y, z);
        return kernel;
    }
}
