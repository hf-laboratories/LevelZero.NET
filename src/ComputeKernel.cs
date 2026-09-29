using System.Runtime.InteropServices;
using LevelZero.Native;

namespace LevelZero;

/// <summary>
/// Wraps a Level Zero kernel handle. Provides typed argument setters and group-size configuration.
/// </summary>
public sealed class ComputeKernel : IDisposable
{
    private IntPtr _handle;
    private bool _disposed;

    internal IntPtr Handle => !_disposed ? _handle : throw new ObjectDisposedException(nameof(ComputeKernel));

    internal ComputeKernel(IntPtr handle)
    {
        _handle = handle;
    }

    /// <summary>Sets the work-group size for this kernel (local_size in OpenCL terms).</summary>
    public void SetGroupSize(uint x, uint y = 1, uint z = 1)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        LevelZeroNative.EnsureSuccess(
            LevelZeroNative.lz_kernel_set_group_size(_handle, x, y, z));
    }

    /// <summary>Binds a USM memory pointer to a kernel argument slot.</summary>
    public void SetArgBuffer(uint index, IntPtr ptr)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        LevelZeroNative.EnsureSuccess(
            LevelZeroNative.lz_kernel_set_arg_mem(_handle, index, ptr));
    }

    /// <summary>Binds a SharedBuffer to a kernel argument slot.</summary>
    public void SetArgBuffer<T>(uint index, SharedBuffer<T> buffer) where T : unmanaged
    {
        SetArgBuffer(index, buffer.Pointer);
    }

    /// <summary>Sets a scalar int kernel argument with zero heap allocations.</summary>
    public void SetArgInt(uint index, int value)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        unsafe
        {
            LevelZeroNative.EnsureSuccess(
                LevelZeroNative.lz_kernel_set_arg_value(_handle, index, (UIntPtr)sizeof(int), (nint)(&value)));
        }
    }

    /// <summary>Sets a scalar float kernel argument with zero heap allocations.</summary>
    public void SetArgFloat(uint index, float value)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        unsafe
        {
            LevelZeroNative.EnsureSuccess(
                LevelZeroNative.lz_kernel_set_arg_value(_handle, index, (UIntPtr)sizeof(float), (nint)(&value)));
        }
    }

    /// <summary>Sets a scalar uint kernel argument with zero heap allocations.</summary>
    public void SetArgUInt(uint index, uint value)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        unsafe
        {
            LevelZeroNative.EnsureSuccess(
                LevelZeroNative.lz_kernel_set_arg_value(_handle, index, (UIntPtr)sizeof(uint), (nint)(&value)));
        }
    }

    /// <summary>Sets a scalar double kernel argument with zero heap allocations.</summary>
    public void SetArgDouble(uint index, double value)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        unsafe
        {
            LevelZeroNative.EnsureSuccess(
                LevelZeroNative.lz_kernel_set_arg_value(_handle, index, (UIntPtr)sizeof(double), (nint)(&value)));
        }
    }

    public void Dispose()
    {
        if (_disposed)
        {
            return;
        }

        _disposed = true;
        if (_handle != IntPtr.Zero)
        {
            _ = LevelZeroNative.lz_kernel_destroy(_handle);
            _handle = IntPtr.Zero;
        }
    }
}

