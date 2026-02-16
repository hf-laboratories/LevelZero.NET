using LevelZero.Native;

namespace LevelZero;

/// <summary>
/// Wraps a SPIR-V module loaded onto a device. Create kernels from this module.
/// </summary>
public sealed class ComputeModule : IDisposable
{
    private IntPtr _handle;
    private readonly IntPtr _contextHandle;
    private bool _disposed;

    /// <summary>Build log from SPIR-V module compilation (may be empty on success).</summary>
    public string BuildLog { get; }

    internal IntPtr Handle => !_disposed ? _handle : throw new ObjectDisposedException(nameof(ComputeModule));

    internal ComputeModule(IntPtr handle, IntPtr contextHandle, string buildLog)
    {
        _handle = handle;
        _contextHandle = contextHandle;
        BuildLog = buildLog;
    }

    /// <summary>
    /// Creates a kernel from this module by name.
    /// </summary>
    public ComputeKernel GetKernel(string name)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        LevelZeroNative.EnsureSuccess(
            LevelZeroNative.lz_kernel_create(_handle, name, out var kernel));
        return new ComputeKernel(kernel);
    }

    /// <summary>
    /// Tries to create a kernel by name. Returns null if the kernel is not found in the module.
    /// </summary>
    public ComputeKernel? TryGetKernel(string name)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        var result = LevelZeroNative.lz_kernel_create(_handle, name, out var kernel);
        return result == 0 ? new ComputeKernel(kernel) : null;
    }

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        if (_handle != IntPtr.Zero)
        {
            LevelZeroNative.lz_module_destroy(_handle);
            _handle = IntPtr.Zero;
        }
    }
}
