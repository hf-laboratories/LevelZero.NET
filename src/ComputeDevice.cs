using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Text;
using LevelZero.Native;

namespace LevelZero;

/// <summary>
/// Represents a Level Zero compute device with its context, command queue, and command list.
/// This is the primary object for GPU interaction — load modules, allocate memory, and launch kernels.
/// </summary>
public sealed class ComputeDevice : IDisposable
{
    private IntPtr _driver;
    private IntPtr _device;
    private IntPtr _context;
    private IntPtr _queue;
    private IntPtr _commandList;
    private bool _disposed;

    /// <summary>Human-readable device name (e.g. "Intel(R) UHD Graphics 770").</summary>
    public string Name { get; }

    /// <summary>Driver index this device belongs to.</summary>
    public uint DriverIndex { get; }

    /// <summary>Device index within the driver.</summary>
    public uint DeviceIndex { get; }

    internal IntPtr ContextHandle => !_disposed ? _context : throw new ObjectDisposedException(nameof(ComputeDevice));
    internal IntPtr DeviceHandle => !_disposed ? _device : throw new ObjectDisposedException(nameof(ComputeDevice));

    private ComputeDevice(IntPtr driver, IntPtr device, IntPtr context, IntPtr queue, IntPtr commandList,
                          string name, uint driverIndex, uint deviceIndex)
    {
        _driver = driver;
        _device = device;
        _context = context;
        _queue = queue;
        _commandList = commandList;
        Name = name;
        DriverIndex = driverIndex;
        DeviceIndex = deviceIndex;
    }

    internal static ComputeDevice Create(uint driverIndex, uint deviceIndex)
    {
        LevelZeroNative.EnsureSuccess(LevelZeroNative.lz_init(0));
        LevelZeroNative.EnsureSuccess(LevelZeroNative.lz_get_driver_handle(driverIndex, out var driver));
        LevelZeroNative.EnsureSuccess(LevelZeroNative.lz_get_device_handle(driverIndex, deviceIndex, out var device));
        LevelZeroNative.EnsureSuccess(LevelZeroNative.lz_context_create(driver, out var context));
        LevelZeroNative.EnsureSuccess(LevelZeroNative.lz_command_queue_create(context, device, out var queue));
        LevelZeroNative.EnsureSuccess(LevelZeroNative.lz_command_list_create(context, device, out var commandList));

        var nameBuf = new StringBuilder(256);
        LevelZeroNative.lz_get_device_name(driverIndex, deviceIndex, nameBuf, (uint)nameBuf.Capacity);
        var name = nameBuf.ToString();

        return new ComputeDevice(driver, device, context, queue, commandList, name, driverIndex, deviceIndex);
    }

    /// <summary>
    /// Loads a SPIR-V module from a file path.
    /// </summary>
    public ComputeModule LoadModule(string spirvPath)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        var spirv = File.ReadAllBytes(spirvPath);
        return LoadModule(spirv);
    }

    /// <summary>
    /// Loads a SPIR-V module from a byte array.
    /// </summary>
    public ComputeModule LoadModule(byte[] spirv)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        var logBuf = new StringBuilder(4096);
        var result = LevelZeroNative.lz_module_create(_context, _device, spirv, (uint)spirv.Length,
            out var module, logBuf, (uint)logBuf.Capacity);
        var buildLog = logBuf.ToString();
        if (result != 0)
            throw new LevelZeroException(result,
                $"Module compilation failed. Build log: {buildLog}");
        return new ComputeModule(module, _context, buildLog);
    }

    /// <summary>
    /// Tries to load a SPIR-V module. Returns null on failure instead of throwing.
    /// </summary>
    public ComputeModule? TryLoadModule(string spirvPath, out string buildLog)
    {
        buildLog = string.Empty;
        if (string.IsNullOrWhiteSpace(spirvPath) || !File.Exists(spirvPath))
            return null;

        var spirv = File.ReadAllBytes(spirvPath);
        return TryLoadModule(spirv, out buildLog);
    }

    /// <summary>
    /// Tries to load a SPIR-V module from bytes. Returns null on failure.
    /// </summary>
    public ComputeModule? TryLoadModule(byte[] spirv, out string buildLog)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        var logBuf = new StringBuilder(4096);
        var result = LevelZeroNative.lz_module_create(_context, _device, spirv, (uint)spirv.Length,
            out var module, logBuf, (uint)logBuf.Capacity);
        buildLog = logBuf.ToString();
        return result == 0 ? new ComputeModule(module, _context, buildLog) : null;
    }

    /// <summary>
    /// Allocates a typed USM shared-memory buffer accessible from both host and device.
    /// </summary>
    public SharedBuffer<T> AllocShared<T>(int count) where T : unmanaged
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        var byteSize = (UIntPtr)(count * Unsafe.SizeOf<T>());
        LevelZeroNative.EnsureSuccess(
            LevelZeroNative.lz_usm_alloc_shared(_context, _device, byteSize, UIntPtr.Zero, out var ptr));
        return new SharedBuffer<T>(_context, ptr, count);
    }

    /// <summary>
    /// Allocates a shared buffer and immediately fills it from a source array.
    /// </summary>
    public SharedBuffer<T> AllocShared<T>(T[] data) where T : unmanaged
    {
        var buffer = AllocShared<T>(data.Length);
        buffer.Write(data);
        return buffer;
    }

    /// <summary>
    /// Launches a kernel with the given work-group counts across up to 3 dimensions,
    /// then synchronizes (blocks until complete).
    /// </summary>
    public void Launch(ComputeKernel kernel, uint groupCountX, uint groupCountY = 1, uint groupCountZ = 1)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        LevelZeroNative.EnsureSuccess(LevelZeroNative.lz_command_list_reset(_commandList));
        LevelZeroNative.EnsureSuccess(
            LevelZeroNative.lz_command_list_append_launch_kernel(_commandList, kernel.Handle,
                groupCountX, groupCountY, groupCountZ));
        LevelZeroNative.EnsureSuccess(LevelZeroNative.lz_command_list_close(_commandList));
        LevelZeroNative.EnsureSuccess(LevelZeroNative.lz_command_queue_execute(_queue, _commandList));
        LevelZeroNative.EnsureSuccess(LevelZeroNative.lz_command_queue_synchronize(_queue, ulong.MaxValue));
    }

    /// <summary>Computes the number of work-groups needed to cover totalItems with the given local size.</summary>
    public static uint GroupCount(int totalItems, uint localSize) =>
        (uint)Math.Max(1, (totalItems + (int)localSize - 1) / (int)localSize);

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;

        if (_commandList != IntPtr.Zero)
            LevelZeroNative.lz_command_list_destroy(_commandList);
        if (_queue != IntPtr.Zero)
            LevelZeroNative.lz_command_queue_destroy(_queue);
        if (_context != IntPtr.Zero)
            LevelZeroNative.lz_context_destroy(_context);

        _commandList = _queue = _context = _device = _driver = IntPtr.Zero;
    }
}
