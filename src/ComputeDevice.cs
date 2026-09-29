using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Text;
using LevelZero.Native;

namespace LevelZero;

/// <summary>
/// Represents a Level Zero compute device with its context, command queue, and command list.
/// This is the primary object for GPU interaction  load modules, allocate memory, and launch kernels.
/// </summary>
public sealed class ComputeDevice : IDisposable
{
    private IntPtr _driver;
    private IntPtr _device;
    private IntPtr _context;
    private IntPtr _queue;
    private IntPtr _commandList;
    private bool _disposed;
    private bool _recording;

    /// <summary>Human-readable device name (e.g. "Intel(R) UHD Graphics 770").</summary>
    public string Name { get; }

    /// <summary>Driver index this device belongs to.</summary>
    public uint DriverIndex { get; }

    /// <summary>Device index within the driver.</summary>
    public uint DeviceIndex { get; }

    internal IntPtr ContextHandle => !_disposed ? _context : throw new ObjectDisposedException(nameof(ComputeDevice));
    public IntPtr DeviceHandle => !_disposed ? _device : throw new ObjectDisposedException(nameof(ComputeDevice));

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
        LevelZeroNative.EnsureSuccess(LevelZeroNative.lz_get_driver_handle(driverIndex, out nint driver));
        LevelZeroNative.EnsureSuccess(LevelZeroNative.lz_get_device_handle(driverIndex, deviceIndex, out nint device));
        LevelZeroNative.EnsureSuccess(LevelZeroNative.lz_context_create(driver, out nint context));
        LevelZeroNative.EnsureSuccess(LevelZeroNative.lz_command_queue_create(context, device, out nint queue));
        LevelZeroNative.EnsureSuccess(LevelZeroNative.lz_command_list_create(context, device, out nint commandList));

        var nameBuf = new StringBuilder(256);
        _ = LevelZeroNative.lz_get_device_name(driverIndex, deviceIndex, nameBuf, (uint)nameBuf.Capacity);
        string name = nameBuf.ToString();

        return new ComputeDevice(driver, device, context, queue, commandList, name, driverIndex, deviceIndex);
    }

    /// <summary>
    /// Loads a module from a file path.
    /// <c>.spv</c> is treated as SPIR-V, while <c>.bin</c>, <c>.native</c>, and <c>.zebin</c>
    /// are treated as native Level Zero binaries.
    /// </summary>
    public ComputeModule LoadModule(string spirvPath)
    {
        return LoadModule(spirvPath, InferModuleFormat(spirvPath));
    }

    /// <summary>
    /// Loads a module from a file path using the specified module format.
    /// </summary>
    public ComputeModule LoadModule(string modulePath, LevelZeroModuleFormat moduleFormat)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        byte[] moduleBytes = File.ReadAllBytes(modulePath);
        return LoadModule(moduleBytes, moduleFormat);
    }

    /// <summary>
    /// Loads a SPIR-V module from a byte array.
    /// </summary>
    public ComputeModule LoadModule(byte[] spirv)
        => LoadModule(spirv, LevelZeroModuleFormat.SpirV);

    /// <summary>
    /// Loads a module from a byte array using the specified module format.
    /// <summary>
    /// Loads a kernel module from bytes using the specified format.
    /// </summary>
    public ComputeModule LoadModule(byte[] moduleBytes, LevelZeroModuleFormat moduleFormat)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        var detectedFormat = DetectModuleFormat(moduleBytes, moduleFormat);
        if (!IsValidModuleFormat(moduleBytes, detectedFormat))
        {
            throw new LevelZeroException(-1, $"Invalid module format or magic bytes for {detectedFormat}.");
        }

        var logBuf = new StringBuilder(262144);
        int result = LevelZeroNative.lz_module_create(_context, _device, moduleBytes, (uint)moduleBytes.Length, (uint)detectedFormat,
            out nint module, logBuf, (uint)logBuf.Capacity);
        string buildLog = logBuf.ToString();
        return result != 0
            ? throw new LevelZeroException(result,
                BuildModuleLoadErrorMessage(detectedFormat, buildLog))
            : new ComputeModule(module, _context, buildLog);
    }

    /// <summary>
    /// Tries to load a module from a file path. Returns null on failure instead of throwing.
    /// </summary>
    public ComputeModule? TryLoadModule(string spirvPath, out string buildLog)
        => TryLoadModule(spirvPath, InferModuleFormat(spirvPath), out buildLog);

    /// <summary>
    /// Tries to load a module from a file path using the specified module format.
    /// Returns null on failure instead of throwing.
    /// </summary>
    public ComputeModule? TryLoadModule(string modulePath, LevelZeroModuleFormat moduleFormat, out string buildLog)
    {
        buildLog = string.Empty;
        if (string.IsNullOrWhiteSpace(modulePath) || !File.Exists(modulePath))
        {
            return null;
        }

        byte[] moduleBytes = File.ReadAllBytes(modulePath);
        return TryLoadModule(moduleBytes, moduleFormat, out buildLog);
    }

    /// <summary>
    /// Tries to load a SPIR-V module from bytes. Returns null on failure.
    /// </summary>
    public ComputeModule? TryLoadModule(byte[] spirv, out string buildLog)
        => TryLoadModule(spirv, LevelZeroModuleFormat.SpirV, out buildLog);

    /// <summary>
    /// Tries to load a module from bytes using the specified format. Returns null on failure.
    /// </summary>
    public ComputeModule? TryLoadModule(byte[] moduleBytes, LevelZeroModuleFormat moduleFormat, out string buildLog)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        buildLog = string.Empty;
        if (moduleBytes == null || moduleBytes.Length < 4)
        {
            buildLog = "Module bytes are empty or less than 4 bytes.";
            return null;
        }

        var detectedFormat = DetectModuleFormat(moduleBytes, moduleFormat);
        if (!IsValidModuleFormat(moduleBytes, detectedFormat))
        {
            buildLog = $"Invalid module format or magic bytes for {detectedFormat}.";
            return null;
        }

        var logBuf = new StringBuilder(262144);
        int result = LevelZeroNative.lz_module_create(_context, _device, moduleBytes, (uint)moduleBytes.Length, (uint)detectedFormat,
            out nint module, logBuf, (uint)logBuf.Capacity);
        buildLog = logBuf.ToString();
        return result == 0 ? new ComputeModule(module, _context, buildLog) : null;
    }

    private static LevelZeroModuleFormat DetectModuleFormat(byte[]? moduleBytes, LevelZeroModuleFormat fallback)
    {
        if (moduleBytes != null && moduleBytes.Length >= 4)
        {
            uint magic = BitConverter.ToUInt32(moduleBytes, 0);
            if (magic == 0x07230203 || magic == 0x03022307)
            {
                return LevelZeroModuleFormat.SpirV;
            }
            if (magic == 0x464C457F) // ELF magic "\x7FELF"
            {
                return LevelZeroModuleFormat.Native;
            }
        }
        return fallback;
    }

    private static bool IsValidModuleFormat(byte[]? moduleBytes, LevelZeroModuleFormat moduleFormat)
    {
        if (moduleBytes == null || moduleBytes.Length < 4)
        {
            return false;
        }

        uint magic = BitConverter.ToUInt32(moduleBytes, 0);

        if (moduleFormat == LevelZeroModuleFormat.SpirV)
        {
            // SPIR-V magic number is 0x07230203 (or byte-swapped 0x03022307)
            return magic == 0x07230203 || magic == 0x03022307;
        }

        if (moduleFormat == LevelZeroModuleFormat.Native)
        {
            // Native format must be an ELF binary (0x464C457F / "\x7FELF")
            return magic == 0x464C457F;
        }

        return true;
    }

    /// <summary>
    /// Allocates a typed USM shared-memory buffer accessible from both host and device.
    /// </summary>
    public SharedBuffer<T> AllocShared<T>(int count) where T : unmanaged
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        nuint byteSize = (UIntPtr)(count * Unsafe.SizeOf<T>());
        LevelZeroNative.EnsureSuccess(
            LevelZeroNative.lz_usm_alloc_shared(_context, _device, byteSize, UIntPtr.Zero, out nint ptr));
        return new SharedBuffer<T>(_context, ptr, count);
    }

    /// <summary>
    /// Allocates a shared buffer and immediately fills it from a source array.
    /// </summary>
    public SharedBuffer<T> AllocShared<T>(T[] data) where T : unmanaged
    {
        SharedBuffer<T> buffer = AllocShared<T>(data.Length);
        buffer.Write(data);
        return buffer;
    }

    /// <summary>
    /// Enqueues a kernel launch into the command stream without blocking the host CPU.
    /// Call <see cref="Synchronize"/> when execution results or barriers are required.
    /// </summary>
    public void EnqueueLaunch(ComputeKernel kernel, uint groupCountX, uint groupCountY = 1, uint groupCountZ = 1)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        LevelZeroNative.EnsureSuccess(LevelZeroNative.lz_command_list_reset(_commandList));
        LevelZeroNative.EnsureSuccess(
            LevelZeroNative.lz_command_list_append_launch_kernel(_commandList, kernel.Handle,
                groupCountX, groupCountY, groupCountZ));
        LevelZeroNative.EnsureSuccess(LevelZeroNative.lz_command_list_close(_commandList));
        LevelZeroNative.EnsureSuccess(LevelZeroNative.lz_command_queue_execute(_queue, _commandList));
    }

    /// <summary>
    /// Synchronizes the command queue, blocking until all enqueued kernel launches complete.
    /// </summary>
    public void Synchronize(ulong timeoutNs = ulong.MaxValue)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        LevelZeroNative.EnsureSuccess(LevelZeroNative.lz_command_queue_synchronize(_queue, timeoutNs));
    }

    /// <summary>
    /// Launches a kernel with the given work-group counts across up to 3 dimensions,
    /// then synchronizes (blocks until complete).
    /// </summary>
    /// <remarks>
    /// While a recording is open (<see cref="BeginRecording"/>) the launch is only appended to the open command
    /// list, followed by a barrier so later launches see its results; nothing runs until <see cref="EndRecording"/>.
    /// </remarks>
    public void Launch(ComputeKernel kernel, uint groupCountX, uint groupCountY = 1, uint groupCountZ = 1)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        if (_recording)
        {
            LevelZeroNative.EnsureSuccess(
                LevelZeroNative.lz_command_list_append_launch_kernel(_commandList, kernel.Handle,
                    groupCountX, groupCountY, groupCountZ));
            LevelZeroNative.EnsureSuccess(LevelZeroNative.lz_command_list_append_barrier(_commandList));
            return;
        }
        EnqueueLaunch(kernel, groupCountX, groupCountY, groupCountZ);
        Synchronize();
    }
    /// <summary>True between <see cref="BeginRecording"/> and <see cref="EndRecording"/>/<see cref="AbortRecording"/>.</summary>
    public bool IsRecording => _recording;
    /// <summary>
    /// Starts collecting <see cref="Launch"/> calls into one command list so a whole forward pass costs a single
    /// submission and a single wait instead of one per kernel. Kernel arguments are captured when each launch is
    /// appended, so the same kernel object can be re-used with different arguments. Host code must not read
    /// results, or rely on device writes, until <see cref="EndRecording"/> returns.
    /// </summary>
    /// <exception cref="InvalidOperationException">A recording is already open.</exception>
    public void BeginRecording()
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        if (_recording)
        {
            throw new InvalidOperationException("A recording is already open.");
        }
        LevelZeroNative.EnsureSuccess(LevelZeroNative.lz_command_list_reset(_commandList));
        _recording = true;
    }
    /// <summary>Submits everything recorded since <see cref="BeginRecording"/> and blocks until it has finished.</summary>
    /// <exception cref="InvalidOperationException">No recording is open.</exception>
    public void EndRecording()
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        if (!_recording)
        {
            throw new InvalidOperationException("No recording is open.");
        }
        _recording = false;
        LevelZeroNative.EnsureSuccess(LevelZeroNative.lz_command_list_close(_commandList));
        LevelZeroNative.EnsureSuccess(LevelZeroNative.lz_command_queue_execute(_queue, _commandList));
        Synchronize();
    }
    /// <summary>Discards an open recording without running it. Safe to call when none is open.</summary>
    public void AbortRecording()
    {
        if (!_recording || _disposed)
        {
            return;
        }
        _recording = false;
        _ = LevelZeroNative.lz_command_list_reset(_commandList);
    }

    /// <summary>Computes the number of work-groups needed to cover totalItems with the given local size.</summary>
    public static uint GroupCount(int totalItems, uint localSize) =>
        (uint)Math.Max(1, (totalItems + (int)localSize - 1) / (int)localSize);

    public void Dispose()
    {
        if (_disposed)
        {
            return;
        }

        _disposed = true;

        if (_commandList != IntPtr.Zero)
        {
            _ = LevelZeroNative.lz_command_list_destroy(_commandList);
        }

        if (_queue != IntPtr.Zero)
        {
            _ = LevelZeroNative.lz_command_queue_destroy(_queue);
        }

        if (_context != IntPtr.Zero)
        {
            _ = LevelZeroNative.lz_context_destroy(_context);
        }

        _commandList = _queue = _context = _device = _driver = IntPtr.Zero;
    }

    private static string BuildModuleLoadErrorMessage(LevelZeroModuleFormat moduleFormat, string buildLog)
        => string.IsNullOrWhiteSpace(buildLog)
            ? $"Module load failed for format {moduleFormat}."
            : $"Module load failed for format {moduleFormat}. Build log: {buildLog}";

    private static LevelZeroModuleFormat InferModuleFormat(string modulePath)
        => Path.GetExtension(modulePath).ToLowerInvariant() switch
        {
            ".bin" or ".native" or ".zebin" => LevelZeroModuleFormat.Native,
            _ => LevelZeroModuleFormat.SpirV,
        };
}

