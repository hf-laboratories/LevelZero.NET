using System.Runtime.InteropServices;
using System.Text;

namespace LevelZero.Native;

/// <summary>
/// P/Invoke declarations for the Level Zero C ABI shim (LevelZeroShim.dll, libLevelZeroShim.so).
/// Internal to the library — consumers use the managed API surface instead.
/// [Lib in C++]->[Shim in C]->[Managed wrapper in C#]->[Public API surface]
/// The shim handles all direct interactions with the Level Zero driver, including error management.
/// </summary>
internal static class LevelZeroNative
{
    private const string DllName = "LevelZeroShim";

    [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
    public static extern int lz_init(uint flags);

    [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
    public static extern int lz_get_driver_count(out uint count);

    [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
    public static extern int lz_get_driver_handle(uint driverIndex, out IntPtr driver);

    [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
    public static extern int lz_get_device_count(uint driverIndex, out uint count);

    [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
    public static extern int lz_get_device_name(uint driverIndex, uint deviceIndex, StringBuilder buffer, uint bufferSize);

    [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
    public static extern int lz_get_device_handle(uint driverIndex, uint deviceIndex, out IntPtr device);

    [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
    public static extern int lz_context_create(IntPtr driver, out IntPtr context);

    [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
    public static extern int lz_context_destroy(IntPtr context);

    [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
    public static extern int lz_command_queue_create(IntPtr context, IntPtr device, out IntPtr queue);

    [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
    public static extern int lz_command_queue_destroy(IntPtr queue);

    [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
    public static extern int lz_command_list_create(IntPtr context, IntPtr device, out IntPtr commandList);

    [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
    public static extern int lz_command_list_reset(IntPtr commandList);

    [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
    public static extern int lz_command_list_close(IntPtr commandList);

    [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
    public static extern int lz_command_list_destroy(IntPtr commandList);

    [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
    public static extern int lz_command_queue_execute(IntPtr queue, IntPtr commandList);

    [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
    public static extern int lz_command_queue_synchronize(IntPtr queue, ulong timeoutNs);

    [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
    public static extern int lz_module_create(IntPtr context, IntPtr device, byte[] spirv, uint size, out IntPtr module, StringBuilder buildLog, uint buildLogSize);

    [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
    public static extern int lz_module_destroy(IntPtr module);

    [DllImport(DllName, CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi)]
    public static extern int lz_kernel_create(IntPtr module, string name, out IntPtr kernel);

    [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
    public static extern int lz_kernel_destroy(IntPtr kernel);

    [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
    public static extern int lz_kernel_set_group_size(IntPtr kernel, uint groupX, uint groupY, uint groupZ);

    [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
    public static extern int lz_kernel_set_arg_value(IntPtr kernel, uint index, UIntPtr size, IntPtr value);

    [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
    public static extern int lz_kernel_set_arg_mem(IntPtr kernel, uint index, IntPtr ptr);

    [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
    public static extern int lz_command_list_append_launch_kernel(IntPtr commandList, IntPtr kernel, uint groupX, uint groupY, uint groupZ);

    [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
    public static extern int lz_command_list_append_barrier(IntPtr commandList);

    [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
    public static extern int lz_command_list_append_memory_copy(IntPtr commandList, IntPtr dst, IntPtr src, UIntPtr size);

    [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
    public static extern int lz_usm_alloc_shared(IntPtr context, IntPtr device, UIntPtr size, UIntPtr alignment, out IntPtr ptr);

    [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
    public static extern int lz_usm_free(IntPtr context, IntPtr ptr);

    [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
    private static extern IntPtr lz_get_last_error();

    [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
    public static extern int lz_get_last_result();

    [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
    public static extern void lz_clear_error();

    public static string GetLastError()
    {
        var ptr = lz_get_last_error();
        return ptr == IntPtr.Zero ? string.Empty : Marshal.PtrToStringAnsi(ptr) ?? string.Empty;
    }

    internal static void EnsureSuccess(int result)
    {
        if (result != 0)
            throw new LevelZeroException(result, GetLastError());
    }
}
