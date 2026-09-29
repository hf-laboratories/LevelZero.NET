using System;
using System.Runtime.InteropServices;

namespace LevelZero.Native;

/// <summary>
/// P/Invoke declarations for Intel Level Zero System Management (Sysman) APIs to query and lock GPU clock frequency.
/// </summary>
internal static class L0SysmanInterop
{
    private const string DllName = "ze_loader";

    [StructLayout(LayoutKind.Sequential)]
    public struct zes_freq_range_t
    {
        public double Min;
        public double Max;
    }

    [DllImport(DllName, EntryPoint = "zesDeviceEnumFrequencyDomains", CallingConvention = CallingConvention.StdCall)]
    public static extern int zesDeviceEnumFrequencyDomains(IntPtr hDevice, ref uint pCount, IntPtr phFrequency);

    [DllImport(DllName, EntryPoint = "zesDeviceEnumFrequencyDomains", CallingConvention = CallingConvention.StdCall)]
    public static extern int zesDeviceEnumFrequencyDomains(IntPtr hDevice, ref uint pCount, [Out] IntPtr[] phFrequency);

    [DllImport(DllName, EntryPoint = "zesFrequencyGetRange", CallingConvention = CallingConvention.StdCall)]
    public static extern int zesFrequencyGetRange(IntPtr hFrequency, ref zes_freq_range_t pLimits);

    [DllImport(DllName, EntryPoint = "zesFrequencySetRange", CallingConvention = CallingConvention.StdCall)]
    public static extern int zesFrequencySetRange(IntPtr hFrequency, ref zes_freq_range_t pLimits);
}
