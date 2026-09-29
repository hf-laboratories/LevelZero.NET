using LevelZero.Native;

namespace LevelZero;

/// <summary>
/// Additive abstraction over Level Zero device discovery/open operations.
/// This keeps the shared L0.NET substrate reusable without forcing all callers
/// to depend directly on the static <see cref="LevelZeroRuntime"/> entry points.
/// </summary>
public interface ILevelZeroDeviceRuntime
{
    /// <summary>
    /// Checks whether the Level Zero runtime is available on this system.
    /// </summary>
    bool IsAvailable();

    /// <summary>
    /// Enumerates all available Level Zero devices across all drivers.
    /// </summary>
    IReadOnlyList<DeviceInfo> EnumerateDevices();

    /// <summary>
    /// Opens the default compute device.
    /// </summary>
    ComputeDevice GetDefaultDevice();

    /// <summary>
    /// Opens a compute device at the specified driver/device indices.
    /// </summary>
    ComputeDevice GetDevice(uint driverIndex, uint deviceIndex);
}

/// <summary>
/// Default adapter that forwards <see cref="ILevelZeroDeviceRuntime"/> calls to
/// the static <see cref="LevelZeroRuntime"/> helpers.
/// </summary>
public sealed class DefaultLevelZeroDeviceRuntime : ILevelZeroDeviceRuntime
{
    /// <inheritdoc />
    public bool IsAvailable() => LevelZeroRuntime.IsAvailable();

    /// <inheritdoc />
    public IReadOnlyList<DeviceInfo> EnumerateDevices() => LevelZeroRuntime.EnumerateDevices();

    /// <inheritdoc />
    public ComputeDevice GetDefaultDevice() => LevelZeroRuntime.GetDefaultDevice();

    /// <inheritdoc />
    public ComputeDevice GetDevice(uint driverIndex, uint deviceIndex) => LevelZeroRuntime.GetDevice(driverIndex, deviceIndex);
    // Null contract: no string? reference fields; all properties are value-type or
    // collection-initialized references with non-null? defaults.
}

/// <summary>
/// Entry point for the LevelZero.NET library.
/// Provides device discovery and a quick way to get a default compute device.
/// </summary>
public static class LevelZeroRuntime
{
    /// <summary>
    /// Returns the default compute device (first device on the first driver).
    /// This is the simplest way to get started.
    /// </summary>
    public static ComputeDevice GetDefaultDevice() => ComputeDevice.Create(0, 0);

    /// <summary>
    /// Returns a compute device at the specified driver/device indices.
    /// </summary>
    public static ComputeDevice GetDevice(uint driverIndex, uint deviceIndex) =>
        ComputeDevice.Create(driverIndex, deviceIndex);

    /// <summary>
    /// Enumerates all available Level Zero devices across all drivers.
    /// </summary>
    public static IReadOnlyList<DeviceInfo> EnumerateDevices()
    {
        LevelZeroNative.EnsureSuccess(LevelZeroNative.lz_init(0));
        LevelZeroNative.EnsureSuccess(LevelZeroNative.lz_get_driver_count(out uint driverCount));

        var devices = new List<DeviceInfo>();
        for (uint d = 0; d < driverCount; d++)
        {
            LevelZeroNative.EnsureSuccess(LevelZeroNative.lz_get_device_count(d, out uint deviceCount));
            for (uint i = 0; i < deviceCount; i++)
            {
                var name = new System.Text.StringBuilder(256);
                _ = LevelZeroNative.lz_get_device_name(d, i, name, (uint)name.Capacity);
                devices.Add(new DeviceInfo(d, i, name.ToString()));
            }
        }

        return devices;
    }

    /// <summary>
    /// Checks whether Level Zero is available on this system (driver loads and at least one device found).
    /// </summary>
    public static bool IsAvailable()
    {
        try
        {
            int result = LevelZeroNative.lz_init(0);
            if (result != 0)
            {
                return false;
            }

            result = LevelZeroNative.lz_get_driver_count(out uint count);
            return result == 0 && count > 0;
        }
        catch (DllNotFoundException) { return false; }
        catch (EntryPointNotFoundException) { return false; }
    }
}

/// <summary>
/// Describes a discovered Level Zero device without opening a full context.
/// </summary>
public readonly record struct DeviceInfo(uint DriverIndex, uint DeviceIndex, string Name)
{
    /// <summary>Opens this device for compute operations.</summary>
    public ComputeDevice Open() => ComputeDevice.Create(DriverIndex, DeviceIndex);
}

