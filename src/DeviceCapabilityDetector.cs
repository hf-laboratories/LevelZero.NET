using System.Text.RegularExpressions;
using LevelZero.Native;

namespace LevelZero;

/// <summary>
/// Detects the Intel GPU present on the system, identifies its architecture
/// family, and maps it to the best ocloc device target for SPIR-V selection.
/// Works bottom-up from oldest supported architecture (Gen12/Tiger Lake) to
/// newest (Xe3/Panther Lake), returning the highest matching level.
/// </summary>
public static class DeviceCapabilityDetector
{
    /// <summary>
    /// Describes a known Intel GPU architecture level, ordered from oldest to newest.
    /// </summary>
    public sealed record ArchitectureLevel(
        string DeviceTarget,
        string Architecture,
        string Family,
        int Rank,
        string[] NamePatterns);

    /// <summary>
    /// Known architecture levels ordered from oldest (rank 0) to newest.
    /// The detector walks this list bottom-up: the highest rank whose name
    /// pattern matches the reported device name wins.
    /// </summary>
    public static IReadOnlyList<ArchitectureLevel> KnownLevels { get; } =
    [
        //                     Name substrings (case-insensitive)
        new("tgllp",   "Gen12",    "Tiger Lake",    0,  ["Tiger Lake", "TGL", "Iris Xe Graphics"]),
        new("dg1",     "Gen12",    "DG1",           1,  ["DG1", "Iris Xe MAX"]),
        new("acm-g12", "Xe-HPG",   "Alchemist G12", 2,  ["A310", "ACM-G12"]),
        new("acm-g11", "Xe-HPG",   "Alchemist G11", 3,  ["A380", "A580", "ACM-G11", "ATS-M75"]),
        new("acm-g10", "Xe-HPG",   "Alchemist G10", 4,  ["A770", "A750", "A580", "ACM-G10", "ATS-M150", "DG2"]),
        new("pvc",     "Xe-HPC",   "Ponte Vecchio", 5,  ["Data Center GPU Max", "Ponte Vecchio", "PVC"]),
        new("mtl",     "Xe-LPG",   "Meteor Lake",   6,  ["Meteor Lake", "MTL", "Core Ultra"]),
        new("arl-h",   "Xe-LPG+",  "Arrow Lake",    7,  ["Arrow Lake", "ARL"]),
        new("bmg-g21", "Xe2-HPG",  "Battlemage",    8,  ["B580", "B570", "Battlemage", "BMG"]),
        new("lnl-m",   "Xe2-LPG",  "Lunar Lake",    9,  ["Lunar Lake", "LNL"]),
        new("ptl-h",   "Xe3-LPG",  "Panther Lake",  10, ["Panther Lake", "PTL"]),
    ];

    /// <summary>
    /// Detects the GPU, determines the best device target, and returns a config.
    /// If no hardware is found or the device doesn't match any known architecture,
    /// falls back to "tgllp" (broadest compatible Gen12 target).
    /// </summary>
    public static LevelZeroConfig Detect()
    {
        var config = new LevelZeroConfig { DetectedAt = DateTime.UtcNow };

        string? deviceName = null;
        try
        {
            if (!LevelZeroRuntime.IsAvailable())
            {
                config.HardwarePresent = false;
                config.DeviceTarget = "tgllp";
                config.Architecture = "Gen12";
                config.DeviceName = "(none)";
                return config;
            }

            var devices = LevelZeroRuntime.EnumerateDevices();
            if (devices.Count == 0)
            {
                config.HardwarePresent = false;
                config.DeviceTarget = "tgllp";
                config.Architecture = "Gen12";
                config.DeviceName = "(none)";
                return config;
            }

            // Use the first device
            deviceName = devices[0].Name;
            config.HardwarePresent = true;
            config.DeviceName = deviceName;
        }
        catch
        {
            config.HardwarePresent = false;
            config.DeviceTarget = "tgllp";
            config.Architecture = "Gen12";
            config.DeviceName = "(unavailable)";
            return config;
        }

        // Walk levels bottom-up to find the highest matching architecture
        var match = MatchDevice(deviceName);
        config.DeviceTarget = match.DeviceTarget;
        config.Architecture = match.Architecture;
        return config;
    }

    /// <summary>
    /// Matches a device name string to the best architecture level.
    /// Walks from highest rank to lowest, returning the first match.
    /// Falls back to tgllp if nothing matches.
    /// </summary>
    public static ArchitectureLevel MatchDevice(string deviceName)
    {
        // Walk from highest rank down — first match wins (highest architecture)
        for (int i = KnownLevels.Count - 1; i >= 0; i--)
        {
            var level = KnownLevels[i];
            foreach (var pattern in level.NamePatterns)
            {
                if (deviceName.Contains(pattern, StringComparison.OrdinalIgnoreCase))
                    return level;
            }
        }

        // Fallback: broadest compatible target
        return KnownLevels[0]; // tgllp
    }

    /// <summary>
    /// Detects the GPU and saves the result to a config file.
    /// Returns the config that was saved.
    /// </summary>
    public static LevelZeroConfig DetectAndSave(string? configPath = null)
    {
        configPath ??= LevelZeroConfig.DefaultPath;
        var config = Detect();
        config.Save(configPath);
        return config;
    }

    /// <summary>
    /// Loads an existing config or runs detection if none exists.
    /// This is the main entry point for auto-configuration.
    /// </summary>
    public static LevelZeroConfig LoadOrDetect(string? configPath = null)
    {
        configPath ??= LevelZeroConfig.DefaultPath;
        var existing = LevelZeroConfig.Load(configPath);
        if (existing is not null)
            return existing;

        return DetectAndSave(configPath);
    }
}
