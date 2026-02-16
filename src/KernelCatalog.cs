using System.Reflection;

namespace LevelZero;

/// <summary>
/// Registry of available SPIR-V kernels. Resolves kernel SPIR-V bytes from
/// embedded resources (device-specific then fallback), environment variables,
/// explicit paths, NuGet content directories, or the auto-detected device config.
/// </summary>
public sealed class KernelCatalog
{
    private readonly ComputeDevice _device;

    /// <summary>Default environment variable prefix for SPIR-V paths.</summary>
    public const string EnvPrefix = "IPU_L0_";

    /// <summary>
    /// Subdirectory where the HFLabs.LevelZero.Kernels.{device} NuGet packages
    /// copy their .spv content files at build time.
    /// </summary>
    public const string NuGetContentDir = "levelzero-kernels";

    /// <summary>Fallback device target when no specific target is configured or detected.</summary>
    public const string FallbackDevice = "tgllp";

    private static readonly Assembly s_assembly = typeof(KernelCatalog).Assembly;

    /// <summary>All device targets that have embedded SPIR-V kernels in the DLL.</summary>
    public static IReadOnlyList<string> EmbeddedDeviceTargets { get; } = GetEmbeddedDeviceTargets();

    /// <summary>
    /// Cached device target from config or explicit override. Loaded lazily on first resolve.
    /// </summary>
    private static string? s_configDeviceTarget;
    private static bool s_configLoaded;
    private static readonly object s_configLock = new();

    internal KernelCatalog(ComputeDevice device)
    {
        _device = device;
    }

    /// <summary>
    /// Explicitly sets the active device target for embedded SPIR-V resolution.
    /// Overrides auto-detection. Call before creating any kernels.
    /// </summary>
    /// <param name="deviceTarget">A device target string (e.g. "bmg-g21", "acm-g10", "pvc").
    /// Use <see cref="EmbeddedDeviceTargets"/> to see available options.</param>
    public static void SetDeviceTarget(string deviceTarget)
    {
        lock (s_configLock)
        {
            s_configDeviceTarget = deviceTarget;
            s_configLoaded = true;
        }
    }

    /// <summary>
    /// Gets the active device target — either explicitly set via <see cref="SetDeviceTarget"/>,
    /// loaded from config file, or auto-detected from the GPU.
    /// </summary>
    public static string GetConfiguredDeviceTarget()
    {
        if (s_configLoaded)
            return s_configDeviceTarget ?? FallbackDevice;

        lock (s_configLock)
        {
            if (s_configLoaded)
                return s_configDeviceTarget ?? FallbackDevice;

            var config = DeviceCapabilityDetector.LoadOrDetect();
            s_configDeviceTarget = config.DeviceTarget;
            s_configLoaded = true;
            return s_configDeviceTarget ?? FallbackDevice;
        }
    }

    /// <summary>
    /// Loads a SPIR-V kernel from the embedded resources baked into the DLL.
    /// Tries device-specific resource first, then falls back to tgllp (Gen12).
    /// </summary>
    /// <param name="kernelName">Kernel name (e.g. "fitness_kernel", "pso_velocity").</param>
    /// <param name="deviceTarget">Device target override. If null, uses <see cref="GetConfiguredDeviceTarget"/>.</param>
    /// <returns>The SPIR-V bytes, or null if no embedded resource matches.</returns>
    public static byte[]? LoadEmbeddedSpirv(string kernelName, string? deviceTarget = null)
    {
        var target = deviceTarget ?? GetConfiguredDeviceTarget();

        // 1. Try device-specific embedded resource
        var deviceResource = $"LevelZero.Kernels.{target}.{kernelName}.spv";
        var bytes = LoadResource(deviceResource);
        if (bytes is not null) return bytes;

        // 2. Fall back to tgllp (broadest compatible target)
        if (!string.Equals(target, FallbackDevice, StringComparison.OrdinalIgnoreCase))
        {
            var fallbackResource = $"LevelZero.Kernels.{FallbackDevice}.{kernelName}.spv";
            bytes = LoadResource(fallbackResource);
            if (bytes is not null) return bytes;
        }

        return null;
    }

    /// <summary>
    /// Resolves a SPIR-V path for a kernel module. Checks (in order):
    /// 1. Explicit path (if non-null and exists)
    /// 2. Environment variable IPU_L0_{NAME}_SPIRV
    /// 3. NuGet content: {appDir}/levelzero-kernels/{name}.spv (flat layout)
    /// 4. Device-specific: {appDir}/levelzero-kernels/{deviceTarget}/{name}.spv
    /// 5. Current directory fallback: {name}.spv
    /// Returns null if no path resolves to an existing file.
    /// Note: does NOT check embedded resources — use <see cref="LoadEmbeddedSpirv"/> or <see cref="LoadModule"/> for that.
    /// </summary>
    public static string? ResolveSpirvPath(string kernelName, string? explicitPath = null)
    {
        if (!string.IsNullOrWhiteSpace(explicitPath) && File.Exists(explicitPath))
            return explicitPath;

        var envVar = $"{EnvPrefix}{kernelName.ToUpperInvariant()}_SPIRV";
        var envPath = Environment.GetEnvironmentVariable(envVar);
        if (!string.IsNullOrWhiteSpace(envPath) && File.Exists(envPath))
            return envPath;

        var appDir = AppContext.BaseDirectory;

        // Flat NuGet content layout (single device package installed)
        var nugetPath = Path.Combine(appDir, NuGetContentDir, $"{kernelName}.spv");
        if (File.Exists(nugetPath))
            return nugetPath;

        // Device-specific subdirectory (multi-device layout or manual deployment)
        var deviceTarget = GetConfiguredDeviceTarget();
        var devicePath = Path.Combine(appDir, NuGetContentDir, deviceTarget, $"{kernelName}.spv");
        if (File.Exists(devicePath))
            return devicePath;

        var localPath = $"{kernelName}.spv";
        if (File.Exists(localPath))
            return localPath;

        return null;
    }

    /// <summary>
    /// Loads a SPIR-V module using the full resolution chain:
    /// file-system paths first, then embedded resources baked into the DLL.
    /// Throws if nothing is found.
    /// </summary>
    public ComputeModule LoadModule(string kernelName, string? explicitPath = null)
    {
        // 1. Try file-system resolution (explicit, env var, NuGet, local)
        var path = ResolveSpirvPath(kernelName, explicitPath);
        if (path is not null)
            return _device.LoadModule(path);

        // 2. Try embedded resource
        var embedded = LoadEmbeddedSpirv(kernelName);
        if (embedded is not null)
            return _device.LoadModule(embedded);

        throw new FileNotFoundException(
            $"SPIR-V not found for kernel '{kernelName}'. " +
            $"Set environment variable {EnvPrefix}{kernelName.ToUpperInvariant()}_SPIRV, " +
            $"place {kernelName}.spv in the working directory, or ensure the DLL was built with embedded kernels.");
    }

    /// <summary>
    /// Tries to load a SPIR-V module using file-system paths then embedded resources.
    /// Returns null if nothing is found or compilation fails.
    /// </summary>
    public ComputeModule? TryLoadModule(string kernelName, out string buildLog, string? explicitPath = null)
    {
        buildLog = string.Empty;

        var path = ResolveSpirvPath(kernelName, explicitPath);
        if (path is not null)
            return _device.TryLoadModule(path, out buildLog);

        var embedded = LoadEmbeddedSpirv(kernelName);
        if (embedded is not null)
            return _device.TryLoadModule(embedded, out buildLog);

        return null;
    }

    /// <summary>Mapping of well-known kernel identifiers to their default environment variable names.</summary>
    public static IReadOnlyDictionary<string, string> WellKnownKernels { get; } = new Dictionary<string, string>
    {
        ["fitness"]              = "IPU_L0_FITNESS_SPIRV",
        ["rastrigin_fitness"]    = "IPU_L0_RASTRIGIN_FITNESS_SPIRV",
        ["pso_velocity"]         = "IPU_L0_PSO_VELOCITY_SPIRV",
        ["snn_neuron_update"]    = "IPU_L0_SNN_NEURON_UPDATE_SPIRV",
        ["snn_correlation"]      = "IPU_L0_SNN_CORRELATION_SPIRV",
        ["stdp_plasticity"]      = "IPU_L0_STDP_PLASTICITY_SPIRV",
        ["nbody_repulsion"]      = "IPU_L0_NBODY_REPULSION_SPIRV",
        ["dominance_matrix"]     = "IPU_L0_DOMINANCE_MATRIX_SPIRV",
        ["exchange_weights"]     = "IPU_L0_EXCHANGE_WEIGHTS_SPIRV",
        ["firefly_attraction"]   = "IPU_L0_FIREFLY_ATTRACTION_SPIRV",
        ["monte_carlo_hypervolume"] = "IPU_L0_MONTE_CARLO_HYPERVOLUME_SPIRV",
        ["pairwise_distance"]    = "IPU_L0_PAIRWISE_DISTANCE_SPIRV",
        ["hypergraph"]          = "IPU_L0_HYPERGRAPH_SPIRV",
        ["tile_placement"]       = "IPU_L0_TILE_PLACEMENT_SPIRV",
    };

    /// <summary>Loads a single embedded resource by logical name.</summary>
    private static byte[]? LoadResource(string resourceName)
    {
        using var stream = s_assembly.GetManifestResourceStream(resourceName);
        if (stream is null) return null;
        var bytes = new byte[stream.Length];
        stream.ReadExactly(bytes);
        return bytes;
    }

    /// <summary>Discovers which device targets have SPIR-V kernels embedded in the DLL.</summary>
    private static IReadOnlyList<string> GetEmbeddedDeviceTargets()
    {
        var names = s_assembly.GetManifestResourceNames();
        var targets = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
        const string prefix = "LevelZero.Kernels.";
        const string suffix = ".spv";
        foreach (var name in names)
        {
            if (!name.StartsWith(prefix, StringComparison.Ordinal) ||
                !name.EndsWith(suffix, StringComparison.Ordinal))
                continue;

            // Format: LevelZero.Kernels.{device}.{kernel}.spv
            var inner = name[prefix.Length..^suffix.Length];
            var dot = inner.IndexOf('.');
            if (dot > 0)
                targets.Add(inner[..dot]);
        }
        return targets.Order().ToList();
    }
}
