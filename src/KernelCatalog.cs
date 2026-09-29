using System.Reflection;

namespace LevelZero;

/// <summary>
/// Registry of available Level Zero kernel modules. Resolves kernel bytes from
/// embedded native-module or SPIR-V resources (device-specific then fallback),
/// environment variables, explicit paths, NuGet content directories, or the
/// auto-detected device config.
/// </summary>
public sealed class KernelCatalog
{
    private readonly ComputeDevice _device;

    /// <summary>Default environment variable prefix for SPIR-V paths.</summary>
    public const string EnvPrefix = "IPU_L0_";

    /// <summary>
    /// Subdirectory where the HFLabs.LevelZero.Kernels.{device} NuGet packages
    /// copy their kernel artifacts at build time.
    /// </summary>
    public const string NuGetContentDir = "levelzero-kernels";

    /// <summary>Fallback device target when no specific target is configured or detected.</summary>
    public const string FallbackDevice = "tgllp";

    /// <summary>Legacy fallback subdirectory for harness compatibility.</summary>
    public const string LegacyKernelsDir = "kernels";

    private static readonly Assembly s_assembly = typeof(KernelCatalog).Assembly;
    private static readonly string[] s_allModuleExtensions = [".zebin", ".bin", ".native", ".spv"];
    private static readonly string[] s_spirvOnlyExtensions = [".spv"];

    /// <summary>
    /// Module extensions honoured by resolution, in preference order (native first).
    /// When <c>IPU_L0_NATIVE_MODULES=0</c> (or <c>false</c>), native device binaries are
    /// excluded so only SPIR-V resolves — a driver/zebin mismatch can hard-abort the process
    /// natively, so test hosts and coverage runs use this to stay driver-independent.
    /// Evaluated per call so tests can toggle the environment variable at runtime.
    /// </summary>
    private static string[] s_supportedModuleExtensions
    {
        get
        {
            string? value = Environment.GetEnvironmentVariable("IPU_L0_NATIVE_MODULES")?.Trim();
            bool nativeExplicitlyEnabled = string.Equals(value, "1", StringComparison.Ordinal)
                || string.Equals(value, "true", StringComparison.OrdinalIgnoreCase);
            return nativeExplicitlyEnabled ? s_allModuleExtensions : s_spirvOnlyExtensions;
        }
    }

    private readonly record struct EmbeddedModuleResource(byte[] Bytes, LevelZeroModuleFormat Format);

    /// <summary>All device targets that have embedded kernel artifacts in the DLL.</summary>
    public static IReadOnlyList<string> EmbeddedDeviceTargets { get; } = GetEmbeddedDeviceTargets();

    /// <summary>
    /// Cached device target from config or explicit override. Loaded lazily on first resolve.
    /// </summary>
    private static string? s_configDeviceTarget;
    private static bool s_configLoaded;
    private static readonly object s_configLock = new();

    public KernelCatalog(ComputeDevice device)
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
    /// Gets the active device target  either explicitly set via <see cref="SetDeviceTarget"/>,
    /// loaded from config file, or auto-detected from the GPU.
    /// </summary>
    public static string GetConfiguredDeviceTarget()
    {
        if (s_configLoaded)
        {
            return s_configDeviceTarget ?? FallbackDevice;
        }

        lock (s_configLock)
        {
            if (s_configLoaded)
            {
                return s_configDeviceTarget ?? FallbackDevice;
            }

            LevelZeroConfig config = DeviceCapabilityDetector.LoadOrDetect();
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
        return LoadEmbeddedResource(kernelName, deviceTarget, [".spv"])?.Bytes;
    }

    /// <summary>
    /// Resolves a SPIR-V path for a kernel module. Checks (in order):
    /// 1. Explicit path (if non-null and exists)
    /// 2. Environment variable IPU_L0_{NAME}_SPIRV
    /// 3. NuGet content: {appDir}/levelzero-kernels/{name}.spv (flat layout)
    /// 4. Device-specific: {appDir}/levelzero-kernels/{deviceTarget}/{name}.spv
    /// 5. Current directory fallback: {name}.spv
    /// Returns null if no path resolves to an existing file.
    /// Note: does NOT check embedded resources  use <see cref="LoadEmbeddedSpirv"/> or <see cref="LoadModule"/> for that.
    /// </summary>
    public static string? ResolveSpirvPath(string kernelName, string? explicitPath = null)
    {
        if (!string.IsNullOrWhiteSpace(explicitPath) && File.Exists(explicitPath))
        {
            return explicitPath;
        }

        foreach (string envVar in GetKernelEnvVarNames(kernelName))
        {
            string? envPath = Environment.GetEnvironmentVariable(envVar);
            if (!string.IsNullOrWhiteSpace(envPath) && File.Exists(envPath))
            {
                return envPath;
            }
        }

        string appDir = AppContext.BaseDirectory;

        // Flat NuGet content layout (single device package installed)
        string nugetPath = Path.Combine(appDir, NuGetContentDir, $"{kernelName}.spv");
        if (File.Exists(nugetPath))
        {
            return nugetPath;
        }

        // Device-specific subdirectory (multi-device layout or manual deployment)
        string deviceTarget = GetConfiguredDeviceTarget();
        string devicePath = Path.Combine(appDir, NuGetContentDir, deviceTarget, $"{kernelName}.spv");
        if (File.Exists(devicePath))
        {
            return devicePath;
        }

        string legacyPath = Path.Combine(appDir, LegacyKernelsDir, $"{kernelName}.spv");
        if (File.Exists(legacyPath))
        {
            return legacyPath;
        }

        string localPath = $"{kernelName}.spv";
        return File.Exists(localPath) ? localPath : null;
    }

    /// <summary>
    /// Resolves a Level Zero module path for a kernel. Checks (in order):
    /// 1. Explicit path (if non-null and exists)
    /// 2. Environment variable IPU_L0_{NAME}_MODULE
    /// 3. Environment variable IPU_L0_{NAME}_SPIRV
    /// 4. NuGet content: {appDir}/levelzero-kernels/{name}.zebin/.bin/.native/.spv
    /// 5. Device-specific: {appDir}/levelzero-kernels/{deviceTarget}/{name}.zebin/.bin/.native/.spv
    /// 6. Current directory fallback: {name}.zebin/.bin/.native/.spv
    /// Returns null if no path resolves to an existing file.
    /// Note: does NOT check embedded resources  use <see cref="LoadEmbeddedSpirv"/> or <see cref="LoadModule"/> for that.
    /// </summary>
    public static string? ResolveModulePath(string kernelName, string? explicitPath = null)
    {
        if (!string.IsNullOrWhiteSpace(explicitPath) && File.Exists(explicitPath))
        {
            return explicitPath;
        }

        foreach (string moduleEnvVar in GetKernelModuleEnvVarNames(kernelName))
        {
            string? moduleEnvPath = Environment.GetEnvironmentVariable(moduleEnvVar);
            if (!string.IsNullOrWhiteSpace(moduleEnvPath) && File.Exists(moduleEnvPath))
            {
                return moduleEnvPath;
            }
        }

        foreach (string spirvEnvVar in GetKernelEnvVarNames(kernelName))
        {
            string? spirvEnvPath = Environment.GetEnvironmentVariable(spirvEnvVar);
            if (!string.IsNullOrWhiteSpace(spirvEnvPath) && File.Exists(spirvEnvPath))
            {
                return spirvEnvPath;
            }
        }

        string appDir = AppContext.BaseDirectory;
        string? flatPath = FindFirstExistingModulePath(Path.Combine(appDir, NuGetContentDir), kernelName);
        if (flatPath is not null)
        {
            return flatPath;
        }

        string deviceTarget = GetConfiguredDeviceTarget();
        string? devicePath = FindFirstExistingModulePath(Path.Combine(appDir, NuGetContentDir, deviceTarget), kernelName);
        if (devicePath is not null)
        {
            return devicePath;
        }

        string? legacyPath = FindFirstExistingModulePath(Path.Combine(appDir, LegacyKernelsDir), kernelName);
        if (legacyPath is not null)
        {
            return legacyPath;
        }

        return FindFirstExistingModulePath(Environment.CurrentDirectory, kernelName);
    }

    /// <summary>
    /// Loads a module using the full resolution chain:
    /// file-system paths first, then embedded native/SPIR-V resources baked into the DLL.
    /// Throws if nothing is found.
    /// </summary>
    public ComputeModule LoadModule(string kernelName, string? explicitPath = null)
    {
        // 1. Try file-system resolution (explicit, env var, NuGet, local)
        string? path = ResolveModulePath(kernelName, explicitPath);
        if (path is not null)
        {
            try
            {
                return _device.LoadModule(path);
            }
            catch (Exception)
            {
                // Fall back to .spv on disk if native format failed
                if (!path.EndsWith(".spv", StringComparison.OrdinalIgnoreCase))
                {
                    string spvPath = Path.ChangeExtension(path, ".spv");
                    if (File.Exists(spvPath))
                    {
                        try
                        {
                            return _device.LoadModule(spvPath);
                        }
                        catch
                        {
                            // Ignore fallback failure, throw original exception
                        }
                    }
                }
                throw;
            }
        }

        // 2. Try embedded resource
        EmbeddedModuleResource? embedded = LoadEmbeddedModule(kernelName);
        if (embedded is not null)
        {
            try
            {
                return _device.LoadModule(embedded.Value.Bytes, embedded.Value.Format);
            }
            catch (Exception)
            {
                // Fall back to embedded .spv if native format failed
                if (embedded.Value.Format == LevelZeroModuleFormat.Native)
                {
                    EmbeddedModuleResource? spvEmbedded = LoadEmbeddedResource(kernelName, null, new[] { ".spv" });
                    if (spvEmbedded is not null)
                    {
                        try
                        {
                            return _device.LoadModule(spvEmbedded.Value.Bytes, spvEmbedded.Value.Format);
                        }
                        catch
                        {
                            // Ignore fallback failure, throw original exception
                        }
                    }
                }
                throw;
            }
        }

        throw new FileNotFoundException(
            $"Level Zero module not found for kernel '{kernelName}'. " +
            $"Set environment variable {GetKernelModuleEnvVarName(kernelName)} or {GetKernelEnvVarName(kernelName)}, " +
            $"place {kernelName}.spv/.bin/.native/.zebin in the working directory, or ensure the DLL was built with embedded kernel artifacts.");
    }

    /// <summary>
    /// Tries to load a module using file-system paths then embedded native/SPIR-V resources.
    /// Returns null if nothing is found or compilation fails.
    /// </summary>
    public ComputeModule? TryLoadModule(string kernelName, out string buildLog, string? explicitPath = null)
    {
        buildLog = string.Empty;

        string? path = ResolveModulePath(kernelName, explicitPath);
        if (path is not null)
        {
            ComputeModule? module = _device.TryLoadModule(path, out buildLog);
            if (module is not null)
            {
                return module;
            }
            // Fall back to .spv on disk if native format failed
            if (!path.EndsWith(".spv", StringComparison.OrdinalIgnoreCase))
            {
                string spvPath = Path.ChangeExtension(path, ".spv");
                if (File.Exists(spvPath))
                {
                    module = _device.TryLoadModule(spvPath, out string fallbackLog);
                    if (module is not null)
                    {
                        buildLog += Environment.NewLine + "Fallback .spv load succeeded. Fallback compiler log: " + fallbackLog;
                        return module;
                    }
                    buildLog += Environment.NewLine + "Fallback .spv load failed: " + fallbackLog;
                }
            }
            return null;
        }

        EmbeddedModuleResource? embedded = LoadEmbeddedModule(kernelName);
        if (embedded is not null)
        {
            ComputeModule? module = _device.TryLoadModule(embedded.Value.Bytes, embedded.Value.Format, out buildLog);
            if (module is not null)
            {
                return module;
            }
            // Fall back to embedded .spv if native format failed
            if (embedded.Value.Format == LevelZeroModuleFormat.Native)
            {
                EmbeddedModuleResource? spvEmbedded = LoadEmbeddedResource(kernelName, null, new[] { ".spv" });
                if (spvEmbedded is not null)
                {
                    module = _device.TryLoadModule(spvEmbedded.Value.Bytes, spvEmbedded.Value.Format, out string fallbackLog);
                    if (module is not null)
                    {
                        buildLog += Environment.NewLine + "Fallback embedded .spv load succeeded. Fallback compiler log: " + fallbackLog;
                        return module;
                    }
                    buildLog += Environment.NewLine + "Fallback embedded .spv load failed: " + fallbackLog;
                }
            }
            return null;
        }

        return null;
    }

    private static EmbeddedModuleResource? LoadEmbeddedModule(string kernelName, string? deviceTarget = null)
        => LoadEmbeddedResource(kernelName, deviceTarget, s_supportedModuleExtensions);

    private static EmbeddedModuleResource? LoadEmbeddedResource(
        string kernelName,
        string? deviceTarget,
        IReadOnlyList<string> extensions)
    {
        string target = deviceTarget ?? GetConfiguredDeviceTarget();

        EmbeddedModuleResource? resource = TryLoadEmbeddedResourceForTarget(target, kernelName, extensions);
        if (resource is not null)
        {
            return resource;
        }

        if (!string.Equals(target, FallbackDevice, StringComparison.OrdinalIgnoreCase))
        {
            return TryLoadEmbeddedResourceForTarget(FallbackDevice, kernelName, extensions);
        }

        return null;
    }

    private static EmbeddedModuleResource? TryLoadEmbeddedResourceForTarget(
        string deviceTarget,
        string kernelName,
        IReadOnlyList<string> extensions)
    {
        foreach (string extension in extensions)
        {
            string resourceName = $"LevelZero.Kernels.{deviceTarget}.{kernelName}{extension}";
            byte[]? bytes = LoadResource(resourceName);
            if (bytes is not null)
            {
                return new EmbeddedModuleResource(bytes, GetModuleFormatFromExtension(extension));
            }
        }

        return null;
    }

    /// <summary>Mapping of well-known kernel identifiers to their canonical SPIR-V environment variable names.</summary>
    public static IReadOnlyDictionary<string, string> WellKnownKernels { get; } = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase)
    {
        ["fitness"] = "IPU_L0_FITNESS_SPIRV",
        ["fitness_kernel"] = "IPU_L0_FITNESS_SPIRV",
        ["rastrigin_fitness"] = "IPU_L0_RASTRIGIN_FITNESS_SPIRV",
        ["pso_velocity"] = "IPU_L0_PSO_VELOCITY_SPIRV",
        ["snn_neuron_update"] = "IPU_L0_SNN_NEURON_UPDATE_SPIRV",
        ["snn_correlation"] = "IPU_L0_SNN_CORRELATION_SPIRV",
        ["stdp_plasticity"] = "IPU_L0_STDP_PLASTICITY_SPIRV",
        ["nbody_repulsion"] = "IPU_L0_NBODY_REPULSION_SPIRV",
        ["dominance_matrix"] = "IPU_L0_DOMINANCE_MATRIX_SPIRV",
        ["exchange_weights"] = "IPU_L0_EXCHANGE_WEIGHTS_SPIRV",
        ["firefly_attraction"] = "IPU_L0_FIREFLY_ATTRACTION_SPIRV",
        ["monte_carlo_hypervolume"] = "IPU_L0_MONTE_CARLO_HYPERVOLUME_SPIRV",
        ["pairwise_distance"] = "IPU_L0_PAIRWISE_DISTANCE_SPIRV",
        ["hypergraph"] = "IPU_L0_HYPERGRAPH_SPIRV",
        ["levelzero-hypergraph"] = "IPU_L0_HYPERGRAPH_SPIRV",
        ["tile_placement"] = "IPU_L0_TILE_PLACEMENT_SPIRV",
        ["levelzero-score"] = "IPU_L0_LEVELZERO_SCORE_SPIRV",
        ["levelzero-rankerprep"] = "IPU_L0_LEVELZERO_RANKERPREP_SPIRV",
        ["levelzero-chainboost"] = "IPU_L0_LEVELZERO_CHAINBOOST_SPIRV",
        ["graphrag_vector_search"] = "IPU_L0_GRAPHRAG_VECTOR_SEARCH_SPIRV",
        ["pixie_worker_pipe"] = "IPU_L0_PIXIE_WORKER_PIPE_SPIRV",
    };

    private static IReadOnlyDictionary<string, string[]> LegacyKernelEnvAliases { get; } = new Dictionary<string, string[]>(StringComparer.OrdinalIgnoreCase)
    {
        ["pso_velocity"] = ["IPU_L0_PSO_SPIRV"],
        ["snn_neuron_update"] = ["IPU_L0_SNN_NEURON_SPIRV"],
        ["snn_correlation"] = ["IPU_L0_SNN_CORR_SPIRV"],
        ["nbody_repulsion"] = ["IPU_L0_NBODY_SPIRV"],
        ["pairwise_distance"] = ["IPU_L0_PAIRWISE_DIST_SPIRV"],
        ["dominance_matrix"] = ["IPU_L0_DOMINANCE_SPIRV"],
        ["firefly_attraction"] = ["IPU_L0_FIREFLY_SPIRV"],
        ["monte_carlo_hypervolume"] = ["IPU_L0_HYPERVOLUME_SPIRV"],
        ["tile_placement"] = ["IPU_L0_TILE_SPIRV"],
        ["levelzero-score"] = ["IPU_L0_SUGGESTION_SPIRV", "IPU_L0_SCORE_SPIRV"],
        ["levelzero-rankerprep"] = ["IPU_L0_RANKER_PREP_SPIRV"],
        ["levelzero-chainboost"] = ["IPU_L0_CHAIN_BOOST_SPIRV"]
    };

    private static string GetKernelEnvVarName(string kernelName)
    {
        if (WellKnownKernels.TryGetValue(kernelName, out string? envVarName))
        {
            return envVarName;
        }

        string normalized = kernelName.Replace('-', '_').ToUpperInvariant();
        return $"{EnvPrefix}{normalized}_SPIRV";
    }

    private static IReadOnlyList<string> GetKernelEnvVarNames(string kernelName)
    {
        List<string> envVarNames = [GetKernelEnvVarName(kernelName)];
        if (LegacyKernelEnvAliases.TryGetValue(kernelName, out string[]? aliases))
        {
            foreach (string alias in aliases)
            {
                if (!envVarNames.Any(existing => string.Equals(existing, alias, StringComparison.OrdinalIgnoreCase)))
                {
                    envVarNames.Add(alias);
                }
            }
        }

        return envVarNames;
    }

    private static string GetKernelModuleEnvVarName(string kernelName)
    {
        if (WellKnownKernels.TryGetValue(kernelName, out string? envVarName))
        {
            return ToModuleEnvVarName(envVarName);
        }

        string normalized = kernelName.Replace('-', '_').ToUpperInvariant();
        return $"{EnvPrefix}{normalized}_MODULE";
    }

    private static IReadOnlyList<string> GetKernelModuleEnvVarNames(string kernelName)
    {
        List<string> envVarNames = [GetKernelModuleEnvVarName(kernelName)];
        if (LegacyKernelEnvAliases.TryGetValue(kernelName, out string[]? aliases))
        {
            foreach (string alias in aliases)
            {
                string moduleAlias = ToModuleEnvVarName(alias);
                if (!envVarNames.Any(existing => string.Equals(existing, moduleAlias, StringComparison.OrdinalIgnoreCase)))
                {
                    envVarNames.Add(moduleAlias);
                }
            }
        }

        return envVarNames;
    }

    private static string ToModuleEnvVarName(string envVarName)
    {
        const string spirvSuffix = "_SPIRV";
        return envVarName.EndsWith(spirvSuffix, StringComparison.Ordinal)
            ? $"{envVarName[..^spirvSuffix.Length]}_MODULE"
            : $"{envVarName}_MODULE";
    }

    private static LevelZeroModuleFormat GetModuleFormatFromExtension(string extension)
        => string.Equals(extension, ".spv", StringComparison.OrdinalIgnoreCase)
            ? LevelZeroModuleFormat.SpirV
            : LevelZeroModuleFormat.Native;

    private static string? FindFirstExistingModulePath(string directory, string kernelName)
    {
        foreach (string extension in s_supportedModuleExtensions)
        {
            string candidate = Path.Combine(directory, $"{kernelName}{extension}");
            if (File.Exists(candidate))
            {
                return candidate;
            }
        }

        return null;
    }

    /// <summary>Loads a single embedded resource by logical name.</summary>
    private static byte[]? LoadResource(string resourceName)
    {
        using Stream? stream = s_assembly.GetManifestResourceStream(resourceName);
        if (stream is null)
        {
            return null;
        }

        byte[] bytes = new byte[stream.Length];
        stream.ReadExactly(bytes);
        return bytes;
    }

    /// <summary>Discovers which device targets have kernel artifacts embedded in the DLL.</summary>
    private static IReadOnlyList<string> GetEmbeddedDeviceTargets()
    {
        string[] names = s_assembly.GetManifestResourceNames();
        var targets = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
        const string prefix = "LevelZero.Kernels.";
        foreach (string name in names)
        {
            if (!name.StartsWith(prefix, StringComparison.Ordinal))
            {
                continue;
            }

            string? suffix = s_supportedModuleExtensions.FirstOrDefault(extension =>
                name.EndsWith(extension, StringComparison.OrdinalIgnoreCase));
            if (suffix is null)
            {
                continue;
            }

            // Format: LevelZero.Kernels.{device}.{kernel}.{extension}
            string inner = name[prefix.Length..^suffix.Length];
            int dot = inner.IndexOf('.', StringComparison.Ordinal);
            if (dot > 0)
            {
                _ = targets.Add(inner[..dot]);
            }
        }

        return targets.Order().ToList();
    }

    /// <summary>
    /// Attempts to extract an embedded kernel resource to a temporary file on disk.
    /// Returns the file path if extracted, or null if not found.
    /// </summary>
    public static string? TryExtractEmbeddedToTempFile(string kernelId)
    {
        string device = GetConfiguredDeviceTarget();
        string[] resourceNames = s_assembly.GetManifestResourceNames();
        string prefix = $"LevelZero.Kernels.{device}.{kernelId}.";
        string? resourceName = resourceNames.FirstOrDefault(r => r.StartsWith(prefix, StringComparison.OrdinalIgnoreCase));
        if (resourceName is null)
        {
            prefix = $"LevelZero.Kernels.{FallbackDevice}.{kernelId}.";
            resourceName = resourceNames.FirstOrDefault(r => r.StartsWith(prefix, StringComparison.OrdinalIgnoreCase));
        }

        if (resourceName is null)
        {
            return null;
        }

        byte[]? bytes = LoadResource(resourceName);
        if (bytes is null)
        {
            return null;
        }

        string ext = Path.GetExtension(resourceName);
        string tempPath = Path.Combine(Path.GetTempPath(), $"{kernelId}_{Guid.NewGuid():N}{ext}");
        File.WriteAllBytes(tempPath, bytes);
        return tempPath;
    }
}

