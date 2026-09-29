namespace LevelZero.Kernels;

/// <summary>
/// Shared SPIR-V resolution logic used by every kernel's single-argument
/// <c>Create(ComputeDevice)</c> factory: prefer an on-disk SPIR-V file, fall back to an
/// embedded blob, otherwise fail loudly. Extracted from 18 identical wrappers.
/// (see azure/dry-cohorts.json, cohort 0B774A33A48E1A6EFF065D07A3DD9A8250D4A2FDD3F879FBA2C6CC38F387AD21)
/// </summary>
internal static class KernelSpirvResolution
{
    /// <summary>
    /// Resolves either a SPIR-V file path or embedded bytes for <paramref name="kernelId"/>.
    /// Exactly one of the two return values is non-null. Throws
    /// <see cref="FileNotFoundException"/> if neither a file nor embedded bytes exist.
    /// </summary>
    public static (string? Path, byte[]? Embedded) Resolve(string kernelId)
    {
        string? path = KernelCatalog.ResolveSpirvPath(kernelId);
        if (path is not null)
        {
            return (path, null);
        }

        byte[]? embedded = KernelCatalog.LoadEmbeddedSpirv(kernelId);
        return embedded is not null
            ? (null, embedded)
            : throw new FileNotFoundException($"SPIR-V not found for {kernelId}.");
    }
}

/// <summary>
/// Shared kernel-entrypoint fallback logic used by kernels that support both a current and a
/// legacy entrypoint name. Extracted from 3 identical implementations.
/// (see azure/dry-cohorts.json, cohort 1B2A5E5549DB...)
/// </summary>
internal static class KernelEntrypointResolution
{
    public static ComputeKernel GetKernelWithFallback(ComputeModule module, string kernelName, string defaultKernelName, string legacyKernelName)
    {
        string requested = string.IsNullOrWhiteSpace(kernelName)
            ? defaultKernelName
            : kernelName;

        ComputeKernel? kernel = module.TryGetKernel(requested);
        if (kernel is not null)
        {
            return kernel;
        }

        if (!string.Equals(requested, defaultKernelName, StringComparison.OrdinalIgnoreCase))
        {
            kernel = module.TryGetKernel(defaultKernelName) ?? module.TryGetKernel(legacyKernelName);
            if (kernel is not null)
            {
                return kernel;
            }
        }
        else
        {
            kernel = module.TryGetKernel(legacyKernelName);
            if (kernel is not null)
            {
                return kernel;
            }
        }

        throw new InvalidOperationException(
            $"Kernel entrypoint '{requested}' not found. Tried '{defaultKernelName}' and '{legacyKernelName}' fallbacks.");
    }
}
