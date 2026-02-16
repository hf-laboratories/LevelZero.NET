namespace LevelZero.Kernels;

/// <summary>
/// GPU-accelerated suggestion scoring kernel.
/// Evaluates feature vectors against a fixed weight profile.
/// </summary>
public sealed class SuggestionKernel : IDisposable
{
    private readonly ComputeDevice _device;
    private readonly ComputeModule _module;
    private readonly ComputeKernel _kernel;

    private SuggestionKernel(ComputeDevice device, ComputeModule module, ComputeKernel kernel)
    {
        _device = device;
        _module = module;
        _kernel = kernel;
    }

    /// <summary>Creates a suggestion kernel, auto-resolving SPIR-V from disk or embedded resources.</summary>
    public static SuggestionKernel Create(ComputeDevice device)
    {
        var path = KernelCatalog.ResolveSpirvPath("levelzero-score");
        if (path is not null) return Create(device, path);
        var embedded = KernelCatalog.LoadEmbeddedSpirv("levelzero-score");
        if (embedded is not null) return Create(device, embedded);
        throw new FileNotFoundException("SPIR-V not found for levelzero-score.");
    }

    public static SuggestionKernel Create(ComputeDevice device, string spirvPath, string kernelName = "suggestion_score")
    {
        var module = device.LoadModule(spirvPath);
        var kernel = module.GetKernel(kernelName);
        return new SuggestionKernel(device, module, kernel);
    }

    public static SuggestionKernel Create(ComputeDevice device, byte[] spirv, string kernelName = "suggestion_score")
    {
        var module = device.LoadModule(spirv);
        var kernel = module.GetKernel(kernelName);
        return new SuggestionKernel(device, module, kernel);
    }

    /// <summary>
    /// Scores feature vectors. Each candidate has 4 features.
    /// </summary>
    /// <param name="features">Flat array [count * 4].</param>
    /// <param name="count">Number of candidates.</param>
    /// <returns>float[count] scores.</returns>
    public float[] Score(float[] features, int count)
    {
        if (features.Length < count * 4)
            throw new ArgumentException("features array must be count*4 length", nameof(features));

        float[] weights = [4f, 5f, 2f, 3f];

        using var featBuf = _device.AllocShared(features);
        using var wBuf = _device.AllocShared(weights);
        using var outBuf = _device.AllocShared<float>(count);

        _kernel.SetArgBuffer(0, featBuf);
        _kernel.SetArgBuffer(1, wBuf);
        _kernel.SetArgBuffer(2, outBuf);
        _kernel.SetArgInt(3, count);

        const uint localSize = 64;
        _kernel.SetGroupSize(localSize);
        _device.Launch(_kernel, ComputeDevice.GroupCount(count, localSize));

        return outBuf.ToArray();
    }

    public void Dispose()
    {
        _kernel.Dispose();
        _module.Dispose();
    }
}
