namespace LevelZero.Kernels;

/// <summary>
/// GPU-accelerated suggestion scoring kernel.
/// Evaluates feature vectors against a fixed weight profile.
/// </summary>
public sealed class SuggestionKernel : IDisposable
{
    public const string DefaultKernelName = "score_kernel";
    public const string LegacyKernelName = "suggestion_score";

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
        var (path, embedded) = KernelSpirvResolution.Resolve("levelzero-score");
        return path is not null ? Create(device, path) : Create(device, embedded!);
    }

    public static SuggestionKernel Create(ComputeDevice device, string spirvPath, string kernelName = DefaultKernelName)
    {
        ComputeModule module = device.LoadModule(spirvPath);
        ComputeKernel kernel = KernelEntrypointResolution.GetKernelWithFallback(module, kernelName, DefaultKernelName, LegacyKernelName);
        return new SuggestionKernel(device, module, kernel);
    }

    public static SuggestionKernel Create(ComputeDevice device, byte[] spirv, string kernelName = DefaultKernelName)
    {
        ComputeModule module = device.LoadModule(spirv);
        ComputeKernel kernel = KernelEntrypointResolution.GetKernelWithFallback(module, kernelName, DefaultKernelName, LegacyKernelName);
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
        {
            throw new ArgumentException("features array must be count*4 length", nameof(features));
        }

        float[] weights = [4f, 5f, 2f, 3f];

        using SharedBuffer<float> featBuf = _device.AllocShared(features);
        using SharedBuffer<float> wBuf = _device.AllocShared(weights);
        using SharedBuffer<float> outBuf = _device.AllocShared<float>(count);

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

