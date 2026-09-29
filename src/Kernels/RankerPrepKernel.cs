namespace LevelZero.Kernels;

/// <summary>
/// GPU-accelerated ranker preparation kernel.
/// Computes adjusted scores by combining base scores and chain boosts.
/// </summary>
public sealed class RankerPrepKernel : IDisposable
{
    public const string DefaultKernelName = "ranker_prep_scores";

    private readonly ComputeDevice _device;
    private readonly ComputeModule _module;
    private readonly ComputeKernel _kernel;

    private RankerPrepKernel(ComputeDevice device, ComputeModule module, ComputeKernel kernel)
    {
        _device = device;
        _module = module;
        _kernel = kernel;
    }

    /// <summary>Creates a ranker-prep kernel, auto-resolving SPIR-V from disk or embedded resources.</summary>
    public static RankerPrepKernel Create(ComputeDevice device)
    {
        var (path, embedded) = KernelSpirvResolution.Resolve("levelzero-rankerprep");
        return path is not null ? Create(device, path) : Create(device, embedded!);
    }

    public static RankerPrepKernel Create(ComputeDevice device, string spirvPath, string kernelName = DefaultKernelName)
    {
        ComputeModule module = device.LoadModule(spirvPath);
        ComputeKernel kernel = module.GetKernel(kernelName);
        return new RankerPrepKernel(device, module, kernel);
    }

    public static RankerPrepKernel Create(ComputeDevice device, byte[] spirv, string kernelName = DefaultKernelName)
    {
        ComputeModule module = device.LoadModule(spirv);
        ComputeKernel kernel = module.GetKernel(kernelName);
        return new RankerPrepKernel(device, module, kernel);
    }

    /// <summary>
    /// Applies chain boosts to base scores: adjusted[i] = baseScores[i] + chainBoosts[i].
    /// </summary>
    public float[] PrepareScores(float[] baseScores, int[] chainBoosts, int count)
    {
        if (count < 0)
        {
            throw new ArgumentOutOfRangeException(nameof(count));
        }

        if (baseScores.Length < count)
        {
            throw new ArgumentException("baseScores length must be >= count", nameof(baseScores));
        }

        if (chainBoosts.Length < count)
        {
            throw new ArgumentException("chainBoosts length must be >= count", nameof(chainBoosts));
        }

        if (count == 0)
        {
            return [];
        }

        using SharedBuffer<float> baseBuf = _device.AllocShared(baseScores);
        using SharedBuffer<int> boostBuf = _device.AllocShared(chainBoosts);
        using SharedBuffer<float> outBuf = _device.AllocShared<float>(count);

        _kernel.SetArgBuffer(0, baseBuf);
        _kernel.SetArgBuffer(1, boostBuf);
        _kernel.SetArgBuffer(2, outBuf);
        _kernel.SetArgInt(3, count);

        const uint localSize = 64;
        _kernel.SetGroupSize(localSize);
        _device.Launch(_kernel, ComputeDevice.GroupCount(count, localSize));

        return outBuf.ToArray();
    }

    /// <summary>
    /// Compatibility alias matching the standalone LevelZeroBindings API.
    /// </summary>
    public float[] Prepare(float[] baseScores, int[] chainBoosts, int count)
        => PrepareScores(baseScores, chainBoosts, count);

    public void Dispose()
    {
        _kernel.Dispose();
        _module.Dispose();
    }
}
