namespace LevelZero.Kernels;

/// <summary>
/// GPU-accelerated Monte Carlo hypervolume estimation.
/// Offloads the sampling-and-dominance-check loop to GPU.
/// </summary>
public sealed class MonteCarloHypervolumeKernel : IDisposable
{
    private readonly ComputeDevice _device;
    private readonly ComputeModule _module;
    private readonly ComputeKernel _kernel;

    private MonteCarloHypervolumeKernel(ComputeDevice device, ComputeModule module, ComputeKernel kernel)
    {
        _device = device;
        _module = module;
        _kernel = kernel;
    }

    /// <summary>Creates a Monte Carlo hypervolume kernel, auto-resolving SPIR-V from disk or embedded resources.</summary>
    public static MonteCarloHypervolumeKernel Create(ComputeDevice device)
    {
        var path = KernelCatalog.ResolveSpirvPath("monte_carlo_hypervolume");
        if (path is not null) return Create(device, path);
        var embedded = KernelCatalog.LoadEmbeddedSpirv("monte_carlo_hypervolume");
        if (embedded is not null) return Create(device, embedded);
        throw new FileNotFoundException("SPIR-V not found for monte_carlo_hypervolume.");
    }

    public static MonteCarloHypervolumeKernel Create(ComputeDevice device, string spirvPath, string kernelName = "mc_hypervolume_stable")
    {
        var module = device.LoadModule(spirvPath);
        var kernel = module.GetKernel(kernelName);
        return new MonteCarloHypervolumeKernel(device, module, kernel);
    }

    public static MonteCarloHypervolumeKernel Create(ComputeDevice device, byte[] spirv, string kernelName = "mc_hypervolume_stable")
    {
        var module = device.LoadModule(spirv);
        var kernel = module.GetKernel(kernelName);
        return new MonteCarloHypervolumeKernel(device, module, kernel);
    }

    /// <summary>
    /// Runs Monte Carlo hypervolume estimation on GPU.
    /// Returns the count of dominated samples out of <paramref name="sampleCount"/>.
    /// </summary>
    public int Evaluate(float[] solutions, float[] idealPoint, float[] referencePoint,
                        int solCount, int objCount, int sampleCount, uint rngSeed = 42)
    {
        // Generate per-work-item seeds (4 uints each for xoshiro128**)
        var seeds = new uint[sampleCount * 4];
        for (int i = 0; i < sampleCount; i++)
        {
            seeds[i * 4 + 0] = rngSeed + (uint)i * 2654435761u;
            seeds[i * 4 + 1] = rngSeed + (uint)i * 1013904223u + 1;
            seeds[i * 4 + 2] = rngSeed + (uint)i * 2246822519u + 2;
            seeds[i * 4 + 3] = rngSeed + (uint)i * 3266489917u + 3;
        }

        // Copy uint[] seeds via byte intermediate
        var seedBytes = new byte[seeds.Length * sizeof(uint)];
        Buffer.BlockCopy(seeds, 0, seedBytes, 0, seedBytes.Length);

        using var solBuf = _device.AllocShared(solutions);
        using var idealBuf = _device.AllocShared(idealPoint);
        using var refBuf = _device.AllocShared(referencePoint);
        using var seedBuf = _device.AllocShared<byte>(seedBytes.Length);
        using var outBuf = _device.AllocShared<int>(sampleCount);

        seedBuf.Write(seedBytes);

        _kernel.SetArgBuffer(0, solBuf);
        _kernel.SetArgBuffer(1, idealBuf);
        _kernel.SetArgBuffer(2, refBuf);
        _kernel.SetArgBuffer(3, seedBuf);
        _kernel.SetArgInt(4, solCount);
        _kernel.SetArgInt(5, objCount);
        _kernel.SetArgBuffer(6, outBuf);

        const uint localSize = 64;
        _kernel.SetGroupSize(localSize);
        _device.Launch(_kernel, ComputeDevice.GroupCount(sampleCount, localSize));

        var dominated = outBuf.ToArray();
        int count = 0;
        for (int i = 0; i < sampleCount; i++)
            count += dominated[i];
        return count;
    }

    public void Dispose()
    {
        _kernel.Dispose();
        _module.Dispose();
    }
}
