namespace LevelZero.Kernels;

/// <summary>
/// GPU-accelerated exchange topology weight computation.
/// Produces an NxN weight matrix based on hop-distance latency modeling.
/// </summary>
public sealed class ExchangeWeightsKernel : IDisposable
{
    private readonly ComputeDevice _device;
    private readonly ComputeModule _module;
    private readonly ComputeKernel _kernel;

    private ExchangeWeightsKernel(ComputeDevice device, ComputeModule module, ComputeKernel kernel)
    {
        _device = device;
        _module = module;
        _kernel = kernel;
    }

    /// <summary>Creates an exchange weights kernel, auto-resolving SPIR-V from disk or embedded resources.</summary>
    public static ExchangeWeightsKernel Create(ComputeDevice device)
    {
        var path = KernelCatalog.ResolveSpirvPath("exchange_weights");
        if (path is not null) return Create(device, path);
        var embedded = KernelCatalog.LoadEmbeddedSpirv("exchange_weights");
        if (embedded is not null) return Create(device, embedded);
        throw new FileNotFoundException("SPIR-V not found for exchange_weights.");
    }

    public static ExchangeWeightsKernel Create(ComputeDevice device, string spirvPath, string kernelName = "exchange_weights")
    {
        var module = device.LoadModule(spirvPath);
        var kernel = module.GetKernel(kernelName);
        return new ExchangeWeightsKernel(device, module, kernel);
    }

    public static ExchangeWeightsKernel Create(ComputeDevice device, byte[] spirv, string kernelName = "exchange_weights")
    {
        var module = device.LoadModule(spirv);
        var kernel = module.GetKernel(kernelName);
        return new ExchangeWeightsKernel(device, module, kernel);
    }

    /// <summary>
    /// Computes exchange topology weights. Returns flat [swarmSize x swarmSize] matrix (2D dispatch).
    /// </summary>
    public float[] Evaluate(int swarmSize, float baseLatency, float latencyPerHop)
    {
        int totalCells = swarmSize * swarmSize;

        using var outBuf = _device.AllocShared<float>(totalCells);

        _kernel.SetArgBuffer(0, outBuf);
        _kernel.SetArgInt(1, swarmSize);
        _kernel.SetArgFloat(2, baseLatency);
        _kernel.SetArgFloat(3, latencyPerHop);

        const uint localX = 8, localY = 8;
        _kernel.SetGroupSize(localX, localY);
        _device.Launch(_kernel,
            ComputeDevice.GroupCount(swarmSize, localX),
            ComputeDevice.GroupCount(swarmSize, localY));

        return outBuf.ToArray();
    }

    public void Dispose()
    {
        _kernel.Dispose();
        _module.Dispose();
    }
}
