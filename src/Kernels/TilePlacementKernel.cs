namespace LevelZero.Kernels;

/// <summary>
/// GPU-accelerated tile placement cost-matrix computation.
/// Evaluates assignment costs of agents to IPU tiles.
/// </summary>
public sealed class TilePlacementKernel : IDisposable
{
    private readonly ComputeDevice _device;
    private readonly ComputeModule _module;
    private readonly ComputeKernel _kernel;

    private TilePlacementKernel(ComputeDevice device, ComputeModule module, ComputeKernel kernel)
    {
        _device = device;
        _module = module;
        _kernel = kernel;
    }

    /// <summary>Creates a tile placement kernel, auto-resolving SPIR-V from disk or embedded resources.</summary>
    public static TilePlacementKernel Create(ComputeDevice device)
    {
        var path = KernelCatalog.ResolveSpirvPath("tile_placement");
        if (path is not null) return Create(device, path);
        var embedded = KernelCatalog.LoadEmbeddedSpirv("tile_placement");
        if (embedded is not null) return Create(device, embedded);
        throw new FileNotFoundException("SPIR-V not found for tile_placement.");
    }

    public static TilePlacementKernel Create(ComputeDevice device, string spirvPath, string kernelName = "tile_placement_cost")
    {
        var module = device.LoadModule(spirvPath);
        var kernel = module.GetKernel(kernelName);
        return new TilePlacementKernel(device, module, kernel);
    }

    public static TilePlacementKernel Create(ComputeDevice device, byte[] spirv, string kernelName = "tile_placement_cost")
    {
        var module = device.LoadModule(spirv);
        var kernel = module.GetKernel(kernelName);
        return new TilePlacementKernel(device, module, kernel);
    }

    /// <summary>
    /// Computes the cost matrix for assigning agents to tiles (2D dispatch).
    /// Returns a flat [agentCount x tileCount] cost array.
    /// </summary>
    public float[] Evaluate(
        float[] agentX, float[] agentY,
        float[] tileX, float[] tileY,
        float[] commWeights,
        int[] tileCapacity, int[] tileUsed,
        int agentCount, int tileCount,
        float latencyPerUnit)
    {
        using var axBuf = _device.AllocShared(agentX);
        using var ayBuf = _device.AllocShared(agentY);
        using var txBuf = _device.AllocShared(tileX);
        using var tyBuf = _device.AllocShared(tileY);
        using var cwBuf = _device.AllocShared(commWeights);
        using var tcBuf = _device.AllocShared(tileCapacity);
        using var tuBuf = _device.AllocShared(tileUsed);
        using var outBuf = _device.AllocShared<float>(agentCount * tileCount);

        _kernel.SetArgBuffer(0, axBuf);
        _kernel.SetArgBuffer(1, ayBuf);
        _kernel.SetArgBuffer(2, txBuf);
        _kernel.SetArgBuffer(3, tyBuf);
        _kernel.SetArgBuffer(4, cwBuf);
        _kernel.SetArgBuffer(5, tcBuf);
        _kernel.SetArgBuffer(6, tuBuf);
        _kernel.SetArgBuffer(7, outBuf);
        _kernel.SetArgInt(8, agentCount);
        _kernel.SetArgInt(9, tileCount);
        _kernel.SetArgFloat(10, latencyPerUnit);

        const uint localX = 8, localY = 8;
        _kernel.SetGroupSize(localX, localY);
        _device.Launch(_kernel,
            ComputeDevice.GroupCount(agentCount, localX),
            ComputeDevice.GroupCount(tileCount, localY));

        return outBuf.ToArray();
    }

    public void Dispose()
    {
        _kernel.Dispose();
        _module.Dispose();
    }
}
