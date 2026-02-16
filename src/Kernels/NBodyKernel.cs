namespace LevelZero.Kernels;

/// <summary>
/// GPU-accelerated N-body repulsion force computation for force-directed graph layout.
/// </summary>
public sealed class NBodyKernel : IDisposable
{
    private readonly ComputeDevice _device;
    private readonly ComputeModule _module;
    private readonly ComputeKernel _kernel;

    private NBodyKernel(ComputeDevice device, ComputeModule module, ComputeKernel kernel)
    {
        _device = device;
        _module = module;
        _kernel = kernel;
    }

    /// <summary>Creates an N-body repulsion kernel, auto-resolving SPIR-V from disk or embedded resources.</summary>
    public static NBodyKernel Create(ComputeDevice device)
    {
        var path = KernelCatalog.ResolveSpirvPath("nbody_repulsion");
        if (path is not null) return Create(device, path);
        var embedded = KernelCatalog.LoadEmbeddedSpirv("nbody_repulsion");
        if (embedded is not null) return Create(device, embedded);
        throw new FileNotFoundException("SPIR-V not found for nbody_repulsion.");
    }

    public static NBodyKernel Create(ComputeDevice device, string spirvPath, string kernelName = "nbody_repulsion")
    {
        var module = device.LoadModule(spirvPath);
        var kernel = module.GetKernel(kernelName);
        return new NBodyKernel(device, module, kernel);
    }

    public static NBodyKernel Create(ComputeDevice device, byte[] spirv, string kernelName = "nbody_repulsion")
    {
        var module = device.LoadModule(spirv);
        var kernel = module.GetKernel(kernelName);
        return new NBodyKernel(device, module, kernel);
    }

    /// <summary>
    /// Computes repulsive forces for all nodes. Returns (forceX, forceY) arrays.
    /// </summary>
    public (float[] forceX, float[] forceY) Evaluate(float[] posX, float[] posY, int nodeCount, float repulsionK, float minDist)
    {
        using var pxBuf = _device.AllocShared(posX);
        using var pyBuf = _device.AllocShared(posY);
        using var fxBuf = _device.AllocShared<float>(nodeCount);
        using var fyBuf = _device.AllocShared<float>(nodeCount);

        _kernel.SetArgBuffer(0, pxBuf);
        _kernel.SetArgBuffer(1, pyBuf);
        _kernel.SetArgInt(2, nodeCount);
        _kernel.SetArgFloat(3, repulsionK);
        _kernel.SetArgFloat(4, minDist);
        _kernel.SetArgBuffer(5, fxBuf);
        _kernel.SetArgBuffer(6, fyBuf);

        const uint localSize = 64;
        _kernel.SetGroupSize(localSize);
        _device.Launch(_kernel, ComputeDevice.GroupCount(nodeCount, localSize));

        return (fxBuf.ToArray(), fyBuf.ToArray());
    }

    public void Dispose()
    {
        _kernel.Dispose();
        _module.Dispose();
    }
}
