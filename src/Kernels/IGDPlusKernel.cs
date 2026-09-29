namespace LevelZero.Kernels;

/// <summary>
/// GPU-accelerated IGD+ (Inverted Generational Distance Plus) directional distance kernel.
/// Computes, for each reference point, the minimum directional (clipped) distance to any
/// solution in the approximation set.
/// </summary>
public sealed class IGDPlusKernel : IDisposable
{
    private readonly ComputeDevice _device;
    private readonly ComputeModule _module;
    private readonly ComputeKernel _minDistKernel;

    private IGDPlusKernel(ComputeDevice device, ComputeModule module, ComputeKernel minDistKernel)
    {
        _device = device;
        _module = module;
        _minDistKernel = minDistKernel;
    }

    /// <summary>Creates an IGDPlus kernel by auto-resolving module bytes from disk/resources.</summary>
    public static IGDPlusKernel Create(ComputeDevice device)
    {
        var (path, embedded) = KernelSpirvResolution.Resolve("igdplus_distances");
        return path is not null ? Create(device, path) : Create(device, embedded!);
    }

    /// <summary>Creates an IGDPlus kernel from a compiled SPIR-V or native module file.</summary>
    public static IGDPlusKernel Create(ComputeDevice device, string modulePath, string kernelName = "igdplus_min_distances")
    {
        ComputeModule module = device.LoadModule(modulePath);
        ComputeKernel kernel = module.GetKernel(kernelName);
        return new IGDPlusKernel(device, module, kernel);
    }

    /// <summary>Creates an IGDPlus kernel from raw module bytes.</summary>
    public static IGDPlusKernel Create(ComputeDevice device, byte[] moduleBytes, string kernelName = "igdplus_min_distances")
    {
        ComputeModule module = device.LoadModule(moduleBytes);
        ComputeKernel kernel = module.GetKernel(kernelName);
        return new IGDPlusKernel(device, module, kernel);
    }

    /// <summary>
    /// Computes IGD+ minimum distances from each reference point to the nearest solution.
    /// </summary>
    /// <param name="solutions">Flat row-major float array [n_sol Ã— dims].</param>
    /// <param name="references">Flat row-major float array [n_ref Ã— dims].</param>
    /// <param name="solCount">Number of approximation solutions.</param>
    /// <param name="refCount">Number of reference points.</param>
    /// <param name="dims">Number of objectives.</param>
    /// <returns>Float array [n_ref] of minimum IGD+ distances.</returns>
    public float[] ComputeMinDistances(float[] solutions, float[] references, int solCount, int refCount, int dims)
    {
        using SharedBuffer<float> solBuf = _device.AllocShared(solutions);
        using SharedBuffer<float> refBuf = _device.AllocShared(references);
        using SharedBuffer<float> outBuf = _device.AllocShared<float>(refCount);

        _minDistKernel.SetArgBuffer(0, solBuf);
        _minDistKernel.SetArgBuffer(1, refBuf);
        _minDistKernel.SetArgBuffer(2, outBuf);
        _minDistKernel.SetArgInt(3, solCount);
        _minDistKernel.SetArgInt(4, refCount);
        _minDistKernel.SetArgInt(5, dims);

        const uint localSize = 64;
        _minDistKernel.SetGroupSize(localSize);
        _device.Launch(_minDistKernel, ComputeDevice.GroupCount(refCount, localSize));

        return outBuf.ToArray();
    }

    public void Dispose()
    {
        _minDistKernel.Dispose();
        _module.Dispose();
    }
}

