namespace LevelZero.Kernels;

/// <summary>
/// GPU-accelerated firefly attraction and flash coupling dynamics.
/// Loads up to 3 kernels: flash coupling (primary), firefly_attraction, kuramoto_components.
/// </summary>
public sealed class FireflyAttractionKernel : IDisposable
{
    private readonly ComputeDevice _device;
    private readonly ComputeModule _module;
    private readonly ComputeKernel _flashCouplingKernel;
    private readonly ComputeKernel? _attractionKernel;
    private readonly ComputeKernel? _kuramotoKernel;

    private FireflyAttractionKernel(ComputeDevice device, ComputeModule module,
                                     ComputeKernel flashCoupling, ComputeKernel? attraction, ComputeKernel? kuramoto)
    {
        _device = device;
        _module = module;
        _flashCouplingKernel = flashCoupling;
        _attractionKernel = attraction;
        _kuramotoKernel = kuramoto;
    }

    /// <summary>Creates a firefly attraction kernel, auto-resolving SPIR-V from disk or embedded resources.</summary>
    public static FireflyAttractionKernel Create(ComputeDevice device)
    {
        var path = KernelCatalog.ResolveSpirvPath("firefly_attraction");
        if (path is not null) return Create(device, path);
        var embedded = KernelCatalog.LoadEmbeddedSpirv("firefly_attraction");
        if (embedded is not null) return Create(device, embedded);
        throw new FileNotFoundException("SPIR-V not found for firefly_attraction.");
    }

    public static FireflyAttractionKernel Create(ComputeDevice device, string spirvPath, string kernelName = "flash_coupling")
    {
        var module = device.LoadModule(spirvPath);
        var flash = module.GetKernel(kernelName);
        var attraction = module.TryGetKernel("firefly_attraction");
        var kuramoto = module.TryGetKernel("kuramoto_components");
        return new FireflyAttractionKernel(device, module, flash, attraction, kuramoto);
    }

    public static FireflyAttractionKernel Create(ComputeDevice device, byte[] spirv, string kernelName = "flash_coupling")
    {
        var module = device.LoadModule(spirv);
        var flash = module.GetKernel(kernelName);
        var attraction = module.TryGetKernel("firefly_attraction");
        var kuramoto = module.TryGetKernel("kuramoto_components");
        return new FireflyAttractionKernel(device, module, flash, attraction, kuramoto);
    }

    /// <summary>
    /// Computes Kuramoto phase coupling perturbations from flashing neighbors.
    /// Returns per-firefly phase deltas.
    /// </summary>
    public float[] ComputeFlashCoupling(
        float[] positions, float[] phases, float[] intensities, int[] flashMask,
        int count, int dims, float couplingK, float flashRadius, float deltaTime)
    {
        using var posBuf = _device.AllocShared(positions);
        using var phBuf = _device.AllocShared(phases);
        using var intBuf = _device.AllocShared(intensities);
        using var maskBuf = _device.AllocShared(flashMask);
        using var outBuf = _device.AllocShared<float>(count);

        _flashCouplingKernel.SetArgBuffer(0, posBuf);
        _flashCouplingKernel.SetArgBuffer(1, phBuf);
        _flashCouplingKernel.SetArgBuffer(2, intBuf);
        _flashCouplingKernel.SetArgBuffer(3, maskBuf);
        _flashCouplingKernel.SetArgInt(4, count);
        _flashCouplingKernel.SetArgInt(5, dims);
        _flashCouplingKernel.SetArgFloat(6, couplingK);
        _flashCouplingKernel.SetArgFloat(7, flashRadius);
        _flashCouplingKernel.SetArgFloat(8, deltaTime);
        _flashCouplingKernel.SetArgBuffer(9, outBuf);

        const uint localSize = 64;
        _flashCouplingKernel.SetGroupSize(localSize);
        _device.Launch(_flashCouplingKernel, ComputeDevice.GroupCount(count, localSize));

        return outBuf.ToArray();
    }

    /// <summary>
    /// Computes standard firefly attraction: each firefly moves toward brighter neighbors.
    /// Returns flat position delta array [count * dims].
    /// </summary>
    public float[] ComputeAttraction(
        float[] positions, float[] intensities,
        int count, int dims, float beta0, float gamma)
    {
        if (_attractionKernel is null)
            throw new InvalidOperationException("Attraction kernel not available in this module");

        using var posBuf = _device.AllocShared(positions);
        using var intBuf = _device.AllocShared(intensities);
        using var outBuf = _device.AllocShared<float>(count * dims);

        _attractionKernel.SetArgBuffer(0, posBuf);
        _attractionKernel.SetArgBuffer(1, intBuf);
        _attractionKernel.SetArgInt(2, count);
        _attractionKernel.SetArgInt(3, dims);
        _attractionKernel.SetArgFloat(4, beta0);
        _attractionKernel.SetArgFloat(5, gamma);
        _attractionKernel.SetArgBuffer(6, outBuf);

        const uint localSize = 64;
        _attractionKernel.SetGroupSize(localSize);
        _device.Launch(_attractionKernel, ComputeDevice.GroupCount(count, localSize));

        return outBuf.ToArray();
    }

    /// <summary>
    /// Computes Kuramoto order parameter components (cos/sin of each phase).
    /// Returns (cosValues, sinValues).
    /// </summary>
    public (float[] cosVals, float[] sinVals) ComputeKuramotoComponents(float[] phases, int count)
    {
        if (_kuramotoKernel is null)
            throw new InvalidOperationException("Kuramoto kernel not available in this module");

        using var phBuf = _device.AllocShared(phases);
        using var cosBuf = _device.AllocShared<float>(count);
        using var sinBuf = _device.AllocShared<float>(count);

        _kuramotoKernel.SetArgBuffer(0, phBuf);
        _kuramotoKernel.SetArgInt(1, count);
        _kuramotoKernel.SetArgBuffer(2, cosBuf);
        _kuramotoKernel.SetArgBuffer(3, sinBuf);

        const uint localSize = 64;
        _kuramotoKernel.SetGroupSize(localSize);
        _device.Launch(_kuramotoKernel, ComputeDevice.GroupCount(count, localSize));

        return (cosBuf.ToArray(), sinBuf.ToArray());
    }

    public void Dispose()
    {
        _kuramotoKernel?.Dispose();
        _attractionKernel?.Dispose();
        _flashCouplingKernel.Dispose();
        _module.Dispose();
    }
}
