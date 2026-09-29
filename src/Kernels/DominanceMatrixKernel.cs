namespace LevelZero.Kernels;

/// <summary>
/// GPU-accelerated dominance counting and matrix computation for multi-objective optimization.
/// Loads up to 3 kernels: dominance_count, dominance_matrix (optional), min_distance_to_set (optional).
/// </summary>
public sealed class DominanceMatrixKernel : IDisposable
{
    private readonly ComputeDevice _device;
    private readonly ComputeModule _module;
    private readonly ComputeKernel _countKernel;
    private readonly ComputeKernel? _matrixKernel;
    private readonly ComputeKernel? _minDistKernel;

    private DominanceMatrixKernel(ComputeDevice device, ComputeModule module,
                                  ComputeKernel countKernel, ComputeKernel? matrixKernel, ComputeKernel? minDistKernel)
    {
        _device = device;
        _module = module;
        _countKernel = countKernel;
        _matrixKernel = matrixKernel;
        _minDistKernel = minDistKernel;
    }

    /// <summary>Creates a dominance matrix kernel, auto-resolving SPIR-V from disk or embedded resources.</summary>
    public static DominanceMatrixKernel Create(ComputeDevice device)
    {
        var (path, embedded) = KernelSpirvResolution.Resolve("dominance_matrix");
        return path is not null ? Create(device, path) : Create(device, embedded!);
    }

    public static DominanceMatrixKernel Create(ComputeDevice device, string spirvPath, string primaryKernelName = "dominance_count")
    {
        ComputeModule module = device.LoadModule(spirvPath);
        ComputeKernel count = module.GetKernel(primaryKernelName);
        ComputeKernel? matrix = module.TryGetKernel("dominance_matrix");
        ComputeKernel? minDist = module.TryGetKernel("min_distance_to_set");
        return new DominanceMatrixKernel(device, module, count, matrix, minDist);
    }

    public static DominanceMatrixKernel Create(ComputeDevice device, byte[] spirv, string primaryKernelName = "dominance_count")
    {
        ComputeModule module = device.LoadModule(spirv);
        ComputeKernel count = module.GetKernel(primaryKernelName);
        ComputeKernel? matrix = module.TryGetKernel("dominance_matrix");
        ComputeKernel? minDist = module.TryGetKernel("min_distance_to_set");
        return new DominanceMatrixKernel(device, module, count, matrix, minDist);
    }

    /// <summary>
    /// Computes domination count for each solution (how many other solutions dominate it).
    /// </summary>
    public int[] ComputeDominationCounts(float[] objectives, int solCount, int objCount)
    {
        using SharedBuffer<float> objBuf = _device.AllocShared(objectives);
        using SharedBuffer<int> outBuf = _device.AllocShared<int>(solCount);

        _countKernel.SetArgBuffer(0, objBuf);
        _countKernel.SetArgInt(1, solCount);
        _countKernel.SetArgInt(2, objCount);
        _countKernel.SetArgBuffer(3, outBuf);

        const uint localSize = 64;
        _countKernel.SetGroupSize(localSize);
        _device.Launch(_countKernel, ComputeDevice.GroupCount(solCount, localSize));

        return outBuf.ToArray();
    }

    /// <summary>
    /// Computes full N x N dominance comparison matrix.
    /// Falls back to count-based approach if matrix kernel not available.
    /// </summary>
    public int[] ComputeDominanceMatrix(float[] objectives, int solCount, int objCount)
    {
        if (_matrixKernel is null)
        {
            return ComputeDominationCounts(objectives, solCount, objCount);
        }

        int totalPairs = solCount * (solCount - 1) / 2;

        using SharedBuffer<float> objBuf = _device.AllocShared(objectives);
        using SharedBuffer<int> outBuf = _device.AllocShared<int>(solCount * solCount);

        _matrixKernel.SetArgBuffer(0, objBuf);
        _matrixKernel.SetArgInt(1, solCount);
        _matrixKernel.SetArgInt(2, objCount);
        _matrixKernel.SetArgBuffer(3, outBuf);

        const uint localSize = 64;
        _matrixKernel.SetGroupSize(localSize);
        _device.Launch(_matrixKernel, ComputeDevice.GroupCount(totalPairs, localSize));

        return outBuf.ToArray();
    }

    /// <summary>
    /// Computes minimum distance from each reference point to the nearest solution (IGD/IGD+).
    /// </summary>
    public float[] ComputeMinDistances(float[] solutions, float[] referenceSet,
                                       int solCount, int refCount, int objCount)
    {
        if (_minDistKernel is null)
        {
            throw new InvalidOperationException("min_distance_to_set kernel not available in this module");
        }

        using SharedBuffer<float> solBuf = _device.AllocShared(solutions);
        using SharedBuffer<float> refBuf = _device.AllocShared(referenceSet);
        using SharedBuffer<float> outBuf = _device.AllocShared<float>(refCount);

        _minDistKernel.SetArgBuffer(0, solBuf);
        _minDistKernel.SetArgBuffer(1, refBuf);
        _minDistKernel.SetArgInt(2, solCount);
        _minDistKernel.SetArgInt(3, refCount);
        _minDistKernel.SetArgInt(4, objCount);
        _minDistKernel.SetArgBuffer(5, outBuf);

        const uint localSize = 64;
        _minDistKernel.SetGroupSize(localSize);
        _device.Launch(_minDistKernel, ComputeDevice.GroupCount(refCount, localSize));

        return outBuf.ToArray();
    }

    public void Dispose()
    {
        _minDistKernel?.Dispose();
        _matrixKernel?.Dispose();
        _countKernel.Dispose();
        _module.Dispose();
    }
}

