namespace LevelZero.Kernels;

/// <summary>
/// Level Zero kernel wrapper for chain-boost accumulation.
/// For each edge, accumulates weighted boosts to target commands based on active sources.
/// </summary>
public sealed class ChainBoostKernel : IDisposable
{
    public const string DefaultKernelName = "chain_boost_accumulate";

    private readonly ComputeDevice _device;
    private readonly ComputeModule _module;
    private readonly ComputeKernel _kernel;

    private ChainBoostKernel(ComputeDevice device, ComputeModule module, ComputeKernel kernel)
    {
        _device = device;
        _module = module;
        _kernel = kernel;
    }

    /// <summary>Creates a chain-boost kernel, auto-resolving SPIR-V from disk or embedded resources.</summary>
    public static ChainBoostKernel Create(ComputeDevice device)
    {
        var (path, embedded) = KernelSpirvResolution.Resolve("levelzero-chainboost");
        return path is not null ? Create(device, path) : Create(device, embedded!);
    }

    public static ChainBoostKernel Create(ComputeDevice device, string spirvPath, string kernelName = DefaultKernelName)
    {
        ComputeModule module = device.LoadModule(spirvPath);
        ComputeKernel kernel = module.GetKernel(kernelName);
        return new ChainBoostKernel(device, module, kernel);
    }

    public static ChainBoostKernel Create(ComputeDevice device, byte[] spirv, string kernelName = DefaultKernelName)
    {
        ComputeModule module = device.LoadModule(spirv);
        ComputeKernel kernel = module.GetKernel(kernelName);
        return new ChainBoostKernel(device, module, kernel);
    }

    /// <summary>
    /// Accumulates chain boosts into an output array of length <paramref name="commandCount"/>.
    /// </summary>
    public int[] Accumulate(
        int[] edgeFrom,
        int[] edgeTo,
        int[] edgeWeight,
        int edgeCount,
        int[] sourceIndices,
        int[] sourceMultipliers,
        int sourceCount,
        int commandCount)
    {
        if (edgeCount < 0)
        {
            throw new ArgumentOutOfRangeException(nameof(edgeCount));
        }

        if (sourceCount < 0)
        {
            throw new ArgumentOutOfRangeException(nameof(sourceCount));
        }

        if (commandCount < 0)
        {
            throw new ArgumentOutOfRangeException(nameof(commandCount));
        }

        if (edgeFrom.Length < edgeCount)
        {
            throw new ArgumentException("edgeFrom length must be >= edgeCount", nameof(edgeFrom));
        }

        if (edgeTo.Length < edgeCount)
        {
            throw new ArgumentException("edgeTo length must be >= edgeCount", nameof(edgeTo));
        }

        if (edgeWeight.Length < edgeCount)
        {
            throw new ArgumentException("edgeWeight length must be >= edgeCount", nameof(edgeWeight));
        }

        if (sourceIndices.Length < sourceCount)
        {
            throw new ArgumentException("sourceIndices length must be >= sourceCount", nameof(sourceIndices));
        }

        if (sourceMultipliers.Length < sourceCount)
        {
            throw new ArgumentException("sourceMultipliers length must be >= sourceCount", nameof(sourceMultipliers));
        }

        if (commandCount == 0)
        {
            return [];
        }

        if (edgeCount == 0 || sourceCount == 0)
        {
            return new int[commandCount];
        }

        using SharedBuffer<int> edgeFromBuf = _device.AllocShared(edgeFrom);
        using SharedBuffer<int> edgeToBuf = _device.AllocShared(edgeTo);
        using SharedBuffer<int> edgeWeightBuf = _device.AllocShared(edgeWeight);
        using SharedBuffer<int> sourceIndicesBuf = _device.AllocShared(sourceIndices);
        using SharedBuffer<int> sourceMultipliersBuf = _device.AllocShared(sourceMultipliers);
        using SharedBuffer<int> outBuf = _device.AllocShared(new int[commandCount]);

        _kernel.SetArgBuffer(0, edgeFromBuf);
        _kernel.SetArgBuffer(1, edgeToBuf);
        _kernel.SetArgBuffer(2, edgeWeightBuf);
        _kernel.SetArgInt(3, edgeCount);
        _kernel.SetArgBuffer(4, sourceIndicesBuf);
        _kernel.SetArgBuffer(5, sourceMultipliersBuf);
        _kernel.SetArgInt(6, sourceCount);
        _kernel.SetArgBuffer(7, outBuf);
        _kernel.SetArgInt(8, commandCount);

        const uint localSize = 64;
        _kernel.SetGroupSize(localSize);
        _device.Launch(_kernel, ComputeDevice.GroupCount(edgeCount, localSize));

        return outBuf.ToArray();
    }

    /// <summary>
    /// Compatibility overload matching standalone LevelZeroBindings signature.
    /// </summary>
    public int[] Accumulate(
        int[] edgeFrom,
        int[] edgeTo,
        int[] edgeWeight,
        int[] sourceIndices,
        int[] sourceMultipliers,
        int commandCount)
        => Accumulate(
            edgeFrom,
            edgeTo,
            edgeWeight,
            edgeFrom.Length,
            sourceIndices,
            sourceMultipliers,
            sourceIndices.Length,
            commandCount);

    public void Dispose()
    {
        _kernel.Dispose();
        _module.Dispose();
    }
}
