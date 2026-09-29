namespace LevelZero.Kernels;

/// <summary>
/// GPU-accelerated Differential Evolution mutation helpers.
/// Currently exposes DE/current-to-best/1 vector mutation.
/// </summary>
public sealed class DeMutationKernel : IDisposable
{
    private readonly ComputeDevice _device;
    private readonly ComputeModule _module;
    private readonly ComputeKernel _currentToBestKernel;
    private readonly SharedBufferPool<float> _bufferPool;

    private DeMutationKernel(ComputeDevice device, ComputeModule module, ComputeKernel currentToBestKernel)
    {
        _device = device;
        _module = module;
        _currentToBestKernel = currentToBestKernel;
        _bufferPool = new SharedBufferPool<float>(device);
    }

    /// <summary>Creates a DE mutation kernel by auto-resolving module bytes from disk/resources.</summary>
    public static DeMutationKernel Create(ComputeDevice device)
    {
        var (path, embedded) = KernelSpirvResolution.Resolve("de_mutation");
        return path is not null ? Create(device, path) : Create(device, embedded!);
    }

    public static DeMutationKernel Create(ComputeDevice device, string modulePath, string kernelName = "de_current_to_best1")
    {
        ComputeModule module = device.LoadModule(modulePath);
        ComputeKernel kernel = module.GetKernel(kernelName);
        return new DeMutationKernel(device, module, kernel);
    }

    public static DeMutationKernel Create(ComputeDevice device, byte[] moduleBytes, string kernelName = "de_current_to_best1")
    {
        ComputeModule module = device.LoadModule(moduleBytes);
        ComputeKernel kernel = module.GetKernel(kernelName);
        return new DeMutationKernel(device, module, kernel);
    }

    /// <summary>
    /// Computes DE/current-to-best/1 mutation vector using pooled USM buffers (PAT-CS-242).
    /// </summary>
    public float[] ComputeCurrentToBest1(float[] current, float[] best, float[] r1, float[] r2, float scalingFactor)
    {
        int dims = current.Length;
        if (dims == 0)
        {
            return [];
        }

        if (best.Length != dims || r1.Length != dims || r2.Length != dims)
        {
            throw new ArgumentException("All mutation vectors must share the same dimensionality.");
        }

        using var currentScope = _bufferPool.RentScoped(dims);
        using var bestScope = _bufferPool.RentScoped(dims);
        using var r1Scope = _bufferPool.RentScoped(dims);
        using var r2Scope = _bufferPool.RentScoped(dims);
        using var outScope = _bufferPool.RentScoped(dims);

        var currentBuf = currentScope.Buffer;
        var bestBuf = bestScope.Buffer;
        var r1Buf = r1Scope.Buffer;
        var r2Buf = r2Scope.Buffer;
        var outBuf = outScope.Buffer;

        currentBuf.Write(current);
        bestBuf.Write(best);
        r1Buf.Write(r1);
        r2Buf.Write(r2);

        ComputeCurrentToBest1(currentBuf, bestBuf, r1Buf, r2Buf, outBuf, dims, scalingFactor);

        float[] result = new float[dims];
        outBuf.ReadTo(result.AsSpan());
        return result;
    }

    /// <summary>
    /// Executes DE/current-to-best/1 mutation directly on persistent device-resident buffers with zero allocations.
    /// </summary>
    public void ComputeCurrentToBest1(
        SharedBuffer<float> currentBuf,
        SharedBuffer<float> bestBuf,
        SharedBuffer<float> r1Buf,
        SharedBuffer<float> r2Buf,
        SharedBuffer<float> outBuf,
        int dims,
        float scalingFactor)
    {
        _currentToBestKernel.SetArgBuffer(0, currentBuf);
        _currentToBestKernel.SetArgBuffer(1, bestBuf);
        _currentToBestKernel.SetArgBuffer(2, r1Buf);
        _currentToBestKernel.SetArgBuffer(3, r2Buf);
        _currentToBestKernel.SetArgInt(4, dims);
        _currentToBestKernel.SetArgFloat(5, scalingFactor);
        _currentToBestKernel.SetArgBuffer(6, outBuf);

        const uint localSize = 64;
        _currentToBestKernel.SetGroupSize(localSize);
        _device.Launch(_currentToBestKernel, ComputeDevice.GroupCount(dims, localSize));
    }

    public void Dispose()
    {
        _bufferPool.Dispose();
        _currentToBestKernel.Dispose();
        _module.Dispose();
    }
}

