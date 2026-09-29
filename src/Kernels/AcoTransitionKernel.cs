namespace LevelZero.Kernels;

/// <summary>
/// GPU-accelerated ACO shortlist transition selection.
/// Each work-item scores one ant's candidate shortlist and samples the next node.
/// </summary>
public sealed class AcoTransitionKernel : IDisposable
{
    private readonly ComputeDevice _device;
    private readonly ComputeModule _module;
    private readonly ComputeKernel _selectionKernel;

    private AcoTransitionKernel(ComputeDevice device, ComputeModule module, ComputeKernel selectionKernel)
    {
        _device = device;
        _module = module;
        _selectionKernel = selectionKernel;
    }

    public static AcoTransitionKernel Create(ComputeDevice device)
    {
        var (path, embedded) = KernelSpirvResolution.Resolve("aco_transition");
        return path is not null ? Create(device, path) : Create(device, embedded!);
    }

    public static AcoTransitionKernel Create(ComputeDevice device, string modulePath, string kernelName = "aco_transition_select")
    {
        ComputeModule module = device.LoadModule(modulePath);
        ComputeKernel kernel = module.GetKernel(kernelName);
        return new AcoTransitionKernel(device, module, kernel);
    }

    public static AcoTransitionKernel Create(ComputeDevice device, byte[] moduleBytes, string kernelName = "aco_transition_select")
    {
        ComputeModule module = device.LoadModule(moduleBytes);
        ComputeKernel kernel = module.GetKernel(kernelName);
        return new AcoTransitionKernel(device, module, kernel);
    }

    public int[] SelectNextNodes(
        int[] currentNodes,
        int[] candidateNodes,
        float[] candidatePheromones,
        int[] candidateCounts,
        float[] selectionRandoms,
        int candidateCapacity,
        float alpha,
        float beta,
        float selectionTemperature)
    {
        ArgumentNullException.ThrowIfNull(currentNodes);
        ArgumentNullException.ThrowIfNull(candidateNodes);
        ArgumentNullException.ThrowIfNull(candidatePheromones);
        ArgumentNullException.ThrowIfNull(candidateCounts);
        ArgumentNullException.ThrowIfNull(selectionRandoms);

        int antCount = currentNodes.Length;
        if (antCount == 0)
        {
            return [];
        }

        if (candidateCounts.Length != antCount || selectionRandoms.Length != antCount)
        {
            throw new ArgumentException("ACO transition inputs must share the same ant count.");
        }

        if (candidateCapacity <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(candidateCapacity), "ACO candidate capacity must be positive.");
        }

        int expectedCandidateLength = antCount * candidateCapacity;
        if (candidateNodes.Length != expectedCandidateLength || candidatePheromones.Length != expectedCandidateLength)
        {
            throw new ArgumentException("ACO candidate node and pheromone buffers must equal antCount * candidateCapacity.");
        }

        using SharedBuffer<int> currentBuf = _device.AllocShared(currentNodes);
        using SharedBuffer<int> candidateNodeBuf = _device.AllocShared(candidateNodes);
        using SharedBuffer<float> candidatePheromoneBuf = _device.AllocShared(candidatePheromones);
        using SharedBuffer<int> candidateCountBuf = _device.AllocShared(candidateCounts);
        using SharedBuffer<float> selectionRandomBuf = _device.AllocShared(selectionRandoms);
        using SharedBuffer<int> selectedNodeBuf = _device.AllocShared<int>(antCount);

        _selectionKernel.SetArgBuffer(0, currentBuf);
        _selectionKernel.SetArgBuffer(1, candidateNodeBuf);
        _selectionKernel.SetArgBuffer(2, candidatePheromoneBuf);
        _selectionKernel.SetArgBuffer(3, candidateCountBuf);
        _selectionKernel.SetArgBuffer(4, selectionRandomBuf);
        _selectionKernel.SetArgInt(5, antCount);
        _selectionKernel.SetArgInt(6, candidateCapacity);
        _selectionKernel.SetArgFloat(7, alpha);
        _selectionKernel.SetArgFloat(8, beta);
        _selectionKernel.SetArgFloat(9, selectionTemperature);
        _selectionKernel.SetArgBuffer(10, selectedNodeBuf);

        const uint localSize = 64;
        _selectionKernel.SetGroupSize(localSize);
        _device.Launch(_selectionKernel, ComputeDevice.GroupCount(antCount, localSize));

        return selectedNodeBuf.ToArray();
    }

    public void Dispose()
    {
        _selectionKernel.Dispose();
        _module.Dispose();
    }
}
