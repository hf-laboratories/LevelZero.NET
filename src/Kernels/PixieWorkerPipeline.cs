namespace LevelZero.Kernels;

using System;
using System.Threading;
using System.Threading.Tasks;

/// <summary>
/// Asynchronous host-to-GPU streaming pipeline for the Pixie cognitive weaver and BESS swarm reasoner.
/// Executes hybrid vector-graph evaluations and parallel particle swarm trajectory updates on Intel Iris Xe GT2 (tgllp).
/// </summary>
public sealed class PixieWorkerPipeline : IDisposable
{
    private readonly ComputeDevice _device;
    private readonly ComputeModule _module;
    private readonly ComputeKernel _hybridKernel;
    private readonly ComputeKernel _swarmKernel;
    private readonly SemaphoreSlim _gate = new(1, 1);
    private bool _disposed;

    private PixieWorkerPipeline(
        ComputeDevice device,
        ComputeModule module,
        ComputeKernel hybridKernel,
        ComputeKernel swarmKernel)
    {
        _device = device;
        _module = module;
        _hybridKernel = hybridKernel;
        _swarmKernel = swarmKernel;
    }

    /// <summary>
    /// Creates a PixieWorkerPipeline, auto-resolving SPIR-V / Zebin from disk or embedded resources.
    /// </summary>
    public static PixieWorkerPipeline Create(ComputeDevice device)
    {
        ArgumentNullException.ThrowIfNull(device);
        ComputeModule module = new KernelCatalog(device).LoadModule("pixie_worker_pipe");
        ComputeKernel hybrid = module.GetKernel("pixie_hybrid_eval_pipe");
        ComputeKernel swarm = module.GetKernel("pixie_swarm_step_pipe");
        return new PixieWorkerPipeline(device, module, hybrid, swarm);
    }

    /// <summary>
    /// Tries to create a PixieWorkerPipeline, returning null if module compilation or loading is unsupported by the driver.
    /// </summary>
    public static PixieWorkerPipeline? TryCreate(ComputeDevice device)
    {
        if (device is null) return null;
        try
        {
            var module = new KernelCatalog(device).TryLoadModule("pixie_worker_pipe", out _);
            if (module is null) return null;
            var hybrid = module.TryGetKernel("pixie_hybrid_eval_pipe");
            var swarm = module.TryGetKernel("pixie_swarm_step_pipe");
            if (hybrid is null || swarm is null)
            {
                module.Dispose();
                return null;
            }
            return new PixieWorkerPipeline(device, module, hybrid, swarm);
        }
        catch
        {
            return null;
        }
    }

    public static PixieWorkerPipeline Create(ComputeDevice device, string modulePath)
    {
        ArgumentNullException.ThrowIfNull(device);
        ArgumentNullException.ThrowIfNull(modulePath);
        ComputeModule module = device.LoadModule(modulePath);
        ComputeKernel hybrid = module.GetKernel("pixie_hybrid_eval_pipe");
        ComputeKernel swarm = module.GetKernel("pixie_swarm_step_pipe");
        return new PixieWorkerPipeline(device, module, hybrid, swarm);
    }

    public static PixieWorkerPipeline Create(ComputeDevice device, byte[] moduleBytes)
    {
        ArgumentNullException.ThrowIfNull(device);
        ArgumentNullException.ThrowIfNull(moduleBytes);
        ComputeModule module = device.LoadModule(moduleBytes);
        ComputeKernel hybrid = module.GetKernel("pixie_hybrid_eval_pipe");
        ComputeKernel swarm = module.GetKernel("pixie_swarm_step_pipe");
        return new PixieWorkerPipeline(device, module, hybrid, swarm);
    }

    /// <summary>
    /// Evaluates hybrid vector cosine similarity blended with graph topology link priors across N candidate nodes.
    /// </summary>
    public async ValueTask<float[]> EvaluateHybridBatchAsync(
        float[] queryVector,
        float[] nodeEmbeddingsFlattened,
        float[] nodeNorms,
        float[] nodePriors,
        int nodeCount,
        int dims,
        float queryNorm,
        float denseWeight = 0.6f,
        float priorWeight = 0.4f,
        CancellationToken ct = default)
    {
        ArgumentNullException.ThrowIfNull(queryVector);
        ArgumentNullException.ThrowIfNull(nodeEmbeddingsFlattened);
        ArgumentNullException.ThrowIfNull(nodeNorms);
        ArgumentNullException.ThrowIfNull(nodePriors);

        if (nodeCount <= 0 || dims <= 0)
        {
            return [];
        }

        await _gate.WaitAsync(ct).ConfigureAwait(false);
        try
        {
            using SharedBuffer<float> qBuf = _device.AllocShared(queryVector);
            using SharedBuffer<float> nodesBuf = _device.AllocShared(nodeEmbeddingsFlattened);
            using SharedBuffer<float> normsBuf = _device.AllocShared(nodeNorms);
            using SharedBuffer<float> priorsBuf = _device.AllocShared(nodePriors);
            using SharedBuffer<float> outBuf = _device.AllocShared<float>(nodeCount);

            _hybridKernel.SetArgBuffer(0, qBuf);
            _hybridKernel.SetArgBuffer(1, nodesBuf);
            _hybridKernel.SetArgBuffer(2, normsBuf);
            _hybridKernel.SetArgBuffer(3, priorsBuf);
            _hybridKernel.SetArgInt(4, nodeCount);
            _hybridKernel.SetArgInt(5, dims);
            _hybridKernel.SetArgFloat(6, queryNorm);
            _hybridKernel.SetArgFloat(7, denseWeight);
            _hybridKernel.SetArgFloat(8, priorWeight);
            _hybridKernel.SetArgBuffer(9, outBuf);

            const uint localSize = 64;
            _hybridKernel.SetGroupSize(localSize);
            _device.Launch(_hybridKernel, ComputeDevice.GroupCount(nodeCount, localSize));

            return outBuf.ToArray();
        }
        finally
        {
            _gate.Release();
        }
    }

    /// <summary>
    /// Executes a parallel BESS evolutionary swarm step in graph embedding space.
    /// </summary>
    public async ValueTask StepSwarmTrajectoryBatchAsync(
        float[] positions,
        float[] velocities,
        float[] pbestPositions,
        float[] gbestPosition,
        int particleCount,
        int dims,
        float inertiaW = 0.729f,
        float c1 = 1.49445f,
        float c2 = 1.49445f,
        uint randomSeed = 1337,
        CancellationToken ct = default)
    {
        ArgumentNullException.ThrowIfNull(positions);
        ArgumentNullException.ThrowIfNull(velocities);
        ArgumentNullException.ThrowIfNull(pbestPositions);
        ArgumentNullException.ThrowIfNull(gbestPosition);

        int totalElements = particleCount * dims;
        if (totalElements <= 0) return;

        await _gate.WaitAsync(ct).ConfigureAwait(false);
        try
        {
            using SharedBuffer<float> posBuf = _device.AllocShared(positions);
            using SharedBuffer<float> velBuf = _device.AllocShared(velocities);
            using SharedBuffer<float> pbestBuf = _device.AllocShared(pbestPositions);
            using SharedBuffer<float> gbestBuf = _device.AllocShared(gbestPosition);

            _swarmKernel.SetArgBuffer(0, posBuf);
            _swarmKernel.SetArgBuffer(1, velBuf);
            _swarmKernel.SetArgBuffer(2, pbestBuf);
            _swarmKernel.SetArgBuffer(3, gbestBuf);
            _swarmKernel.SetArgInt(4, particleCount);
            _swarmKernel.SetArgInt(5, dims);
            _swarmKernel.SetArgFloat(6, inertiaW);
            _swarmKernel.SetArgFloat(7, c1);
            _swarmKernel.SetArgFloat(8, c2);
            _swarmKernel.SetArgUInt(9, randomSeed);

            const uint localSize = 64;
            _swarmKernel.SetGroupSize(localSize);
            _device.Launch(_swarmKernel, ComputeDevice.GroupCount(totalElements, localSize));

            posBuf.ReadTo(positions);
            velBuf.ReadTo(velocities);
        }
        finally
        {
            _gate.Release();
        }
    }

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        _hybridKernel.Dispose();
        _swarmKernel.Dispose();
        _module.Dispose();
        _gate.Dispose();
    }
}
