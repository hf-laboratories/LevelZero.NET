namespace LevelZero.Kernels;

using System;

/// <summary>
/// GPU-accelerated dense vector similarity search for GraphRAG.
/// Tailored for Intel Xe-LP (Iris Xe GT2 / tgllp) utilizing Shared Local Memory (SLM) query caching
/// and 128-bit float4 streaming.
/// </summary>
public sealed class GraphRagVectorSearchKernel : IDisposable
{
    private readonly ComputeDevice _device;
    private readonly ComputeModule _module;
    private readonly ComputeKernel _cosineKernel;
    private readonly ComputeKernel? _dotProductKernel;

    private GraphRagVectorSearchKernel(
        ComputeDevice device,
        ComputeModule module,
        ComputeKernel cosineKernel,
        ComputeKernel? dotProductKernel)
    {
        _device = device;
        _module = module;
        _cosineKernel = cosineKernel;
        _dotProductKernel = dotProductKernel;
    }

    /// <summary>
    /// Creates a GraphRAG vector search kernel, auto-resolving SPIR-V / Zebin from disk or embedded resources.
    /// </summary>
    public static GraphRagVectorSearchKernel Create(ComputeDevice device)
    {
        ArgumentNullException.ThrowIfNull(device);
        ComputeModule module = new KernelCatalog(device).LoadModule("graphrag_vector_search");
        ComputeKernel cosine = module.GetKernel("graphrag_cosine_similarity_batch");
        ComputeKernel? dot = module.TryGetKernel("graphrag_dot_product_batch");
        return new GraphRagVectorSearchKernel(device, module, cosine, dot);
    }

    /// <summary>
    /// Tries to create a GraphRAG vector search kernel, returning null if the module or kernels cannot be loaded.
    /// </summary>
    public static GraphRagVectorSearchKernel? TryCreate(ComputeDevice device)
    {
        if (device == null) return null;
        try
        {
            ComputeModule? module = new KernelCatalog(device).TryLoadModule("graphrag_vector_search", out _);
            if (module is null) return null;
            ComputeKernel? cosine = module.TryGetKernel("graphrag_cosine_similarity_batch");
            if (cosine is null) return null;
            ComputeKernel? dot = module.TryGetKernel("graphrag_dot_product_batch");
            return new GraphRagVectorSearchKernel(device, module, cosine, dot);
        }
        catch
        {
            return null;
        }
    }

    public static GraphRagVectorSearchKernel Create(ComputeDevice device, string spirvPath)
    {
        ComputeModule module = device.LoadModule(spirvPath);
        ComputeKernel cosine = module.GetKernel("graphrag_cosine_similarity_batch");
        ComputeKernel? dot = module.TryGetKernel("graphrag_dot_product_batch");
        return new GraphRagVectorSearchKernel(device, module, cosine, dot);
    }

    public static GraphRagVectorSearchKernel Create(ComputeDevice device, byte[] spirv)
    {
        ComputeModule module = device.LoadModule(spirv);
        ComputeKernel cosine = module.GetKernel("graphrag_cosine_similarity_batch");
        ComputeKernel? dot = module.TryGetKernel("graphrag_dot_product_batch");
        return new GraphRagVectorSearchKernel(device, module, cosine, dot);
    }

    /// <summary>
    /// Computes normalized cosine similarities in [0, 1] for N candidate node embeddings against query q.
    /// </summary>
    public float[] EvaluateCosineSimilarityBatch(
        float[] queryVector,
        float[] nodeEmbeddingsFlattened,
        float[] nodeNorms,
        int nodeCount,
        int dims,
        float queryNorm)
    {
        if (nodeCount <= 0 || dims <= 0)
        {
            return [];
        }

        using SharedBuffer<float> qBuf = _device.AllocShared(queryVector);
        using SharedBuffer<float> nodesBuf = _device.AllocShared(nodeEmbeddingsFlattened);
        using SharedBuffer<float> normsBuf = _device.AllocShared(nodeNorms);
        using SharedBuffer<float> outBuf = _device.AllocShared<float>(nodeCount);

        _cosineKernel.SetArgBuffer(0, qBuf);
        _cosineKernel.SetArgBuffer(1, nodesBuf);
        _cosineKernel.SetArgBuffer(2, normsBuf);
        _cosineKernel.SetArgInt(3, nodeCount);
        _cosineKernel.SetArgInt(4, dims);
        _cosineKernel.SetArgFloat(5, queryNorm);
        _cosineKernel.SetArgBuffer(6, outBuf);

        const uint localSize = 64;
        _cosineKernel.SetGroupSize(localSize);
        _device.Launch(_cosineKernel, ComputeDevice.GroupCount(nodeCount, localSize));

        return outBuf.ToArray();
    }

    /// <summary>
    /// Computes raw inner products for N candidate embeddings against query q.
    /// </summary>
    public float[] EvaluateDotProductBatch(
        float[] queryVector,
        float[] nodeEmbeddingsFlattened,
        int nodeCount,
        int dims)
    {
        if (nodeCount <= 0 || dims <= 0 || _dotProductKernel is null)
        {
            return [];
        }

        using SharedBuffer<float> qBuf = _device.AllocShared(queryVector);
        using SharedBuffer<float> nodesBuf = _device.AllocShared(nodeEmbeddingsFlattened);
        using SharedBuffer<float> outBuf = _device.AllocShared<float>(nodeCount);

        _dotProductKernel.SetArgBuffer(0, qBuf);
        _dotProductKernel.SetArgBuffer(1, nodesBuf);
        _dotProductKernel.SetArgInt(2, nodeCount);
        _dotProductKernel.SetArgInt(3, dims);
        _dotProductKernel.SetArgBuffer(4, outBuf);

        const uint localSize = 64;
        _dotProductKernel.SetGroupSize(localSize);
        _device.Launch(_dotProductKernel, ComputeDevice.GroupCount(nodeCount, localSize));

        return outBuf.ToArray();
    }

    public void Dispose()
    {
        _dotProductKernel?.Dispose();
        _cosineKernel.Dispose();
        _module.Dispose();
    }
}
