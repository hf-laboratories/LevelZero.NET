namespace LevelZero.Kernels;

/// <summary>
/// GPU-accelerated pairwise Euclidean distance computation.
/// Supports upper-triangle output (compact) and full N x N matrix output.
/// </summary>
public sealed class PairwiseDistanceKernel : IDisposable
{
    private readonly ComputeDevice _device;
    private readonly ComputeModule _module;
    private readonly ComputeKernel _triangleKernel;
    private readonly ComputeKernel? _matrixKernel;

    private PairwiseDistanceKernel(ComputeDevice device, ComputeModule module,
                                    ComputeKernel triangleKernel, ComputeKernel? matrixKernel)
    {
        _device = device;
        _module = module;
        _triangleKernel = triangleKernel;
        _matrixKernel = matrixKernel;
    }

    /// <summary>Creates a pairwise distance kernel, auto-resolving SPIR-V from disk or embedded resources.</summary>
    public static PairwiseDistanceKernel Create(ComputeDevice device)
    {
        var path = KernelCatalog.ResolveSpirvPath("pairwise_distance");
        if (path is not null) return Create(device, path);
        var embedded = KernelCatalog.LoadEmbeddedSpirv("pairwise_distance");
        if (embedded is not null) return Create(device, embedded);
        throw new FileNotFoundException("SPIR-V not found for pairwise_distance.");
    }

    public static PairwiseDistanceKernel Create(ComputeDevice device, string spirvPath, string kernelName = "pairwise_distance")
    {
        var module = device.LoadModule(spirvPath);
        var triangle = module.GetKernel(kernelName);
        var matrix = module.TryGetKernel("pairwise_distance_matrix");
        return new PairwiseDistanceKernel(device, module, triangle, matrix);
    }

    public static PairwiseDistanceKernel Create(ComputeDevice device, byte[] spirv, string kernelName = "pairwise_distance")
    {
        var module = device.LoadModule(spirv);
        var triangle = module.GetKernel(kernelName);
        var matrix = module.TryGetKernel("pairwise_distance_matrix");
        return new PairwiseDistanceKernel(device, module, triangle, matrix);
    }

    /// <summary>
    /// Computes upper-triangle pairwise distances. Returns array of size N*(N-1)/2.
    /// </summary>
    public float[] EvaluateTriangle(float[] points, int pointCount, int dims)
    {
        int totalPairs = pointCount * (pointCount - 1) / 2;
        if (totalPairs <= 0) return [];

        using var ptsBuf = _device.AllocShared(points);
        using var outBuf = _device.AllocShared<float>(totalPairs);

        _triangleKernel.SetArgBuffer(0, ptsBuf);
        _triangleKernel.SetArgInt(1, pointCount);
        _triangleKernel.SetArgInt(2, dims);
        _triangleKernel.SetArgBuffer(3, outBuf);

        const uint localSize = 64;
        _triangleKernel.SetGroupSize(localSize);
        _device.Launch(_triangleKernel, ComputeDevice.GroupCount(totalPairs, localSize));

        return outBuf.ToArray();
    }

    /// <summary>
    /// Computes full N x N distance matrix. Uses the matrix kernel if available,
    /// otherwise falls back to triangle kernel and expands.
    /// </summary>
    public float[] EvaluateMatrix(float[] points, int pointCount, int dims)
    {
        if (_matrixKernel is not null)
        {
            using var ptsBuf = _device.AllocShared(points);
            using var outBuf = _device.AllocShared<float>(pointCount * pointCount);

            _matrixKernel.SetArgBuffer(0, ptsBuf);
            _matrixKernel.SetArgInt(1, pointCount);
            _matrixKernel.SetArgInt(2, dims);
            _matrixKernel.SetArgBuffer(3, outBuf);

            const uint localSize = 64;
            _matrixKernel.SetGroupSize(localSize);
            _device.Launch(_matrixKernel, ComputeDevice.GroupCount(pointCount, localSize));

            return outBuf.ToArray();
        }

        // Fallback: expand triangle to full matrix
        var triangle = EvaluateTriangle(points, pointCount, dims);
        var matrix = new float[pointCount * pointCount];
        int k = 0;
        for (int i = 0; i < pointCount; i++)
        {
            for (int j = i + 1; j < pointCount; j++)
            {
                matrix[i * pointCount + j] = triangle[k];
                matrix[j * pointCount + i] = triangle[k];
                k++;
            }
        }
        return matrix;
    }

    public void Dispose()
    {
        _matrixKernel?.Dispose();
        _triangleKernel.Dispose();
        _module.Dispose();
    }
}
