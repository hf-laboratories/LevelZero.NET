namespace LevelZero.Kernels;

/// <summary>
/// GPU-accelerated hypergraph scoring kernel.
/// Evaluates a parametric score function over (x, y, value) triplets.
/// </summary>
public sealed class HypergraphKernel : IDisposable
{
    private readonly ComputeDevice _device;
    private readonly ComputeModule _module;
    private readonly ComputeKernel _kernel;

    private HypergraphKernel(ComputeDevice device, ComputeModule module, ComputeKernel kernel)
    {
        _device = device;
        _module = module;
        _kernel = kernel;
    }

    /// <summary>Creates a hypergraph kernel, auto-resolving SPIR-V from disk or embedded resources.</summary>
    public static HypergraphKernel Create(ComputeDevice device)
    {
        var path = KernelCatalog.ResolveSpirvPath("levelzero-hypergraph");
        if (path is not null) return Create(device, path);
        var embedded = KernelCatalog.LoadEmbeddedSpirv("levelzero-hypergraph");
        if (embedded is not null) return Create(device, embedded);
        throw new FileNotFoundException("SPIR-V not found for levelzero-hypergraph.");
    }

    public static HypergraphKernel Create(ComputeDevice device, string spirvPath, string kernelName = "hypergraph_score")
    {
        var module = device.LoadModule(spirvPath);
        var kernel = module.GetKernel(kernelName);
        return new HypergraphKernel(device, module, kernel);
    }

    public static HypergraphKernel Create(ComputeDevice device, byte[] spirv, string kernelName = "hypergraph_score")
    {
        var module = device.LoadModule(spirv);
        var kernel = module.GetKernel(kernelName);
        return new HypergraphKernel(device, module, kernel);
    }

    /// <summary>
    /// Scores (x, y, value) triplets with parameters (a, b, c).
    /// All input arrays must have the same length.
    /// </summary>
    public float[] Score(float[] x, float[] y, float[] values, float a, float b, float c)
    {
        if (x.Length != y.Length || x.Length != values.Length)
            throw new ArgumentException("Input arrays must have matching lengths.");

        int count = x.Length;

        using var xBuf = _device.AllocShared(x);
        using var yBuf = _device.AllocShared(y);
        using var vBuf = _device.AllocShared(values);
        using var outBuf = _device.AllocShared<float>(count);

        _kernel.SetArgBuffer(0, xBuf);
        _kernel.SetArgBuffer(1, yBuf);
        _kernel.SetArgBuffer(2, vBuf);
        _kernel.SetArgBuffer(3, outBuf);
        _kernel.SetArgFloat(4, a);
        _kernel.SetArgFloat(5, b);
        _kernel.SetArgFloat(6, c);
        _kernel.SetArgInt(7, count);

        const uint localSize = 64;
        _kernel.SetGroupSize(localSize);
        _device.Launch(_kernel, ComputeDevice.GroupCount(count, localSize));

        return outBuf.ToArray();
    }

    public void Dispose()
    {
        _kernel.Dispose();
        _module.Dispose();
    }
}
