using System.Runtime.InteropServices;

namespace LevelZero.Kernels;

/// <summary>
/// GPU-accelerated pairwise spike correlation matrix computation (2D dispatch).
/// </summary>
public sealed class CorrelationKernel : IDisposable
{
    private readonly ComputeDevice _device;
    private readonly ComputeModule _module;
    private readonly ComputeKernel _kernel;

    private CorrelationKernel(ComputeDevice device, ComputeModule module, ComputeKernel kernel)
    {
        _device = device;
        _module = module;
        _kernel = kernel;
    }

    /// <summary>Creates a correlation kernel, auto-resolving SPIR-V from disk or embedded resources.</summary>
    public static CorrelationKernel Create(ComputeDevice device)
    {
        var path = KernelCatalog.ResolveSpirvPath("snn_correlation");
        if (path is not null) return Create(device, path);
        var embedded = KernelCatalog.LoadEmbeddedSpirv("snn_correlation");
        if (embedded is not null) return Create(device, embedded);
        throw new FileNotFoundException("SPIR-V not found for snn_correlation.");
    }

    public static CorrelationKernel Create(ComputeDevice device, string spirvPath, string kernelName = "snn_correlation")
    {
        var module = device.LoadModule(spirvPath);
        var kernel = module.GetKernel(kernelName);
        return new CorrelationKernel(device, module, kernel);
    }

    public static CorrelationKernel Create(ComputeDevice device, byte[] spirv, string kernelName = "snn_correlation")
    {
        var module = device.LoadModule(spirv);
        var kernel = module.GetKernel(kernelName);
        return new CorrelationKernel(device, module, kernel);
    }

    /// <summary>
    /// Computes pairwise spike correlation matrix.
    /// </summary>
    /// <param name="spikeTimes">Concatenated spike times for all groups.</param>
    /// <param name="groupOffsets">int[numGroups+1]: start offset of each group's spikes.</param>
    /// <param name="numGroups">Number of neuron groups.</param>
    /// <param name="window">Coincidence window (seconds).</param>
    /// <returns>float[numGroups * numGroups] correlation matrix (row-major).</returns>
    public float[] Evaluate(float[] spikeTimes, int[] groupOffsets, int numGroups, float window)
    {
        int matrixSize = numGroups * numGroups;

        using var stBuf = _device.AllocShared(spikeTimes);
        using var goBuf = _device.AllocShared(groupOffsets);
        using var outBuf = _device.AllocShared<float>(matrixSize);

        _kernel.SetArgBuffer(0, stBuf);
        _kernel.SetArgBuffer(1, goBuf);
        _kernel.SetArgInt(2, numGroups);
        _kernel.SetArgFloat(3, window);
        _kernel.SetArgBuffer(4, outBuf);

        const uint localX = 8, localY = 8;
        _kernel.SetGroupSize(localX, localY);
        _device.Launch(_kernel,
            ComputeDevice.GroupCount(numGroups, localX),
            ComputeDevice.GroupCount(numGroups, localY));

        return outBuf.ToArray();
    }

    public void Dispose()
    {
        _kernel.Dispose();
        _module.Dispose();
    }
}
