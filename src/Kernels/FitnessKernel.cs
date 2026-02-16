using System.Runtime.InteropServices;

namespace LevelZero.Kernels;

/// <summary>
/// GPU-accelerated fitness evaluation kernel (sum of squares or custom objective).
/// </summary>
public sealed class FitnessKernel : IDisposable
{
    private readonly ComputeDevice _device;
    private readonly ComputeModule _module;
    private readonly ComputeKernel _kernel;

    private FitnessKernel(ComputeDevice device, ComputeModule module, ComputeKernel kernel)
    {
        _device = device;
        _module = module;
        _kernel = kernel;
    }

    /// <summary>Creates a fitness kernel, auto-resolving SPIR-V from disk or embedded resources.</summary>
    public static FitnessKernel Create(ComputeDevice device, string kernelName = "fitness_kernel")
    {
        var path = KernelCatalog.ResolveSpirvPath("fitness_kernel");
        if (path is not null) return Create(device, path, kernelName);
        var embedded = KernelCatalog.LoadEmbeddedSpirv("fitness_kernel");
        if (embedded is not null) return Create(device, embedded, kernelName);
        throw new FileNotFoundException("SPIR-V not found for fitness_kernel.");
    }

    /// <summary>Creates a fitness kernel from a SPIR-V file path.</summary>
    public static FitnessKernel Create(ComputeDevice device, string spirvPath, string kernelName = "fitness_kernel")
    {
        var module = device.LoadModule(spirvPath);
        var kernel = module.GetKernel(kernelName);
        return new FitnessKernel(device, module, kernel);
    }

    /// <summary>Creates a fitness kernel from SPIR-V bytes.</summary>
    public static FitnessKernel Create(ComputeDevice device, byte[] spirv, string kernelName = "fitness_kernel")
    {
        var module = device.LoadModule(spirv);
        var kernel = module.GetKernel(kernelName);
        return new FitnessKernel(device, module, kernel);
    }

    /// <summary>
    /// Evaluates fitness for each row in positions.
    /// </summary>
    /// <param name="positions">Flat array [count * dimensions].</param>
    /// <param name="count">Number of candidate solutions.</param>
    /// <param name="dimensions">Dimensionality of each solution.</param>
    /// <returns>float[count] fitness values.</returns>
    public float[] Evaluate(float[] positions, int count, int dimensions)
    {
        using var posBuffer = _device.AllocShared(positions);
        using var outBuffer = _device.AllocShared<float>(count);

        _kernel.SetArgBuffer(0, posBuffer);
        _kernel.SetArgInt(1, count);
        _kernel.SetArgInt(2, dimensions);
        _kernel.SetArgBuffer(3, outBuffer);

        const uint localSize = 64;
        _kernel.SetGroupSize(localSize);
        _device.Launch(_kernel, ComputeDevice.GroupCount(count, localSize));

        return outBuffer.ToArray();
    }

    public void Dispose()
    {
        _kernel.Dispose();
        _module.Dispose();
    }
}
