using System.Runtime.InteropServices;

namespace LevelZero.Kernels;

/// <summary>
/// GPU-accelerated PSO velocity and position update kernel.
/// Modifies velocities and positions in-place.
/// </summary>
public sealed class PSOVelocityKernel : IDisposable
{
    private readonly ComputeDevice _device;
    private readonly ComputeModule _module;
    private readonly ComputeKernel _kernel;

    private PSOVelocityKernel(ComputeDevice device, ComputeModule module, ComputeKernel kernel)
    {
        _device = device;
        _module = module;
        _kernel = kernel;
    }

    /// <summary>Creates a PSO velocity kernel, auto-resolving SPIR-V from disk or embedded resources.</summary>
    public static PSOVelocityKernel Create(ComputeDevice device)
    {
        var path = KernelCatalog.ResolveSpirvPath("pso_velocity");
        if (path is not null) return Create(device, path);
        var embedded = KernelCatalog.LoadEmbeddedSpirv("pso_velocity");
        if (embedded is not null) return Create(device, embedded);
        throw new FileNotFoundException("SPIR-V not found for pso_velocity.");
    }

    public static PSOVelocityKernel Create(ComputeDevice device, string spirvPath, string kernelName = "pso_velocity_update")
    {
        var module = device.LoadModule(spirvPath);
        var kernel = module.GetKernel(kernelName);
        return new PSOVelocityKernel(device, module, kernel);
    }

    public static PSOVelocityKernel Create(ComputeDevice device, byte[] spirv, string kernelName = "pso_velocity_update")
    {
        var module = device.LoadModule(spirv);
        var kernel = module.GetKernel(kernelName);
        return new PSOVelocityKernel(device, module, kernel);
    }

    /// <summary>
    /// Runs PSO velocity+position update. Modifies velocities and positions arrays in-place.
    /// </summary>
    public void Evaluate(float[] velocities, float[] positions, float[] pBest, float[] gBest,
                         float[] randoms, int count, int dims, float w, float c1, float c2,
                         float loBound, float hiBound)
    {
        int total = count * dims;

        using var velBuf = _device.AllocShared(velocities);
        using var posBuf = _device.AllocShared(positions);
        using var pbBuf = _device.AllocShared(pBest);
        using var gbBuf = _device.AllocShared(gBest);
        using var rndBuf = _device.AllocShared(randoms);

        _kernel.SetArgBuffer(0, velBuf);
        _kernel.SetArgBuffer(1, posBuf);
        _kernel.SetArgBuffer(2, pbBuf);
        _kernel.SetArgBuffer(3, gbBuf);
        _kernel.SetArgBuffer(4, rndBuf);
        _kernel.SetArgInt(5, count);
        _kernel.SetArgInt(6, dims);
        _kernel.SetArgFloat(7, w);
        _kernel.SetArgFloat(8, c1);
        _kernel.SetArgFloat(9, c2);
        _kernel.SetArgFloat(10, loBound);
        _kernel.SetArgFloat(11, hiBound);

        const uint localSize = 64;
        _kernel.SetGroupSize(localSize);
        _device.Launch(_kernel, ComputeDevice.GroupCount(total, localSize));

        velBuf.ReadTo(velocities);
        posBuf.ReadTo(positions);
    }

    public void Dispose()
    {
        _kernel.Dispose();
        _module.Dispose();
    }
}
