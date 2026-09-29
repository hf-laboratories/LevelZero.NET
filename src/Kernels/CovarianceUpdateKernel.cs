namespace LevelZero.Kernels;

/// <summary>
/// GPU-accelerated CMA-ES covariance matrix update (rank-one + rank-mu).
/// Updates the full DÃ—D matrix in a single kernel dispatch with DÂ² work-items.
/// </summary>
public sealed class CovarianceUpdateKernel : IDisposable
{
    private readonly ComputeDevice _device;
    private readonly ComputeModule _module;
    private readonly ComputeKernel _kernel;

    private CovarianceUpdateKernel(ComputeDevice device, ComputeModule module, ComputeKernel kernel)
    {
        _device = device;
        _module = module;
        _kernel = kernel;
    }

    /// <summary>Creates a covariance update kernel from a module file path.</summary>
    public static CovarianceUpdateKernel Create(ComputeDevice device, string modulePath, string kernelName = "cmaes_covariance_update")
    {
        ComputeModule module = device.LoadModule(modulePath);
        ComputeKernel kernel = module.GetKernel(kernelName);
        return new CovarianceUpdateKernel(device, module, kernel);
    }

    /// <summary>Creates a covariance update kernel from pre-loaded module bytes.</summary>
    public static CovarianceUpdateKernel Create(ComputeDevice device, byte[] moduleBytes, string kernelName = "cmaes_covariance_update")
    {
        ComputeModule module = device.LoadModule(moduleBytes);
        ComputeKernel kernel = module.GetKernel(kernelName);
        return new CovarianceUpdateKernel(device, module, kernel);
    }

    /// <summary>
    /// Applies the rank-one + rank-mu update to the covariance matrix and returns the updated flat matrix.
    /// </summary>
    /// <param name="C">Flat [dims*dims] covariance matrix (row-major), modified in-place on GPU.</param>
    /// <param name="pc">Evolution path vector [dims].</param>
    /// <param name="deviations">Offspring deviations [mu*dims] row-major: dev[k*dims+d] = (pos_k[d]-mean[d])/sigma.</param>
    /// <param name="weights">Recombination weights [mu].</param>
    /// <param name="dims">Problem dimensionality D.</param>
    /// <param name="c1">Rank-one learning rate.</param>
    /// <param name="cmu">Rank-mu learning rate.</param>
    /// <param name="oldFactor">Decay factor applied to existing C values.</param>
    /// <returns>Updated flat [dims*dims] covariance matrix.</returns>
    public float[] UpdateCovariance(float[] C, float[] pc, float[] deviations, float[] weights, int dims, float c1, float cmu, float oldFactor)
    {
        int mu = weights.Length;
        using SharedBuffer<float> cBuf = _device.AllocShared(C);
        using SharedBuffer<float> pcBuf = _device.AllocShared(pc);
        using SharedBuffer<float> devBuf = _device.AllocShared(deviations);
        using SharedBuffer<float> wBuf = _device.AllocShared(weights);

        _kernel.SetArgBuffer(0, cBuf);
        _kernel.SetArgBuffer(1, pcBuf);
        _kernel.SetArgBuffer(2, devBuf);
        _kernel.SetArgBuffer(3, wBuf);
        _kernel.SetArgInt(4, dims);
        _kernel.SetArgInt(5, mu);
        _kernel.SetArgFloat(6, c1);
        _kernel.SetArgFloat(7, cmu);
        _kernel.SetArgFloat(8, oldFactor);

        const uint localSize = 64;
        _device.Launch(_kernel, ComputeDevice.GroupCount(dims * dims, localSize));

        return cBuf.ToArray();
    }

    /// <inheritdoc />
    public void Dispose()
    {
        _kernel.Dispose();
        _module.Dispose();
    }
}

