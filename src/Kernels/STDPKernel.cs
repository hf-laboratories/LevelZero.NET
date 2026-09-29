namespace LevelZero.Kernels;

/// <summary>
/// GPU-accelerated STDP (Spike-Timing-Dependent Plasticity) weight updates and trace decay.
/// Loads two kernels from one module: stdp_plasticity and stdp_trace_decay.
/// </summary>
public sealed class STDPKernel : IDisposable
{
    private readonly ComputeDevice _device;
    private readonly ComputeModule _module;
    private readonly ComputeKernel _plasticityKernel;
    private readonly ComputeKernel _decayKernel;

    private STDPKernel(ComputeDevice device, ComputeModule module,
                       ComputeKernel plasticityKernel, ComputeKernel decayKernel)
    {
        _device = device;
        _module = module;
        _plasticityKernel = plasticityKernel;
        _decayKernel = decayKernel;
    }

    /// <summary>Creates an STDP kernel, auto-resolving SPIR-V from disk or embedded resources.</summary>
    public static STDPKernel Create(ComputeDevice device)
    {
        var (path, embedded) = KernelSpirvResolution.Resolve("stdp_plasticity");
        return path is not null ? Create(device, path) : Create(device, embedded!);
    }

    public static STDPKernel Create(ComputeDevice device, string spirvPath)
    {
        ComputeModule module = device.LoadModule(spirvPath);
        ComputeKernel plasticity = module.GetKernel("stdp_plasticity");
        ComputeKernel decay = module.GetKernel("stdp_trace_decay");
        return new STDPKernel(device, module, plasticity, decay);
    }

    public static STDPKernel Create(ComputeDevice device, byte[] spirv)
    {
        ComputeModule module = device.LoadModule(spirv);
        ComputeKernel plasticity = module.GetKernel("stdp_plasticity");
        ComputeKernel decay = module.GetKernel("stdp_trace_decay");
        return new STDPKernel(device, module, plasticity, decay);
    }

    /// <summary>
    /// Applies STDP weight update. Modifies weights in-place.
    /// </summary>
    public void UpdateWeights(
        float[] weights, float[] preTraces, float[] postTraces,
        int[] srcGroup, int[] dstGroup, int connCount,
        float aPlus, float aMinus, float wMin, float wMax)
    {
        using SharedBuffer<float> wBuf = _device.AllocShared(weights);
        using SharedBuffer<float> preBuf = _device.AllocShared(preTraces);
        using SharedBuffer<float> postBuf = _device.AllocShared(postTraces);
        using SharedBuffer<int> srcBuf = _device.AllocShared(srcGroup);
        using SharedBuffer<int> dstBuf = _device.AllocShared(dstGroup);

        _plasticityKernel.SetArgBuffer(0, wBuf);
        _plasticityKernel.SetArgBuffer(1, preBuf);
        _plasticityKernel.SetArgBuffer(2, postBuf);
        _plasticityKernel.SetArgBuffer(3, srcBuf);
        _plasticityKernel.SetArgBuffer(4, dstBuf);
        _plasticityKernel.SetArgInt(5, connCount);
        _plasticityKernel.SetArgFloat(6, aPlus);
        _plasticityKernel.SetArgFloat(7, aMinus);
        _plasticityKernel.SetArgFloat(8, wMin);
        _plasticityKernel.SetArgFloat(9, wMax);

        const uint localSize = 64;
        _plasticityKernel.SetGroupSize(localSize);
        _device.Launch(_plasticityKernel, ComputeDevice.GroupCount(connCount, localSize));

        wBuf.ReadTo(weights);
    }

    /// <summary>
    /// Decays trace arrays on GPU. Modifies traces in-place.
    /// </summary>
    public void DecayTraces(float[] traces, int count, float decayFactor)
    {
        using SharedBuffer<float> buf = _device.AllocShared(traces);

        _decayKernel.SetArgBuffer(0, buf);
        _decayKernel.SetArgInt(1, count);
        _decayKernel.SetArgFloat(2, decayFactor);

        const uint localSize = 64;
        _decayKernel.SetGroupSize(localSize);
        _device.Launch(_decayKernel, ComputeDevice.GroupCount(count, localSize));

        buf.ReadTo(traces);
    }

    public void Dispose()
    {
        _decayKernel.Dispose();
        _plasticityKernel.Dispose();
        _module.Dispose();
    }
}

