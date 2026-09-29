using System.Runtime.InteropServices;

namespace LevelZero.Kernels;

/// <summary>
/// GPU-accelerated LIF neuron membrane potential update.
/// </summary>
public sealed class NeuronUpdateKernel : IDisposable
{
    private readonly ComputeDevice _device;
    private readonly ComputeModule _module;
    private readonly ComputeKernel _kernel;

    private NeuronUpdateKernel(ComputeDevice device, ComputeModule module, ComputeKernel kernel)
    {
        _device = device;
        _module = module;
        _kernel = kernel;
    }

    /// <summary>Creates a neuron update kernel, auto-resolving SPIR-V from disk or embedded resources.</summary>
    public static NeuronUpdateKernel Create(ComputeDevice device)
    {
        var (path, embedded) = KernelSpirvResolution.Resolve("snn_neuron_update");
        return path is not null ? Create(device, path) : Create(device, embedded!);
    }

    public static NeuronUpdateKernel Create(ComputeDevice device, string spirvPath, string kernelName = "snn_neuron_update")
    {
        ComputeModule module = device.LoadModule(spirvPath);
        ComputeKernel kernel = module.GetKernel(kernelName);
        return new NeuronUpdateKernel(device, module, kernel);
    }

    public static NeuronUpdateKernel Create(ComputeDevice device, byte[] spirv, string kernelName = "snn_neuron_update")
    {
        ComputeModule module = device.LoadModule(spirv);
        ComputeKernel kernel = module.GetKernel(kernelName);
        return new NeuronUpdateKernel(device, module, kernel);
    }

    /// <summary>
    /// Runs the neuron update kernel. Modifies potentials and refractoryCounters in-place.
    /// Returns int[neuronCount] spiked flags (1 = spiked, 0 = did not).
    /// </summary>
    public int[] Evaluate(float[] potentials, int[] refractoryCounters, float[] inputs,
                          int neuronCount, float leak, float threshold, float resetValue, int refractoryPhases)
    {
        using SharedBuffer<float> potBuf = _device.AllocShared(potentials);
        using SharedBuffer<int> refBuf = _device.AllocShared(refractoryCounters);
        using SharedBuffer<float> inBuf = _device.AllocShared(inputs);
        using SharedBuffer<int> spkBuf = _device.AllocShared<int>(neuronCount);

        _kernel.SetArgBuffer(0, potBuf);
        _kernel.SetArgBuffer(1, refBuf);
        _kernel.SetArgBuffer(2, inBuf);
        _kernel.SetArgBuffer(3, spkBuf);
        _kernel.SetArgInt(4, neuronCount);
        _kernel.SetArgFloat(5, leak);
        _kernel.SetArgFloat(6, threshold);
        _kernel.SetArgFloat(7, resetValue);
        _kernel.SetArgInt(8, refractoryPhases);

        const uint localSize = 64;
        _kernel.SetGroupSize(localSize);
        _device.Launch(_kernel, ComputeDevice.GroupCount(neuronCount, localSize));

        potBuf.ReadTo(potentials);
        refBuf.ReadTo(refractoryCounters);

        return spkBuf.ToArray();
    }

    public void Dispose()
    {
        _kernel.Dispose();
        _module.Dispose();
    }
}

