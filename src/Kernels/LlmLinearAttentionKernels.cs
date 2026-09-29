using System;
namespace LevelZero.Kernels;
/// <summary>GPU elementwise <c>x[i] *= sigmoid(gate[i])</c>, the output gate of Qwen3.5 full-attention layers.</summary>
public sealed class LlmSigmoidGateKernel : IDisposable
{
    private const uint LocalSize = 64;
    private readonly ComputeKernel _kernel;
    internal LlmSigmoidGateKernel(ComputeKernel kernel)
    {
        _kernel = kernel;
    }
    /// <summary>Multiplies the first <paramref name="count"/> elements of <paramref name="x"/> by sigmoid of <paramref name="gate"/>.</summary>
    public void Execute(ComputeDevice device, SharedBuffer<float> x, SharedBuffer<float> gate, int count)
    {
        ArgumentNullException.ThrowIfNull(device);
        ArgumentNullException.ThrowIfNull(x);
        ArgumentNullException.ThrowIfNull(gate);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(count);
        if (x.Count < count || gate.Count < count)
        {
            throw new ArgumentException($"Buffers hold {x.Count} and {gate.Count} elements but count is {count}.");
        }
        _ = _kernel
            .WithArg(0, x)
            .WithArg(1, gate)
            .WithArg(2, count)
            .WithGroupSize(LocalSize);
        device.Launch(_kernel, ComputeDevice.GroupCount(count, LocalSize));
    }
    /// <summary>Disposes the underlying kernel handle.</summary>
    public void Dispose() => _kernel.Dispose();
}
/// <summary>GPU depthwise causal conv1d plus SiLU with a persistent per-channel history (Gated DeltaNet's short conv).</summary>
public sealed class LlmCausalConvKernel : IDisposable
{
    /// <summary>Largest supported kernel size.</summary>
    public const int MaxKernelSize = 8;
    private const uint LocalSize = 64;
    private readonly ComputeKernel _kernel;
    internal LlmCausalConvKernel(ComputeKernel kernel)
    {
        _kernel = kernel;
    }
    /// <summary>
    /// Convolves <paramref name="numTokens"/> consecutive tokens, in order, and leaves the last
    /// <c>kernelSize - 1</c> inputs in <paramref name="state"/> for the next call.
    /// </summary>
    /// <param name="device">The L0 compute device.</param>
    /// <param name="x">Inputs [numTokens, channels].</param>
    /// <param name="state">History [channels, kernelSize - 1], oldest first. Zero for a new sequence. Updated.</param>
    /// <param name="weights">Filter taps [channels, kernelSize], oldest tap first.</param>
    /// <param name="output">Outputs [numTokens, channels] (after SiLU).</param>
    /// <param name="channels">Number of channels.</param>
    /// <param name="kernelSize">Taps per channel, 2 to <see cref="MaxKernelSize"/>.</param>
    /// <param name="numTokens">Tokens to process.</param>
    public void Execute(
        ComputeDevice device,
        SharedBuffer<float> x,
        SharedBuffer<float> state,
        SharedBuffer<float> weights,
        SharedBuffer<float> output,
        int channels,
        int kernelSize,
        int numTokens)
    {
        ArgumentNullException.ThrowIfNull(device);
        ArgumentNullException.ThrowIfNull(x);
        ArgumentNullException.ThrowIfNull(state);
        ArgumentNullException.ThrowIfNull(weights);
        ArgumentNullException.ThrowIfNull(output);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(channels);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(numTokens);
        if (kernelSize < 2 || kernelSize > MaxKernelSize)
        {
            throw new ArgumentOutOfRangeException(nameof(kernelSize), kernelSize, $"Kernel size must be 2 to {MaxKernelSize}.");
        }
        long rows = (long)numTokens * channels;
        if (x.Count < rows || output.Count < rows)
        {
            throw new ArgumentException($"x and output must hold at least {rows} elements.");
        }
        if (state.Count < (long)channels * (kernelSize - 1) || weights.Count < (long)channels * kernelSize)
        {
            throw new ArgumentException("state or weights are too small for the channel count.");
        }
        _ = _kernel
            .WithArg(0, x)
            .WithArg(1, state)
            .WithArg(2, weights)
            .WithArg(3, output)
            .WithArg(4, channels)
            .WithArg(5, kernelSize)
            .WithArg(6, numTokens)
            .WithGroupSize(LocalSize);
        device.Launch(_kernel, ComputeDevice.GroupCount(channels, LocalSize));
    }
    /// <summary>Disposes the underlying kernel handle.</summary>
    public void Dispose() => _kernel.Dispose();
}
/// <summary>
/// GPU gated delta rule (Qwen3.5 / Qwen3-Next linear attention) with its output gated RMSNorm. One work-group
/// per head, tokens in order, a float32 recurrent state that carries across calls.
/// </summary>
public sealed class LlmGatedDeltaKernel : IDisposable
{
    private readonly ComputeKernel _kernel;
    internal LlmGatedDeltaKernel(ComputeKernel kernel)
    {
        _kernel = kernel;
    }
    /// <summary>
    /// Runs <paramref name="numTokens"/> tokens through the delta rule. Key and value heads must be equal in
    /// number, and <paramref name="valueDim"/> is also the work-group size (at most 1024).
    /// </summary>
    /// <param name="device">The L0 compute device.</param>
    /// <param name="qkv">Post-conv rows [numTokens, 2 * numHeads * keyDim + numHeads * valueDim] as q, k, v.</param>
    /// <param name="z">Output gate inputs [numTokens, numHeads * valueDim].</param>
    /// <param name="bRaw">Beta pre-activations [numTokens, numHeads].</param>
    /// <param name="aRaw">Decay pre-activations [numTokens, numHeads].</param>
    /// <param name="aLog"><c>A_log</c> [numHeads].</param>
    /// <param name="dtBias"><c>dt_bias</c> [numHeads].</param>
    /// <param name="normWeight">Gated RMSNorm weight [valueDim].</param>
    /// <param name="state">Recurrent state [numHeads, keyDim, valueDim]; zero for a new sequence. Updated.</param>
    /// <param name="output">Outputs [numTokens, numHeads * valueDim].</param>
    /// <param name="numHeads">Number of heads.</param>
    /// <param name="keyDim">Key head size.</param>
    /// <param name="valueDim">Value head size.</param>
    /// <param name="numTokens">Tokens to process.</param>
    /// <param name="epsilon">RMSNorm epsilon.</param>
    public void Execute(
        ComputeDevice device,
        SharedBuffer<float> qkv,
        SharedBuffer<float> z,
        SharedBuffer<float> bRaw,
        SharedBuffer<float> aRaw,
        SharedBuffer<float> aLog,
        SharedBuffer<float> dtBias,
        SharedBuffer<float> normWeight,
        SharedBuffer<float> state,
        SharedBuffer<float> output,
        int numHeads,
        int keyDim,
        int valueDim,
        int numTokens,
        float epsilon)
    {
        ArgumentNullException.ThrowIfNull(device);
        ArgumentNullException.ThrowIfNull(qkv);
        ArgumentNullException.ThrowIfNull(z);
        ArgumentNullException.ThrowIfNull(bRaw);
        ArgumentNullException.ThrowIfNull(aRaw);
        ArgumentNullException.ThrowIfNull(aLog);
        ArgumentNullException.ThrowIfNull(dtBias);
        ArgumentNullException.ThrowIfNull(normWeight);
        ArgumentNullException.ThrowIfNull(state);
        ArgumentNullException.ThrowIfNull(output);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(numHeads);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(keyDim);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(numTokens);
        if (valueDim <= 0 || valueDim > 1024 || (valueDim & (valueDim - 1)) != 0)
        {
            throw new ArgumentOutOfRangeException(nameof(valueDim), valueDim, "valueDim must be a power of two up to 1024.");
        }
        long convDim = (2L * numHeads * keyDim) + ((long)numHeads * valueDim);
        long valueTotal = (long)numHeads * valueDim;
        if (qkv.Count < numTokens * convDim
            || z.Count < numTokens * valueTotal
            || output.Count < numTokens * valueTotal
            || bRaw.Count < (long)numTokens * numHeads
            || aRaw.Count < (long)numTokens * numHeads
            || aLog.Count < numHeads
            || dtBias.Count < numHeads
            || normWeight.Count < valueDim
            || state.Count < (long)numHeads * keyDim * valueDim)
        {
            throw new ArgumentException("A buffer is too small for the given head sizes and token count.");
        }
        int localBytes = ((2 * keyDim) + (2 * valueDim)) * sizeof(float);
        _ = _kernel
            .WithArg(0, qkv)
            .WithArg(1, z)
            .WithArg(2, bRaw)
            .WithArg(3, aRaw)
            .WithArg(4, aLog)
            .WithArg(5, dtBias)
            .WithArg(6, normWeight)
            .WithArg(7, state)
            .WithArg(8, output)
            .WithArg(9, numHeads)
            .WithArg(10, keyDim)
            .WithArg(11, valueDim)
            .WithArg(12, numTokens)
            .WithArg(13, epsilon)
            .WithLocalArg(14, localBytes)
            .WithGroupSize((uint)valueDim);
        device.Launch(_kernel, (uint)numHeads);
    }
    /// <summary>Disposes the underlying kernel handle.</summary>
    public void Dispose() => _kernel.Dispose();
}