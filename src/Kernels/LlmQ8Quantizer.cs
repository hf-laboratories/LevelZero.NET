using System;
using System.Threading.Tasks;
namespace LevelZero.Kernels;

/// <summary>
/// Host-side int8 quantization in the layout <see cref="LlmGemvQ8Kernel"/> reads, plus a bit-faithful CPU model of the
/// device activation quantizer and matrix product for tests.
/// </summary>
public static class LlmQ8Quantizer
{
    /// <summary>
    /// Quantizes a HuggingFace <c>[out, in]</c> matrix (each output's inputs contiguous) into device layout:
    /// <paramref name="words"/> <c>[in / 4][out]</c> and <paramref name="scales"/> <c>[in / GroupSize][out]</c>.
    /// The scale of a group is <c>max|w| / 127</c> rounded to half; values are rounded to nearest with that stored scale.
    /// </summary>
    public static void QuantizeOutIn(ReadOnlySpan<float> hf, int outDim, int inDim, out int[] words, out Half[] scales)
    {
        if (!LlmGemvQ8Kernel.Supports(outDim, inDim))
        {
            throw new NotSupportedException($"Int8 quantization needs out % 4 == 0 and in % {LlmGemvQ8Kernel.GroupSize} == 0, got {outDim} x {inDim}.");
        }

        if (hf.Length != (long)outDim * inDim)
        {
            throw new ArgumentException($"Expected {(long)outDim * inDim} elements, got {hf.Length}.", nameof(hf));
        }

        int groups = inDim / LlmGemvQ8Kernel.GroupSize;
        int[] w = new int[(long)(inDim / 4) * outDim];
        Half[] s = new Half[(long)groups * outDim];
        float[] source = hf.ToArray();
        _ = Parallel.For(0, outDim, n =>
        {
            int rowStart = n * inDim;
            for (int g = 0; g < groups; g++)
            {
                int k0 = g * LlmGemvQ8Kernel.GroupSize;
                float amax = 0f;
                for (int i = 0; i < LlmGemvQ8Kernel.GroupSize; i++)
                {
                    amax = MathF.Max(amax, MathF.Abs(source[rowStart + k0 + i]));
                }

                Half scaleHalf = (Half)(amax / 127f);
                float scale = (float)scaleHalf;
                float inv = scale > 0f ? 1f / scale : 0f;
                s[(g * outDim) + n] = scaleHalf;
                for (int q = 0; q < LlmGemvQ8Kernel.GroupSize / 4; q++)
                {
                    int packed = 0;
                    for (int b = 0; b < 4; b++)
                    {
                        float v = source[rowStart + k0 + (q * 4) + b] * inv;
                        int qi = Math.Clamp((int)MathF.Round(v, MidpointRounding.ToEven), -127, 127);
                        packed |= (qi & 0xFF) << (8 * b);
                    }

                    w[((((k0 / 4) + q)) * (long)outDim) + n] = packed;
                }
            }
        });
        words = w;
        scales = s;
    }

    /// <summary>
    /// CPU model of <c>quantize_act_q8</c> for one row: packed quads and per-group scales.
    /// </summary>
    public static void QuantizeRow(ReadOnlySpan<float> x, out int[] quads, out float[] scales)
    {
        int k = x.Length;
        if (k <= 0 || k % LlmGemvQ8Kernel.GroupSize != 0)
        {
            throw new ArgumentException($"Length must be a positive multiple of {LlmGemvQ8Kernel.GroupSize}.", nameof(x));
        }

        int groups = k / LlmGemvQ8Kernel.GroupSize;
        quads = new int[k / 4];
        scales = new float[groups];
        for (int g = 0; g < groups; g++)
        {
            int k0 = g * LlmGemvQ8Kernel.GroupSize;
            float amax = 0f;
            for (int i = 0; i < LlmGemvQ8Kernel.GroupSize; i++)
            {
                amax = MathF.Max(amax, MathF.Abs(x[k0 + i]));
            }

            float inv = amax > 0f ? 127f / amax : 0f;
            scales[g] = amax / 127f;
            for (int q = 0; q < LlmGemvQ8Kernel.GroupSize / 4; q++)
            {
                int packed = 0;
                for (int b = 0; b < 4; b++)
                {
                    int qi = Math.Clamp((int)MathF.Round(x[k0 + (q * 4) + b] * inv, MidpointRounding.ToEven), -127, 127);
                    packed |= (qi & 0xFF) << (8 * b);
                }

                quads[(k0 / 4) + q] = packed;
            }
        }
    }

    /// <summary>
    /// CPU model of <c>gemv_q8</c>: integer group dot products scaled by both group scales, accumulated in float.
    /// </summary>
    public static float[] ReferenceGemv(ReadOnlySpan<float> x, int[] words, Half[] scales, int outDim, int inDim)
    {
        QuantizeRow(x, out int[] xq, out float[] xs);
        int groups = inDim / LlmGemvQ8Kernel.GroupSize;
        var y = new float[outDim];
        for (int n = 0; n < outDim; n++)
        {
            float acc = 0f;
            for (int g = 0; g < groups; g++)
            {
                int dot = 0;
                for (int i = 0; i < LlmGemvQ8Kernel.GroupSize / 4; i++)
                {
                    int q = (g * (LlmGemvQ8Kernel.GroupSize / 4)) + i;
                    int a = xq[q];
                    int b = words[((long)q * outDim) + n];
                    for (int j = 0; j < 4; j++)
                    {
                        dot += (sbyte)(a >> (8 * j)) * (sbyte)(b >> (8 * j));
                    }
                }

                acc += dot * (float)scales[((long)g * outDim) + n] * xs[g];
            }

            y[n] = acc;
        }

        return y;
    }
}
