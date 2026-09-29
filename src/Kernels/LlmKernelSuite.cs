using System;
using System.IO;

namespace LevelZero.Kernels;

/// <summary>
/// Orchestrates loading the LLM OpenCL/SPIR-V binary once and instantiating the individual sub-kernels.
/// </summary>
public sealed class LlmKernelSuite : IDisposable
{
    private readonly ComputeModule _module;

    /// <summary>Retrieves the Embedding Lookup kernel.</summary>
    public LlmEmbeddingKernel Embedding { get; }

    /// <summary>Retrieves the Fused RMSNorm kernel.</summary>
    public LlmRMSNormKernel RMSNorm { get; }

    /// <summary>Retrieves the GEMM matrix multiplication kernel.</summary>
    public LlmGemmKernel Gemm { get; }
    /// <summary>Retrieves the single-row GEMV kernels (float32 and half weights) used for decode.</summary>
    public LlmGemvKernel Gemv { get; }

    /// <summary>Retrieves the Fused RoPE rotary embedding kernel.</summary>
    public LlmRoPEKernel RoPE { get; }

    /// <summary>Retrieves the SwiGLU activation kernel.</summary>
    public LlmSwiGLUKernel SwiGLU { get; }

    /// <summary>Retrieves the Fused Self-Attention (SDPA) kernel.</summary>
    public LlmAttentionKernel Attention { get; }

    /// <summary>Retrieves the Logits Softmax and Sampling kernel.</summary>
    public LlmSamplingKernel Sampling { get; }
    /// <summary>Retrieves the broadcast bias-add kernel (QKV projection bias).</summary>
    public LlmBiasAddKernel BiasAdd { get; }
    /// <summary>Retrieves the in-place elementwise add kernel (residual connections).</summary>
    public LlmAddKernel Add { get; }
    /// <summary>Retrieves the KV cache store kernel (writes a token's K and V into the caches).</summary>
    public LlmKvCacheStoreKernel KvCacheStore { get; }
    /// <summary>Retrieves the slice-copy kernel (lifts one row out of a batch).</summary>
    public LlmCopyKernel Copy { get; }

    /// <summary>Retrieves the sigmoid output-gate kernel (Qwen3.5 attention).</summary>

    public LlmSigmoidGateKernel SigmoidGate { get; }

    

    /// <summary>Retrieves the causal conv1d + SiLU kernel (Gated DeltaNet).</summary>

    public LlmCausalConvKernel CausalConv { get; }

    

    /// <summary>Retrieves the gated delta rule kernel (Gated DeltaNet).</summary>

    public LlmGatedDeltaKernel GatedDelta { get; }

    

    /// <summary>Retrieves the int8 DP4A GEMV/GEMM kernels, or null when the loaded binary predates them.</summary>
    public LlmGemvQ8Kernel? GemvQ8 { get; }
    /// <summary>Activation scratch shared by every int8 projection on this suite; null until <see cref="EnsureQ8Scratch"/>.</summary>
    public LlmQ8Activations? Q8Scratch { get; private set; }
    private readonly System.Collections.Generic.List<LlmQ8Activations> _retiredScratch = new();
    /// <summary>
    /// Makes sure the shared int8 activation scratch holds at least <paramref name="rows"/> rows of
    /// <paramref name="k"/> inputs. Call before recording any command list that uses int8 projections: a replaced
    /// scratch is kept alive (not disposed) so lists recorded earlier stay valid.
    /// </summary>
    public void EnsureQ8Scratch(ComputeDevice device, int rows, int k)
    {
        ArgumentNullException.ThrowIfNull(device);
        LlmQ8Activations? current = Q8Scratch;
        if (current is not null && current.RowCapacity >= rows && current.KCapacity >= k)
        {
            return;
        }
        int newRows = Math.Max(rows, current?.RowCapacity ?? 0);
        int newK = Math.Max(k, current?.KCapacity ?? 0);
        if (current is not null)
        {
            _retiredScratch.Add(current);
        }
        Q8Scratch = new LlmQ8Activations(device, newRows, newK);
    }
    private LlmKernelSuite(ComputeModule module)
    {
        _module = module;
        Embedding = new LlmEmbeddingKernel(module.GetKernel("embedding_lookup"));
        RMSNorm = new LlmRMSNormKernel(module.GetKernel("rmsnorm_fused"));
        Gemm = new LlmGemmKernel(module.GetKernel("gemm_tgllp"));
        Gemv = new LlmGemvKernel(module.GetKernel("gemv_f32"), module.GetKernel("gemv_f16"), module.GetKernel("gemm_rows_f32"), module.GetKernel("gemm_rows_f16"));
        RoPE = new LlmRoPEKernel(module.GetKernel("rope_fused"), module.TryGetKernel("rope_fused_freq"));
        SwiGLU = new LlmSwiGLUKernel(module.GetKernel("swiglu_activation"));
        Attention = new LlmAttentionKernel(module.GetKernel("fused_sdpa"), module.GetKernel("fused_sdpa_prefill"));
        Sampling = new LlmSamplingKernel(module.GetKernel("logits_sample"));
        BiasAdd = new LlmBiasAddKernel(module.GetKernel("bias_add"));
        Add = new LlmAddKernel(module.GetKernel("add_inplace"));
        Copy = new LlmCopyKernel(module.GetKernel("copy_slice"));
        KvCacheStore = new LlmKvCacheStoreKernel(module.GetKernel("kv_cache_store"), module.GetKernel("kv_cache_store_rows"));
        SigmoidGate = new LlmSigmoidGateKernel(module.GetKernel("sigmoid_gate_mul"));
        CausalConv = new LlmCausalConvKernel(module.GetKernel("dn_conv1d_silu"));
        GatedDelta = new LlmGatedDeltaKernel(module.GetKernel("gated_delta_rule"));
        ComputeKernel? quantize = module.TryGetKernel("quantize_act_q8");
        ComputeKernel? gemvQ8 = module.TryGetKernel("gemv_q8");
        ComputeKernel? rowsQ8 = module.TryGetKernel("gemm_rows_q8");
        if (quantize is not null && gemvQ8 is not null && rowsQ8 is not null)
        {
            GemvQ8 = new LlmGemvQ8Kernel(quantize, gemvQ8, rowsQ8);
        }
        else
        {
            quantize?.Dispose();
            gemvQ8?.Dispose();
            rowsQ8?.Dispose();
        }
    }

    /// <summary>
    /// Creates the LLM kernel suite by auto-resolving or loading the embedded SPIR-V binary.
    /// </summary>
    public static LlmKernelSuite Create(ComputeDevice device)
    {
        var catalog = new KernelCatalog(device);
        ComputeModule module = catalog.LoadModule("llm_kernels");
        return new LlmKernelSuite(module);
    }

    /// <summary>
    /// Disposes the module and all child kernels.
    /// </summary>
    public void Dispose()
    {
        Embedding.Dispose();
        RMSNorm.Dispose();
        Gemm.Dispose();
        Gemv.Dispose();
        RoPE.Dispose();
        SwiGLU.Dispose();
        Attention.Dispose();
        Sampling.Dispose();
        BiasAdd.Dispose();
        Add.Dispose();
        KvCacheStore.Dispose();
        Copy.Dispose();
        SigmoidGate.Dispose();
        CausalConv.Dispose();
        GatedDelta.Dispose();
        _module.Dispose();
    }
}
