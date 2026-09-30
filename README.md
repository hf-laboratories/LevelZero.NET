# HFLabs.LevelZero.NET (v2.0)

[![NuGet](https://img.shields.io/badge/nuget-v2.0.0-blue.svg)](https://nuget.org)
[![Framework](https://img.shields.io/badge/.NET-10.0-purple.svg)](https://dotnet.microsoft.com)
[![Hardware](https://img.shields.io/badge/targets-11%20Intel%20Architectures-orange.svg)](https://github.com)
[![License](https://img.shields.io/badge/license-Dual%20(Community%20%2F%20Commercial)-green.svg)](https://github.com)

**`HFLabs.LevelZero.NET`** provides high-performance, zero-allocation managed .NET bindings and runtime abstractions for the **Intel oneAPI Level Zero API** (`ze_api.h`).

It enables C# developers to allocate shared/device GPU memory, record and dispatch low-latency command lists, load pre-compiled `.zebin` / SPIR-V compute kernels, and orchestrate heterogeneous hardware acceleration without writing C++ shims.

---

## 🚀 Key Features

* **Zero-Allocation Managed Abstractions:** Clean, safe C# wrappers over unmanaged Level Zero driver handles, command queues, and shared memory allocations (`SharedBuffer<T>`).
* **Multi-Architecture Native Support:** Pre-compiled `.zebin` machine code catalogs for 11 Intel GPU device targets:
  * `tgllp` (Tiger Lake LP / Iris Xe)
  * `dg1` (Iris Xe MAX)
  * `acm-g10`, `acm-g11`, `acm-g12` (Intel Arc A-Series / Alchemist)
  * `mtl` (Meteor Lake / Core Ultra 100 series Arc iGPU)
  * `arl-h` (Arrow Lake - H / Core Ultra 200 series)
  * `lnl-m` (Lunar Lake - M / Core Ultra 200V series Xe2)
  * `bmg-g21` (Intel Arc B-Series / Battlemage)
  * `ptl-h` (Panther Lake - H / Xe3)
  * `pvc` (Ponte Vecchio / Intel Data Center GPU Max 1100 & 1550)
* **High-Performance Memory Management:** Native USM (Unified Shared Memory) support for host, device, and shared buffer allocations with zero-copy memory transfers.
* **Transformer & Compute Kernel Integration:** Out-of-the-box support for accelerated matrix multiplication (GEMM), attention mechanisms (SDPA), normalization (RMSNorm), and evolutionary swarm computing.

---

## 💻 Code Example: Device Discovery & Kernel Dispatch

```csharp
using LevelZero;
using LevelZero.Kernels;

// 1. Initialize Runtime and Default Intel GPU
using ComputeDevice device = LevelZeroRuntime.GetDefaultDevice();
Console.WriteLine($"Discovered GPU: {device.Name}");

// 2. Allocate Unified Shared Memory (USM)
float[] hostData = new float[1024];
using SharedBuffer<float> buffer = device.AllocShared(hostData);

// 3. Load pre-compiled SPIR-V or native .zebin kernel module
byte[] kernelBytes = File.ReadAllBytes("my_kernel.zebin");
using ComputeModule module = device.LoadModule(kernelBytes, LevelZeroModuleFormat.Zebin);

// 4. Dispatch kernel across compute units
// (Memory is unified; results are immediately accessible on CPU upon queue synchronization)
buffer.CopyTo(hostData);
```

---

## 📦 Related Packages

* **`HFLabs.LevelZero.Kernels.<target>`** — Dedicated binary packages containing pre-compiled `.zebin` and `.spv` modules for specific Intel architectures (e.g. `HFLabs.LevelZero.Kernels.tgllp`, `HFLabs.LevelZero.Kernels.acm-g10`).
* **`HFLabs.ML.Agentic`** — Zero-dependency multi-agent orchestration, MCP, and A2A protocols for .NET.
* **`l0llm`** — Self-contained command line LLM inference engine and OpenAI-compatible REST server.

---

## 📄 Licensing & Commercial Use

* **Community / Research License:** Free for open-source development, academic research, and evaluation.
* **Commercial License:** Required for embedding in proprietary or commercial desktop/server software. Contact alix@HFLabs.dev for commercial licensing and custom kernel engineering services.

## Trademarks and non-affiliation

Product and company names in this repository belong to their owners and are used only to say what this project works with. HF Laboratories is not affiliated with, endorsed by, or sponsored by any of them.

- Intel, oneAPI, Level Zero are trademarks or registered trademarks of Intel Corporation.
- Microsoft, Windows, .NET, PowerShell, NuGet are trademarks or registered trademarks of Microsoft Corporation.
- OpenAI is a trademark or registered trademark of OpenAI.
- Hugging Face is a trademark or registered trademark of Hugging Face, Inc.
- macOS is a trademark or registered trademark of Apple Inc.
- Linux is a trademark or registered trademark of Linus Torvalds.
- Ubuntu is a trademark or registered trademark of Canonical Ltd.

See [hflabs.dev/legal/trademarks](https://hflabs.dev/legal/trademarks) for the full list.
