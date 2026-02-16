# LevelZero.NET

Managed .NET wrapper for [Intel Level Zero](https://github.com/oneapi-src/level-zero) GPU compute, with a built-in SPIR-V kernel catalog for swarm optimization, spiking neural networks, and multi-objective algorithms.

## Features

- **Zero-config GPU access** — discover and use Intel GPUs via Level Zero with a clean managed API
- **Built-in SPIR-V kernels** — pre-compiled kernels for 11 Intel GPU targets (Tiger Lake through Panther Lake)
- **Kernel catalog** — automatic device detection and kernel resolution with fallback chain
- **Native shim embedding** — `LevelZeroShim.dll` / `libLevelZeroShim.so` extracted at runtime, no manual setup
- **High-level kernel wrappers** — `FitnessKernel`, `PSOVelocityKernel`, `NeuronUpdateKernel`, `STDPKernel`, `NBodyKernel`, and more

## Installation

```bash
dotnet add package LevelZero.NET
```

## Quick Start

```csharp
using LevelZero;

// Check GPU availability
if (LevelZeroRuntime.IsAvailable())
{
    var device = LevelZeroRuntime.GetDefaultDevice();
    Console.WriteLine($"GPU: {device}");

    // Load a SPIR-V module
    byte[] spirv = KernelCatalog.GetKernel("fitness_eval");
    using var module = device.LoadModule(spirv);

    // Allocate shared memory and launch
    using var buffer = device.AllocShared<float>(1024);
    device.Launch(module, "fitness_eval", workItems: 1024);
}
```

## Supported GPU Targets

| Codename | Architecture | Examples |
|----------|-------------|----------|
| tgllp | Gen12 | Intel Iris Xe (11th Gen) |
| dg1 | Gen12 | Intel Iris Xe MAX |
| acm-g10/g11/g12 | Xe-HPG | Intel Arc A-series |
| pvc | Xe-HPC | Intel Data Center GPU Max |
| mtl | Xe-LPG | Intel Core Ultra |
| arl-h | Xe-LPG+ | Intel Arrow Lake |
| bmg-g21 | Xe2 | Intel Arc B-series (Battlemage) |
| lnl-m | Xe2-LPG | Intel Lunar Lake |
| ptl-h | Xe3-LPG | Intel Panther Lake |

## Kernel Catalog

The `KernelCatalog` resolves SPIR-V binaries using this chain:
1. Explicit file path
2. `LEVELZERO_SPIRV_DIR` environment variable
3. NuGet package content
4. Embedded assembly resources
5. Fallback to `tgllp` device target

## Publishing

Tag a release to trigger NuGet publishing:

```bash
git tag v1.0.0
git push origin v1.0.0
```

Requires the `NUGET_API_KEY` secret configured in the repository settings.

## License

MIT
