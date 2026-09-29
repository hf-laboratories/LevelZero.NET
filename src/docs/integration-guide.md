# LevelZero.NET Integration Guide

End-to-end guide for integrating Intel GPU compute into your .NET application using LevelZero.NET.

---

## 1. Prerequisites

| Requirement | Notes |
|---|---|
| .NET 10 SDK | `dotnet --version` must report 10.x |
| Intel GPU with Level Zero driver | Integrated (Tiger Lake+) or discrete (Arc, Data Center Max) |
| `LevelZeroShim.dll` / `libLevelZeroShim.so` | C ABI shim — built from `level-zero-interop/shim.cpp` via CMake |

Check that Level Zero is visible:

```csharp
Console.WriteLine(LevelZeroRuntime.IsAvailable());  // true
```

---

## 2. NuGet Packages

### Core library

```xml
<PackageReference Include="LevelZero.NET" Version="1.0.0" />
```

This gives you the full managed API: device management, module loading, memory allocation, kernel launch, and all 14 kernel wrappers.

### Pre-compiled kernel packages

Install the package matching your GPU:

```xml
<!-- Example: Arc B580 -->
<PackageReference Include="HFLabs.LevelZero.Kernels.bmg-g21" Version="1.0.0" />
```

Full device list:

| Device Target | GPU Family | Example Hardware |
|---|---|---|
| `tgllp` | Gen12 | 11th Gen Intel Iris Xe |
| `dg1` | Gen12 | Intel Iris Xe MAX |
| `acm-g10` | Xe-HPG | Arc A770, A750 |
| `acm-g11` | Xe-HPG | Arc A580, A380 |
| `acm-g12` | Xe-HPG | Arc A310 |
| `pvc` | Xe-HPC | Data Center GPU Max (Ponte Vecchio) |
| `mtl` | Xe-LPG | Core Ultra 100 series (Meteor Lake) |
| `arl-h` | Xe-LPG+ | Core Ultra 200 series (Arrow Lake) |
| `bmg-g21` | Xe2-HPG | Arc B580, B570 (Battlemage) |
| `lnl-m` | Xe2-LPG | Core Ultra 200V (Lunar Lake) |
| `ptl-h` | Xe3-LPG | Core Ultra 300 (Panther Lake) |

Not sure which package to install? Or, you would rather do it programmatically? Use the detector:

```csharp
var config = DeviceCapabilityDetector.Detect();
Console.WriteLine(config.DeviceTarget);  // e.g. "bmg-g21"
```

---

## 3. Device Discovery & Setup

### Get the default device

```csharp
using var device = LevelZeroRuntime.GetDefaultDevice();
Console.WriteLine(device.Name);  // "Intel(R) Arc(TM) B580 Graphics"
```

### Enumerate all devices

```csharp
foreach (var info in LevelZeroRuntime.EnumerateDevices())
{
    Console.WriteLine($"[{info.DriverIndex}:{info.DeviceIndex}] {info.Name}");
}
```

### Open a specific device

```csharp
var info = LevelZeroRuntime.EnumerateDevices()
    .First(d => d.Name.Contains("Arc"));
using var device = info.Open();
```

Or by explicit indices:

```csharp
using var device = LevelZeroRuntime.GetDevice(driverIndex: 0, deviceIndex: 1);
```

---

## 4. Auto Device Detection & Config

The `DeviceCapabilityDetector` probes the system GPU, walks the architecture hierarchy, and persists the result to `levelzero-config.json`:

```csharp
// First call: detects + saves. Subsequent calls: loads from file.
var config = DeviceCapabilityDetector.LoadOrDetect();

Console.WriteLine(config.DeviceTarget);    // "bmg-g21"
Console.WriteLine(config.Architecture);    // "Xe2-HPG"
Console.WriteLine(config.DeviceName);      // "Intel(R) Arc(TM) B580 Graphics"
Console.WriteLine(config.HardwarePresent); // true
Console.WriteLine(config.DetectedAt);      // 2025-06-14T...
```

Force re-detection:

```csharp
var config = DeviceCapabilityDetector.DetectAndSave();
```

The `KernelCatalog` reads this config automatically to locate device-specific SPIR-V subdirectories.

### Config file format

```json
{
  "DeviceTarget": "bmg-g21",
  "Architecture": "Xe2-HPG",
  "DeviceName": "Intel(R) Arc(TM) B580 Graphics",
  "HardwarePresent": true,
  "DetectedAt": "2025-06-14T12:00:00Z"
}
```

Default location: `levelzero-config.json` in the application's base directory. Override with a custom path:

```csharp
var config = DeviceCapabilityDetector.LoadOrDetect("C:/myapp/gpu-config.json");
```

---

## 5. Using Kernel Wrappers

Every kernel wrapper provides three factory methods:

| Method | When to use |
|---|---|
| `Create(device)` | Auto-resolves SPIR-V via `KernelCatalog` (recommended) |
| `Create(device, spirvPath)` | Explicit `.spv` file path |
| `Create(device, byte[])` | SPIR-V loaded from a stream, embedded resource, etc. |

### FitnessKernel — Objective function evaluation

```csharp
using var device = LevelZeroRuntime.GetDefaultDevice();
using var fitness = FitnessKernel.Create(device);

int count = 1024;
int dims = 10;
float[] positions = new float[count * dims];

// Fill positions with candidate solutions...

float[] results = fitness.Evaluate(positions, count, dims);
// results[i] = fitness of candidate i
```

### PSOVelocityKernel — Particle swarm velocity/position update

```csharp
using var pso = PSOVelocityKernel.Create(device);

float[] velocities = new float[count * dims];
float[] positions  = new float[count * dims];
float[] pBest      = new float[count * dims];
float[] gBest      = new float[dims];
float[] randoms    = new float[count * dims * 2]; // r1 + r2

// Fill arrays...

pso.Evaluate(velocities, positions, pBest, gBest, randoms,
             count, dims, w: 0.7f, c1: 1.5f, c2: 1.5f,
             loBound: -5.12f, hiBound: 5.12f);

// velocities and positions are updated in-place
```

### CorrelationKernel — SNN spike correlation

```csharp
using var corr = CorrelationKernel.Create(device);
float[] correlations = corr.Evaluate(spikeTimes, groupOffsets, results,
                                     groupCount, maxSpikes, tau);
```

### NeuronUpdateKernel — SNN integrate-and-fire

```csharp
using var neurons = NeuronUpdateKernel.Create(device);
int[] spikes = neurons.Evaluate(potentials, refractory, inputs,
                                count, threshold, resetValue, dt);
```

### STDPKernel — Synaptic plasticity

```csharp
using var stdp = STDPKernel.Create(device);

// Update weights based on pre/post timing
stdp.UpdateWeights(weights, preTimes, postTimes, synCount, aPlus, aMinus, tauPlus, tauMinus);

// Decay eligibility traces
stdp.DecayTraces(traces, traceCount, decayFactor);
```

### NBodyKernel — Repulsive force computation

```csharp
using var nbody = NBodyKernel.Create(device);
var (forceX, forceY) = nbody.Evaluate(posX, posY, count, repulsionStrength);
```

### DominanceMatrixKernel — Pareto dominance

```csharp
using var dom = DominanceMatrixKernel.Create(device);
int[] dominationCounts = dom.ComputeDominationCounts(objectives, popSize, numObjectives);
```

### FireflyAttractionKernel — Firefly optimization

```csharp
using var firefly = FireflyAttractionKernel.Create(device);
float[] coupling = firefly.ComputeFlashCoupling(positions, brightness, count, dims, beta0, gamma);
```

### MonteCarloHypervolumeKernel — Hypervolume estimation

```csharp
using var hv = MonteCarloHypervolumeKernel.Create(device);
int dominated = hv.Evaluate(solutions, idealPoint, refPoint,
                            randomSamples, popSize, numObjectives, numSamples);
double volume = (double)dominated / numSamples * refVolume;
```

### PairwiseDistanceKernel — Distance matrices

```csharp
using var dist = PairwiseDistanceKernel.Create(device);

// Upper-triangle (N*(N-1)/2 entries)
float[] triangle = dist.EvaluateTriangle(positions, count, dims);

// Full NxN matrix
float[] matrix = dist.EvaluateMatrix(positions, count, dims);
```

### TilePlacementKernel — IPU tile mapping scores

```csharp
using var tile = TilePlacementKernel.Create(device);
float[] scores = tile.Evaluate(agentX, agentY, tileX, tileY,
                               agentCount, tileCount, bandwidth, latency);
```

---

## 6. Manual Module & Kernel Usage

For kernels not covered by the 14 wrappers, or for custom module files (`.spv`, `.bin`, `.zebin`):

```csharp
using var device = LevelZeroRuntime.GetDefaultDevice();

// Load a module (.spv => SPIR-V, .bin/.native/.zebin => native binary)
using var module = device.LoadModule("my_custom_kernel.spv");

// Get a kernel handle
using var kernel = module.GetKernel("my_entry_point");

// Allocate shared memory
using var input  = device.AllocShared(new float[] { 1, 2, 3, 4 });
using var output = device.AllocShared<float>(4);

// Bind arguments (by index, matching the kernel signature)
kernel.SetArgBuffer(0, input);
kernel.SetArgBuffer(1, output);
kernel.SetArgInt(2, 4);

// Set local work-group size and launch
uint localSize = 64;
kernel.SetGroupSize(localSize);
device.Launch(kernel, ComputeDevice.GroupCount(4, localSize));

// Read results back
float[] results = output.ToArray();
```

### TryLoadModule for graceful fallback

```csharp
var module = device.TryLoadModule("optional_kernel.spv", out string buildLog);
if (module is null)
{
    Console.WriteLine($"Kernel unavailable: {buildLog}");
    // Fall back to CPU implementation
}
```

### Loading from embedded resources

```csharp
var spirvBytes = GetEmbeddedResource("HFLabs.LevelZero.Kernels.[ Kernel Name ].spv");
using var module = device.LoadModule(spirvBytes);

byte[] nativeBytes = File.ReadAllBytes("my_custom_kernel.zebin");
using var nativeModule = device.LoadModule(nativeBytes, LevelZeroModuleFormat.Native);
```

---

## 7. KernelCatalog — Module Resolution

The `KernelCatalog` searches for custom module files in this order:

1. **Explicit path** — passed directly to `ResolveSpirvPath`
2. **Environment variable** — `IPU_L0_{KERNEL_NAME}_MODULE` (native-aware)
3. **Legacy environment variable** — `IPU_L0_{KERNEL_NAME}_SPIRV`
4. **NuGet flat** — `{outputDir}/levelzero-kernels/{name}.zebin/.bin/.native/.spv`
5. **NuGet device-specific** — `{outputDir}/levelzero-kernels/{deviceTarget}/{name}.zebin/.bin/.native/.spv`
6. **Working directory** — `{name}.zebin/.bin/.native/.spv`

### Environment variable override

All embedded kernel → env var mappings:

| Kernel Name | Environment Variable |
|---|---|
| `fitness` | `IPU_L0_FITNESS_SPIRV` |
| `rastrigin_fitness` | `IPU_L0_RASTRIGIN_FITNESS_SPIRV` |
| `pso_velocity` | `IPU_L0_PSO_VELOCITY_SPIRV` |
| `snn_neuron_update` | `IPU_L0_SNN_NEURON_UPDATE_SPIRV` |
| `snn_correlation` | `IPU_L0_SNN_CORRELATION_SPIRV` |
| `stdp_plasticity` | `IPU_L0_STDP_PLASTICITY_SPIRV` |
| `nbody_repulsion` | `IPU_L0_NBODY_REPULSION_SPIRV` |
| `dominance_matrix` | `IPU_L0_DOMINANCE_MATRIX_SPIRV` |
| `exchange_weights` | `IPU_L0_EXCHANGE_WEIGHTS_SPIRV` |
| `firefly_attraction` | `IPU_L0_FIREFLY_ATTRACTION_SPIRV` |
| `monte_carlo_hypervolume` | `IPU_L0_MONTE_CARLO_HYPERVOLUME_SPIRV` |
| `pairwise_distance` | `IPU_L0_PAIRWISE_DISTANCE_SPIRV` |
| `tile_placement` | `IPU_L0_TILE_PLACEMENT_SPIRV` |

### Programmatic resolution

```csharp
// Resolve path without loading
string? path = KernelCatalog.ResolveModulePath("fitness_kernel");

// Resolve with explicit override (checked first)
string? path = KernelCatalog.ResolveModulePath("fitness_kernel", 
    explicitPath: "/opt/kernels/fitness.spv");
```

---

## 8. Memory Management

### SharedBuffer lifecycle

`SharedBuffer<T>` wraps a Level Zero USM (Unified Shared Memory) allocation. The buffer is accessible from both CPU and GPU without explicit copies.

```csharp
// Allocate empty buffer
using var buf = device.AllocShared<float>(1024);

// Allocate and fill from array
using var buf = device.AllocShared(new float[] { 1, 2, 3, 4 });

// Write new data
buf.Write(new float[] { 5, 6, 7, 8 });

// Read back
float[] results = buf.ToArray();

// Or read into existing array
float[] target = new float[1024];
buf.ReadTo(target);
```

Rules:

- Always `Dispose()` (or use `using`) to free the USM allocation
- Don't read from a buffer while a kernel is still running — `Launch()` synchronizes automatically
- `Write()` throws if the source is larger than the buffer capacity
- The generic constraint is `unmanaged` — works with `float`, `int`, `double`, `uint`, custom structs, etc.

---

## 10. Error Handling

All Level Zero errors throw `LevelZeroException`:

```csharp
try
{
    using var device = LevelZeroRuntime.GetDefaultDevice();
}
catch (LevelZeroException ex)
{
    Console.WriteLine($"L0 error {ex.NativeResult}: {ex.NativeMessage}");
}
catch (DllNotFoundException)
{
    Console.WriteLine("LevelZeroShim not found — is it in PATH?");
}
```

Module compilation failures include the build log:

```csharp
var module = device.TryLoadModule("bad_kernel.spv", out string log);
if (module is null)
    Console.WriteLine($"SPIR-V compilation failed:\n{log}");
```

---

## 11. Deployment Checklist

- [ ] `LevelZeroShim.dll` (Win) or `libLevelZeroShim.so` (Linux) in app directory or PATH
- [ ] Intel Level Zero runtime driver installed on target machine
- [ ] Correct `HFLabs.LevelZero.Kernels.{device}` NuGet package referenced
- [ ] If shipping multiple device targets: use `DeviceCapabilityDetector` + device-specific subdirs

### Multi-device deployment

Install multiple kernel packages side by side:

```xml
<PackageReference Include="HFLabs.LevelZero.Kernels.acm-g10" Version="1.0.0" />
<PackageReference Include="HFLabs.LevelZero.Kernels.bmg-g21" Version="1.0.0" />
```

---

## 12. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `DllNotFoundException: LevelZeroShim` | Native shim not found | Copy `LevelZeroShim.dll` to output dir or add to PATH |
| `LevelZeroRuntime.IsAvailable()` returns `false` | No Intel GPU or driver missing | Install Intel GPU driver with Level Zero support |
| `FileNotFoundException: SPIR-V not found` | Kernel package not installed | Add `HFLabs.LevelZero.Kernels.{device}` NuGet reference |
| Wrong SPIR-V loaded for GPU | Device target mismatch | Run `DeviceCapabilityDetector.DetectAndSave()` to refresh config |
| Module compilation fails | SPIR-V incompatible with device | Recompile with correct `ocloc -device` target |
| `LevelZeroException` with obscure error code | Native API failure | Check Intel Level Zero spec for the error code value |

## 13. API Reference Summary

### Core Types

| Type | Purpose |
|---|---|
| `LevelZeroRuntime` | Static entry point — `GetDefaultDevice()`, `EnumerateDevices()`, `IsAvailable()` |
| `ComputeDevice` | GPU context + queue + command list — `LoadModule()`, `AllocShared<T>()`, `Launch()` |
| `ComputeModule` | SPIR-V module — `GetKernel()`, `TryGetKernel()` |
| `ComputeKernel` | Kernel handle — `SetArgBuffer()`, `SetArgInt()`, `SetArgFloat()`, `SetGroupSize()` |
| `SharedBuffer<T>` | Typed USM allocation — `Write()`, `ReadTo()`, `ToArray()` |
| `KernelCatalog` | Module path resolution — `ResolveModulePath()`, `ResolveSpirvPath()`, `LoadModule()`, `TryLoadModule()` |
| `DeviceCapabilityDetector` | GPU detection — `Detect()`, `LoadOrDetect()`, `DetectAndSave()` |
| `LevelZeroConfig` | Persisted config — `DeviceTarget`, `Architecture`, `DeviceName` |
| `LevelZeroException` | Error wrapper — `NativeResult`, `NativeMessage` |
| `DeviceInfo` | Discovered device — `Name`, `DriverIndex`, `DeviceIndex`, `Open()` |

---
