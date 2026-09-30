# Level Zero shim

This shim exposes a minimal C ABI over the Level Zero loader so it can be called from C# via P/Invoke.

## Level Zero loader

The Level Zero loader (`ze_loader.dll`) is not included in this repository. On Windows it is installed with the Intel GPU driver (`System32\ze_loader.dll`). You can also point `LEVELZERO_NET_ZE_LOADER_PATH` at your own copy, or build the loader from the upstream project.

## Build

Set `LEVEL_ZERO_ROOT` to the Level Zero loader install (contains `ze_loader`), then:

```
cmake -B build -S .
cmake --build build --config Release
```

If the loader is not found, the shim builds in stub mode and returns dependency-unavailable errors.

### Windows prerequisites

- Visual Studio Build Tools 2022 (C++ build tools) and Windows 10/11 SDK.
- Run from a Developer PowerShell (or call `VsDevCmd.bat`) so the MSVC/SDK libraries are on `PATH`/`LIB`.
- Recommended generator:

```
cmake -B build -S . -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

If you only have LLVM installed, you still need the Windows SDK libraries available (from Build Tools) for linking.

## Exports
- `lz_init`
- `lz_get_driver_count`
- `lz_get_device_count`
- `lz_get_device_name`
- `lz_get_last_error`
- `lz_get_last_result`
- `lz_clear_error`
