using System.Reflection;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

namespace LevelZero.Native;

/// <summary>
/// Extracts embedded native DLLs (LevelZeroShim + ze_loader) to a cache directory
/// and registers a NativeLibrary import resolver so P/Invoke finds them automatically.
/// Triggered once via <see cref="ModuleInitializerAttribute"/>.
/// </summary>
internal static class NativeResolver
{
    private static readonly object s_lock = new();
    private static string? s_extractDir;
    private static bool s_initialized;

#pragma warning disable CA2255 // ModuleInitializer in library is intentional
    [ModuleInitializer]
#pragma warning restore CA2255
    internal static void Initialize()
    {
        if (s_initialized) return;
        lock (s_lock)
        {
            if (s_initialized) return;
            s_initialized = true;
            NativeLibrary.SetDllImportResolver(typeof(NativeResolver).Assembly, ResolveNativeLibrary);
        }
    }

    private static IntPtr ResolveNativeLibrary(string libraryName, Assembly assembly, DllImportSearchPath? searchPath)
    {
        if (!string.Equals(libraryName, "LevelZeroShim", StringComparison.OrdinalIgnoreCase))
            return IntPtr.Zero;

        var dir = EnsureExtracted();

        string shimFile = RuntimeInformation.IsOSPlatform(OSPlatform.Windows)
            ? Path.Combine(dir, "LevelZeroShim.dll")
            : Path.Combine(dir, "libLevelZeroShim.so");

        if (NativeLibrary.TryLoad(shimFile, out var handle))
            return handle;

        return IntPtr.Zero;
    }

    /// <summary>
    /// Extracts embedded native DLLs to a version-stamped cache directory.
    /// Only runs once — subsequent calls return the cached path.
    /// </summary>
    private static string EnsureExtracted()
    {
        if (s_extractDir is not null) return s_extractDir;

        lock (s_lock)
        {
            if (s_extractDir is not null) return s_extractDir;

            var asm = typeof(NativeResolver).Assembly;
            var version = asm.GetName().Version?.ToString() ?? "0.0.0";
            var cacheDir = Path.Combine(Path.GetTempPath(), "LevelZero.NET", version);
            Directory.CreateDirectory(cacheDir);

            if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
            {
                ExtractResource(asm, "LevelZero.Native.ze_loader.dll",
                    Path.Combine(cacheDir, "ze_loader.dll"));
                ExtractResource(asm, "LevelZero.Native.LevelZeroShim.dll",
                    Path.Combine(cacheDir, "LevelZeroShim.dll"));
            }
            else
            {
                ExtractResource(asm, "LevelZero.Native.libLevelZeroShim.so",
                    Path.Combine(cacheDir, "libLevelZeroShim.so"));
            }

            s_extractDir = cacheDir;
            return cacheDir;
        }
    }

    private static void ExtractResource(Assembly assembly, string resourceName, string targetPath)
    {
        if (File.Exists(targetPath)) return;

        using var stream = assembly.GetManifestResourceStream(resourceName);
        if (stream is null) return;

        var tempPath = targetPath + ".tmp";
        try
        {
            using (var fs = new FileStream(tempPath, FileMode.Create, FileAccess.Write, FileShare.None))
                stream.CopyTo(fs);

            File.Move(tempPath, targetPath, overwrite: true);
        }
        catch (IOException)
        {
            // Another process may have written the file concurrently — that's fine
            try { File.Delete(tempPath); } catch { /* best effort */ }
        }
    }
}
