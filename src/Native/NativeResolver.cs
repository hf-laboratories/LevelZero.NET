using System.Reflection;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

namespace LevelZero.Native;

/// <summary>
/// Extracts the native shim to a cache directory and registers a NativeLibrary
/// import resolver so P/Invoke finds it automatically.
/// Triggered once via <see cref="ModuleInitializerAttribute"/>.
/// </summary>
internal static class NativeResolver
{
    private const string IpuLoaderPathEnv = "IPU_L0_ZE_LOADER_PATH";
    private const string LevelZeroLoaderPathEnv = "LEVELZERO_NET_ZE_LOADER_PATH";

    private static readonly object s_lock = new();
    private static string? s_extractDir;
    private static bool s_initialized;

#pragma warning disable CA2255 // ModuleInitializer in library is intentional
    [ModuleInitializer]
#pragma warning restore CA2255
    internal static void Initialize()
    {
        if (s_initialized)
        {
            return;
        }

        lock (s_lock)
        {
            if (s_initialized)
            {
                return;
            }

            s_initialized = true;
            NativeLibrary.SetDllImportResolver(typeof(NativeResolver).Assembly, ResolveNativeLibrary);
        }
    }

    private static IntPtr ResolveNativeLibrary(string libraryName, Assembly assembly, DllImportSearchPath? searchPath)
    {
        if (!string.Equals(libraryName, "LevelZeroShim", StringComparison.OrdinalIgnoreCase))
        {
            return IntPtr.Zero;
        }

        string dir = EnsureExtracted();

        string shimFile = RuntimeInformation.IsOSPlatform(OSPlatform.Windows)
            ? Path.Combine(dir, "LevelZeroShim.dll")
            : Path.Combine(dir, "libLevelZeroShim.so");

        return NativeLibrary.TryLoad(shimFile, out nint handle) ? handle : nint.Zero;
    }

    /// <summary>
    /// Extracts embedded native DLLs to a version-stamped cache directory.
    /// Only runs once  subsequent calls return the cached path.
    /// </summary>
    private static string EnsureExtracted()
    {
        if (s_extractDir is not null)
        {
            return s_extractDir;
        }

        lock (s_lock)
        {
            if (s_extractDir is not null)
            {
                return s_extractDir;
            }

            Assembly asm = typeof(NativeResolver).Assembly;
            string version = asm.GetName().Version?.ToString() ?? "0.0.0";
            string rootDir = Path.Combine(Path.GetTempPath(), "LevelZero.NET");
            string cacheDir = Path.Combine(rootDir, version);
            PurgeStaleVersionDirectories(rootDir, version);
            _ = Directory.CreateDirectory(cacheDir);

            if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
            {
                PrepareWindowsZeLoader(cacheDir);
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

    private static void PrepareWindowsZeLoader(string cacheDir)
    {
        string targetPath = Path.Combine(cacheDir, "ze_loader.dll");
        string? overridePath = GetConfiguredLoaderPath();
        if (overridePath is not null)
        {
            CopyFileReplacing(overridePath, targetPath);
            return;
        }

        if (File.Exists(GetSystemZeLoaderPath()))
        {
            // Driver updates can make a previously cached bundled loader stale.
            // Removing it lets Windows bind LevelZeroShim to the installed driver loader.
            TryDelete(targetPath);
            return;
        }

        // No system loader and no override: the Intel Level Zero loader is not redistributed
        // with LevelZero.NET. It is installed with the Intel GPU driver.
    }

    private static string? GetConfiguredLoaderPath()
    {
        string? path = Environment.GetEnvironmentVariable(IpuLoaderPathEnv);
        if (string.IsNullOrWhiteSpace(path))
        {
            path = Environment.GetEnvironmentVariable(LevelZeroLoaderPathEnv);
        }

        if (string.IsNullOrWhiteSpace(path))
        {
            return null;
        }

        path = Environment.ExpandEnvironmentVariables(path.Trim('"'));
        return File.Exists(path) ? path : null;
    }

    private static string GetSystemZeLoaderPath() =>
        Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.System), "ze_loader.dll");

    private static void CopyFileReplacing(string sourcePath, string targetPath)
    {
        string tempPath = targetPath + ".tmp";
        try
        {
            File.Copy(sourcePath, tempPath, overwrite: true);
            File.Move(tempPath, targetPath, overwrite: true);
        }
        finally
        {
            TryDelete(tempPath);
        }
    }

    /// <summary>
    /// Removes cache directories left behind by other assembly versions so driver or
    /// packaging updates cannot keep serving a stale loader/shim. Best effort: files
    /// locked by a live process are simply skipped.
    /// </summary>
    private static void PurgeStaleVersionDirectories(string rootDir, string currentVersion)
    {
        if (!Directory.Exists(rootDir))
        {
            return;
        }

        foreach (string dir in Directory.EnumerateDirectories(rootDir))
        {
            if (string.Equals(Path.GetFileName(dir), currentVersion, StringComparison.OrdinalIgnoreCase))
            {
                continue;
            }

            try
            { Directory.Delete(dir, recursive: true); }
            catch { /* in use elsewhere — skip */ }
        }
    }

    private static void TryDelete(string path)
    {
        try
        { File.Delete(path); }
        catch { /* best effort */ }
    }

    private static void ExtractResource(Assembly assembly, string resourceName, string targetPath)
    {
        if (File.Exists(targetPath))
        {
            return;
        }

        using Stream? stream = assembly.GetManifestResourceStream(resourceName);
        if (stream is null)
        {
            return;
        }

        string tempPath = targetPath + ".tmp";
        try
        {
            using (var fs = new FileStream(tempPath, FileMode.Create, FileAccess.Write, FileShare.None))
            {
                stream.CopyTo(fs);
            }

            File.Move(tempPath, targetPath, overwrite: true);
        }
        catch (IOException)
        {
            // Another process may have written the file concurrently — that's fine
            try
            { File.Delete(tempPath); }
            catch { /* best effort */ }
        }
    }

}
