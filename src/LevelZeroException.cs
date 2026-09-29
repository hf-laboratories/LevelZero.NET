namespace LevelZero;

/// <summary>
/// Exception thrown when a Level Zero API call fails.
/// </summary>
public sealed class LevelZeroException(int resultCode, string message) : Exception($"Level Zero error (0x{resultCode:X8}): {message}")
{
    public int NativeResultCode { get; } = resultCode;
    // Null contract: no string? reference fields; all properties are value-type or
    // collection-initialized references with non-null? defaults.
}

