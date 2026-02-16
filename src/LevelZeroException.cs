namespace LevelZero;

/// <summary>
/// Exception thrown when a Level Zero API call fails.
/// </summary>
public sealed class LevelZeroException : Exception
{
    public int NativeResultCode { get; }

    public LevelZeroException(int resultCode, string message)
        : base($"Level Zero error (0x{resultCode:X8}): {message}")
    {
        NativeResultCode = resultCode;
    }
}
