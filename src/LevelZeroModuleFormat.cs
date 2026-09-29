namespace LevelZero;

/// <summary>
/// Module container formats accepted by the Level Zero runtime.
/// </summary>
public enum LevelZeroModuleFormat : uint
{
    /// <summary>SPIR-V intermediate language module.</summary>
    SpirV = 0,

    /// <summary>Native device-specific binary such as <c>.bin</c>, <c>.native</c>, or <c>.zebin</c>.</summary>
    Native = 1,
}
