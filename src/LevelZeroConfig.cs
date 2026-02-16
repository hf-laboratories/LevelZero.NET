using System.Text.Json;
using System.Text.Json.Serialization;

namespace LevelZero;

/// <summary>
/// Persisted configuration for the detected Level Zero device target.
/// Stored as levelzero-config.json next to the application.
/// </summary>
public sealed class LevelZeroConfig
{
    /// <summary>Default config file name.</summary>
    public const string FileName = "levelzero-config.json";

    /// <summary>The best-matching ocloc device target (e.g. "bmg-g21", "tgllp").</summary>
    [JsonPropertyName("deviceTarget")]
    public string DeviceTarget { get; set; } = "";

    /// <summary>The architecture family name (e.g. "Xe2-HPG", "Gen12").</summary>
    [JsonPropertyName("architecture")]
    public string Architecture { get; set; } = "";

    /// <summary>The GPU device name string reported by Level Zero.</summary>
    [JsonPropertyName("deviceName")]
    public string DeviceName { get; set; } = "";

    /// <summary>When this detection was performed (UTC).</summary>
    [JsonPropertyName("detectedAt")]
    public DateTime DetectedAt { get; set; }

    /// <summary>Whether Level Zero hardware was actually detected.</summary>
    [JsonPropertyName("hardwarePresent")]
    public bool HardwarePresent { get; set; }

    private static readonly JsonSerializerOptions s_jsonOptions = new()
    {
        WriteIndented = true,
        DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingDefault
    };

    /// <summary>Saves this config to the specified path.</summary>
    public void Save(string path)
    {
        var json = JsonSerializer.Serialize(this, s_jsonOptions);
        File.WriteAllText(path, json);
    }

    /// <summary>Loads a config from the specified path. Returns null if file doesn't exist.</summary>
    public static LevelZeroConfig? Load(string path)
    {
        if (!File.Exists(path))
            return null;
        var json = File.ReadAllText(path);
        return JsonSerializer.Deserialize<LevelZeroConfig>(json, s_jsonOptions);
    }

    /// <summary>
    /// Returns the default config file path: {AppContext.BaseDirectory}/levelzero-config.json
    /// </summary>
    public static string DefaultPath => Path.Combine(AppContext.BaseDirectory, FileName);
}
