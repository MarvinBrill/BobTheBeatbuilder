using System.Reflection;
using System.Text;
using System.Text.Json;
using osu.Game.Beatmaps;
using osu.Game.Rulesets.Osu;

// No-mod osu!standard difficulty attributes for a list of local .osu files.
// Input: a UTF-8 text file with one path per line. Output: one JSON object per line, same order.
// Results go to a file because osu! libraries may log to the console.
if (args.Length != 2 || !File.Exists(args[0]))
{
    Console.Error.WriteLine("Usage: BobDifficulty <paths.txt> <results.jsonl>");
    return 2;
}

string[] paths = File.ReadAllLines(args[0], Encoding.UTF8).Where(line => line.Length > 0).ToArray();
string calculator = "ppy.osu.Game.Rulesets.Osu " + (typeof(OsuRuleset).Assembly.GetName().Version?.ToString() ?? "unknown");
var results = new string[paths.Length];
using var rulesets = new ThreadLocal<OsuRuleset>(() => new OsuRuleset());

string Calculate(string path)
{
    var output = new Dictionary<string, object?> { ["path"] = path, ["calculator"] = calculator };
    try
    {
        var working = new FlatWorkingBeatmap(path);
        var attributes = rulesets.Value!.CreateDifficultyCalculator(working).Calculate();
        foreach (var property in attributes.GetType().GetProperties(BindingFlags.Public | BindingFlags.Instance))
        {
            if (!property.CanRead || property.GetIndexParameters().Length > 0) continue;
            switch (property.GetValue(attributes))
            {
                case double value when double.IsFinite(value):
                    output[Snake(property.Name)] = value;
                    break;
                case float value when float.IsFinite(value):
                    output[Snake(property.Name)] = (double)value;
                    break;
                case int value:
                    output[Snake(property.Name)] = value;
                    break;
                case long value:
                    output[Snake(property.Name)] = value;
                    break;
            }
        }
    }
    catch (Exception exception)
    {
        output["error"] = $"{exception.GetType().Name}: {exception.Message}";
    }
    return JsonSerializer.Serialize(output);
}

static string Snake(string name)
{
    var builder = new StringBuilder(name.Length + 8);
    for (int index = 0; index < name.Length; index++)
    {
        char character = name[index];
        if (char.IsUpper(character) && index > 0) builder.Append('_');
        builder.Append(char.ToLowerInvariant(character));
    }
    return builder.ToString();
}

if (paths.Length > 0)
{
    // The decoder initialises its shared ruleset store on first use; do that once before threads start.
    results[0] = Calculate(paths[0]);
    Parallel.For(1, paths.Length, new ParallelOptions { MaxDegreeOfParallelism = Math.Max(1, Environment.ProcessorCount - 1) },
        index => results[index] = Calculate(paths[index]));
}

File.WriteAllLines(args[1], results, new UTF8Encoding(false));
return 0;
