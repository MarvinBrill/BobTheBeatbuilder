using System.Text.Json;
using System.Text.RegularExpressions;
using Realms;

// Only reads a caller-created snapshot. No network, extraction, or game writes.
if (args.Length is < 1 or > 2 || !File.Exists(args[0]) || (args.Length == 2 && args[1] != "--skins"))
{
    Console.Error.WriteLine("Usage: LazerReader <snapshot/client.realm> [--skins]");
    return 2;
}

try
{
    var configuration = new RealmConfiguration(Path.GetFullPath(args[0]))
    {
        IsReadOnly = true,
        IsDynamic = true
    };

    using var realm = OpenSnapshot(configuration);
    using var output = new Utf8JsonWriter(Console.OpenStandardOutput());
    output.WriteStartArray();

    if (args.Length == 2)
    {
        foreach (dynamic skin in realm.DynamicApi.All("Skin"))
        {
            if ((bool)skin.DeletePending || (bool)skin.Protected) continue;
            output.WriteStartObject();
            output.WriteString("id", ((Guid)skin.ID).ToString());
            output.WriteString("name", (string)skin.Name);
            output.WriteStartArray("files");
            foreach (dynamic entry in skin.Files)
            {
                output.WriteStartObject();
                output.WriteString("name", (string)entry.Filename);
                output.WriteString("hash", (string)entry.File.Hash);
                output.WriteEndObject();
            }
            output.WriteEndArray();
            output.WriteEndObject();
        }
        output.WriteEndArray();
        output.Flush();
        return 0;
    }

    foreach (dynamic set in realm.DynamicApi.All("BeatmapSet"))
    {
        if ((bool)set.DeletePending || (bool)set.Protected) continue;

        output.WriteStartObject();
        output.WriteString("id", ((Guid)set.ID).ToString());
        var stars = new Dictionary<string, double>(StringComparer.OrdinalIgnoreCase);
        foreach (dynamic beatmap in set.Beatmaps)
        {
            double rating = (double)beatmap.StarRating;
            if (double.IsFinite(rating) && rating >= 0)
                stars[(string)beatmap.Hash] = rating;
        }
        output.WriteStartArray("files");
        foreach (dynamic entry in set.Files)
        {
            output.WriteStartObject();
            output.WriteString("name", (string)entry.Filename);
            output.WriteString("hash", (string)entry.File.Hash);
            if (stars.TryGetValue((string)entry.File.Hash, out double rating))
                output.WriteNumber("star_rating", rating);
            else
                output.WriteNull("star_rating");
            output.WriteEndObject();
        }
        output.WriteEndArray();
        output.WriteEndObject();
    }

    output.WriteEndArray();
    output.Flush();
    return 0;
}
catch (Exception exception)
{
    Console.Error.WriteLine($"Cannot read lazer snapshot: {exception.Message}");
    return 1;
}

static Realm OpenSnapshot(RealmConfiguration configuration)
{
    try { return Realm.GetInstance(configuration); }
    catch (Realms.Exceptions.RealmException exception)
    {
        // Read the actual version; never migrate the user's database.
        var match = Regex.Match(exception.Message, @"last set version (\d+)");
        if (!match.Success) throw;
        configuration.SchemaVersion = ulong.Parse(match.Groups[1].Value);
        return Realm.GetInstance(configuration);
    }
}
