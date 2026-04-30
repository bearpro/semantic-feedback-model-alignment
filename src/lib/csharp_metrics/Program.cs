using System.Text.Json;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp;
using Microsoft.CodeAnalysis.CSharp.Syntax;

var results = args
    .Select(AnalyzeFile)
    .OrderBy(result => result.Path, StringComparer.Ordinal)
    .ToArray();

Console.WriteLine(
    JsonSerializer.Serialize(
        results,
        new JsonSerializerOptions
        {
            PropertyNamingPolicy = JsonNamingPolicy.CamelCase,
            WriteIndented = true,
        }
    )
);

return;

static FileMetrics AnalyzeFile(string path)
{
    var fullPath = Path.GetFullPath(path);

    try
    {
        var source = File.ReadAllText(fullPath);
        var tree = CSharpSyntaxTree.ParseText(source, path: fullPath);
        var root = tree.GetRoot();
        var definedTypeCount = root
            .DescendantNodes()
            .Count(node => node is BaseTypeDeclarationSyntax || node is DelegateDeclarationSyntax);
        var parseErrors = tree
            .GetDiagnostics()
            .Where(diagnostic => diagnostic.Severity == DiagnosticSeverity.Error)
            .Select(diagnostic => diagnostic.ToString())
            .ToArray();

        return new FileMetrics(
            Path: fullPath,
            ByteCount: new FileInfo(fullPath).Length,
            LineCount: CountLines(source),
            DefinedTypeCount: definedTypeCount,
            ParseErrorCount: parseErrors.Length,
            ParseErrors: parseErrors,
            AnalysisError: null
        );
    }
    catch (Exception ex)
    {
        return new FileMetrics(
            Path: fullPath,
            ByteCount: File.Exists(fullPath) ? new FileInfo(fullPath).Length : 0,
            LineCount: 0,
            DefinedTypeCount: 0,
            ParseErrorCount: 0,
            ParseErrors: Array.Empty<string>(),
            AnalysisError: ex.Message
        );
    }
}

static int CountLines(string source)
{
    if (string.IsNullOrEmpty(source))
    {
        return 0;
    }

    using var reader = new StringReader(source);
    var count = 0;

    while (reader.ReadLine() is not null)
    {
        count++;
    }

    return count;
}

internal sealed record FileMetrics(
    string Path,
    long ByteCount,
    int LineCount,
    int DefinedTypeCount,
    int ParseErrorCount,
    string[] ParseErrors,
    string? AnalysisError
);
