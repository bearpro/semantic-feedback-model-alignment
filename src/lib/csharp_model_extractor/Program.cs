using System.Text;
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

static FileExtractionResult AnalyzeFile(string path)
{
    var fullPath = Path.GetFullPath(path);

    try
    {
        var source = File.ReadAllText(fullPath);
        var tree = CSharpSyntaxTree.ParseText(source, path: fullPath);
        var root = tree.GetCompilationUnitRoot();
        var parseErrors = tree
            .GetDiagnostics()
            .Where(diagnostic => diagnostic.Severity == DiagnosticSeverity.Error)
            .Select(diagnostic => diagnostic.ToString())
            .ToArray();

        var declaredTypes = root
            .DescendantNodes()
            .OfType<BaseTypeDeclarationSyntax>()
            .Select(node => node.Identifier.ValueText)
            .Where(name => !string.IsNullOrWhiteSpace(name))
            .ToHashSet(StringComparer.Ordinal);

        var declaredEnums = root
            .DescendantNodes()
            .OfType<EnumDeclarationSyntax>()
            .Select(node => node.Identifier.ValueText)
            .Where(name => !string.IsNullOrWhiteSpace(name))
            .ToHashSet(StringComparer.Ordinal);

        var walker = new ModelWalker(declaredTypes, declaredEnums);
        walker.Visit(root);

        return new FileExtractionResult(
            Path: fullPath,
            ParseErrorCount: parseErrors.Length,
            ParseErrors: parseErrors,
            AnalysisError: null,
            Elements: walker.Elements.ToArray()
        );
    }
    catch (Exception ex)
    {
        return new FileExtractionResult(
            Path: fullPath,
            ParseErrorCount: 0,
            ParseErrors: Array.Empty<string>(),
            AnalysisError: ex.Message,
            Elements: Array.Empty<ElementRecord>()
        );
    }
}

internal sealed class ModelWalker : CSharpSyntaxWalker
{
    private readonly HashSet<string> _declaredTypes;
    private readonly HashSet<string> _declaredEnums;
    private readonly List<string> _namespaceStack = [];
    private readonly List<string> _typeStack = [];
    private readonly HashSet<string> _seenKeys = new(StringComparer.Ordinal);

    public ModelWalker(HashSet<string> declaredTypes, HashSet<string> declaredEnums)
        : base(SyntaxWalkerDepth.Node)
    {
        _declaredTypes = declaredTypes;
        _declaredEnums = declaredEnums;
    }

    public List<ElementRecord> Elements { get; } = [];

    public override void VisitNamespaceDeclaration(NamespaceDeclarationSyntax node)
    {
        _namespaceStack.Add(node.Name.ToString());
        base.VisitNamespaceDeclaration(node);
        _namespaceStack.RemoveAt(_namespaceStack.Count - 1);
    }

    public override void VisitFileScopedNamespaceDeclaration(FileScopedNamespaceDeclarationSyntax node)
    {
        _namespaceStack.Add(node.Name.ToString());
        base.VisitFileScopedNamespaceDeclaration(node);
        _namespaceStack.RemoveAt(_namespaceStack.Count - 1);
    }

    public override void VisitClassDeclaration(ClassDeclarationSyntax node) =>
        VisitNamedType(node, "class");

    public override void VisitStructDeclaration(StructDeclarationSyntax node) =>
        VisitNamedType(node, "struct");

    public override void VisitInterfaceDeclaration(InterfaceDeclarationSyntax node) =>
        VisitNamedType(node, "interface");

    public override void VisitRecordDeclaration(RecordDeclarationSyntax node) =>
        VisitNamedType(node, "record");

    public override void VisitEnumDeclaration(EnumDeclarationSyntax node)
    {
        var typePath = QualifySymbol(node.Identifier.ValueText);
        AddElement(
            "type",
            typePath,
            node.Identifier.ValueText,
            CurrentParentTypePath(),
            "enum",
            "enum",
            isNullable: false,
            isCollection: false,
            collectionItemType: null,
            isUserDefinedType: false,
            relationTargetType: null,
            commentText: CommentExtraction.ExtractCommentText(node)
        );

        _typeStack.Add(node.Identifier.ValueText);
        foreach (var member in node.Members)
        {
            AddElement(
                "enum_member",
                QualifySymbol(member.Identifier.ValueText),
                member.Identifier.ValueText,
                typePath,
                node.Identifier.ValueText,
                "enum_member",
                isNullable: false,
                isCollection: false,
                collectionItemType: null,
                isUserDefinedType: false,
                relationTargetType: null,
                commentText: CommentExtraction.ExtractCommentText(member)
            );
        }
        _typeStack.RemoveAt(_typeStack.Count - 1);
    }

    private void VisitNamedType(TypeDeclarationSyntax node, string declarationKind)
    {
        var typePath = QualifySymbol(node.Identifier.ValueText);
        AddElement(
            "type",
            typePath,
            node.Identifier.ValueText,
            CurrentParentTypePath(),
            declarationKind,
            declarationKind,
            isNullable: false,
            isCollection: false,
            collectionItemType: null,
            isUserDefinedType: false,
            relationTargetType: null,
            commentText: CommentExtraction.ExtractCommentText(node)
        );

        _typeStack.Add(node.Identifier.ValueText);

        if (node is RecordDeclarationSyntax record && record.ParameterList is not null)
        {
            foreach (var parameter in record.ParameterList.Parameters)
            {
                AddPropertyLikeElement(
                    containingTypePath: typePath,
                    propertyName: parameter.Identifier.ValueText,
                    typeSyntax: parameter.Type,
                    commentText: CommentExtraction.ExtractCommentText(parameter),
                    sourceKind: "property"
                );
            }
        }

        foreach (var member in node.Members)
        {
            switch (member)
            {
                case PropertyDeclarationSyntax property when ShouldIncludeProperty(property, node is InterfaceDeclarationSyntax):
                    AddPropertyLikeElement(
                        containingTypePath: typePath,
                        propertyName: property.Identifier.ValueText,
                        typeSyntax: property.Type,
                        commentText: CommentExtraction.ExtractCommentText(property),
                        sourceKind: "property"
                    );
                    break;
                case FieldDeclarationSyntax field when IsPublic(field.Modifiers):
                    foreach (var variable in field.Declaration.Variables)
                    {
                        AddPropertyLikeElement(
                            containingTypePath: typePath,
                            propertyName: variable.Identifier.ValueText,
                            typeSyntax: field.Declaration.Type,
                            commentText: CommentExtraction.ExtractCommentText(field),
                            sourceKind: "field"
                        );
                    }
                    break;
            }
        }

        foreach (var member in node.Members)
        {
            if (member is BaseTypeDeclarationSyntax nestedType)
            {
                Visit(nestedType);
            }
        }

        _typeStack.RemoveAt(_typeStack.Count - 1);
    }

    private void AddPropertyLikeElement(
        string containingTypePath,
        string propertyName,
        TypeSyntax? typeSyntax,
        string? commentText,
        string sourceKind)
    {
        var features = TypeAnalysis.AnalyzeType(typeSyntax, _declaredTypes, _declaredEnums);
        var propertyPath = $"{containingTypePath}.{propertyName}";
        AddElement(
            "property",
            propertyPath,
            propertyName,
            containingTypePath,
            features.CSharpType,
            features.NormalizedType,
            features.IsNullable,
            features.IsCollection,
            features.CollectionItemType,
            features.IsUserDefinedType,
            features.RelationTargetType,
            commentText
        );

        if (features.RelationTargetType is not null)
        {
            AddElement(
                "relation",
                $"relation.{containingTypePath}.{propertyName}.{features.RelationTargetType}",
                propertyName,
                containingTypePath,
                features.CSharpType,
                features.NormalizedType,
                features.IsNullable,
                features.IsCollection,
                features.CollectionItemType,
                features.IsUserDefinedType,
                features.RelationTargetType,
                commentText
            );
        }
    }

    private void AddElement(
        string elementKind,
        string symbolPath,
        string name,
        string? parentSymbolPath,
        string? csharpType,
        string? normalizedType,
        bool isNullable,
        bool isCollection,
        string? collectionItemType,
        bool isUserDefinedType,
        string? relationTargetType,
        string? commentText)
    {
        var key = $"{elementKind}|{symbolPath}";
        if (!_seenKeys.Add(key))
        {
            return;
        }

        Elements.Add(
            new ElementRecord(
                ElementKind: elementKind,
                SymbolPath: symbolPath,
                Name: name,
                ParentSymbolPath: parentSymbolPath,
                CSharpType: csharpType,
                NormalizedType: normalizedType,
                IsNullable: isNullable,
                IsCollection: isCollection,
                CollectionItemType: collectionItemType,
                IsUserDefinedType: isUserDefinedType,
                RelationTargetType: relationTargetType,
                CommentText: string.IsNullOrWhiteSpace(commentText) ? null : commentText
            )
        );
    }

    private string QualifySymbol(string name)
    {
        var parts = _namespaceStack.Concat(_typeStack).Append(name).Where(part => !string.IsNullOrWhiteSpace(part));
        return string.Join(".", parts);
    }

    private string? CurrentParentTypePath() =>
        _typeStack.Count == 0 ? null : string.Join(".", _namespaceStack.Concat(_typeStack));

    private static bool ShouldIncludeProperty(PropertyDeclarationSyntax property, bool isInterfaceMember) =>
        isInterfaceMember || IsPublic(property.Modifiers);

    private static bool IsPublic(SyntaxTokenList modifiers) =>
        modifiers.Any(token => token.IsKind(SyntaxKind.PublicKeyword));
}

internal static class TypeAnalysis
{
    public static TypeFeatures AnalyzeType(
        TypeSyntax? typeSyntax,
        HashSet<string> declaredTypes,
        HashSet<string> declaredEnums)
    {
        if (typeSyntax is null)
        {
            return new TypeFeatures(
                CSharpType: string.Empty,
                NormalizedType: "unknown",
                IsNullable: false,
                IsCollection: false,
                CollectionItemType: null,
                IsUserDefinedType: false,
                RelationTargetType: null
            );
        }

        var original = typeSyntax.ToString().Trim();
        var isNullable = false;
        var isCollection = false;
        TypeSyntax currentType = typeSyntax;

        if (currentType is NullableTypeSyntax nullableType)
        {
            isNullable = true;
            currentType = nullableType.ElementType;
        }

        if (TryMatchGeneric(currentType, "Nullable", out var nullableArgument))
        {
            isNullable = true;
            currentType = nullableArgument!;
        }

        string? collectionItemType = null;
        if (currentType is ArrayTypeSyntax arrayType)
        {
            isCollection = true;
            collectionItemType = SimplifyTypeName(arrayType.ElementType);
            currentType = arrayType.ElementType;
        }
        else if (TryUnwrapCollection(currentType, out var collectionElementType))
        {
            isCollection = true;
            collectionItemType = SimplifyTypeName(collectionElementType!);
            currentType = collectionElementType!;
        }

        var simpleName = SimplifyTypeName(currentType);
        var isEnum = declaredEnums.Contains(simpleName);
        var isUserDefinedType = declaredTypes.Contains(simpleName) && !isEnum;
        var relationTargetType = isUserDefinedType ? simpleName : null;
        var normalizedType = NormalizeSimpleType(simpleName, isEnum, isUserDefinedType);

        return new TypeFeatures(
            CSharpType: original,
            NormalizedType: normalizedType,
            IsNullable: isNullable,
            IsCollection: isCollection,
            CollectionItemType: collectionItemType,
            IsUserDefinedType: isUserDefinedType,
            RelationTargetType: relationTargetType
        );
    }

    private static bool TryUnwrapCollection(TypeSyntax typeSyntax, out TypeSyntax? elementType)
    {
        elementType = null;
        if (typeSyntax is not GenericNameSyntax genericName)
        {
            if (typeSyntax is QualifiedNameSyntax qualified)
            {
                return TryUnwrapCollection(qualified.Right, out elementType);
            }

            return false;
        }

        var collectionNames = new HashSet<string>(StringComparer.Ordinal)
        {
            "IEnumerable",
            "ICollection",
            "IList",
            "IReadOnlyCollection",
            "IReadOnlyList",
            "List",
            "HashSet",
            "Collection",
            "ObservableCollection",
            "Queue",
            "Stack",
        };

        if (!collectionNames.Contains(genericName.Identifier.ValueText))
        {
            return false;
        }

        if (genericName.TypeArgumentList.Arguments.Count == 0)
        {
            return false;
        }

        elementType = genericName.TypeArgumentList.Arguments[^1];
        return true;
    }

    private static bool TryMatchGeneric(TypeSyntax typeSyntax, string genericIdentifier, out TypeSyntax? argument)
    {
        argument = null;
        if (typeSyntax is GenericNameSyntax genericName && genericName.Identifier.ValueText == genericIdentifier)
        {
            argument = genericName.TypeArgumentList.Arguments.FirstOrDefault();
            return argument is not null;
        }

        if (typeSyntax is QualifiedNameSyntax qualified)
        {
            return TryMatchGeneric(qualified.Right, genericIdentifier, out argument);
        }

        return false;
    }

    private static string SimplifyTypeName(TypeSyntax typeSyntax) =>
        typeSyntax switch
        {
            IdentifierNameSyntax identifier => identifier.Identifier.ValueText,
            PredefinedTypeSyntax predefined => predefined.Keyword.ValueText,
            QualifiedNameSyntax qualified => SimplifyTypeName(qualified.Right),
            AliasQualifiedNameSyntax aliasQualified => SimplifyTypeName(aliasQualified.Name),
            GenericNameSyntax generic => generic.Identifier.ValueText,
            NullableTypeSyntax nullable => SimplifyTypeName(nullable.ElementType),
            ArrayTypeSyntax array => SimplifyTypeName(array.ElementType),
            _ => typeSyntax.ToString().Split('.').Last().Split('<').First().TrimEnd('?'),
        };

    private static string NormalizeSimpleType(string simpleName, bool isEnum, bool isUserDefinedType)
    {
        var integerTypes = new HashSet<string>(StringComparer.Ordinal)
        {
            "byte", "sbyte", "short", "ushort", "int", "uint", "long", "ulong", "nint", "nuint"
        };
        var numberTypes = new HashSet<string>(StringComparer.Ordinal)
        {
            "float", "double", "decimal"
        };

        if (integerTypes.Contains(simpleName))
        {
            return "integer";
        }

        if (numberTypes.Contains(simpleName))
        {
            return "number";
        }

        return simpleName switch
        {
            "bool" => "boolean",
            "string" => "string",
            "char" => "string",
            "DateTime" => "datetime",
            "DateTimeOffset" => "datetime",
            "TimeSpan" => "duration",
            "Guid" => "guid",
            "object" => "object",
            _ when isEnum => "enum",
            _ when isUserDefinedType => "user_defined",
            _ => "unknown",
        };
    }
}

internal static class CommentExtraction
{
    public static string? ExtractCommentText(SyntaxNode node)
    {
        var triviaText = string.Concat(
            node.GetLeadingTrivia()
                .Where(trivia => trivia.IsKind(SyntaxKind.SingleLineDocumentationCommentTrivia)
                    || trivia.IsKind(SyntaxKind.MultiLineDocumentationCommentTrivia)
                    || trivia.IsKind(SyntaxKind.SingleLineCommentTrivia)
                    || trivia.IsKind(SyntaxKind.MultiLineCommentTrivia))
                .Select(trivia => trivia.ToFullString())
        );

        if (string.IsNullOrWhiteSpace(triviaText))
        {
            return null;
        }

        var builder = new StringBuilder();
        using var reader = new StringReader(triviaText);

        while (reader.ReadLine() is { } line)
        {
            var cleaned = line.Trim();
            cleaned = cleaned.TrimStart('/');
            cleaned = cleaned.TrimStart('*');
            cleaned = cleaned.Trim();
            cleaned = cleaned.Replace("<summary>", string.Empty, StringComparison.OrdinalIgnoreCase);
            cleaned = cleaned.Replace("</summary>", string.Empty, StringComparison.OrdinalIgnoreCase);
            if (!string.IsNullOrWhiteSpace(cleaned))
            {
                if (builder.Length > 0)
                {
                    builder.Append(' ');
                }
                builder.Append(cleaned);
            }
        }

        var result = builder.ToString().Trim();
        return string.IsNullOrWhiteSpace(result) ? null : result;
    }
}

internal sealed record FileExtractionResult(
    string Path,
    int ParseErrorCount,
    string[] ParseErrors,
    string? AnalysisError,
    ElementRecord[] Elements
);

internal sealed record ElementRecord(
    string ElementKind,
    string SymbolPath,
    string Name,
    string? ParentSymbolPath,
    string? CSharpType,
    string? NormalizedType,
    bool IsNullable,
    bool IsCollection,
    string? CollectionItemType,
    bool IsUserDefinedType,
    string? RelationTargetType,
    string? CommentText
);

internal sealed record TypeFeatures(
    string CSharpType,
    string NormalizedType,
    bool IsNullable,
    bool IsCollection,
    string? CollectionItemType,
    bool IsUserDefinedType,
    string? RelationTargetType
);
