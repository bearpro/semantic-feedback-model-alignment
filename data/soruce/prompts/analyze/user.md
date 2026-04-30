Проанализируй C# код и найди только нарушения правил семантического именования.

Проверяй только эти кейсы:
- `SMN001` — ambiguous type name
- `SMN002` — ambiguous property name
- `SMN003` — quantity without unit suffix
- `SMN004` — duplicate semantic role under different names
- `SMN005` — bool property without boolean naming pattern
- `SMN006` — collection named as singular / singular named as collection
- `SMN007` — generic placeholder name (Data, Info, Details, etc.)
- `SMN008` — inconsistent synonym usage across model

Если нарушений нет, верни пустой массив `warnings`.

Формат ответа:
```json
{
  "summary": "short summary",
  "warnings": [
    {
      "code": "SMN001",
      "symbol": "CustomerData",
      "message": "why this is a problem",
      "suggestion": "better name or rename strategy"
    }
  ]
}
```

C# код:
---
{{CSharpCode}}
---
