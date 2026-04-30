# General pipeline

## Notebooks

- `1-sources.ipynb`: source matrix build
- `2-infer.ipynb`: prompt execution and `final.cs` artifacts
- `3-project.ipynb`: `final.cs` -> tabular model projections
- `4-align.ipynb`: pairwise schema alignment candidates
- `5-score.ipynb`: pair/model/scenario alignment statistics

## Матрица эксперимента

- Сценарии промптинга (тривильный/с инструкциями/с фидбеком)
- Модели (openai o4mini/gemini/anthropic/gemma)
- Документы
- Алгоритмы выравнивания (Valentine/BDI-kit/Magneto)
