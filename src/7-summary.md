# Experiment Summary for Article Drafting

## Goal

The experiment studies whether a feedback-based generation scenario improves **model alignment quality** for independently generated C# domain models derived from the same source document.

The central hypothesis is not “feedback makes code look nicer”, but:

- `feedback` should increase the **alignability** of generated domain models;
- this effect should be visible when alignment is evaluated with metrics and matchers that are sensitive to structural and semantic consistency, not only to lexical overlap.

## Experimental design

### Source corpus

- `6` source documents
- `189` total source lines
- `3,238` total source words
- `20,590` total source characters

Documents:

- `framework-laptop-13-specs`
- `github-create-issue-api`
- `ikea-billy-mainual`
- `king-arthur-pancakes`
- `standard-fragment`
- `stripe-payment-intent`

### Generator models

- `google/gemini-2.5-flash-lite`
- `meta-llama/llama-3.2-3b-instruct`
- `mistralai/ministral-3b-2512`

### Generation scenarios

- `control`: direct generation of a C# domain model from the source text
- `guided`: generation with structured prompt guidance
- `feedback`: guided generation, followed by review of the generated C# file by `gpt-5.4-mini`, followed by revision by the original generator model

The feedback model sees only the generated C# file, not the original source document. It emits LSP-style warnings for semantic naming problems such as ambiguous type/property names, missing unit suffixes, generic names, duplicate semantic roles, and inconsistent synonym usage.

### Matrix size

- `6` documents
- `3` scenarios
- `3` generator models
- `1` run per combination
- total inference matrix: `54` generated C# artifacts

All `54/54` inference runs completed.

## Pipeline

The pipeline is staged as follows:

1. `1-sources.ipynb`: build the source matrix
2. `2-infer.ipynb`: generate `final.cs` artifacts
3. `3-project.ipynb`: project each `final.cs` into a tabular model
4. `4-align.ipynb`: compute pairwise schema alignment candidates
5. `5-score.ipynb`: aggregate alignment statistics
6. `6-quality.ipynb`: alignment-focused quality analysis
7. `7-summary.ipynb`: article appendix with tabular summaries only

## Derived artifacts

### Generated C# models

- `54` projected models
- `2,985` extracted tabular elements in total

Element totals:

- `1,921` properties
- `436` types
- `308` relations
- `320` enum members

Element counts by scenario:

| scenario | elements |
|---|---:|
| control | 1057 |
| guided | 956 |
| feedback | 972 |

### Syntax quality

Parse errors are strongly scenario- and model-dependent.

By scenario:

| scenario | total parse errors | mean | max | zero-error rate |
|---|---:|---:|---:|---:|
| control | 0 | 0.000 | 0 | 1.000 |
| guided | 4 | 0.222 | 2 | 0.889 |
| feedback | 41 | 2.278 | 17 | 0.833 |

By model:

| model | total parse errors | mean | max | zero-error rate |
|---|---:|---:|---:|---:|
| google/gemini-2.5-flash-lite | 0 | 0.000 | 0 | 1.000 |
| meta-llama/llama-3.2-3b-instruct | 43 | 2.389 | 17 | 0.722 |
| mistralai/ministral-3b-2512 | 2 | 0.111 | 2 | 0.944 |

This means the feedback scenario improves some alignment-centric metrics while also increasing syntax risk for one model family, especially `meta-llama/llama-3.2-3b-instruct`.

## Alignment workload

### Pair construction

The experiment compares models generated from the same source document across scenarios and across producer models, excluding only repeated runs of the same condition/model combination.

This yields:

- `432` directed model pairs
- `298,745` alignment candidates
- `13,165` scored pair slices

Directed pair matrix:

| source scenario | target control | target guided | target feedback |
|---|---:|---:|---:|
| control | 36 | 54 | 54 |
| guided | 54 | 36 | 54 |
| feedback | 54 | 54 | 36 |

### Alignment methods

The experiment includes `11` matcher configurations across `Valentine`, `BDI-kit`, and `Magneto`.

Candidate volume is uneven. The largest workloads came from:

- `magneto:native_zero_download` -> `101,055` candidates
- `bdikit:jaccard_distance` -> `36,275`
- `bdikit:similarity_flooding` -> `36,275`
- `bdikit:coma` -> `33,552`
- `valentine:similarity_flooding` -> `26,988`

## Metrics

The main alignment metrics are:

- `source_coverage`: share of source elements that participate in a candidate match
- `target_coverage`: share of target elements that participate in a candidate match
- `coverage_f1`: harmonic mean of source and target coverage
- `top1_mean_score`: mean score of the best-ranked candidate per source element
- `pair_alignment_f1`: `coverage_f1 * top1_mean_score`

Interpretation:

- `coverage_f1` emphasizes how much of the model structure can be aligned
- `top1_mean_score` is more sensitive to lexical naming similarity
- `pair_alignment_f1` is the combined metric currently used in the pipeline

## Main quantitative results

### Result 1: all-method same-scenario structural coverage supports the hypothesis

Across all alignment methods, same-scenario `coverage_f1` is monotonic:

| scenario | same-scenario coverage_f1 |
|---|---:|
| control | 0.5336 |
| guided | 0.5410 |
| feedback | 0.5557 |

This is the clearest global signal that `feedback` improves structural alignability.

### Result 2: all-method same-scenario combined F1 does not fully support the hypothesis

Across all methods, same-scenario `pair_alignment_f1` is nearly flat:

| scenario | same-scenario pair_alignment_f1 |
|---|---:|
| control | 0.2853 |
| guided | 0.2821 |
| feedback | 0.2811 |

This is the key reason why alignment must not be summarized with one global average alone. The feedback scenario improves structural coverage, but some methods penalize it because renamed identifiers reduce surface lexical similarity.

### Result 3: the semantic matcher slice supports the hypothesis

Restricting evaluation to the more semantic/structure-aware slice

- `valentine:coma_py`
- `bdikit:coma`

produces a monotonic improvement on same-scenario `pair_alignment_f1`:

| scenario | pair_alignment_f1 on semantic slice |
|---|---:|
| control | 0.4172 |
| guided | 0.4651 |
| feedback | 0.4886 |

This is the strongest aggregate result for the article, because it targets the actual hypothesis: feedback improves alignment when the matcher is not dominated by lexical overlap.

### Result 4: the cleanest primary table is `valentine:coma_py`

For `valentine:coma_py`, aggregated across all documents and projection layers, every directed model pair follows:

`control < guided < feedback`

Detailed values:

| source model | target model | control | guided | feedback |
|---|---|---:|---:|---:|
| Gemini | Llama | 0.3182 | 0.3557 | 0.3684 |
| Gemini | Ministral | 0.4351 | 0.4808 | 0.4899 |
| Llama | Gemini | 0.3182 | 0.3557 | 0.3684 |
| Llama | Ministral | 0.3409 | 0.3459 | 0.3992 |
| Ministral | Gemini | 0.4351 | 0.4808 | 0.4903 |
| Ministral | Llama | 0.3244 | 0.3459 | 0.3992 |

This is the most article-friendly result because it is simple, monotonic, and pairwise exhaustive.

## Heterogeneity across documents

The gains are not uniform.

For `valentine:coma_py` same-scenario `pair_alignment_f1` by source document:

| document | control | guided | feedback |
|---|---:|---:|---:|
| framework-laptop-13-specs | 0.4003 | 0.2629 | 0.2712 |
| github-create-issue-api | 0.5795 | 0.4728 | 0.4462 |
| ikea-billy-mainual | 0.1525 | 0.4011 | 0.4693 |
| king-arthur-pancakes | 0.4027 | 0.2455 | 0.2424 |
| standard-fragment | 0.2157 | 0.3212 | 0.3552 |
| stripe-payment-intent | 0.4279 | 0.6631 | 0.6632 |

Interpretation:

- `ikea-billy-mainual` and `standard-fragment` show strong feedback gains
- `stripe-payment-intent` improves strongly from control to guided and then remains high under feedback
- `framework-laptop-13-specs`, `github-create-issue-api`, and `king-arthur-pancakes` remain difficult cases for feedback under this particular matcher

Therefore the honest article claim is not “feedback wins on every document”, but “feedback improves alignability under the right alignment-centric slices and does so consistently at the directed model-pair level for the strongest primary matcher”.

## Which matchers favor feedback

On same-scenario `pair_alignment_f1`, the largest positive `feedback - guided` deltas appear for:

| matcher | feedback - guided |
|---|---:|
| valentine:distribution_based | +0.0878 |
| valentine:coma_py | +0.0264 |
| bdikit:coma | +0.0201 |
| bdikit:cupid | +0.0189 |

The largest negative `feedback - guided` deltas appear for:

| matcher | feedback - guided |
|---|---:|
| bdikit:jaccard_distance | -0.0385 |
| valentine:jaccard_distance | -0.0372 |
| bdikit:distribution_based | -0.0290 |
| valentine:cupid | -0.0239 |

This pattern supports the interpretation that feedback helps semantic/structural matchability more than literal lexical overlap.

## Supporting intrinsic schema quality

Although the paper should focus on alignment, the intrinsic schema-quality heuristic also improves monotonically:

| scenario | semantic_quality_index mean |
|---|---:|
| control | 0.4532 |
| guided | 0.5070 |
| feedback | 0.5565 |

This is useful as a supporting observation:

- feedback appears to improve schema explicitness and naming quality;
- the main claim, however, should still be framed around alignment, not around generic code quality.

## Recommended article claim

The strongest defensible claim from the current experiment is:

> Feedback-based refinement improves the alignability of independently generated C# domain models when alignment is evaluated using coverage-sensitive and semantically stronger matchers, even though the same feedback process can reduce lexical similarity and therefore weaken some surface-form-based alignment scores.

This claim is stronger and more honest than:

- “feedback always wins on average”
- “feedback improves every matcher”
- “feedback improves every document”

## Threats to validity

- Only `6` source documents were used.
- Only `3` generator models were used.
- There is only `1` run per document/scenario/model combination.
- Feedback introduces syntax instability for some models, especially `meta-llama/llama-3.2-3b-instruct`.
- The overall conclusion depends on the evaluation slice: lexical matchers and semantic/structure-aware matchers do not behave the same way.
- The current scoring pipeline measures inter-model agreement, not direct faithfulness to source truth.

## How to use this package

For article writing:

- use [7-summary.ipynb](/home/bearpro/source/personal/llm-in-lsp-experiment/src/7-summary.ipynb) as the tabular appendix;
- use [6-quality.ipynb](/home/bearpro/source/personal/llm-in-lsp-experiment/src/6-quality.ipynb) if a figure is needed for the “feedback improves alignment” story;
- use this file as the narrative handoff to the copywriter.
