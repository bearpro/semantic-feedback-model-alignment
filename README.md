# LLM in LSP Experiment

This repository explores whether semantic naming guidance can make LLM-generated code easier to align across independently produced models.

The broader problem is formalizing normative and technical documents so they can later be checked for contradictions. A practical obstacle is that different authors or models may describe the same concepts with different type names, property names, and structure. This experiment tests whether that mismatch can be reduced earlier, during authoring.

The setup is simple:

- Several LLMs generate C# models from the same source documents.
- The generation is repeated under different conditions:
  - plain prompting,
  - prompting with naming guidance,
  - prompting with language-server-style feedback.
- The resulting models are projected, aligned, and scored to see whether stronger semantic guidance improves cross-model alignment.

The repository contains the experiment code, notebooks, prompts, and source documents. Generated artifacts are not tracked directly in Git; they are stored in `artifacts.tar.zst`.
