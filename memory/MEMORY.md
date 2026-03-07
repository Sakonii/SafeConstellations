# SteeringFail Project Memory

## Project Overview
**Predecessor:** SafeConstellations — mitigates LLM over-refusal using task-specific constellation steering.
**Successor Paper:** "Why Steering Methods Fail? A Mechanistic Analysis from Safety Perspective"
- Central thesis: Linear Representation Hypothesis (LRH) is *incomplete* — refusal is task-conditioned, not a single direction.
- Key framing: "Refusal is not mediated by a single direction — it is task-conditioned, and the causal circuits upstream of refusal activation differ systematically by task type."

## SafeConstellations Framework (Existing Work)
- Dataset: 270 samples × 32 layers × 4096-dim embeddings (LLaMA 3.1-8B, Qwen 1.5-7B)
- Tasks: sentiment_analysis, translate, cryptanalysis, rag_qa, rephrase
- Labels: response_labels (cautious/harmful/not_harmful) + refusal_class (direct_answer/direct_refusal/indirect_refusal)
- TARGET_BEHAVIOR_MASK: (cautious | not_harmful) & direct_answer
- OVER_REFUSAL_MASK: refusing & benign_intent_tasks
- Key finding: Mid-layers (11–15) show best separation between target and over-refusal behaviors
- Embeddings stored as .pt files at layer_{n}_input_norm and final_norm keys

## Notebook Structure (Existing)
- 1: Dataset construction
- 2: Over-refusal evaluation (heatmaps)
- 3: Spider plots
- 4: Memory Bank Construction (LLaMA + Qwen)
- 5: SafeConstellations steering (LLaMA + Qwen)
- 6: MMLU utility evaluation
- 7: Clustering analysis (silhouette, Davies-Bouldin, centroid distance per layer/task)

## Planned Successor Notebooks (see research_plan.md for full details)
- 7 (extend): Add per-task constellation UMAP grid + cross-task centroid distance matrix + "galaxy map"
- 8: Arditi et al. Replication — compute refusal direction, ablate, eval on harmful/harmless
- 9: Test Universality — Q1: task-direction cosine similarity, Q2: per-task ablation suppression, Q3: cross-task transfer matrix
- 10: Failure Mode Taxonomy — classify under/over-steering, task-mismatch, layer-mismatch; regime diagram
- 11: Vocabulary Projection — logit lens + α sweep + "assistant" degeneration explanation
- 12: Task-Conditioned Attention Circuits — per-task head attribution heatmaps + causal patching

## Arditi et al. Key Details
- Paper: "Refusal in Language Models Is Mediated by a Single Direction" (arXiv:2406.11717)
- GitHub: https://github.com/andyrdt/refusal_direction
- Direction = mean(refused_harmful_embeddings) - mean(answered_harmless_embeddings) at each layer
- Ablation: h = h - (h·d̂)d̂ (project out direction, applied at ALL layers via hooks)
- Our data already has the needed classes: harmful_instruction+refused → "refused harmful"; benign_instruction+direct_answer → "harmless answered"

## Key Technical Details
- Model: meta-llama/Meta-Llama-3.1-8B-Instruct (32 layers, 4096-dim)
- Embeddings dir: ./embeddings/ (CSV + .pt torch files)
- CSV has columns: torch_path, llm_evaluation, refusal_class
- Torch file keys: embeddings, thinking_content, responses, texts, text_type_labels, intended_task_labels
- UMAP: n_components=2, n_neighbors=15, min_dist=0.1, random_state=42
- Layer naming: layer_{n}_input_norm (n=0..31), final_norm

## Links
- research_plan.md: Full notebook designs with code sketches
