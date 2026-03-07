# Research Plan: "Why Steering Methods Fail?"
## Successor to SafeConstellations — A* Target

**Core Thesis:**
The Linear Representation Hypothesis is *incomplete*. Refusal is not mediated by a single universal direction — it is task-conditioned. A global steering vector captures only the *final common pathway* of refusal while missing the upstream causal structure. This explains *when* and *why* current steering methods fail.

---

## Paper Structure (Map to Notebooks)

| Section | Claim | Notebook |
|---------|-------|----------|
| §3 Constellation Evidence | Task-specific behavioral clusters exist and are geometrically stable | 7 (extended) |
| §4 Baseline (Arditi) | Single-direction ablation works globally but unevenly | 8 |
| §5 Universality Test | Task-specific directions diverge; global direction is a lossy approximation | 9 |
| §6 Failure Modes | Failures are geometrically predictable, not random | 10 |
| §7 Logit Lens | Over-steering pushes activations OOD; "assistant" token explained | 11 |
| §8 Circuits | Different attention heads mediate refusal per task | 12 |

---

## Notebook 7 (Extended): Task-Specific Constellation Patterns

**Status:** Exists. Extend with two new parts.

### What Already Exists
- Silhouette Score, Davies-Bouldin, centroid distance per layer per task
- Individual sample trajectory plots (layers 14–18)

### Part A (New): Per-Task Constellation UMAP Grid
For each task in {sentiment_analysis, translate, cryptanalysis, rag_qa, rephrase}:
- Create a 5×6 subplot grid (5 tasks × 6 layer groups)
- UMAP at each layer group, colored by TARGET (green) vs OVER_REFUSAL (red)
- Show that constellations are not only different behaviorally but geometrically distinct per task
- Key plot: overlay all tasks on same UMAP space to show their non-overlap

### Part B (New): Cross-Task Constellation Geometry
- Compute centroid vectors for each task at each layer
- Cross-task centroid distance matrix (5×5 heatmap per layer group): do different tasks occupy different regions?
- "Galaxy map" figure: each task constellation = a cluster of stars in UMAP, show they are well-separated galaxies

### Key Message for Paper
"Constellation patterns are not a single behavioral cluster — each task occupies a distinct region of activation space. The behavioral geometry is task-conditioned from mid-layers onward. This motivates examining whether the refusal direction itself is universal."

---

## Notebook 8: Arditi et al. Replication (Stage 1 Baseline)

**Goal:** Reproduce the single-direction finding on LLaMA-3.1-8B and Qwen-1.5-7B. This is the *null hypothesis* you subsequently complicate.

### Dataset Mapping
From existing data:
```python
# "Harmful refused" class (Arditi: harmful prompts with high refusal score)
harmful_refused_mask = (text_types == 'harmful_instruction') & (
    (refusal_labels == 'direct_refusal') | (refusal_labels == 'indirect_refusal')
)

# "Harmless answered" class (Arditi: harmless prompts with low refusal score)
harmless_answered_mask = (text_types == 'benign_instruction') & (
    refusal_labels == 'direct_answer'
)
```

### Step 1: Compute the Refusal Direction (at each layer)
```python
for layer in range(32):
    layer_name = f'layer_{layer}_input_norm'
    mu_refused = embeddings_np[layer_name][harmful_refused_mask].mean(axis=0)
    mu_answered = embeddings_np[layer_name][harmless_answered_mask].mean(axis=0)
    candidate_directions[layer] = mu_refused - mu_answered  # shape: (4096,)
    candidate_directions[layer] /= np.linalg.norm(candidate_directions[layer])
```

### Step 2: Select Best Layer/Direction
- For each candidate direction d_L, measure discrimination:
  - Project all embeddings onto d_L: scores = embeddings @ d_L
  - Compute AUROC separating harmful_refused vs harmless_answered
  - Best layer = argmax(AUROC)
  - Also compute: variance explained by d_L within each class (intra-class), vs. between classes
- Report the selected layer and direction quality metrics

### Step 3: Ablation Intervention (requires model)
Implement Arditi-style hook to project out the refusal direction:
```python
def make_ablation_hook(direction):
    d = direction / direction.norm()
    def hook(module, input, output):
        if isinstance(output, tuple):
            h = output[0]
        else:
            h = output
        h = h - (h @ d).unsqueeze(-1) * d  # project out direction
        return (h,) + output[1:] if isinstance(output, tuple) else h
    return hook
```
Apply at ALL layers (Arditi applies at all layers by default).

### Step 4: Evaluation
Run both baseline (no hook) and ablated model on:
- **Harmful test set**: measure refusal rate before/after ablation
- **Harmless test set**: measure refusal rate (should stay low = utility preserved)
- Report: ASR (Attack Success Rate = % harmful queries answered after ablation), utility preservation %

Use LLM-as-judge (existing evaluator from notebook 5/6).

### Step 5: Report Alongside Arditi's Numbers
Table comparing: Arditi (LLaMA-3-8B-Instruct) vs. Ours (LLaMA-3.1-8B-Instruct)
- Refusal rate before ablation
- Refusal rate after ablation (harmful)
- Refusal rate after ablation (harmless)
- MMLU accuracy delta

### Key Message
"We successfully replicate the Arditi et al. finding: a single direction ablated across all layers suppresses refusal by X% on harmful prompts while preserving harmless compliance at Y%. This establishes our null hypothesis."

---

## Notebook 9: Test Universality Across Tasks (Stage 2)

**Goal:** Break the null hypothesis systematically with three targeted experiments.

### Q1: Is the Refusal Direction the Same Across Tasks?

Compute task-specific refusal directions:
```python
task_directions = {}
for task in TASKS:
    task_refused = harmful_refused_mask & (intended_tasks == task)
    task_answered = harmless_answered_mask & (intended_tasks == task)

    if task_refused.sum() < 3 or task_answered.sum() < 3:
        continue

    for layer in range(32):
        layer_name = f'layer_{layer}_input_norm'
        mu_r = embeddings_np[layer_name][task_refused].mean(axis=0)
        mu_a = embeddings_np[layer_name][task_answered].mean(axis=0)
        d = mu_r - mu_a
        task_directions[(task, layer)] = d / np.linalg.norm(d)
```

**Analysis:**
- Cosine similarity matrix (Tasks × Tasks) at key layers (e.g., 12, 15, 20)
- Show as heatmap: if values near 1.0 → Arditi is right; if near 0 or negative → task-conditioned
- Alignment of each task direction with global Arditi direction (barplot)
- Layer-wise evolution: how does the cross-task agreement evolve from layer 0 → 31?
  - Hypothesis: early layers diverge (task-specific computation); late layers converge (shared final pathway)

**Figure:** Two-panel. Left: cosine similarity heatmap at layer 15 (divergence peak). Right: mean cross-task cosine similarity vs. layer (U-shaped curve with convergence at final layers).

### Q2: Does Single Direction Ablation Suppress Refusal Equally Across Tasks?

Using the single global Arditi direction:
- Run ablated model on test set for each task separately
- Report: per-task refusal suppression rate after ablation
- Expected finding: some tasks (e.g., translate) are barely affected; others (e.g., sentiment) are strongly affected
- Show as barplot: baseline refusal rate (red) vs. ablated (blue) per task

**Interpretation:** "The single direction captures the refusal pathway for Task A much better than Task B — because their refusal directions are not the same."

### Q3: Does a Task-Specific Steering Vector Transfer to Other Tasks?

Take task-specific directions learned from SafeConstellations / notebook 9Q1:
```python
# Apply v_sentiment to translate samples
results = cross_task_transfer_matrix  # shape: (n_tasks, n_tasks)
# results[i][j] = refusal suppression when direction from task_i applied to task_j samples
```
- Show as matrix heatmap: diagonal = within-task (should be best); off-diagonal = transfer
- If off-diagonal is near zero → directions are task-specific and non-transferable

**Figure:** Transfer matrix heatmap. The diagonal should be bright (high suppression), off-diagonal dim (low transfer). This is a visually clean single figure for the paper.

### Key Message
"Task-specific refusal directions have cosine similarity of X.XX ± Y (vs. 1.0 if universal). Single-direction ablation suppresses refusal by Z% for translate tasks but only W% for sentiment tasks. Task-specific directions do not transfer: applying the translate-steering vector to sentiment analysis yields only P% suppression (baseline: Q%). Collectively, these results demonstrate that refusal is task-conditioned, not mediated by a single universal direction."

---

## Notebook 10: Failure Mode Taxonomy

**Goal:** Show that steering failures are *geometrically predictable*, not random.

### Setup
Load all ablation results from existing notebooks (fixed layers, dynamic layers, task-specific vs global).

### Failure Mode Classification
Assign each steered output a failure mode:
```python
def classify_failure(response, eval_label, refusal_class, alpha, layer_config, task):
    if refusal_class in ['direct_refusal', 'indirect_refusal']:
        return 'under_steering'  # Still refused after steering
    elif is_degenerate(response):  # detect "assistant assistant", repetition
        return 'over_steering'
    elif refusal_class == 'direct_answer' and task != best_task_for_direction:
        return 'task_direction_mismatch'
    elif layer_config == 'fixed' and alpha > alpha_threshold:
        return 'layer_mismatch'
    else:
        return 'success'

def is_degenerate(response, threshold=3):
    tokens = response.split()
    # Check for repetition: same token appearing > threshold times consecutively
    for i in range(len(tokens) - threshold):
        if len(set(tokens[i:i+threshold])) == 1:
            return True
    return False
```

### Analysis
1. **Distribution barplot**: stacked bar per task × ablation type → show failure mode breakdown
2. **Geometric correlates of each failure mode:**
   - Under-steering: plot distribution of steering-vector alignment for under-steered vs. successful
   - Over-steering: plot per-token probability of "assistant" token as function of α
   - Task-direction mismatch: correlation between cosine_sim(v_global, v_task) and success rate
   - Layer-mismatch: show trajectory collapse metric when steering at wrong layers

3. **Decision boundary visualization:**
   - α × cosine_sim(v_global, v_task) space: color-code by failure mode
   - This creates a clean "regime diagram" figure for the paper

### Key Message
"Failure mode is predictable from two quantities: (1) the steering coefficient α and (2) the alignment between the global direction and the task-specific direction. Low alignment + high α → always fails. Task-specific steering avoids this quadrant entirely."

---

## Notebook 11: Vocabulary Projection Under Steering (Logit Lens)

**Goal:** Explain the "assistant assistant…" degeneration mechanistically.

### Setup (requires model)
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct")
```

### Experiment 11a: Logit Lens Baseline
Extract hidden states at each layer and decode via the final LM head (logit lens):
```python
def get_logit_lens(model, tokenizer, prompt, layer_range=range(32)):
    with torch.no_grad():
        inputs = tokenizer(prompt, return_tensors='pt').to(model.device)
        outputs = model(**inputs, output_hidden_states=True)

    results = {}
    lm_head = model.lm_head
    norm = model.model.norm  # final layer norm

    for layer_idx in layer_range:
        h = outputs.hidden_states[layer_idx + 1]  # +1 for embedding layer
        logits = lm_head(norm(h[:, -1, :]))       # decode last token position
        top5 = torch.topk(logits, 5, dim=-1)
        results[layer_idx] = [tokenizer.decode([t]) for t in top5.indices[0]]

    return results
```
Visualize as heatmap: rows=layers, columns=rank (1-5), content=token string.
Do this for: benign prompt, harmful prompt, harmful + over-steered.

### Experiment 11b: Steering Intensity Sweep
For α ∈ {0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0}:
- Apply steering hook at layer 15 (best SafeConstellations layer): h' = h + α * v_refusal
  - Note: positive α pushes TOWARD target behavior (anti-refusal direction)
  - Negative or large α → OOD
- Record: top-1 predicted token at final layer
- Record: probability of "assistant" / "<|start_header_id|>" tokens
- Record: perplexity of output

**Figure A:** Line plot of P("assistant") vs. α — shows phase transition at collapse threshold.
**Figure B:** Token prediction heatmap: rows=α values, columns=top-5 tokens (shows the transition from meaningful output → "assistant" domination).

### Experiment 11c: WHY "assistant"?
Mechanistic explanation:
- LLaMA chat template: `<|start_header_id|>user<|end_header_id|>\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n`
- At position of generating first output token, the most probable token is always "assistant" from the chat template structure
- When over-steered, hidden state falls outside trained distribution → model falls back to maximum prior
- Show: the "assistant" embedding vector's cosine similarity with the OOD activation region
- Measure: distance from steered activations to training manifold (approximate via PCA reconstruction error)

### Experiment 11d: Per-Layer Semantic Stability
For each layer, measure: does the top-1 predicted token change between no-steering and steered (α=1)?
- "Stable layers": layers where steering doesn't disrupt next-token prediction
- "Disrupted layers": layers where steering causes prediction to flip
- This identifies the layers where semantic content is committed vs. still plastic

### Key Message
"Over-steering pushes hidden states outside the training manifold. At α ≈ {threshold}, the model's next-token prediction collapses from meaningful content to 'assistant' — the highest-prior token from the chat template. This is not a semantically meaningful prediction but a failure mode: the model has been steered into an out-of-distribution state where it defaults to the most statistically frequent token from its position in the sequence."

---

## Notebook 12: Task-Conditioned Attention Circuits

**Goal:** Identify *mechanistically* which attention heads mediate refusal per task.

### Setup
Requires model + hooks. Use TransformerLens or manual hooks.

### Experiment 12a: Attention Head Contribution to Refusal Direction
For each head (L, H) compute its contribution to the refusal direction:
```python
def get_head_refusal_contribution(model, prompts_refused, prompts_answered, direction):
    """
    For each attention head, measure how much its output is aligned with the refusal direction.
    """
    contributions = torch.zeros(n_layers, n_heads)

    for (prompts, label) in [(prompts_refused, +1), (prompts_answered, -1)]:
        for prompt in prompts:
            with hooks_to_capture_head_outputs(model) as head_outputs:
                model(prompt)

            for layer in range(n_layers):
                for head in range(n_heads):
                    h_out = head_outputs[(layer, head)]  # (seq, d_head)
                    h_out_proj = project_to_model_dim(h_out, layer, head)  # (seq, d_model)
                    alignment = (h_out_proj[:, -1] @ direction).item()
                    contributions[layer, head] += label * alignment

    return contributions / len(prompts_refused)
```

Show as: 32×32 (layers × heads) heatmap — "refusal attribution map."

### Experiment 12b: Per-Task Attribution Maps
Repeat 12a for each task separately:
- 5 heatmaps (one per task)
- Compute overlap: |top-10 heads for task_A ∩ top-10 heads for task_B| / 10
- Expected finding: early-mid layers → task-specific heads dominate; late layers → shared "final refusal head"

**Figure:** 5-panel subplot of attribution heatmaps (one per task). Shared heads highlighted with border.

### Experiment 12c: Causal Tracing (activation patching)
For each attention head (L, H) on a subset of 20 examples:
- Run model on refused_harmful prompt → get hidden states H_refused
- Run model on answered_harmless prompt → get hidden states H_answered
- Patch: replace head (L, H) output in H_answered run with the output from H_refused run
- Measure: does output flip to refusal?
- Causally necessary heads: those where patching flips the output

This is compute-intensive; run on 20 examples × top-20 heads (by attribution score).

### Experiment 12d: MLP Neuron Analysis (optional extension)
Following the brainstorming idea "locate neurons":
- For each MLP layer, identify top-k neurons that fire for refused vs answered
- Show: per-task neuron overlap (analogous to 12b but for MLP neurons)
- Connect: are the "constellation layers" (11-15) also the "refusal neuron layers"?

### Key Message
"Refusal in sentiment-analysis tasks is primarily mediated by attention heads H{L1}.{H1} and H{L2}.{H2}, while translation refusal depends on H{L3}.{H3} and H{L4}.{H4}. These non-overlapping circuits share only {N} heads in the final 3 layers — the 'final common pathway' that the Arditi direction captures. A global steering vector corrects the shared pathway but cannot simultaneously address the task-specific upstream circuits, explaining the uneven per-task suppression observed in Notebook 9."

---

## Priority & Compute Requirements

| Notebook | Compute | Timeline | Existing Data? |
|----------|---------|----------|---------------|
| 7 (extend) | Low (UMAP only) | 1 day | Yes — existing embeddings |
| 8 (Arditi) | Medium (model inference) | 2 days | Partial — need ablation runs |
| 9 (Universality) | Low + Medium | 2 days | Yes for Q1; need model for Q2/Q3 |
| 10 (Taxonomy) | Low | 1 day | Yes — existing ablation outputs |
| 11 (Logit Lens) | Medium (model + hooks) | 2 days | Need model |
| 12 (Circuits) | High (patching) | 3 days | Need model |

---

## Key Figures for the Paper (A* quality)

1. **Fig 1 — Galaxy Map**: UMAP of all tasks in shared space, showing each task's constellation is a distinct "galaxy." (Notebook 7)
2. **Fig 2 — Single Direction Works (Baseline)**: Before/after ablation refusal rates, matching Arditi numbers. (Notebook 8)
3. **Fig 3 — Task-Direction Divergence Heatmap**: Cosine similarity matrix of task-specific directions at peak-divergence layer. (Notebook 9 Q1)
4. **Fig 4 — Uneven Suppression Barplot**: Per-task refusal suppression from global ablation. (Notebook 9 Q2)
5. **Fig 5 — Transfer Matrix**: Cross-task transfer heatmap showing diagonal dominance. (Notebook 9 Q3)
6. **Fig 6 — Regime Diagram**: α × cosine_sim(v_global, v_task) → failure mode. (Notebook 10)
7. **Fig 7 — Logit Lens + α Sweep**: Token prediction evolution under increasing steering pressure. (Notebook 11)
8. **Fig 8 — Attribution Maps**: Per-task attention head refusal attribution heatmaps. (Notebook 12)

---

## Notebook Number Assignment

- **7** (extend in-place) — Constellation patterns per task + cross-task geometry
- **8** — Arditi et al. Replication
- **9** — Universality Test (Q1 + Q2 + Q3)
- **10** — Failure Mode Taxonomy
- **11** — Vocabulary Projection (Logit Lens)
- **12** — Task-Conditioned Attention Circuits
