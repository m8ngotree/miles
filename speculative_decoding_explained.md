# Speculative Decoding: 2-3x Speedup for Inference

Speculative decoding is a clever optimization that accelerates autoregressive generation by **2-3x** without changing the output distribution!

---

## The Problem: Sequential Token Generation is Slow

### Standard Autoregressive Generation

```python
# Generate 100 tokens (SLOW!)
prompt = "Solve: What is 2+2?"
tokens = tokenize(prompt)

for i in range(100):
    # Full forward pass for EACH token
    logits = target_model(tokens)  # 70B model - very expensive!

    # Sample next token
    next_token = sample(logits[-1])

    # Append and repeat
    tokens.append(next_token)
```

**Problem:**
- Each token requires full 70B model forward pass
- Cannot parallelize (each token depends on previous)
- For 100 tokens: 100 sequential forward passes
- **Very slow!**

**Wallclock time example:**
```
70B model forward pass: 100ms
100 tokens: 100 × 100ms = 10 seconds
```

---

## The Insight: Most Tokens are Predictable

When generating coherent text, many tokens are **highly predictable**:

```
Prompt: "The capital of France is"
Next tokens: " Paris" (very predictable!)

Prompt: "2 + 2 ="
Next tokens: " 4" (very predictable!)
```

**Key observation:** The large model is "overkill" for easy predictions!

**Idea:** Use a small, fast model to guess multiple tokens ahead, then verify in batch!

---

## Speculative Decoding: The Algorithm

### Architecture

```
┌─────────────────────┐
│   Draft Model       │  Small, fast (e.g., 7B params)
│   (cheap)           │  Generates K candidate tokens
└─────────────────────┘
          ↓
    [t₁, t₂, t₃, t₄]  ← Draft tokens
          ↓
┌─────────────────────┐
│   Target Model      │  Large, accurate (e.g., 70B params)
│   (expensive)       │  Verifies all K tokens in parallel
└─────────────────────┘
          ↓
    [✓, ✓, ✓, ✗]      ← Accept/reject
          ↓
  Final: [t₁, t₂, t₃]  ← Keep accepted tokens
```

### Step-by-Step Process

**Input:**
- Prompt: "The capital of France is"
- Num draft tokens: K = 4

**Step 1: Draft Phase (cheap)**
```python
# Small draft model generates K tokens quickly
draft_tokens = draft_model.generate(prompt, num_tokens=4)
# Output: [" Paris", ",", " a", " city"]
```

**Step 2: Verification Phase (expensive, but parallel!)**
```python
# Target model verifies ALL K tokens in ONE forward pass
target_logits = target_model([prompt + draft_tokens[:0],  # P(" Paris" | prompt)
                               prompt + draft_tokens[:1],  # P("," | prompt + " Paris")
                               prompt + draft_tokens[:2],  # P(" a" | prompt + " Paris,")
                               prompt + draft_tokens[:3]]) # P(" city" | prompt + " Paris, a")
```

**Step 3: Accept/Reject Decision**
```python
for i in range(K):
    # Sample from target distribution
    target_token = sample(target_logits[i])

    if target_token == draft_tokens[i]:
        accept()  # ✓ Draft token matches target
    else:
        reject()  # ✗ Draft token rejected
        break     # Stop and use target's token instead
```

**Result:**
```
Draft:  [" Paris", ",", " a", " city"]
Verify: [✓,       ✓,    ✓,    ✗      ]
Accept: [" Paris", ",", " a"]          ← 3 tokens in one iteration!
```

---

## Why This Works: Mathematically Correct

**Key property:** The final output distribution is **identical** to standard autoregressive sampling!

### Proof Sketch

For each token position, we:
1. **Sample from target distribution** (using rejection sampling)
2. **Accept draft token if it matches** (common case - fast path)
3. **Reject and resample if mismatch** (rare case - fall back to target)

This implements **exact sampling** from the target model's distribution, but with amortized speedup when draft and target agree!

**Acceptance probability:**
```
P(accept draft[i]) = min(1, P_target(draft[i]) / P_draft(draft[i]))
```

When draft model is well-aligned with target, acceptance rate is high!

---

## Speedup Analysis

### Without Speculative Decoding
```
Generate N tokens:
  Time = N × T_target

Example (N=100, T_target=100ms):
  Time = 100 × 100ms = 10,000ms = 10 seconds
```

### With Speculative Decoding
```
Generate N tokens with K draft tokens per iteration:
  Num iterations ≈ N / (K × acceptance_rate)
  Time per iteration = T_draft + T_target
  Total time ≈ (N / (K × α)) × (T_draft + T_target)

Example (K=4, α=0.75, T_draft=10ms, T_target=100ms):
  Iterations = 100 / (4 × 0.75) = 33.3
  Time = 33.3 × (10 + 100) = 3,663ms ≈ 3.7 seconds

Speedup = 10s / 3.7s = 2.7x ✓
```

**Key factors:**
- **K (num draft tokens):** More draft tokens = higher speedup (but lower acceptance)
- **α (acceptance rate):** Higher alignment = more speedup
- **T_draft / T_target:** Smaller draft model = more speedup

---

## In Miles: Implementation

### Configuration (from docs/en/advanced/speculative-decoding.md)

```bash
# Enable speculative decoding with EAGLE algorithm
--sglang-speculative-algorithm EAGLE
--sglang-speculative-num-steps 3
--sglang-speculative-eagle-topk 1
--sglang-speculative-num-draft-tokens 4
--sglang-enable-draft-weights-cpu-backup

# Optional: Use external draft model
--sglang-speculative-draft-model-path /path/to/draft/model
```

**Parameters:**
- `EAGLE`: Speculative decoding variant (uses MTP layers)
- `num-steps`: How many verification rounds
- `num-draft-tokens`: K (how many tokens draft generates)
- `draft-model-path`: Separate draft model (or use MTP layers)

### Tracking Metrics (from miles/utils/types.py:41-76)

```python
class SpecInfo:
    spec_accept_token_num: int = 0    # How many draft tokens accepted
    spec_draft_token_num: int = 0     # How many draft tokens generated
    spec_verify_ct: int = 0           # How many verification steps
    spec_accept_rate: float = 0.0     # Acceptance rate
    spec_accept_length: float = 0.0   # Avg tokens per verification
```

**Computed metrics:**
```python
# Acceptance rate: What fraction of draft tokens were accepted?
accept_rate = spec_accept_token_num / spec_draft_token_num

# Average accepted length: How many tokens per verification step?
accept_length = response_length / spec_verify_ct
```

### During Generation (from miles/rollout/sglang_rollout.py:177-182)

```python
if args.sglang_speculative_algorithm:
    sample.spec_info.add(
        meta_info=output["meta_info"],
        response_length=sample.response_length,
    )
```

SGLang returns `meta_info` with speculative decoding statistics, which Miles tracks per sample.

---

## MTP Layers: Medusa-style Speculation

**MTP (Multi-Token Prediction) layers** are lightweight prediction heads attached to intermediate layers:

```
┌──────────────────────────────────────┐
│       Target Model (70B)              │
│                                       │
│  Layer 1  ──→ [MTP head] ──→ draft_1 │  ← Predict from layer 1
│  Layer 2  ──→ [MTP head] ──→ draft_2 │  ← Predict from layer 2
│    ...                                │
│  Layer N  ──→ final logits            │  ← Target prediction
└──────────────────────────────────────┘
```

**Benefits:**
- No separate draft model needed!
- MTP heads are tiny (~1% of model size)
- Can train MTP during RL to stay aligned

**In Miles:**
```bash
--mtp-num-layers 1           # Use 1 MTP layer
--enable-mtp-training        # Train MTP during RL
--mtp-loss-scaling-factor 0.2  # Weight MTP loss
```

During training, MTP heads learn to predict future tokens from intermediate representations, staying aligned with the target model as it improves via RL!

---

## Example: Concrete Walkthrough

### Setup
```
Prompt: "Solve: 2 + 2 = ?"
Draft model: 7B (T_draft = 10ms)
Target model: 70B (T_target = 100ms)
Num draft tokens: K = 4
```

### Iteration 1

**Draft phase:**
```python
draft_model("Solve: 2 + 2 = ?")
→ [" The", " answer", " is", " 4"]  # 4 tokens in 10ms
```

**Verify phase:**
```python
# Verify all 4 in parallel (100ms for one forward pass!)
target_model(["Solve: 2 + 2 = ? The",
              "Solve: 2 + 2 = ? The answer",
              "Solve: 2 + 2 = ? The answer is",
              "Solve: 2 + 2 = ? The answer is 4"])

→ Sample from each position:
  Position 0: " The" (matches draft ✓)
  Position 1: " answer" (matches draft ✓)
  Position 2: " is" (matches draft ✓)
  Position 3: " 4" (matches draft ✓)
```

**Result:** Accepted all 4 tokens!
**Progress:** Generated 4 tokens in 110ms (vs. 400ms normally!)
**Speedup:** 3.6x for this iteration

### Iteration 2

**Draft phase:**
```python
draft_model("Solve: 2 + 2 = ? The answer is 4")
→ [".", " Hope", " this", " helps"]  # 4 more tokens
```

**Verify phase:**
```python
target_model([..., "4.", "4. Hope", "4. Hope this", ...])

→ Sample:
  Position 0: "." (matches ✓)
  Position 1: " Let" (DOESN'T match draft ✗)
  → Reject! Stop here and use " Let"
```

**Result:** Accepted 1 token, rejected rest
**Progress:** Generated 2 tokens (1 accepted + 1 from target) in 110ms
**Speedup:** Still faster than 200ms for 2 tokens normally

**Average over many iterations:** 2-3x speedup!

---

## Challenges & Solutions

### Challenge 1: Draft-Target Drift During RL

**Problem:**
- Training updates target model
- Draft model stays frozen
- Distributions drift apart → lower acceptance rate → slower!

**Solution in Miles:**
```bash
--enable-mtp-training  # Train MTP heads during RL
```

Online training keeps draft aligned with target as policy improves!

### Challenge 2: Memory Overhead

**Problem:**
- Need to store both draft and target models
- MTP layers add extra memory

**Solution:**
```bash
--sglang-enable-draft-weights-cpu-backup  # Offload draft to CPU
```

Keep draft model on CPU, load for generation, offload after.

### Challenge 3: Batch Scheduling

**Problem:**
- Different samples have different acceptance rates
- Some finish quickly, others slowly
- Hard to batch efficiently

**Solution:**
- SGLang's continuous batching handles this
- Dynamic scheduling based on completion

---

## When to Use Speculative Decoding

### ✅ Good use cases:
- **Predictable outputs:** Math, coding, structured generation
- **Long sequences:** More tokens = more amortized speedup
- **Inference-bound:** When generation is the bottleneck

### ❌ Not ideal for:
- **Highly creative generation:** Low acceptance rate
- **Very short sequences:** Overhead dominates
- **Training-bound:** If training is slow anyway

---

## Summary: Key Takeaways

### **How it works:**
1. **Draft model** generates K candidate tokens (fast)
2. **Target model** verifies all K in parallel (one forward pass)
3. **Accept/reject** based on token matching
4. **Repeat** until sequence complete

### **Why it's fast:**
- Amortizes expensive target model calls
- Parallelizes verification across K tokens
- 2-3x speedup in practice!

### **Why it's correct:**
- Implements exact sampling from target distribution
- Uses rejection sampling for unmatched tokens
- Output distribution identical to standard generation

### **In Miles:**
- Configured via `--sglang-speculative-algorithm EAGLE`
- Tracks acceptance rate in `Sample.spec_info`
- Can train MTP layers online during RL
- Integrated with SGLang for continuous batching

### **Key metrics:**
```python
acceptance_rate = accepted_tokens / draft_tokens  # Want high (>70%)
speedup ≈ K × acceptance_rate                     # Theoretical upper bound
actual_speedup ≈ 2-3x                             # In practice
```

**Bottom line:** Speculative decoding makes rollouts 2-3x faster without sacrificing quality - a critical optimization for production RL training!
