# Data Packing and Batching in Miles

## The Problem: Variable-Length Sequences

Training on language models has a challenge: **sequences have different lengths!**

```
Sample 1: "What is 2+2?" → "4"                    (10 tokens total)
Sample 2: "Explain quantum physics" → "Quantum..." (50 tokens total)
Sample 3: "Hi" → "Hello!"                          (5 tokens total)
```

**Naive batching** would pad everything to max length:
```
Batch (padded to 50):
  Sample 1: [tokens...] + [PAD PAD PAD PAD ...] (40 wasted tokens!)
  Sample 2: [tokens...]                          (no padding)
  Sample 3: [tokens...] + [PAD PAD PAD ...]     (45 wasted tokens!)
```

**Problem:** 85 wasted tokens out of 150 total = **57% waste!**

---

## The Solution: Sequence Packing

Instead of padding, **concatenate sequences** into dense packs:

```
Pack 1: [Sample1][Sample3][Sample2]
        [10 tokens][5 tokens][50 tokens] = 65 tokens, 0% waste!
```

**How do we know where each sequence starts/ends?**
→ Use **cumulative sequence lengths** (`cu_seqlens`)

---

## Visual Example: Packing 3 Samples

### Input Samples
```python
Sample 1: tokens = [101, 2054, 2003, 1017, 1009, 1017, 102]  # "What is 2+2?"
          length = 7

Sample 2: tokens = [101, 7592, 102]                           # "Hi"
          length = 3

Sample 3: tokens = [101, 4863, 102, 2009, 2003, 1012, 102]   # "Hello! It is..."
          length = 7
```

### Packing Process

**Step 1: Concatenate all tokens**
```python
flat_tokens = [101, 2054, 2003, 1017, 1009, 1017, 102,  # Sample 1 (7 tokens)
               101, 7592, 102,                            # Sample 2 (3 tokens)
               101, 4863, 102, 2009, 2003, 1012, 102]    # Sample 3 (7 tokens)
# Total: 17 tokens (no padding!)
```

**Step 2: Build cumulative sequence lengths**
```python
cu_seqlens = [0, 7, 10, 17]
#             ↑  ↑   ↑   ↑
#             │  │   │   └─ End of Sample 3
#             │  │   └───── End of Sample 2 (start of Sample 3)
#             │  └───────── End of Sample 1 (start of Sample 2)
#             └──────────── Start of Sample 1
```

**Interpretation:**
- Sample 1: `flat_tokens[0:7]`   (cu_seqlens[0] to cu_seqlens[1])
- Sample 2: `flat_tokens[7:10]`  (cu_seqlens[1] to cu_seqlens[2])
- Sample 3: `flat_tokens[10:17]` (cu_seqlens[2] to cu_seqlens[3])

**Step 3: Pack all metadata**
```python
packed_batch = {
    "tokens": [101, 2054, ..., 102],        # Flat concatenated tokens
    "cu_seqlens": [0, 7, 10, 17],           # Sequence boundaries
    "position_ids": [0,1,2,3,4,5,6, 0,1,2, 0,1,2,3,4,5,6],  # Reset for each seq
    "loss_masks": [0,0,0,0,0,0,1, 0,0,1, 0,0,0,0,0,1,1],     # Which tokens train
    "rewards": [1.0, 0.5, 1.0],             # Per-sequence rewards
    "advantages": [...],                     # Flat advantages (17 values)
    "returns": [...],                        # Flat returns (17 values)
}
```

---

## How Flash Attention Uses `cu_seqlens`

Flash Attention is designed for variable-length sequences:

```python
# Instead of:
attention(tokens, mask)  # Needs padding + mask

# Flash Attention uses:
flash_attn_varlen_func(
    q, k, v,
    cu_seqlens_q=cu_seqlens,  # [0, 7, 10, 17]
    cu_seqlens_k=cu_seqlens,
    max_seqlen_q=7,           # Longest sequence
    max_seqlen_k=7,
)
```

**Flash Attention knows:**
- Tokens 0-6 attend to each other (Sample 1)
- Tokens 7-9 attend to each other (Sample 2)
- Tokens 10-16 attend to each other (Sample 3)
- **No cross-sample attention!** (prevented by cu_seqlens)

---

## Code Walkthrough: `pack_sequences()` (data_packing.py)

### Part 1: Determine Number of Packs (lines 49-55)

```python
if num_packs:
    k_partitions = num_packs
elif max_tokens_per_gpu:
    total_tokens = sum(seq_lengths)
    k_partitions = max(1, math.ceil(total_tokens / max_tokens_per_gpu))
else:
    k_partitions = 1
```

**Example:**
```python
seq_lengths = [100, 200, 150, 50, 300]  # 5 samples, 800 total tokens
max_tokens_per_gpu = 300

k_partitions = ceil(800 / 300) = 3 packs
```

**Why multiple packs?**
- Each pack fits in GPU memory (≤ max_tokens_per_gpu)
- Process multiple packs sequentially or distribute across GPUs

### Part 2: Balance Partitions (lines 58-60)

```python
partitions = get_seqlen_balanced_partitions(
    seq_lengths, k_partitions=k_partitions, equal_size=False
)
```

**Goal:** Distribute sequences evenly across packs

**Example:**
```python
seq_lengths = [100, 200, 150, 50, 300]
k_partitions = 3

# Bad partitioning:
# Pack 1: [100, 200] = 300 tokens
# Pack 2: [150, 50] = 200 tokens
# Pack 3: [300] = 300 tokens
# Imbalanced! Pack 2 underutilized.

# Good partitioning (balanced):
# Pack 1: [200, 50] = 250 tokens
# Pack 2: [100, 150] = 250 tokens
# Pack 3: [300] = 300 tokens
# Much better balance!

partitions = [[1, 3], [0, 2], [4]]  # Indices into seq_lengths
```

### Part 3: Pack Each Partition (lines 64-86)

```python
for indices in partitions:
    cu_seqlens = [0]
    flat_tokens = []
    flat_masks = []
    flat_positionids = []
    flat_advantages = []
    flat_returns = []

    for i in indices:
        seq_tokens = tokens[i]
        seq_mask = loss_masks[i]
        seq_positionids = list(range(len(seq_tokens)))

        flat_tokens.extend(seq_tokens)           # Concatenate tokens
        flat_positionids.extend(seq_positionids) # Reset position IDs
        flat_masks.extend(seq_mask)
        flat_advantages.extend(advantages[i])
        flat_returns.extend(returns[i])
        cu_seqlens.append(cu_seqlens[-1] + len(seq_tokens))  # Track boundaries
```

**Trace through example:**
```python
# Pack 1: indices = [1, 3] (samples with lengths [200, 50])

# Iteration 1 (i=1, length=200):
flat_tokens = tokens[1]           # 200 tokens
flat_positionids = [0,1,2,...,199]
cu_seqlens = [0, 200]

# Iteration 2 (i=3, length=50):
flat_tokens.extend(tokens[3])     # Now 250 tokens total
flat_positionids.extend([0,1,...,49])  # Reset to 0!
cu_seqlens = [0, 200, 250]

# Result:
cu_seqlens = [0, 200, 250]
flat_tokens has 250 tokens
position_ids = [0,1,...,199, 0,1,...,49]  # Resets for each sequence!
```

**Key insight:** Position IDs **reset for each sequence** so each sample thinks it starts at position 0.

### Part 4: Create Packed Batch (lines 88-100)

```python
packed_batch = {
    "tokens": torch.tensor(flat_tokens, dtype=torch.long),
    "loss_masks": torch.tensor(flat_masks, dtype=torch.int),
    "position_ids": torch.tensor(flat_positionids, dtype=torch.int),
    "cu_seqlens": torch.tensor(cu_seqlens, dtype=torch.int32),
    "rewards": torch.tensor([rewards[i] for i in indices], dtype=torch.float32),
    "response_lengths": [response_lengths[i] for i in indices],
    "advantages": torch.tensor(flat_advantages, dtype=torch.float32),
    "returns": torch.tensor(flat_returns, dtype=torch.float32),
    "rollout_log_probs": torch.tensor(flat_rollout_log_probs, dtype=torch.float32),
}
```

**Result:** One dense packed batch ready for training!

---

## Benefits of Packing

### 1. **Memory Efficiency**
```
Padded batching:  50% wasted on padding
Packed batching:  0% waste!
```

### 2. **Compute Efficiency**
- No computation on padding tokens
- Flash Attention optimized for packed sequences
- 2-3x throughput improvement!

### 3. **Load Balancing**
- Balanced partitioning ensures even GPU utilization
- No GPU sits idle while another processes huge batch

---

## How Training Uses Packed Batches

```python
# 1. Pack sequences
packed_batches = pack_sequences(
    tokens=tokens,
    loss_masks=loss_masks,
    advantages=advantages,
    returns=returns,
    max_tokens_per_gpu=4096
)

# 2. For each pack, forward pass
for batch in packed_batches:
    # Flash Attention uses cu_seqlens internally
    logits = model(
        input_ids=batch["tokens"],
        position_ids=batch["position_ids"],
        cu_seqlens=batch["cu_seqlens"],  # Tells model where sequences are
    )

    # 3. Compute loss (respects cu_seqlens boundaries)
    loss = compute_loss(
        logits,
        batch["advantages"],
        batch["loss_masks"],
    )

    # 4. Backprop
    loss.backward()
```

---

## Summary: Key Concepts

1. **Problem:** Variable-length sequences waste memory with padding

2. **Solution:** Concatenate sequences into dense packs

3. **cu_seqlens:** Cumulative sequence lengths track boundaries
   - `[0, 7, 10, 17]` means sequences at [0:7], [7:10], [10:17]

4. **Position IDs:** Reset for each sequence
   - `[0,1,2,3,4,5,6, 0,1,2, 0,1,2,3,4,5,6]`

5. **Balanced partitioning:** Distribute load evenly across packs

6. **Flash Attention:** Natively supports packed format via cu_seqlens

7. **Benefits:**
   - 0% memory waste
   - 2-3x compute efficiency
   - Better GPU utilization

---

## Visual Summary

```
INPUT:
  Sample 1: [t1, t2, t3]        (3 tokens)
  Sample 2: [t4, t5]            (2 tokens)
  Sample 3: [t6, t7, t8, t9]    (4 tokens)

PACKING:
  flat_tokens = [t1, t2, t3, t4, t5, t6, t7, t8, t9]  (9 tokens, no padding!)
  cu_seqlens = [0, 3, 5, 9]
                ↑  ↑  ↑  ↑
                │  │  │  └─ End of Sample 3
                │  │  └──── End of Sample 2
                │  └─────── End of Sample 1
                └────────── Start

ATTENTION:
  Sample 1 attends to: [t1, t2, t3]         (indices 0:3)
  Sample 2 attends to: [t4, t5]             (indices 3:5)
  Sample 3 attends to: [t6, t7, t8, t9]     (indices 5:9)
  NO cross-sample attention!

EFFICIENCY:
  Padded: 9 real + 3 pad = 12 tokens (25% waste)
  Packed: 9 real + 0 pad = 9 tokens  (0% waste!) ✓
```
