# Complete Training Loop in Miles

This document explains the **complete training flow** from receiving rollout data to updating model weights.

Location: `miles/backends/fsdp_utils/actor.py`

---

## Overview: The Training Pipeline

```
1. Receive rollout data (samples from generation)
   ↓
2. Compute advantages (GAE)
   ↓
3. Pack sequences into batches
   ↓
4. Forward pass (actor model)
   ↓
5. Compute PPO loss
   ↓
6. Backward pass (gradients)
   ↓
7. Update weights (optimizer step)
   ↓
8. Sync to rollout workers
```

---

## Step-by-Step Breakdown

### **1. Entry Point: `train()` method (line 468)**

```python
def train(self, rollout_id: int, rollout_data_ref: Box) -> None:
    rollout_data = process_rollout_data(
        self.args, rollout_data_ref, self.dp_rank, self.dp_size
    )
    self._train_core(rollout_id=rollout_id, rollout_data=rollout_data)
```

**What happens:**
- Receives `rollout_data_ref`: Ray object reference containing samples
- `process_rollout_data()`: Partitions data across Data Parallel (DP) ranks
  - DP rank 0 gets samples [0, 4, 8, 12, ...]
  - DP rank 1 gets samples [1, 5, 9, 13, ...]
  - Each GPU trains on different samples (data parallelism)
- Calls `_train_core()` to do actual training

**Input `rollout_data` contains:**
```python
{
    "tokens": [[101, 2054, ...], [101, 7592, ...], ...],  # List of token sequences
    "loss_masks": [[0, 0, 1, 1], [0, 1, 1], ...],         # Which tokens to train on
    "rewards": [1.0, 0.5, 1.0, ...],                      # Per-sample rewards
    "response_lengths": [10, 5, 15, ...],                 # Length of each response
    "rollout_log_probs": [[−0.5, −0.3, ...], ...],        # Log probs from generation
}
```

---

### **2. Compute Advantages: `_train_core()` (line 531)**

```python
def _train_core(self, rollout_id: int, rollout_data) -> None:
    # For GRPO/GSPO: Simple advantages (just rewards)
    if self.args.advantage_estimator in ["grpo", "gspo"]:
        rollout_data["advantages"] = rollout_data["returns"] = [
            torch.tensor([rollout_data["rewards"][i]] * rollout_data["response_lengths"][i])
            for i in range(len(rollout_data["rewards"]))
        ]
```

**Two modes:**

**Mode A: GAE (Generalized Advantage Estimation)**
- Would call `get_advantages_and_returns()` from `ppo_utils.py`
- Computes `A[t] = δ[t] + γλ*A[t+1]` (what we studied!)
- Not shown in this code snippet (alternative implementation)

**Mode B: GRPO/GSPO (shown above)**
- Simple: broadcast reward to all tokens
- `advantages[i] = [reward, reward, reward, ...]` (same value repeated)
- Example: reward=1.0, response_length=5 → advantages=[1.0, 1.0, 1.0, 1.0, 1.0]

---

### **3. Pack Sequences: `_packed_data()` (line 540)**

```python
packed_batches, grad_accum = self._packed_data(rollout_data)
```

**What it does:**
- Calls `pack_sequences()` from `data_packing.py`
- Groups samples into dense packs (no padding!)
- Creates `cu_seqlens` to track sequence boundaries
- Returns list of packed batches ready for GPU

**Example:**
```python
# Input: 3 samples
Sample 1: 100 tokens
Sample 2: 50 tokens
Sample 3: 150 tokens

# Output: 2 packed batches (max_tokens_per_gpu=200)
packed_batches = [
    {  # Pack 1: Sample 1 + Sample 2
        "tokens": [concat(sample1, sample2)],  # 150 tokens
        "cu_seqlens": [0, 100, 150],
        "advantages": [advantages1 + advantages2],
        ...
    },
    {  # Pack 2: Sample 3
        "tokens": [sample3],  # 150 tokens
        "cu_seqlens": [0, 150],
        "advantages": [advantages3],
        ...
    }
]
```

---

### **4. Compute Reference Log Probs (Optional, line 546-548)**

```python
if self.ref_model is not None:
    self._compute_log_prob("ref", packed_batches, store_prefix="ref_")
```

**Purpose:** For KL penalty against reference model

- Load frozen reference model (original SFT model)
- Forward pass to get `ref_log_probs`
- Used to compute KL divergence: `KL = log_probs - ref_log_probs`
- Prevents policy from deviating too far from SFT model

---

### **5. Main Training Loop: `actor_train` (line 552-563)**

```python
self.optimizer.zero_grad(set_to_none=True)
for mbs_id, packed_batch in enumerate(packed_batches):
    self._train_step(
        packed_batch=packed_batch,
        reported_accum=reported_accum,
        mbs_id=mbs_id,
        grad_accum=grad_accum,
    )
```

**Key points:**
- Zero gradients once at start
- Loop through each packed batch
- Each batch: forward → loss → backward
- Gradients **accumulate** across batches
- Optimizer step happens periodically (controlled by `grad_accum`)

---

### **6. Training Step: `_train_step()` - The Heart of Training**

This is where all the math happens! Let me break it down section by section.

#### **6a. Forward Pass (lines 584-599)**

```python
# Get model inputs (tokens, position_ids, cu_seqlens)
model_args = self._get_model_inputs_args(packed_batch)

# Forward pass through actor model
logits = self.model(**model_args).logits.squeeze(0).float()

# Compute log probabilities
log_probs, entropy_result = get_logprob_and_entropy_with_cp(
    logits=logits,
    target_tokens=packed_batch["tokens"],
    cp_rank=self.cp_rank,
    cp_size=self.cp_size,
    cp_group=self.cp_group,
    ...
)
```

**What happens:**
1. **Model forward pass:** Input tokens → logits [vocab_size]
2. **Compute log probabilities:** For each generated token, what's log P(token)?
3. **Handle Context Parallel:** If CP enabled, gather logits across ranks
4. **Compute entropy:** For entropy bonus

**Example:**
```python
# Token: "4"
logits = [−10.2, −8.5, ..., −0.5, ...]  # Raw scores for all vocab
                           ↑ "4" has highest logit
log_prob = −0.5  # log P(token="4" | context)
```

#### **6b. Unpack and Prepare (lines 601-621)**

```python
unpacked_batches = unpack_sequences(packed_batch)

# Extract old log probs (from rollout)
old_log_probs = torch.cat([batch["rollout_log_probs"] for batch in unpacked_batches], dim=0)

# Extract new log probs (just computed)
log_probs = torch.cat([batch["cur_log_probs"] for batch in unpacked_batches], dim=0)

# Extract advantages
advantages = torch.cat([batch["advantages"] for batch in unpacked_batches], dim=0)

# Compute PPO KL
ppo_kl = old_log_probs - log_probs
```

**Key variables now:**
```python
old_log_probs = [−2.0, −1.8, −1.5, ...]  # From generation (old policy)
log_probs     = [−1.5, −1.5, −1.2, ...]  # From forward pass (new policy)
ppo_kl        = [−0.5, −0.3, −0.3, ...]  # Difference
advantages    = [ 0.37, 0.28, 0.19, ...]  # From GAE
```

#### **6c. Compute PPO Policy Loss (line 641)**

```python
pg_loss, pg_clipfrac = compute_policy_loss(
    ppo_kl, advantages, self.args.eps_clip, self.args.eps_clip_high
)
```

**This calls `ppo_utils.py:compute_policy_loss()`:**

```python
def compute_policy_loss(ppo_kl, advantages, eps_clip, eps_clip_high):
    ratio = (-ppo_kl).exp()  # π_new / π_old
    pg_losses1 = -ratio * advantages  # Unclipped
    pg_losses2 = -ratio.clamp(1 - eps_clip, 1 + eps_clip_high) * advantages  # Clipped
    clip_pg_losses1 = torch.maximum(pg_losses1, pg_losses2)  # Take max
    return clip_pg_losses1, clipfrac
```

**Example with numbers:**
```python
ppo_kl = −0.5
ratio = exp(0.5) = 1.649
advantage = 0.37
eps_clip = 0.2

unclipped = −1.649 * 0.37 = −0.610
clipped = −1.2 * 0.37 = −0.444  # ratio clamped to 1.2
final = max(−0.610, −0.444) = −0.444  # Take less negative (conservative)
```

#### **6d. Aggregate Loss Across Samples (line 674)**

```python
pg_loss = sum_of_sample_mean(pg_loss, response_lengths, loss_masks)
```

**What this does:**
- Averages loss within each sample (respecting loss_mask)
- Then averages across all samples

```python
# Example: 2 samples
Sample 1: loss_per_token = [−0.5, −0.4, −0.3], loss_mask = [1, 1, 1]
  → mean = (−0.5 + −0.4 + −0.3) / 3 = −0.4

Sample 2: loss_per_token = [−0.2, −0.1], loss_mask = [1, 1]
  → mean = (−0.2 + −0.1) / 2 = −0.15

Final: (−0.4 + −0.15) / 2 = −0.275
```

#### **6e. Entropy Bonus (lines 686-689)**

```python
entropy_loss = sum_of_sample_mean(entropy, response_lengths, loss_masks)
loss = pg_loss - self.args.entropy_coef * entropy_loss
```

**Entropy bonus encourages exploration:**
- High entropy = diverse outputs (good for exploration)
- Loss = policy_loss − α*entropy (subtract because we want to minimize loss, maximize entropy)
- Typical α = 0.01

#### **6f. KL Penalty (Optional, lines 691-704)**

```python
if self.args.use_kl_loss:
    ref_log_probs = torch.cat([batch["ref_log_probs"] for batch in unpacked_batches], dim=0)
    kl = compute_approx_kl(log_probs, ref_log_probs, ...)
    kl_loss = sum_of_sample_mean(kl, response_lengths, loss_masks)
    loss = loss + self.args.kl_loss_coef * kl_loss
```

**KL penalty prevents drift from reference model:**
```
KL(new || ref) = Σ new_log_prob − ref_log_prob
```

Penalizes policy for deviating from original SFT model.

**Final loss:**
```python
loss = pg_loss - α*entropy_loss + β*kl_loss
     = (PPO clipped loss) - (exploration bonus) + (stay close to ref penalty)
```

---

### **7. Backward Pass (line 730)**

```python
loss = loss * self.dp_size / self.args.global_batch_size
loss.backward()
```

**Gradient accumulation scaling:**
- `loss` is averaged over microbatch
- Scale by `dp_size / global_batch_size` for correct gradient magnitude
- `backward()` accumulates gradients in `param.grad`

**Example:**
```python
global_batch_size = 128
dp_size = 4 (4 GPUs)
microbatch_size = 8

Each GPU processes 128/4 = 32 samples total
Divided into 32/8 = 4 microbatches per GPU

loss = microbatch_loss * 4/128  # Scale down
loss.backward()  # Gradients accumulate across 4 microbatches
```

---

### **8. Optimizer Step (Periodic, lines 736-745)**

```python
if (mbs_id + 1) in grad_accum:
    # Clip gradients to prevent explosion
    grad_norm = torch.nn.utils.clip_grad_norm_(
        self.model.parameters(), self.args.clip_grad
    )

    # Update weights
    self.optimizer.step()

    # Update learning rate
    self.lr_scheduler.step()

    # Reset gradients
    self.optimizer.zero_grad(set_to_none=True)
```

**When does this run?**
- `grad_accum` is a list like `[4, 8, 12, 16]` (accumulate every 4 microbatches)
- After processing mbs_id=3, 7, 11, 15: do optimizer step
- Between steps: gradients accumulate

**Gradient clipping:**
```python
# Before clipping:
grad_norm = 15.2  (too large!)

# Clip to max_norm=1.0:
for param in model.parameters():
    param.grad *= 1.0 / 15.2  # Scale down all gradients
```

Prevents gradient explosion.

---

## Complete Flow Example

Let's trace one sample through the entire pipeline:

```python
# ═══════════════════════════════════════════════════════════════
# INPUT: Rollout data from generation
# ═══════════════════════════════════════════════════════════════
sample = {
    "tokens": [101, 2054, 2003, 1017, 1009, 1017, 102, 19],  # "What is 2+2? 4"
    "loss_mask": [0, 0, 0, 0, 0, 0, 0, 1],  # Only train on "4"
    "reward": 1.0,
    "response_length": 8,
    "rollout_log_probs": [−2.0, −1.8, −1.5, −1.2, −1.0, −0.8, −0.5, −0.2],
}

# ═══════════════════════════════════════════════════════════════
# STEP 1: Compute advantages (GRPO mode)
# ═══════════════════════════════════════════════════════════════
advantages = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]  # Broadcast reward

# ═══════════════════════════════════════════════════════════════
# STEP 2: Pack sequences
# ═══════════════════════════════════════════════════════════════
# (Assume single sample for simplicity)
packed_batch = {
    "tokens": tensor([101, 2054, ..., 19]),
    "cu_seqlens": [0, 8],
    "advantages": tensor([1.0, 1.0, ..., 1.0]),
    "rollout_log_probs": tensor([−2.0, −1.8, ..., −0.2]),
    "loss_mask": tensor([0, 0, 0, 0, 0, 0, 0, 1]),
}

# ═══════════════════════════════════════════════════════════════
# STEP 3: Forward pass
# ═══════════════════════════════════════════════════════════════
logits = model(tokens=packed_batch["tokens"])
# logits.shape = [8, 50000]  (8 tokens, vocab_size=50000)

log_probs = compute_log_probs(logits, tokens)
# log_probs = [−1.5, −1.5, −1.2, −1.0, −0.8, −0.6, −0.4, −0.1]

# ═══════════════════════════════════════════════════════════════
# STEP 4: Compute PPO loss
# ═══════════════════════════════════════════════════════════════
old_log_probs = packed_batch["rollout_log_probs"]
# [−2.0, −1.8, −1.5, −1.2, −1.0, −0.8, −0.5, −0.2]

ppo_kl = old_log_probs - log_probs
# [−0.5, −0.3, −0.3, −0.2, −0.2, −0.2, −0.1, −0.1]

ratio = exp(−ppo_kl)
# [1.649, 1.350, 1.350, 1.221, 1.221, 1.221, 1.105, 1.105]

# Clipping (eps=0.2):
clipped_ratio = clip(ratio, 0.8, 1.2)
# [1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 1.105, 1.105]

advantages = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]

pg_loss = −clipped_ratio * advantages
# [−1.2, −1.2, −1.2, −1.2, −1.2, −1.2, −1.105, −1.105]

# Apply loss mask (only token 7 trains):
masked_loss = pg_loss * loss_mask
# [0, 0, 0, 0, 0, 0, 0, −1.105]

final_loss = masked_loss.sum() / loss_mask.sum()
# −1.105 / 1 = −1.105

# Add entropy bonus (assume entropy=2.5):
loss = −1.105 − 0.01*2.5 = −1.130

# ═══════════════════════════════════════════════════════════════
# STEP 5: Backward pass
# ═══════════════════════════════════════════════════════════════
loss.backward()
# Gradients flow: loss → pg_loss → ratio → log_probs → logits → weights

# ═══════════════════════════════════════════════════════════════
# STEP 6: Optimizer step (if accumulated enough)
# ═══════════════════════════════════════════════════════════════
clip_grad_norm_(model.parameters(), max_norm=1.0)
optimizer.step()  # weights ← weights − lr * grad
optimizer.zero_grad()
```

---

## Summary: Key Takeaways

### **The 8-Step Pipeline:**
1. **Receive rollout data** (samples from generation)
2. **Compute advantages** (GAE or simple reward broadcast)
3. **Pack sequences** (efficient batching with cu_seqlens)
4. **Forward pass** (actor model → logits → log_probs)
5. **Compute PPO loss** (clipped policy gradient + entropy + KL)
6. **Backward pass** (compute gradients, accumulate)
7. **Optimizer step** (clip gradients, update weights)
8. **Sync to rollout** (new weights → SGLang engines)

### **Important Mechanisms:**

**Gradient Accumulation:**
- Process small microbatches
- Accumulate gradients across batches
- Update weights periodically
- Enables large effective batch sizes on limited memory

**Data Parallelism:**
- Each GPU gets different samples
- Gradients averaged across GPUs
- Scales to many GPUs efficiently

**Context Parallelism:**
- Each GPU gets chunk of sequence
- Requires gathering for GAE
- Flash Attention handles automatically

**Loss Components:**
```
Total Loss = PPO Loss - α*Entropy + β*KL
           = (clipped policy gradient) - (exploration) + (prevent drift)
```

### **Connection to Theory:**

All the math we studied shows up here:

| Concept | Where in Code |
|---------|---------------|
| TD Error | `delta = rewards[t] + gamma * nextvalues - values[t]` (ppo_utils.py:354) |
| GAE | `lastgaelam = delta + gamma * lambd * lastgaelam` (ppo_utils.py:355) |
| Importance Ratio | `ratio = (-ppo_kl).exp()` (actor.py / ppo_utils.py:132) |
| PPO Clipping | `ratio.clamp(1 - eps_clip, 1 + eps_clip_high)` (ppo_utils.py:134) |
| Policy Loss | `max(-ratio*A, -clip(ratio)*A)` (ppo_utils.py:135) |

You now understand how the **math becomes code**, and how the code becomes **trained models**!
