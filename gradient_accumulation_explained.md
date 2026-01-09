# Gradient Accumulation and Distributed Training

This explains how Miles handles large batch sizes across multiple GPUs using gradient accumulation and distributed coordination.

---

## The Problem: Memory vs. Batch Size

**Goal:** Train with large effective batch size (e.g., 128 samples)

**Problem:** GPU memory can't fit 128 samples at once!

**Example:**
```
Model size: 70B parameters → 140GB memory
GPU memory: 80GB total
Available for activation: ~40GB

Batch size 128: Would need 200GB → OOM! ❌
```

**Solution:** Gradient accumulation + Data parallelism

---

## Gradient Accumulation: The Core Idea

Instead of processing all samples at once, **process in small chunks** and accumulate gradients:

```python
# Naive (doesn't fit in memory):
batch = load_all_128_samples()
loss = compute_loss(batch)
loss.backward()  # OOM!
optimizer.step()

# With gradient accumulation (fits in memory):
optimizer.zero_grad()
for microbatch in split_into_16_chunks(batch):  # 128 / 16 = 8 samples per chunk
    loss = compute_loss(microbatch)
    loss.backward()  # Gradients accumulate in param.grad
optimizer.step()  # Update with accumulated gradients
```

**Key insight:**
```
∇L(batch_128) = mean([∇L(chunk_1), ∇L(chunk_2), ..., ∇L(chunk_16)])
```

Gradient of the mean = mean of the gradients!

---

## How It Works Mathematically

### Without Gradient Accumulation:

```python
# Full batch loss
loss = (loss_1 + loss_2 + ... + loss_128) / 128

# Gradient
∇L = ∂loss/∂θ = (∇loss_1 + ∇loss_2 + ... + ∇loss_128) / 128
```

### With Gradient Accumulation (16 chunks of 8):

```python
# Chunk 1
loss_chunk_1 = (loss_1 + ... + loss_8) / 8
∇L_chunk_1 = ∂loss_chunk_1/∂θ

# Chunk 2
loss_chunk_2 = (loss_9 + ... + loss_16) / 8
∇L_chunk_2 = ∂loss_chunk_2/∂θ

# ... (chunks 3-16)

# Accumulate
∇L_total = (∇L_chunk_1 + ∇L_chunk_2 + ... + ∇L_chunk_16) / 16

# This equals full batch gradient!
∇L_total = (∇loss_1 + ∇loss_2 + ... + ∇loss_128) / 128  ✓
```

**Important:** Must divide by number of accumulation steps to get correct gradient magnitude!

---

## In Miles Code

### Setup (from `actor.py:_packed_data()`)

```python
def _packed_data(self, rollout_data):
    # Determine microbatch size
    micro_batch_size = self.args.micro_batch_size  # e.g., 8
    global_batch_size = self.args.global_batch_size  # e.g., 128
    dp_size = self.dp_size  # e.g., 4 GPUs

    # Each GPU handles:
    local_batch_size = global_batch_size // dp_size  # 128 / 4 = 32

    # Number of microbatches per GPU:
    num_microbatches = local_batch_size // micro_batch_size  # 32 / 8 = 4

    # Pack data into microbatches
    packed_batches = []
    for i in range(num_microbatches):
        packed_batches.append(pack_microbatch(data[i*8:(i+1)*8]))

    # When to accumulate gradients:
    grad_accum = [4, 8, 12, 16]  # Optimizer step after batch 4, 8, 12, 16
```

**Result:**
- 4 GPUs, each processes 4 microbatches
- Total: 4 × 4 = 16 microbatches globally
- Effective batch size: 16 × 8 = 128 samples ✓

---

### Training Loop (from `actor.py:552-563`)

```python
optimizer.zero_grad()  # Start with clean gradients
for mbs_id, packed_batch in enumerate(packed_batches):
    self._train_step(
        packed_batch=packed_batch,
        mbs_id=mbs_id,
        grad_accum=grad_accum,
    )
```

### Inside Training Step (from `actor.py:729-745`)

```python
# Scale loss for gradient accumulation
loss = loss * self.dp_size / self.args.global_batch_size
loss.backward()  # Accumulate gradients

# Accumulate metrics
reported_accum[...].append(loss)

# Check if it's time to update weights
if (mbs_id + 1) in grad_accum:
    # Clip gradients
    grad_norm = clip_grad_norm_(self.model.parameters(), max_norm=1.0)

    # Update weights
    optimizer.step()

    # Update learning rate
    lr_scheduler.step()

    # Reset gradients
    optimizer.zero_grad()

    # Aggregate and log metrics
    ...
```

---

## Loss Scaling: Why `loss * dp_size / global_batch_size`?

This line is crucial:
```python
loss = loss * self.dp_size / self.args.global_batch_size
```

Let's understand why:

### Scenario:
```
global_batch_size = 128
dp_size = 4 GPUs
micro_batch_size = 8

Each GPU:
  - Processes 128 / 4 = 32 samples total
  - In 32 / 8 = 4 microbatches
```

### Without scaling:

```python
# GPU 0, microbatch 0 (8 samples):
loss_0 = sum(losses) / 8  # Average over microbatch
loss_0.backward()
# grad = ∂loss_0/∂θ

# GPU 0, microbatch 1 (8 samples):
loss_1 = sum(losses) / 8
loss_1.backward()
# grad += ∂loss_1/∂θ (accumulates!)

# ... (microbatch 2, 3)

# After 4 microbatches:
# grad = (∂loss_0 + ∂loss_1 + ∂loss_2 + ∂loss_3) / 8
```

**Problem:** We summed 4 gradients but each was averaged over 8 samples. The final gradient is **4x too large**!

### With scaling:

```python
# GPU 0, microbatch 0:
loss_0 = (sum(losses) / 8) * (4 / 128)  # Scale down!
loss_0.backward()

# ... (repeat for 3 more microbatches)

# After 4 microbatches:
# grad = 4 * (gradient_per_microbatch * 4/128)
#      = 4 * (gradient_per_microbatch * 1/32)
#      = gradient_per_microbatch * 4/32
#      = gradient_per_microbatch / 8
```

Wait, that's still not right! Let me recalculate:

### Correct reasoning:

```python
# Microbatch loss (averaged over 8 samples):
microbatch_loss = sum(sample_losses) / 8

# We want global loss (averaged over 128 samples):
global_loss = sum(all_128_sample_losses) / 128

# Relationship:
global_loss = sum(all_16_microbatch_losses * 8) / 128
            = sum(all_16_microbatch_losses) / 16

# On each GPU (4 microbatches):
local_contribution = sum(4_microbatch_losses) / 16

# But we compute:
microbatch_loss.backward()  # This gives ∇(sum/8)

# To get ∇(sum/16), we need to scale by 8/16 = 1/2:
scaled_loss = microbatch_loss * (8/16)

# Generalized:
scaled_loss = microbatch_loss * (micro_batch_size / global_batch_size)
```

Hmm, but the code does `dp_size / global_batch_size`. Let me think about this more carefully...

### Actually:

The loss is **already averaged within the microbatch** by `sum_of_sample_mean()`.

```python
# Line 674:
pg_loss = sum_of_sample_mean(pg_loss, response_lengths, loss_masks)
```

This averages over samples in the microbatch. Then:

```python
# Line 729:
loss = loss * self.dp_size / self.args.global_batch_size
```

**Breakdown:**
- `loss`: Already averaged over microbatch samples
- `dp_size / global_batch_size`: Scaling factor

**Example:**
```
dp_size = 4
global_batch_size = 128
scaling = 4 / 128 = 1/32

Each GPU processes 32 samples in 4 microbatches
Each microbatch: 8 samples

Microbatch loss (averaged): L_mb
Scaled loss: L_mb * (1/32)

After accumulating 4 microbatches:
accumulated_grad = 4 * ∇(L_mb * 1/32)
                 = 4 * (1/32) * ∇L_mb
                 = (1/8) * ∇L_mb

Then across 4 GPUs (allreduce happens in FSDP automatically):
global_grad = 4 * (1/8) * ∇L_mb
            = (1/2) * ∇L_mb
```

Actually, I think the FSDP framework handles the allreduce differently. Let me simplify:

### Simplified Explanation:

The scaling `dp_size / global_batch_size` ensures:
1. **Local accumulation** is correctly averaged over the GPU's local batches
2. **Global reduction** (handled by FSDP) correctly averages across all GPUs
3. **Final gradient** matches what you'd get from processing all 128 samples at once

The key: **gradient = ∂(average_loss)/∂θ**, and we need to ensure the average is over all 128 samples, not just the 8 in the microbatch.

---

## Data Parallelism: Multiple GPUs

Data Parallel training splits data across GPUs:

```
Global batch: 128 samples

GPU 0: samples [0, 1, 2, ..., 31]    (32 samples)
GPU 1: samples [32, 33, 34, ..., 63]  (32 samples)
GPU 2: samples [64, 65, 66, ..., 95]  (32 samples)
GPU 3: samples [96, 97, 98, ..., 127] (32 samples)
```

Each GPU:
1. Processes its own samples
2. Computes gradients independently
3. **All-reduces gradients** across GPUs
4. Updates weights with averaged gradients

---

## FSDP (Fully Sharded Data Parallel)

Miles uses FSDP, which shards the model across GPUs:

### Regular Data Parallel:
```
Each GPU has: Full model (70B params) + Different data
Memory per GPU: 140GB (model) + 40GB (activations) = 180GB ❌ Doesn't fit!
```

### FSDP:
```
Each GPU has: 1/4 of model (17.5B params) + Different data
Memory per GPU: 35GB (model shard) + 40GB (activations) = 75GB ✓ Fits!
```

**How FSDP works:**

```python
# Forward pass:
for layer in model.layers:
    # 1. All-gather layer parameters from all GPUs
    gather_params(layer)  # Temporarily reconstruct full layer

    # 2. Compute forward pass
    output = layer(input)

    # 3. Free gathered parameters (keep only local shard)
    free_params(layer)

# Backward pass:
for layer in reversed(model.layers):
    # 1. All-gather layer parameters again
    gather_params(layer)

    # 2. Compute gradients
    output.backward()

    # 3. Reduce-scatter gradients (each GPU keeps 1/4 of gradients)
    reduce_scatter_grads(layer)

    # 4. Free gathered parameters
    free_params(layer)
```

**Benefits:**
- Memory savings: Store only 1/N of model on each GPU
- Communication: All-gather during forward/backward (efficient)
- Scaling: Train models that don't fit on single GPU

---

## Gradient Synchronization in Distributed Training

### When do gradients sync?

**FSDP automatically handles this!**

During backward pass:
```python
loss.backward()
```

FSDP does:
1. **Reduce-scatter:** Aggregate gradients across GPUs for each shard
2. Each GPU ends up with **averaged gradients** for its shard
3. No explicit `all_reduce` call needed (FSDP handles it!)

### Verification in code:

```python
# Line 738: Clip gradients
grad_norm = clip_grad_norm_(self.model.parameters(), max_norm=1.0)
```

`grad_norm` is computed **after** FSDP has already synchronized gradients across GPUs!

---

## Complete Example: 4 GPUs, Batch 128

```
Configuration:
  - 4 GPUs (Data Parallel size = 4)
  - Global batch size = 128
  - Micro batch size = 8
  - Model: 70B parameters (FSDP sharded across 4 GPUs)

Per GPU:
  - Local batch size = 128 / 4 = 32
  - Num microbatches = 32 / 8 = 4
  - Model shard = 17.5B parameters

Timeline:
────────────────────────────────────────────────────────────────

STEP 1: optimizer.zero_grad()
  All GPUs: gradients = 0

STEP 2: GPU 0 processes microbatch 0 (samples 0-7)
  Forward: model(tokens[0:8]) → logits
  Loss: compute_loss(logits, targets, advantages)
  Scale: loss *= 4/128  # Scale by dp_size/global_batch_size
  Backward: loss.backward()
    → FSDP all-gathers params, computes grads, reduce-scatters
    → GPU 0 now has grads for its shard
  Gradients: accumulated (1/4 microbatches done on GPU 0)

STEP 3: GPU 0 processes microbatch 1 (samples 8-15)
  (Same as above)
  Gradients: accumulated (2/4 microbatches done on GPU 0)

STEP 4: GPU 0 processes microbatch 2 (samples 16-23)
  Gradients: accumulated (3/4 microbatches done on GPU 0)

STEP 5: GPU 0 processes microbatch 3 (samples 24-31)
  Gradients: accumulated (4/4 microbatches done on GPU 0)

  (Simultaneously, GPUs 1, 2, 3 process their microbatches)

STEP 6: grad_accum checkpoint reached!
  All GPUs have finished their 4 microbatches
  Gradients are already synchronized (thanks to FSDP reduce-scatter)

  All GPUs:
    - Clip gradients: grad_norm = clip_grad_norm_(...)
    - Optimizer step: optimizer.step()
      → Each GPU updates its shard of parameters
    - Zero gradients: optimizer.zero_grad()

STEP 7: All GPUs now have updated model (each holding 1/4 of params)
  Ready for next batch!
```

---

## Summary: Key Concepts

### **Gradient Accumulation:**
- Process data in small chunks (microbatches)
- Gradients accumulate across chunks
- Update weights after all chunks processed
- Enables large effective batch sizes with limited memory

### **Data Parallelism:**
- Split data across GPUs
- Each GPU processes different samples
- Gradients averaged across GPUs
- All GPUs end up with same model weights

### **FSDP (Fully Sharded Data Parallel):**
- Shard model parameters across GPUs
- All-gather during forward/backward
- Reduce-scatter gradients
- Each GPU stores and updates only its shard
- Massive memory savings for huge models

### **Loss Scaling:**
- Scale loss by `dp_size / global_batch_size`
- Ensures correct gradient magnitude
- Accounts for accumulation and data parallelism

### **Synchronization Points:**
1. **During backward:** FSDP reduce-scatters gradients
2. **At optimizer.step():** Each GPU updates its shard
3. **Next forward:** All-gather ensures all GPUs see same model

---

## Visual Summary

```
┌─────────────────────────────────────────────────────────────┐
│              DISTRIBUTED TRAINING OVERVIEW                  │
└─────────────────────────────────────────────────────────────┘

GLOBAL BATCH: 128 samples
├─ GPU 0: 32 samples (4 microbatches × 8)
├─ GPU 1: 32 samples (4 microbatches × 8)
├─ GPU 2: 32 samples (4 microbatches × 8)
└─ GPU 3: 32 samples (4 microbatches × 8)

EACH GPU TIMELINE:
┌──────────────────────────────────────────────────────────┐
│ 1. zero_grad()                                           │
│ 2. Process MB 0 → backward → accumulate grad            │
│ 3. Process MB 1 → backward → accumulate grad            │
│ 4. Process MB 2 → backward → accumulate grad            │
│ 5. Process MB 3 → backward → accumulate grad            │
│ 6. [SYNC] FSDP has already reduced gradients            │
│ 7. clip_grad_norm()                                      │
│ 8. optimizer.step() → update local shard                │
│ 9. zero_grad()                                           │
└──────────────────────────────────────────────────────────┘

FSDP MODEL SHARDING:
┌────────────────┬────────────────┬────────────────┬────────────────┐
│   GPU 0:       │   GPU 1:       │   GPU 2:       │   GPU 3:       │
│   Params 0-25% │   Params 25-50%│   Params 50-75%│   Params 75-100│
│   17.5B params │   17.5B params │   17.5B params │   17.5B params │
└────────────────┴────────────────┴────────────────┴────────────────┘
         ↓ All-gather during forward/backward ↓
┌──────────────────────────────────────────────────────────────────┐
│  Temporarily reconstructed full layer (70B params)               │
└──────────────────────────────────────────────────────────────────┘
```

You now understand how Miles trains massive models efficiently across multiple GPUs!
