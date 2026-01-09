# Memory Management in Miles

Training large MoE models (70B+ parameters) requires careful memory management. Miles provides several strategies to fit massive models in limited GPU memory.

---

## The Memory Challenge

### Memory Breakdown for 70B Model

```
Model weights (fp16):     140 GB
Optimizer states (Adam):  280 GB  (2× for momentum + variance)
Gradients (fp16):         140 GB
Activations (batch=8):     40 GB
─────────────────────────────
Total:                    600 GB per GPU! ❌
```

**Problem:** High-end GPUs have 80GB memory. We need 600GB!

**Solution:** Offloading + Model/Optimizer Sharding

---

## Strategy 1: FSDP Sharding (Always On)

**Fully Sharded Data Parallel** splits model across GPUs:

```
4 GPUs, 70B model:

Each GPU stores:
  Weights:    140/4 = 35 GB  ✓
  Optimizer:  280/4 = 70 GB  ✓
  Gradients:  140/4 = 35 GB  ✓
  Activations:       40 GB  ✓
  ─────────────────────────
  Total:            180 GB  ❌ Still doesn't fit in 80GB!
```

FSDP helps, but not enough for very large models!

---

## Strategy 2: CPU Offloading

Move inactive tensors to CPU RAM during different phases.

### **2a. Training Offload (`--offload-train`)**

**When training is NOT active:**
- Rollout workers are generating samples
- Training GPUs sit idle
- Wasteful to keep model on GPU!

**Solution:** Offload to CPU between training iterations

```python
# From miles/backends/fsdp_utils/actor.py:295-318

@timer
def sleep(self) -> None:
    """Pause CUDA memory - offload to CPU"""
    if not self.args.offload_train:
        return

    self.model.cpu()                        # Move model to CPU
    move_torch_optimizer(self.optimizer, "cpu")  # Move optimizer to CPU
    clear_memory()                          # Clear GPU cache
    dist.barrier()                          # Sync all GPUs

@timer
def wake_up(self) -> None:
    """Resume CUDA memory - load from CPU"""
    if not self.args.offload_train:
        return

    self.model.cuda()                       # Move model back to GPU
    move_torch_optimizer(self.optimizer, "cuda") # Move optimizer to GPU
    dist.barrier()
```

**Timeline:**
```
t=0:  Rollout starts → sleep() → Model moved to CPU
t=10: Rollout ends
t=11: Training starts → wake_up() → Model moved to GPU
t=15: Training completes → sleep() → Model moved to CPU
t=16: Next rollout starts...
```

**Benefits:**
- GPU memory free during rollout
- Can run more rollout workers
- Prevents GPU memory fragmentation

**Cost:**
- Transfer time: ~5-10 seconds for 70B model
- Adds latency between rollout and training

### **2b. Rollout Offload (`--offload-rollout`)**

Similar idea for rollout workers:

```python
# When rollout is NOT active (training is running):
rollout_worker.sleep()   # Offload to CPU

# When new batch arrives:
rollout_worker.wake_up() # Load back to GPU
rollout_worker.generate()
rollout_worker.sleep()   # Offload again
```

### **2c. FSDP CPU Offload (`--fsdp-cpu-offload`)**

**Most aggressive:** Keep optimizer on CPU **all the time**!

```
GPU:
  Weights (sharded):  35 GB
  Gradients:          35 GB
  Activations:        40 GB
  ────────────────────────
  Total:             110 GB  ← Closer to fitting!

CPU:
  Optimizer states:   70 GB  ← Moved to CPU RAM
```

**Tradeoff:**
- ✅ Saves 70GB GPU memory
- ❌ Optimizer step runs on CPU (slower!)
- ❌ Need to copy gradients CPU→GPU every step

**When to use:**
- Model barely fits on GPU
- Optimizer step not the bottleneck
- Have plenty of CPU RAM

**Configuration:**
```bash
--fsdp-cpu-offload        # Keep optimizer on CPU
--fsdp-cpu-offload-backend gloo  # Use Gloo for CPU communication
```

---

## Strategy 3: Colocated Mode

**Problem with separate rollout + training:**
```
GPU 0-3: Rollout workers  (70B model loaded)
GPU 4-7: Training workers (70B model loaded)
```
Total memory: 2× model size!

**Colocated mode:** Same GPUs for rollout AND training!

```
GPU 0-3: Rollout + Training (70B model loaded ONCE)
```

### How It Works

```python
# Pseudocode
while training:
    # Phase 1: Rollout (using current weights)
    samples = rollout_manager.generate(batch_size=128)

    # Phase 2: Training (update weights)
    train_actor.train(samples)

    # Loop continues with updated weights
```

**Benefits:**
- ✅ 2x memory savings (don't duplicate model)
- ✅ No weight syncing needed (same GPU!)
- ✅ Simpler deployment

**Drawbacks:**
- ❌ Sequential (can't overlap rollout + training)
- ❌ Lower throughput
- ❌ GPUs idle during opposite phase

**When to use:**
- Limited GPUs available
- Memory constrained
- Simpler setup preferred

**Configuration:**
```bash
--colocated-mode      # Enable colocated rollout + training
--offload-train       # Offload during rollout phase
```

---

## Strategy 4: Activation Checkpointing

**Problem:** Activations grow with sequence length!

```
Batch size = 8
Sequence length = 2048 tokens
Model layers = 80

Activations per layer ≈ 8 × 2048 × hidden_size
Total activations ≈ 40 GB  ← Can't reduce batch/sequence!
```

**Solution:** Recompute activations during backward instead of storing

### Gradient Checkpointing

```python
# Normal (store all activations):
def forward(x):
    for layer in layers:
        x = layer(x)
        save_for_backward(x)  # ← Stores activation
    return x

# With checkpointing (recompute):
def forward(x):
    for layer in layers:
        if layer.checkpoint:
            x = checkpoint(layer, x)  # Don't store, recompute in backward
        else:
            x = layer(x)
            save_for_backward(x)
    return x
```

**Tradeoff:**
- ✅ Memory: ~50% reduction in activation memory
- ❌ Time: ~30% slower (recompute cost)

**Configuration:**
```bash
--gradient-checkpointing       # Enable for all layers
--gradient-checkpointing-ratio 0.5  # Checkpoint every other layer
```

---

## Strategy 5: Flash Attention

Flash Attention reduces memory for attention mechanism:

**Standard Attention:**
```python
# Compute full attention matrix (memory expensive!)
scores = Q @ K.T           # [batch, heads, seq, seq] ← O(seq²) memory!
attn = softmax(scores)
output = attn @ V
```

For seq_len=2048:
```
Attention matrix: batch × heads × 2048 × 2048 = huge!
Example: 8 × 32 × 2048 × 2048 × 2 bytes = 2 GB per layer!
```

**Flash Attention:**
```python
# Fused kernel, never materializes full matrix
output = flash_attn(Q, K, V)  # ← O(seq) memory!
```

**Benefits:**
- ✅ Massive memory savings (O(seq²) → O(seq))
- ✅ Faster (fused CUDA kernel)
- ✅ Numerically accurate

**In Miles:**
Flash Attention is used automatically (via HuggingFace models with `use_flash_attention_2=True`)

---

## Memory Management in Practice

### Example Configuration for 70B Model on 4×80GB GPUs

```bash
# Strategy 1: FSDP (automatic)
# Each GPU stores 1/4 of model (35GB)

# Strategy 2: Training offload
--offload-train
# Offload to CPU when rollout active

# Strategy 3: Gradient checkpointing
--gradient-checkpointing-ratio 0.5
# Checkpoint half the layers

# Strategy 4: Flash Attention (automatic)
# Reduces attention memory

# Result:
# GPU memory per rank:
#   Model shard:    35 GB
#   Gradients:      35 GB
#   Activations:    20 GB (with checkpointing)
#   ─────────────────────
#   Total:          90 GB
#
# Still over 80GB! Need one more optimization...

# Strategy 5: Reduce batch size or use colocated mode
--micro-batch-size 4  # Instead of 8
# OR
--colocated-mode --offload-train
```

---

## OOM (Out of Memory) Handling

### Detection

Miles monitors memory usage:

```python
import torch

def print_memory(prefix=""):
    allocated = torch.cuda.memory_allocated() / 1e9
    reserved = torch.cuda.memory_reserved() / 1e9
    print(f"{prefix} Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB")
```

Called at key points:
- Before/after model loading
- Before/after offloading
- During training steps

### Recovery Strategies

**1. Automatic retry with smaller batch:**
```python
try:
    train(batch_size=16)
except torch.cuda.OutOfMemoryError:
    clear_memory()
    train(batch_size=8)  # Retry with half batch
```

**2. Emergency offload:**
```python
except torch.cuda.OutOfMemoryError:
    # Offload everything possible
    model.cpu()
    optimizer.cpu()
    clear_memory()
    # Try again
```

**3. Graceful degradation:**
```python
if check_memory_available() < required_memory:
    # Enable more aggressive optimizations
    enable_gradient_checkpointing()
    reduce_micro_batch_size()
```

---

## Memory Profiling

### Tools

**1. PyTorch Memory Profiler:**
```python
with torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA],
    record_shapes=True,
    with_stack=True,
) as prof:
    train_step()

print(prof.key_averages().table())
```

**2. NVIDIA nvidia-smi:**
```bash
nvidia-smi -l 1  # Monitor every second
```

**3. PyTorch memory snapshot:**
```python
torch.cuda.memory._record_memory_history()
train_step()
torch.cuda.memory._dump_snapshot("memory.pkl")
```

Analyze with:
```bash
python -m torch.cuda._memory_viz trace_plot memory.pkl -o memory.html
```

---

## Best Practices

### 1. Start Conservative
```bash
# First attempt - most aggressive memory saving
--offload-train
--fsdp-cpu-offload
--gradient-checkpointing
--micro-batch-size 1
```

### 2. Profile and Optimize
```
- Run with profiling
- Identify memory bottlenecks
- Relax constraints gradually
- Measure throughput vs memory tradeoff
```

### 3. Memory-Throughput Tradeoff

```
Configuration         Memory    Throughput
─────────────────────────────────────────
All optimizations     60 GB     100 tok/s
Some offloading       70 GB     200 tok/s
Minimal offload       75 GB     300 tok/s
No offload            90 GB     ❌ OOM
```

Choose based on your constraints!

### 4. Monitor in Production

```python
# Log memory usage
if iteration % 10 == 0:
    log_memory_stats()

# Alert on high usage
if torch.cuda.memory_allocated() > 0.9 * total_memory:
    trigger_alert("Memory usage > 90%!")
```

---

## Summary: Memory Management Strategies

| Strategy | Memory Saved | Performance Cost | When to Use |
|----------|--------------|------------------|-------------|
| **FSDP Sharding** | 75% (on 4 GPUs) | Minimal | Always |
| **Training Offload** | Frees GPU during rollout | 5-10s transfer | Colocated mode |
| **FSDP CPU Offload** | ~40% optimizer | Slower optimizer | Very tight memory |
| **Activation Checkpointing** | ~50% activations | ~30% slower | Memory constrained |
| **Flash Attention** | ~50% attention | None (faster!) | Always |
| **Colocated Mode** | 50% (no duplicate) | Sequential phases | Limited GPUs |
| **Smaller Batch** | Linear with batch | Linear throughput | Last resort |

### Typical Configuration Hierarchy

```
Level 1 (Baseline):
  - FSDP sharding
  - Flash Attention

Level 2 (Moderate):
  + Gradient checkpointing (ratio=0.5)
  + Training offload (if colocated)

Level 3 (Aggressive):
  + FSDP CPU offload
  + Higher checkpointing ratio (0.75)

Level 4 (Desperate):
  + Smaller micro batch size
  + Shorter sequence lengths
```

**Start at Level 1, escalate as needed!**

---

## Code References

**Offloading:**
- `miles/backends/fsdp_utils/actor.py:295-318` - sleep/wake_up methods
- `miles/utils/arguments.py` - Offload configuration flags

**Memory Utilities:**
- `miles/utils/memory_utils.py` - Memory monitoring helpers
- `clear_memory()` - Clear CUDA cache

**Configuration Examples:**
- `scripts/train/*.sh` - See production configs for different model sizes

You now understand how to fit massive models in limited memory! 🎯
