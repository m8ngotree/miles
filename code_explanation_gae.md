# Detailed Code Explanation: `get_advantages_and_returns()`

## Location
`miles/utils/ppo_utils.py:309-370`

## Purpose
Computes GAE (Generalized Advantage Estimation) advantages and returns for PPO training.

---

## Function Signature (lines 309-316)

```python
def get_advantages_and_returns(
    total_len: int,         # Full sequence length (prompt + response)
    response_len: int,      # Response length only
    values: torch.Tensor,   # Critic predictions [response_len]
    rewards: torch.Tensor,  # Rewards [response_len]
    gamma: float,           # Discount factor (typically 1.0)
    lambd: float,           # GAE lambda (typically 0.95)
) -> tuple[torch.Tensor, torch.Tensor]:
```

**Parameters explained:**
- `total_len`: Total sequence = prompt + response (e.g., 50 tokens)
- `response_len`: Just the response part (e.g., 30 tokens)
- `values`: Critic's value predictions for each response token
- `rewards`: Rewards for each response token (often sparse - only last token)
- `gamma`: Temporal discount (typically 0.99 or 1.0)
- `lambd`: GAE lambda for bias-variance tradeoff (typically 0.95)

**Returns:**
- `advantages`: How good each action was vs. expected
- `returns`: Target values for critic training

---

## Section 1: Handle Context Parallel (lines 339-347)

```python
from megatron.core import mpu

cp_size = mpu.get_context_parallel_world_size()
if cp_size > 1:
    from miles.backends.megatron_utils.cp_utils import all_gather_with_cp

    full_rewards = all_gather_with_cp(rewards, total_len, response_len)
    full_values = all_gather_with_cp(values, total_len, response_len)
else:
    full_rewards = rewards
    full_values = values
```

**What's happening:**

When using **Context Parallel (CP)**, each GPU only has a **chunk** of the sequence:

```
GPU 0: tokens [0-12]   (has values[0-12], rewards[0-12])
GPU 1: tokens [13-25]  (has values[13-25], rewards[13-25])
GPU 2: tokens [26-38]  (has values[26-38], rewards[26-38])
GPU 3: tokens [39-50]  (has values[39-50], rewards[39-50])
```

**Problem:** GAE needs the COMPLETE sequence because:
```python
A[t] = δ[t] + γλ * A[t+1]
```
To compute A[13] on GPU 1, you need A[14], A[15], ... A[50] (which GPU 2 and 3 have!).

**Solution:**
- `all_gather_with_cp()` collects chunks from all GPUs
- Each GPU now has `full_rewards` and `full_values` for entire response
- Compute GAE on full sequence
- Later: slice back to local chunk

**If cp_size == 1:** No Context Parallel, just use local tensors directly.

---

## Section 2: GAE Computation - The Core Algorithm (lines 349-357)

```python
lastgaelam = 0
advantages_reversed = []

for t in reversed(range(response_len)):
    nextvalues = full_values[t + 1] if t < response_len - 1 else 0.0
    delta = full_rewards[t] + gamma * nextvalues - full_values[t]
    lastgaelam = delta + gamma * lambd * lastgaelam
    advantages_reversed.append(lastgaelam)
full_advantages = torch.tensor(advantages_reversed[::-1], dtype=full_values.dtype, device=full_values.device)
```

**Step-by-step breakdown:**

### Line 349-350: Initialize
```python
lastgaelam = 0              # Starting advantage (for last token)
advantages_reversed = []     # Will store advantages in reverse order
```

### Line 352: Backward iteration
```python
for t in reversed(range(response_len)):
```
**Why reversed?** Because `A[t]` depends on `A[t+1]`!
- If `response_len = 30`, loop goes: 29, 28, 27, ..., 1, 0

### Line 353: Get next value
```python
nextvalues = full_values[t + 1] if t < response_len - 1 else 0.0
```
- For tokens 0 to 28: use `full_values[t+1]`
- For token 29 (last token): use 0.0 (no future value)

Example:
```python
t = 29: nextvalues = 0.0       (last token, terminal state)
t = 28: nextvalues = full_values[29]
t = 0:  nextvalues = full_values[1]
```

### Line 354: Compute TD error (delta)
```python
delta = full_rewards[t] + gamma * nextvalues - full_values[t]
```

This is the **TD error**: "How much better than expected?"

Formula: `δ[t] = r[t] + γ*V[t+1] - V[t]`

Example:
```python
# Token 28:
r[28] = 0.0
V[28] = 0.8
V[29] = 0.9
gamma = 1.0

delta = 0.0 + 1.0 * 0.9 - 0.8 = 0.1
```

### Line 355: GAE recursive formula
```python
lastgaelam = delta + gamma * lambd * lastgaelam
```

This is **THE CORE OF GAE**!

Formula: `A[t] = δ[t] + γλ * A[t+1]`

**Trace through example** (response_len = 3):

```python
# Iteration 1 (t=2, last token):
delta_2 = 0.1
lastgaelam = 0.1 + 1.0 * 0.95 * 0 = 0.1
advantages_reversed = [0.1]

# Iteration 2 (t=1):
delta_1 = 0.1
lastgaelam = 0.1 + 1.0 * 0.95 * 0.1 = 0.195
advantages_reversed = [0.1, 0.195]

# Iteration 3 (t=0):
delta_0 = 0.1
lastgaelam = 0.1 + 1.0 * 0.95 * 0.195 = 0.285
advantages_reversed = [0.1, 0.195, 0.285]
```

### Line 356: Store advantage
```python
advantages_reversed.append(lastgaelam)
```
Append to list (will be in reverse order: [A[T-1], A[T-2], ..., A[0]])

### Line 357: Reverse back to correct order
```python
full_advantages = torch.tensor(advantages_reversed[::-1], dtype=full_values.dtype, device=full_values.device)
```
- `[::-1]` reverses the list: [A[0], A[1], ..., A[T-1]]
- Convert to tensor with correct dtype/device

---

## Section 3: Compute Returns (line 358)

```python
full_returns = full_advantages + full_values
```

**Returns** = target values for critic training

Formula: `G[t] = A[t] + V[t]`

**Why?** The critic should predict returns, and we have:
```
A[t] = G[t] - V[t]  (definition of advantage)
So: G[t] = A[t] + V[t]
```

Example:
```python
A = [0.285, 0.195, 0.1]
V = [0.6,   0.7,   0.8]
G = [0.885, 0.895, 0.9]  ← Critic should learn to predict these!
```

---

## Section 4: Slice Back to Local Chunk (lines 360-367)

```python
if cp_size > 1:
    from miles.backends.megatron_utils.cp_utils import slice_log_prob_with_cp

    advantages = slice_log_prob_with_cp(full_advantages, total_len, response_len)
    returns = slice_log_prob_with_cp(full_returns, total_len, response_len)
else:
    advantages = full_advantages
    returns = full_returns
```

**With Context Parallel:**
- We gathered full sequence from all GPUs
- Computed GAE on full sequence
- Now slice back to just our local chunk

Example:
```
full_advantages = [A[0], A[1], ..., A[50]]  (all GPUs)

GPU 0: advantages = [A[0], ..., A[12]]
GPU 1: advantages = [A[13], ..., A[25]]
GPU 2: advantages = [A[26], ..., A[38]]
GPU 3: advantages = [A[39], ..., A[50]]
```

**Without Context Parallel (cp_size == 1):**
- Already have full sequence locally
- No slicing needed

---

## Section 5: Return Results (line 369)

```python
return advantages.detach(), returns
```

- `.detach()`: Detach advantages from computation graph (no gradients through GAE)
- Return both advantages (for PPO loss) and returns (for value loss)

**Why detach?**
- Advantages are targets for policy gradient
- Don't want gradients flowing through GAE computation
- Only backprop through the actual policy/value networks

---

## Complete Flow Example

```python
# Input:
total_len = 50        # Full sequence
response_len = 30     # Response only
values = [0.5, 0.6, 0.7, ...]      # 30 values
rewards = [0, 0, 0, ..., 0, 1.0]   # Sparse reward at end
gamma = 1.0
lambd = 0.95

# Step 1: Gather (if CP):
full_rewards, full_values = all_gather(...)  # All GPUs have full sequence

# Step 2: Compute GAE (backward):
for t in [29, 28, 27, ..., 1, 0]:
    delta = rewards[t] + gamma * values[t+1] - values[t]
    advantage[t] = delta + gamma * lambd * advantage[t+1]

# Result: advantages = [0.371, 0.285, 0.195, ..., 0.1]

# Step 3: Compute returns:
returns = advantages + values
# returns = [0.871, 0.885, 0.895, ..., 1.0]

# Step 4: Slice back (if CP):
advantages, returns = slice_back_to_local_chunk(...)

# Step 5: Return:
return advantages.detach(), returns
```

---

## Key Insights

1. **Context Parallel requires gathering:**
   - Each GPU has a chunk
   - GAE needs full sequence (backward dependencies)
   - Gather → Compute → Slice back

2. **Backward iteration is required:**
   - `A[t]` depends on `A[t+1]`
   - Must compute from end to start

3. **TD error uses bootstrapping:**
   - `delta = r[t] + γ*V[t+1] - V[t]`
   - Uses V[t+1] instead of waiting for actual rewards

4. **GAE weighs TD errors:**
   - `A[t] = δ[t] + (γλ)*δ[t+1] + (γλ)²*δ[t+2] + ...`
   - Exponential weighting by γλ

5. **Returns train the critic:**
   - `G[t] = A[t] + V[t]`
   - Critic learns: V → G (predict returns)

---

## Where This Is Used

This function is called during training:

1. **Rollout phase:** Generate samples, save old_log_probs
2. **Training forward pass:** Get new_log_probs and critic values
3. **Call this function:** Compute advantages and returns
4. **PPO loss:** Use advantages with importance ratio
5. **Value loss:** MSE between critic predictions and returns
6. **Backprop:** Update both actor and critic

Location in code:
- Called from `miles/backends/fsdp_utils/actor.py` (FSDP training)
- Or from `miles/backends/megatron_utils/loss.py` (Megatron training)
