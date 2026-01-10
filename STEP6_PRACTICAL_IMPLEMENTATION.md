# Step 6: Practical Implementation Guide

This guide shows you how to actually **run training**, **customize Miles**, and **apply common patterns** in practice.

---

## Part 1: Running Training

### 1.1 Quick Start

**Simplest way to start training:**

```bash
# 1. Clone and setup
git clone https://github.com/bytedance/miles
cd miles

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run a pre-configured training script
bash scripts/run-qwen3-4B_4xgpu.sh
```

That's it! The script handles everything:
- Starting Ray cluster
- Loading model
- Launching rollout workers
- Running training
- Logging to Weights & Biases

---

### 1.2 Understanding Configuration Scripts

**Script Structure:**

Miles uses modular shell scripts with clear sections:

```bash
#!/bin/bash

# ═══════════════════════════════════════════════════════════════
# SECTION 1: Cleanup (optional)
# ═══════════════════════════════════════════════════════════════
pkill -9 sglang  # Kill previous SGLang processes
ray stop --force # Stop Ray cluster

# ═══════════════════════════════════════════════════════════════
# SECTION 2: Environment Setup
# ═══════════════════════════════════════════════════════════════
export CUDA_VISIBLE_DEVICES=4,5,6,7  # Which GPUs to use
export PYTHONBUFFERED=16             # Disable Python buffering

# ═══════════════════════════════════════════════════════════════
# SECTION 3: Model Architecture
# ═══════════════════════════════════════════════════════════════
source "/root/miles/scripts/models/qwen3-4B.sh"
# Defines MODEL_ARGS: layers, hidden size, attention heads, etc.

# ═══════════════════════════════════════════════════════════════
# SECTION 4: Checkpoint Paths
# ═══════════════════════════════════════════════════════════════
CKPT_ARGS=(
   --hf-checkpoint /root/Qwen3-4B          # HuggingFace model
   --ref-load /root/Qwen3-4B_torch_dist    # Reference model (for KL)
   --load /root/Qwen3-4B_miles/            # Resume from checkpoint
   --save /root/Qwen3-4B_miles/            # Save checkpoints here
   --save-interval 20                      # Save every 20 iterations
)

# ═══════════════════════════════════════════════════════════════
# SECTION 5: Rollout Configuration
# ═══════════════════════════════════════════════════════════════
ROLLOUT_ARGS=(
   --prompt-data /root/data.jsonl          # Training data
   --input-key prompt                      # JSON key for prompts
   --label-key label                       # JSON key for labels
   --apply-chat-template                   # Format with chat template

   --rm-type deepscaler                    # Reward model type

   --num-rollout 3000                      # Total training iterations
   --rollout-batch-size 32                 # Samples per iteration
   --n-samples-per-prompt 8                # Samples per prompt (for GRPO)
   --rollout-max-response-len 8192         # Max generation length
   --rollout-temperature 1                 # Sampling temperature

   --global-batch-size 256                 # Total batch size for training
)

# ═══════════════════════════════════════════════════════════════
# SECTION 6: Evaluation Configuration
# ═══════════════════════════════════════════════════════════════
EVAL_ARGS=(
   --eval-interval 20                      # Evaluate every 20 iterations
   --eval-prompt-data aime /root/aime.jsonl  # Eval dataset
   --n-samples-per-eval-prompt 16          # Samples per eval prompt
)

# ═══════════════════════════════════════════════════════════════
# SECTION 7: Performance Tuning
# ═══════════════════════════════════════════════════════════════
PERF_ARGS=(
   --tensor-model-parallel-size 2          # Split model across 2 GPUs
   --sequence-parallel                     # Enable sequence parallelism
   --context-parallel-size 1               # Context parallel size

   --recompute-granularity full            # Activation checkpointing
   --recompute-num-layers 1                # Checkpoint every N layers

   --use-dynamic-batch-size                # Auto batch sizing
   --max-tokens-per-gpu 9216               # Max tokens per GPU
)

# ═══════════════════════════════════════════════════════════════
# SECTION 8: Algorithm Configuration (GRPO/PPO)
# ═══════════════════════════════════════════════════════════════
GRPO_ARGS=(
   --advantage-estimator grpo              # Use GRPO (simpler than GAE)
   --use-kl-loss                           # Add KL penalty
   --kl-loss-coef 0.00                     # KL coefficient
   --entropy-coef 0.00                     # Entropy bonus
   --eps-clip 0.2                          # PPO clipping epsilon
   --eps-clip-high 0.28                    # Asymmetric clipping
)

# ═══════════════════════════════════════════════════════════════
# SECTION 9: Optimizer Settings
# ═══════════════════════════════════════════════════════════════
OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-6                               # Learning rate
   --lr-decay-style constant               # No LR decay
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98
)

# ═══════════════════════════════════════════════════════════════
# SECTION 10: Logging
# ═══════════════════════════════════════════════════════════════
WANDB_ARGS=(
   --use-wandb
   --wandb-project my-rl-project
   --wandb-group qwen3-4B-experiment
   --wandb-key ${WANDB_KEY}                # Set env var first!
)

# ═══════════════════════════════════════════════════════════════
# SECTION 11: Launch Ray and Start Training
# ═══════════════════════════════════════════════════════════════
ray start --head --num-gpus 4

ray job submit -- python3 train.py \
   --actor-num-nodes 1 \
   --actor-num-gpus-per-node 4 \
   --colocate \                            # Same GPUs for rollout+training
   ${MODEL_ARGS[@]} \
   ${CKPT_ARGS[@]} \
   ${ROLLOUT_ARGS[@]} \
   ${OPTIMIZER_ARGS[@]} \
   ${GRPO_ARGS[@]} \
   ${WANDB_ARGS[@]} \
   ${PERF_ARGS[@]}
```

---

### 1.3 Model-Specific Configurations

**Model architecture files:** `scripts/models/*.sh`

**Example: Qwen3-4B** (`scripts/models/qwen3-4B.sh`):

```bash
MODEL_ARGS=(
   --swiglu                                # SwiGLU activation
   --num-layers 36                         # Transformer layers
   --hidden-size 2560                      # Hidden dimension
   --ffn-hidden-size 9728                  # FFN dimension
   --num-attention-heads 32                # Attention heads
   --group-query-attention                 # GQA enabled
   --num-query-groups 8                    # GQA groups
   --use-rotary-position-embeddings        # RoPE
   --disable-bias-linear                   # No bias in linear layers
   --normalization "RMSNorm"               # RMSNorm (not LayerNorm)
   --norm-epsilon 1e-6
   --rotary-base 1000000                   # RoPE base frequency
   --vocab-size 151936                     # Vocabulary size
   --kv-channels 128                       # K/V dimension per head
   --qk-layernorm                          # QK normalization
)
```

**To add support for a new model:**

1. Create `scripts/models/your-model.sh`
2. Specify architecture parameters
3. Reference in your training script: `source scripts/models/your-model.sh`

**Common model architectures supported:**
- Qwen2.5/3 (various sizes: 0.5B to 235B)
- GLM-4 (9B to 355B)
- DeepSeek-R1
- Kimi-k2
- Custom models (configure manually)

---

### 1.4 Command-Line Arguments Reference

**Key Arguments by Category:**

#### **Data:**
```bash
--prompt-data PATH                  # Training data (JSONL)
--input-key KEY                     # JSON field for prompt
--label-key KEY                     # JSON field for label/answer
--apply-chat-template               # Auto-format with model's chat template
--rollout-shuffle                   # Shuffle data each iteration
```

#### **Rollout:**
```bash
--num-rollout N                     # Total training iterations
--rollout-batch-size N              # Prompts per iteration
--n-samples-per-prompt N            # Samples per prompt (GRPO)
--rollout-max-response-len N        # Max generation length
--rollout-temperature T             # Sampling temperature
--rollout-top-p P                   # Nucleus sampling
```

#### **Training:**
```bash
--global-batch-size N               # Effective batch size
--micro-batch-size N                # Per-GPU batch size
--use-dynamic-batch-size            # Auto-determine batch size
--max-tokens-per-gpu N              # Memory constraint
```

#### **Algorithm:**
```bash
--advantage-estimator TYPE          # grpo | gspo | gae
--eps-clip EPSILON                  # PPO clipping (0.2 typical)
--kl-loss-coef COEF                 # KL penalty weight
--entropy-coef COEF                 # Entropy bonus weight
```

#### **Parallelism:**
```bash
--tensor-model-parallel-size N      # Split model across N GPUs
--pipeline-model-parallel-size N    # Pipeline parallelism
--context-parallel-size N           # Context parallelism
--sequence-parallel                 # Enable sequence parallelism
```

#### **Memory:**
```bash
--offload-train                     # Offload during rollout
--offload-rollout                   # Offload during training
--colocate                          # Same GPUs for both (auto offload)
--recompute-granularity LEVEL       # Activation checkpointing
```

#### **Checkpoints:**
```bash
--hf-checkpoint PATH                # HuggingFace model path
--load PATH                         # Resume from Miles checkpoint
--save PATH                         # Save checkpoints here
--save-interval N                   # Save every N iterations
```

**Full reference:** `python train.py --help`

---

## Part 2: Customization

### 2.1 Custom Reward Functions

**Reward models are in:** `miles/rollout/rm_hub/`

**Example: Math Problem Reward** (simplified from `math_dapo_utils.py`):

```python
# File: miles/rollout/rm_hub/my_math_reward.py

import re
from miles.utils.types import Sample

def extract_answer(text: str) -> str | None:
    """Extract answer from generated text.

    Example: "The answer is \\boxed{42}" → "42"
    """
    # Find last \boxed{...} expression
    match = re.search(r'\\boxed\{([^}]+)\}', text)
    if match:
        return match.group(1).strip()
    return None

def normalize_answer(ans: str) -> str:
    """Normalize answer for comparison."""
    # Remove whitespace, convert to lowercase
    ans = ans.replace(" ", "").lower()
    # Remove common LaTeX commands
    ans = ans.replace("\\text{", "").replace("}", "")
    return ans

async def my_math_reward(args, sample: Sample) -> float:
    """Compute reward for math problem.

    Args:
        args: Command-line arguments
        sample: Generated sample

    Returns:
        1.0 if correct, 0.0 if wrong
    """
    # Extract predicted answer from response
    predicted = extract_answer(sample.response)
    if predicted is None:
        return 0.0  # No answer found

    # Get ground truth from sample
    ground_truth = sample.label

    # Normalize both answers
    pred_norm = normalize_answer(predicted)
    truth_norm = normalize_answer(ground_truth)

    # Check if they match
    if pred_norm == truth_norm:
        return 1.0  # Correct!
    else:
        return 0.0  # Wrong

# Register reward function
REWARD_FUNCTIONS = {
    "my_math": my_math_reward,
}
```

**To use your custom reward:**

1. Add file to `miles/rollout/rm_hub/`
2. Register in `miles/rollout/rm_hub/__init__.py`:
```python
from .my_math_reward import my_math_reward

REWARD_REGISTRY = {
    ...
    "my_math": my_math_reward,
}
```

3. Use in training script:
```bash
--rm-type my_math
```

**Advanced: Process Reward Models**

Reward at each token, not just final:

```python
async def process_reward(args, sample: Sample) -> list[float]:
    """Return reward for each token."""
    rewards = []
    for i, token in enumerate(sample.tokens):
        if is_correct_step(token):
            rewards.append(0.1)  # Small positive reward
        else:
            rewards.append(0.0)

    # Big reward at end if final answer correct
    if is_final_answer_correct(sample):
        rewards[-1] = 1.0

    return rewards
```

---

### 2.2 Custom Data Sources

**Data format:** JSONL (one JSON object per line)

**Basic format:**

```json
{"prompt": "What is 2+2?", "label": "4"}
{"prompt": "Solve x^2 = 9", "label": "x = 3 or x = -3"}
```

**With chat template:**

```python
# Your data
{"prompt": "What is 2+2?", "label": "4"}

# With --apply-chat-template, becomes:
{
  "prompt": "<|im_start|>user\nWhat is 2+2?<|im_end|>\n<|im_start|>assistant\n",
  "label": "4"
}
```

**Multi-turn conversations:**

```json
{
  "prompt": [
    {"role": "user", "content": "What is 2+2?"},
    {"role": "assistant", "content": "4"},
    {"role": "user", "content": "What about 3+3?"}
  ],
  "label": "6"
}
```

**Custom data loading:**

Implement in `miles/utils/data.py`:

```python
class CustomDataset(Dataset):
    def __init__(self, path: str, tokenizer):
        self.data = self.load_data(path)
        self.tokenizer = tokenizer

    def load_data(self, path: str):
        # Your custom loading logic
        with open(path) as f:
            data = custom_parse(f.read())
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        return {
            "prompt": item["question"],
            "label": item["answer"],
            "metadata": item.get("difficulty", "medium"),
        }
```

---

### 2.3 Custom Generation Strategies

**Sampling parameters:**

```bash
# Temperature (randomness)
--rollout-temperature 1.0    # Default: balanced
--rollout-temperature 0.0    # Greedy (deterministic)
--rollout-temperature 2.0    # Very random

# Top-p (nucleus sampling)
--rollout-top-p 0.9          # Sample from top 90% probability mass
--rollout-top-p 1.0          # No filtering (use all tokens)

# Top-k sampling
--rollout-top-k 50           # Sample from top 50 tokens

# Stop conditions
--rollout-stop "</s>"        # Stop at specific token
--rollout-stop-token-ids 2   # Stop at token ID
--rollout-max-response-len 8192  # Maximum length
```

**Beam search (deterministic):**

```bash
--rollout-num-beams 4        # Use beam search with 4 beams
--rollout-temperature 0.0    # Greedy within beams
```

**Custom sampling in code:**

Modify `miles/rollout/sglang_rollout.py`:

```python
# Custom sampling strategy
sampling_params = {
    "temperature": 1.0,
    "top_p": 0.95,
    "top_k": 100,

    # Custom: Penalty for repetition
    "repetition_penalty": 1.2,

    # Custom: Length penalty
    "length_penalty": 0.9,

    # Custom: Presence penalty
    "presence_penalty": 0.5,
}
```

---

### 2.4 Plugin System

**Miles plugins:** `miles_plugins/`

**Structure:**
```
miles_plugins/
├── __init__.py
├── mbridge/                 # Model bridge (adapters)
│   ├── glm4moe.py          # GLM-4 MoE adapter
│   └── ...
├── megatron_bridge/         # Megatron-specific adapters
│   └── ...
└── models/                  # Custom model implementations
    └── ...
```

**Creating a custom plugin:**

**Example: Custom model adapter**

```python
# File: miles_plugins/models/my_custom_model.py

import torch
from torch import nn

class MyCustomModel(nn.Module):
    """Custom model implementation."""

    def __init__(self, config):
        super().__init__()
        self.config = config

        # Your custom architecture
        self.layers = nn.ModuleList([
            MyCustomLayer(config) for _ in range(config.num_layers)
        ])

    def forward(self, input_ids, attention_mask=None):
        # Your forward logic
        hidden_states = self.embed(input_ids)

        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask)

        logits = self.lm_head(hidden_states)
        return logits
```

**Register plugin:**

```python
# File: miles_plugins/__init__.py

from .models.my_custom_model import MyCustomModel

MODEL_REGISTRY = {
    "my_custom_model": MyCustomModel,
}
```

**Use in training:**

```bash
--model-type my_custom_model
--hf-checkpoint /path/to/my/model
```

---

## Part 3: Common Patterns

### 3.1 SFT → RL Workflow

**Step 1: Supervised Fine-Tuning (SFT)**

First, train on high-quality demonstrations:

```bash
# Use your SFT framework (e.g., HuggingFace, Megatron)
python sft_train.py \
    --model qwen3-4B \
    --data sft_data.jsonl \
    --epochs 3
```

**Step 2: Convert to Miles format**

```bash
# Convert HuggingFace checkpoint to torch distributed format
python tools/convert_checkpoint.py \
    --model-type qwen3 \
    --input /path/to/hf/checkpoint \
    --output /path/to/torch_dist/checkpoint
```

**Step 3: RL Training**

```bash
bash scripts/run-qwen3-4B.sh
# With checkpoints from SFT:
#   --hf-checkpoint /path/to/hf/checkpoint
#   --load /path/to/torch_dist/checkpoint
```

**Step 4: Iterate**

```
SFT → RL (iteration 1) → RL (iteration 2) → ... → Final model
     ↑__________________|
        Use RL checkpoint for next iteration
```

---

### 3.2 Hyperparameter Tuning

**Key hyperparameters to tune:**

#### **1. Learning Rate**
```bash
--lr 1e-6    # Start conservative
--lr 5e-7    # If training unstable
--lr 2e-6    # If training too slow
```

**Rule of thumb:** Start with 1e-6, adjust based on training curves.

#### **2. Batch Size**
```bash
--global-batch-size 256    # Larger = more stable, slower
--global-batch-size 128    # Smaller = less stable, faster
--global-batch-size 512    # Very large (if you have GPUs!)
```

**Trade-off:** Stability vs. throughput

#### **3. PPO Clipping**
```bash
--eps-clip 0.2      # Standard
--eps-clip 0.1      # More conservative
--eps-clip 0.3      # More aggressive
```

**Watch:** `train/pg_clipfrac` metric
- If clipfrac > 0.3: Reduce eps-clip
- If clipfrac < 0.1: Can increase eps-clip

#### **4. KL Penalty**
```bash
--kl-loss-coef 0.0    # No KL penalty
--kl-loss-coef 0.01   # Light penalty
--kl-loss-coef 0.1    # Strong penalty
```

**Purpose:** Keep policy close to SFT model

#### **5. Temperature**
```bash
--rollout-temperature 1.0    # Standard
--rollout-temperature 0.8    # More focused
--rollout-temperature 1.2    # More exploratory
```

**For GRPO:** Higher temperature = more diversity = better

#### **6. Samples per Prompt**
```bash
--n-samples-per-prompt 8     # GRPO default
--n-samples-per-prompt 16    # More samples = better estimates
--n-samples-per-prompt 4     # Fewer = faster
```

**Tuning workflow:**

```python
# 1. Start with defaults
# 2. Monitor key metrics:
#    - train/loss (should decrease)
#    - train/pg_clipfrac (0.1-0.3 is good)
#    - train/ppo_kl (should be small, < 0.01)
#    - eval/reward (should increase)

# 3. Adjust based on observations:
if train_loss_unstable:
    decrease_lr()  # or decrease_batch_size()

if pg_clipfrac_too_high:
    decrease_eps_clip()

if eval_reward_plateaus:
    increase_temperature()  # more exploration
    increase_n_samples_per_prompt()
```

---

### 3.3 Debugging Failed Runs

**Common issues and fixes:**

#### **Issue 1: OOM (Out of Memory)**

**Symptoms:**
```
RuntimeError: CUDA out of memory. Tried to allocate X GB
```

**Solutions:**
```bash
# 1. Reduce batch size
--micro-batch-size 4  # Instead of 8

# 2. Enable offloading
--offload-train

# 3. Increase gradient checkpointing
--recompute-granularity full
--recompute-num-layers 2  # Checkpoint more layers

# 4. Use colocated mode
--colocate  # Share GPUs between rollout and training

# 5. Reduce sequence length
--rollout-max-response-len 4096  # Instead of 8192
```

#### **Issue 2: Ray Worker Crashes**

**Symptoms:**
```
RayActorError: The actor died unexpectedly
```

**Debug:**
```bash
# 1. Check Ray logs
cat /tmp/ray/session_latest/logs/worker-*.out

# 2. Increase logging
export RAY_LOG_TO_STDERR=1
ray start --head --log-to-driver

# 3. Check GPU health
nvidia-smi
```

**Common causes:**
- GPU OOM (see above)
- NCCL communication errors
- Model initialization failures

#### **Issue 3: NaN Loss**

**Symptoms:**
```
train/loss: nan
```

**Solutions:**
```bash
# 1. Reduce learning rate
--lr 5e-7  # Much lower

# 2. Enable mixed precision safety
--accumulate-allreduce-grads-in-fp32
--attention-softmax-in-fp32

# 3. Gradient clipping
--clip-grad 1.0  # Already default, can try 0.5

# 4. Check data quality
# Bad data can cause NaN!
```

#### **Issue 4: Slow Training**

**Symptoms:**
- Iterations take > 5 minutes
- GPU utilization < 50%

**Optimizations:**
```bash
# 1. Enable speculative decoding
--sglang-speculative-algorithm EAGLE
--sglang-speculative-num-draft-tokens 4

# 2. Use Flash Attention (should be automatic)
--attention-backend flash

# 3. Increase batch size (if memory allows)
--global-batch-size 512

# 4. Reduce eval frequency
--eval-interval 100  # Instead of 20

# 5. Use fewer samples per prompt
--n-samples-per-prompt 4  # Instead of 8
```

#### **Issue 5: No Learning Progress**

**Symptoms:**
- `eval/reward` stays flat
- Model outputs don't improve

**Debug:**
```bash
# 1. Check if reward model works
# Add logging to your reward function

# 2. Verify advantages are computed
# Check train/advantages in W&B

# 3. Check if policy is updating
# Monitor train/ppo_kl (should be > 0)

# 4. Increase learning rate if too small
--lr 2e-6  # Instead of 1e-6

# 5. Reduce KL penalty if too strong
--kl-loss-coef 0.0  # Disable to debug
```

---

### 3.4 Performance Optimization

**Optimization checklist:**

#### **1. GPU Utilization**
```bash
# Monitor:
nvidia-smi -l 1

# Target: >80% GPU utilization during training

# If low:
- Increase batch size (--micro-batch-size)
- Reduce gradient checkpointing
- Check for CPU bottlenecks
```

#### **2. Throughput (tokens/sec)**
```bash
# Log throughput metrics
# Check in W&B: rollout/tokens_per_second

# To improve:
--sglang-speculative-algorithm EAGLE  # 2-3x speedup
--max-tokens-per-gpu 16384           # Bigger batches
--use-dynamic-batch-size             # Auto-optimize
```

#### **3. Memory Efficiency**
```bash
# Minimize memory waste:
--use-dynamic-batch-size             # Pack efficiently
# (Uses data packing we learned in Step 4!)

# Monitor:
# Check GPU memory usage in nvidia-smi
```

#### **4. Distributed Scaling**
```bash
# Scale to multiple nodes:
--actor-num-nodes 2                  # Use 2 nodes
--actor-num-gpus-per-node 8          # 8 GPUs each

# Total: 16 GPUs
# Should see ~16x throughput (if network fast)
```

---

## Quick Reference: Common Commands

### Start Training
```bash
bash scripts/run-qwen3-4B.sh
```

### Resume from Checkpoint
```bash
--load /path/to/checkpoint
```

### Monitor Training
```bash
# Weights & Biases dashboard
wandb login
# Then check: wandb.ai/your-project

# Or local logs
tail -f /tmp/ray/session_latest/logs/worker-*.out
```

### Debug
```bash
# Enable verbose logging
export RAY_LOG_TO_STDERR=1
export PYTHONBUFFERED=1

# Check GPU usage
watch -n 1 nvidia-smi
```

### Kill Everything
```bash
pkill -9 python
pkill -9 ray
pkill -9 sglang
ray stop --force
```

---

## Summary: Key Takeaways

✅ **Configuration:** Modular shell scripts with clear sections
✅ **Model Support:** Use `scripts/models/*.sh` for architectures
✅ **Custom Rewards:** Add to `miles/rollout/rm_hub/`
✅ **Custom Data:** JSONL format, support chat templates
✅ **Plugins:** Extend via `miles_plugins/`
✅ **SFT→RL:** Standard workflow pattern
✅ **Hyperparameters:** Start conservative, tune based on metrics
✅ **Debugging:** Check logs, reduce batch size, enable safety features
✅ **Performance:** Speculative decoding, Flash Attention, dynamic batching

**You're now ready to run your own RL training!** 🚀
