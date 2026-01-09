#!/usr/bin/env python3
"""
Concrete example of PPO math - No dependencies required!
"""

import math

print("=" * 80)
print("PPO MATH EXAMPLE: Understanding GAE and Clipped Loss")
print("=" * 80)

# ============================================================================
# Setup: Response generation
# ============================================================================
print("\n[SCENARIO]")
print("Prompt: 'What is the capital of France?'")
print("Response: 'Paris'  (2 tokens for simplicity)")
print()

tokens = ["Paris", "."]
rewards = [0.0, 1.0]  # Only final token gets reward
values = [0.5, 0.8]   # Critic predictions

print(f"Tokens:         {tokens}")
print(f"Rewards:        {rewards}")
print(f"Critic values:  {values}")

gamma = 1.0
lambd = 0.95

print(f"\nγ (gamma) = {gamma}")
print(f"λ (lambda) = {lambd}")

# ============================================================================
# STEP 1: GAE - BACKWARD PASS
# ============================================================================
print("\n" + "=" * 80)
print("STEP 1: Computing Advantages (GAE) - BACKWARD ITERATION")
print("=" * 80)

response_len = len(rewards)
lastgaelam = 0
advantages = []

print("\nBackward pass (t=1 → t=0):")
print("-" * 40)

for t in reversed(range(response_len)):
    nextvalues = values[t + 1] if t < response_len - 1 else 0.0

    # TD error: δ_t = r_t + γ*V_{t+1} - V_t
    delta = rewards[t] + gamma * nextvalues - values[t]

    # GAE: A_t = δ_t + γλ*A_{t+1}
    lastgaelam = delta + gamma * lambd * lastgaelam
    advantages.insert(0, lastgaelam)  # Insert at front to maintain order

    print(f"\nt={t} (token '{tokens[t]}')")
    print(f"  Next value: V[{t+1}] = {nextvalues}")
    print(f"  δ[{t}] = r[{t}] + γ*V[{t+1}] - V[{t}]")
    print(f"       = {rewards[t]} + {gamma}*{nextvalues} - {values[t]}")
    print(f"       = {delta:.3f}")
    print(f"  A[{t}] = δ[{t}] + γλ*A[{t+1}]")
    print(f"       = {delta:.3f} + {gamma}*{lambd}*{lastgaelam:.3f}")
    print(f"       = {lastgaelam:.3f}")

print("\n" + "-" * 40)
print(f"RESULT: advantages = {[f'{a:.3f}' for a in advantages]}")
print()
print("KEY INSIGHT:")
print(f"  • A[0] = {advantages[0]:.3f} > A[1] = {advantages[1]:.3f}")
print("  • First token gets HIGHER advantage even though reward=0!")
print("  • This is GAE propagating reward backward")
print(f"  • A[0] includes λ={lambd} * A[1] = {lambd * advantages[1]:.3f}")

# ============================================================================
# STEP 2: PPO Clipped Loss
# ============================================================================
print("\n" + "=" * 80)
print("STEP 2: PPO Clipped Loss")
print("=" * 80)

old_log_probs = [-2.0, -0.5]  # From rollout
print(f"\nOld policy log P (from rollout): {old_log_probs}")

# Scenario A: Small update
print("\n" + "-" * 40)
print("SCENARIO A: Small policy update")
print("-" * 40)

new_log_probs_a = [-1.5, -0.3]
print(f"New policy log P: {new_log_probs_a}")

ratios_a = [math.exp(new - old) for new, old in zip(new_log_probs_a, old_log_probs)]
print(f"Ratio (π_new/π_old): {[f'{r:.3f}' for r in ratios_a]}")

eps_clip = 0.2
unclipped_loss_a = [-r * a for r, a in zip(ratios_a, advantages)]
clipped_ratios_a = [max(1-eps_clip, min(r, 1+eps_clip)) for r in ratios_a]
clipped_loss_a = [-cr * a for cr, a in zip(clipped_ratios_a, advantages)]
final_loss_a = [max(u, c) for u, c in zip(unclipped_loss_a, clipped_loss_a)]

print(f"\nUnclipped loss: {[f'{l:.3f}' for l in unclipped_loss_a]}")
print(f"Clipped ratios: {[f'{r:.3f}' for r in clipped_ratios_a]}")
print(f"Clipped loss:   {[f'{l:.3f}' for l in clipped_loss_a]}")
print(f"Final (max):    {[f'{l:.3f}' for l in final_loss_a]}")
print("\n✓ Ratios within [0.8, 1.2] → Clipping NOT active")

# Scenario B: Large update
print("\n" + "-" * 40)
print("SCENARIO B: LARGE policy update (dangerous!)")
print("-" * 40)

new_log_probs_b = [0.5, 1.0]  # Much higher!
print(f"New policy log P: {new_log_probs_b}")

ratios_b = [math.exp(new - old) for new, old in zip(new_log_probs_b, old_log_probs)]
print(f"Ratio (π_new/π_old): {[f'{r:.3f}' for r in ratios_b]}")
print("  → Policy is 12x and 4.5x more likely! HUGE change!")

unclipped_loss_b = [-r * a for r, a in zip(ratios_b, advantages)]
clipped_ratios_b = [max(1-eps_clip, min(r, 1+eps_clip)) for r in ratios_b]
clipped_loss_b = [-cr * a for cr, a in zip(clipped_ratios_b, advantages)]
final_loss_b = [max(u, c) for u, c in zip(unclipped_loss_b, clipped_loss_b)]

print(f"\nUnclipped loss: {[f'{l:.3f}' for l in unclipped_loss_b]}")
print(f"Clipped ratios: {[f'{r:.3f}' for r in clipped_ratios_b]}")
print(f"Clipped loss:   {[f'{l:.3f}' for l in clipped_loss_b]}")
print(f"Final (max):    {[f'{l:.3f}' for l in final_loss_b]}")
print("\n✗ Clipping ACTIVE! Ratios clamped to 1.2")
print("  → Prevents catastrophic policy update")

# ============================================================================
# Why backward pass is required
# ============================================================================
print("\n" + "=" * 80)
print("STEP 3: Why GAE Must Be Computed BACKWARD")
print("=" * 80)

print("""
Look at the GAE formula:
  A[t] = δ[t] + γλ * A[t+1]

DEPENDENCY: Computing A[t] requires A[t+1] (future advantage)!

Example from our calculation:
  A[1] = δ[1] + γλ * A[2]
       = 0.200 + 0          (A[2]=0, last token)
       = 0.200

  A[0] = δ[0] + γλ * A[1]
       = -0.300 + 1.0*0.95*0.200
       = -0.300 + 0.190
       = -0.110

You CANNOT compute A[0] without first computing A[1]!

If you tried FORWARD:
  t=0: A[0] = δ[0] + γλ * A[1]  ← What is A[1]?? Doesn't exist yet!

This is why lines 352-356 in ppo_utils.py loop BACKWARD:
  for t in reversed(range(response_len)):
      lastgaelam = delta + gamma * lambd * lastgaelam
""")

# ============================================================================
# Complete training flow
# ============================================================================
print("\n" + "=" * 80)
print("COMPLETE TRAINING STEP SUMMARY")
print("=" * 80)

print("""
1. ROLLOUT (Generation):
   ┌─────────────────────────────────────┐
   │ Old policy generates: "Paris ."     │
   │ Save old_log_probs = [-2.0, -0.5]  │
   │ Get reward = 1.0 (correct!)         │
   └─────────────────────────────────────┘

2. ADVANTAGE (GAE):
   ┌─────────────────────────────────────┐
   │ Critic predicts: V = [0.5, 0.8]    │
   │ Compute δ (TD errors)               │
   │ Backward pass: A[0]=f(A[1])         │
   │ Result: A = [-0.110, 0.200]         │
   └─────────────────────────────────────┘
            ↓ Credit assignment!

3. POLICY LOSS (PPO):
   ┌─────────────────────────────────────┐
   │ New policy forward: new_log_probs   │
   │ ratio = exp(new_log - old_log)      │
   │ loss = -min(r*A, clip(r,0.8,1.2)*A)│
   │ Backprop through actor              │
   └─────────────────────────────────────┘
            ↓ Clipping prevents collapse

4. VALUE LOSS (Critic):
   ┌─────────────────────────────────────┐
   │ returns = advantages + values       │
   │ loss = MSE(critic_output, returns)  │
   │ Backprop through critic             │
   └─────────────────────────────────────┘

5. SYNC WEIGHTS:
   New weights → Rollout workers → Next iteration!
""")

print("=" * 80)
print("Key Takeaways:")
print("  1. GAE propagates rewards backward (credit assignment)")
print("  2. Backward iteration is REQUIRED (temporal dependency)")
print("  3. PPO clipping prevents destructive updates")
print("  4. old_log_probs from rollout are used for importance ratio")
print("=" * 80)
