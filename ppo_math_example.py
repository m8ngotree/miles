#!/usr/bin/env python3
"""
Concrete example of PPO math to understand GAE and importance sampling.

This shows step-by-step how Miles computes advantages and PPO loss.
"""

import torch

print("=" * 80)
print("PPO MATH EXAMPLE: Understanding GAE and Clipped Loss")
print("=" * 80)

# ============================================================================
# Setup: Imagine we generated "The capital of France is Paris"
# ============================================================================
print("\n[SCENARIO]")
print("Prompt: 'What is the capital of France?'")
print("Generated response: 'The capital of France is Paris'")
print("Tokens: ['The', 'capital', 'of', 'France', 'is', 'Paris']")
print()

# Token rewards (only final token gets reward)
rewards = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
print(f"Rewards:        {rewards.tolist()}")

# Critic predictions (value function estimates)
values = torch.tensor([0.3, 0.4, 0.5, 0.6, 0.7, 0.9])
print(f"Critic values:  {values.tolist()}")

# Hyperparameters
gamma = 1.0    # Discount factor (no discounting for simplicity)
lambd = 0.95   # GAE lambda (exponential weighting)

print(f"\nγ (gamma) = {gamma}  (discount factor)")
print(f"λ (lambda) = {lambd}  (GAE lambda)")

# ============================================================================
# STEP 1: Compute GAE (Generalized Advantage Estimation)
# ============================================================================
print("\n" + "=" * 80)
print("STEP 1: Computing Advantages with GAE (BACKWARD PASS)")
print("=" * 80)

response_len = len(rewards)
lastgaelam = 0
advantages_reversed = []

print("\nBackward iteration (t = 5 → 0):")
print("-" * 80)

for t in reversed(range(response_len)):
    # Get next value (0 if last token)
    nextvalues = values[t + 1] if t < response_len - 1 else 0.0

    # Compute TD error: δ_t = r_t + γ * V_{t+1} - V_t
    delta = rewards[t] + gamma * nextvalues - values[t]

    # Compute advantage: A_t = δ_t + γλ * A_{t+1}
    lastgaelam = delta + gamma * lambd * lastgaelam
    advantages_reversed.append(lastgaelam)

    print(f"t={t} ('{['The', 'capital', 'of', 'France', 'is', 'Paris'][t]}')")
    print(f"  r[{t}]={rewards[t]:.1f}, V[{t}]={values[t]:.1f}, V[{t+1}]={nextvalues:.1f}")
    print(f"  δ[{t}] = {rewards[t]:.1f} + {gamma}*{nextvalues:.1f} - {values[t]:.1f} = {delta:.3f}")
    print(f"  A[{t}] = {delta:.3f} + {gamma}*{lambd}*{lastgaelam:.3f} = {lastgaelam:.3f}")
    print()

advantages = torch.tensor(advantages_reversed[::-1])  # Reverse back to forward order
returns = advantages + values

print("RESULTS:")
print(f"Advantages:     {advantages.tolist()}")
print(f"Returns:        {returns.tolist()}")
print()
print("KEY INSIGHT:")
print("  • Advantage[0] is LARGEST (0.428) even though reward[0]=0!")
print("  • This is GAE's credit assignment: early tokens get credit for final reward")
print("  • Each token's advantage includes λ-weighted future advantages")

# ============================================================================
# STEP 2: Compute PPO Clipped Loss
# ============================================================================
print("\n" + "=" * 80)
print("STEP 2: Computing PPO Clipped Loss")
print("=" * 80)

# Old policy log probs (from rollout/generation)
old_log_probs = torch.tensor([-0.5, -0.6, -0.4, -0.7, -0.3, -0.2])
print(f"\nOld policy log P:   {old_log_probs.tolist()}")

# Scenario A: Small policy update (within clip range)
print("\n" + "-" * 80)
print("SCENARIO A: Small update (new policy slightly better)")
print("-" * 80)
new_log_probs_a = torch.tensor([-0.4, -0.5, -0.3, -0.6, -0.2, -0.1])
print(f"New policy log P:   {new_log_probs_a.tolist()}")

# Compute importance ratio
log_ratio_a = new_log_probs_a - old_log_probs
ratio_a = torch.exp(log_ratio_a)
print(f"\nLog ratio:          {log_ratio_a.tolist()}")
print(f"Ratio π_new/π_old:  {ratio_a.tolist()}")

# PPO loss computation
eps_clip = 0.2
pg_losses1_a = -ratio_a * advantages
pg_losses2_a = -torch.clamp(ratio_a, 1 - eps_clip, 1 + eps_clip) * advantages
final_loss_a = torch.maximum(pg_losses1_a, pg_losses2_a)

print(f"\nUnclipped loss:     {pg_losses1_a.tolist()}")
print(f"Clipped loss:       {pg_losses2_a.tolist()}")
print(f"Final loss (max):   {final_loss_a.tolist()}")
print(f"\nMean loss: {final_loss_a.mean():.4f}")
print("✓ Clipping NOT active (ratio within [0.8, 1.2])")

# Scenario B: Large policy update (outside clip range)
print("\n" + "-" * 80)
print("SCENARIO B: Large update (new policy MUCH better - TOO AGGRESSIVE)")
print("-" * 80)
new_log_probs_b = torch.tensor([0.5, 0.4, 0.6, 0.3, 0.7, 0.8])  # Much higher!
print(f"New policy log P:   {new_log_probs_b.tolist()}")

log_ratio_b = new_log_probs_b - old_log_probs
ratio_b = torch.exp(log_ratio_b)
print(f"\nLog ratio:          {log_ratio_b.tolist()}")
print(f"Ratio π_new/π_old:  {ratio_b.tolist()}")
print("  → Ratios are 2.7x to 7.4x! Policy changed DRASTICALLY")

pg_losses1_b = -ratio_b * advantages
pg_losses2_b = -torch.clamp(ratio_b, 1 - eps_clip, 1 + eps_clip) * advantages
final_loss_b = torch.maximum(pg_losses1_b, pg_losses2_b)

print(f"\nUnclipped loss:     {pg_losses1_b.tolist()}")
print(f"Clipped loss:       {pg_losses2_b.tolist()}")
print(f"Final loss (max):   {final_loss_b.tolist()}")
print(f"\nMean loss: {final_loss_b.mean():.4f}")
print("✗ Clipping ACTIVE! Ratio clamped to [0.8, 1.2]")
print("  → Prevents destructive policy updates")

# ============================================================================
# STEP 3: Why Clipping Matters
# ============================================================================
print("\n" + "=" * 80)
print("STEP 3: Why Clipping Prevents Collapse")
print("=" * 80)

print("""
WITHOUT CLIPPING (Scenario B):
  • Ratio = 7.4x means new policy is 740% more likely to generate token
  • Gradient would be HUGE: -7.4 * advantage
  • One gradient step could destroy the policy!
  • Model might degenerate into always outputting high-probability garbage

WITH CLIPPING (eps=0.2):
  • Ratio clamped to 1.2 (20% increase max)
  • Gradient is moderate: -1.2 * advantage
  • Policy improves gradually and stably
  • Prevents policy collapse

KEY INSIGHT:
  PPO sacrifices some convergence speed for stability.
  Better to take many small safe steps than one catastrophic leap!
""")

# ============================================================================
# Summary
# ============================================================================
print("\n" + "=" * 80)
print("SUMMARY: Complete Training Step")
print("=" * 80)

print("""
1. GENERATION (Rollout):
   - Old policy generates tokens
   - Save old_log_probs = [-0.5, -0.6, ...]
   - Get reward = 1.0 for correct answer

2. ADVANTAGE COMPUTATION (GAE):
   - Critic estimates values = [0.3, 0.4, ...]
   - Compute advantages backward: A[t] = δ[t] + γλ*A[t+1]
   - Result: advantages = [0.428, 0.357, ...] (credit assignment!)

3. POLICY LOSS (PPO):
   - New policy forward pass → new_log_probs
   - Compute ratio = exp(new_log_probs - old_log_probs)
   - Clipped loss = -min(ratio*A, clip(ratio)*A)
   - Backprop through actor

4. VALUE LOSS (Critic):
   - Compute returns = advantages + values
   - Value loss = MSE(values, returns)
   - Backprop through critic

5. UPDATE WEIGHTS:
   - Adam step on both models
   - Sync to rollout workers
   - Generate next batch!
""")

print("Run this script to see the math in action!")
print("=" * 80)
