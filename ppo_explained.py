#!/usr/bin/env python3
"""
PPO (Proximal Policy Optimization) Explained Step-by-Step

This shows why PPO clipping prevents policy collapse.
"""

import math

def explain_importance_sampling():
    """Explain what the importance sampling ratio means."""

    print("=" * 80)
    print("PART 1: IMPORTANCE SAMPLING RATIO")
    print("=" * 80)

    print("""
SCENARIO:
  We generated text with policy π_old (the "rollout" policy)
  We want to train policy π_new (the "training" policy)

PROBLEM:
  Our samples came from π_old, but we're updating π_new!
  The two policies are different - how do we account for this?

SOLUTION: Importance Sampling Ratio

  ratio = π_new(action) / π_old(action)

In log space (what we actually use):

  log_ratio = log π_new - log π_old
  ratio = exp(log_ratio)

INTERPRETATION:
  ratio = 1.0  → New and old policies agree perfectly
  ratio = 1.5  → New policy is 50% MORE likely to take this action
  ratio = 0.5  → New policy is 50% LESS likely to take this action
  ratio = 3.0  → New policy is 200% MORE likely (dangerous!)
    """)

    print("\n" + "-" * 80)
    print("CONCRETE EXAMPLE")
    print("-" * 80)

    # Example token
    old_log_prob = -2.0
    scenarios = [
        ("Small update (good)", -1.8, 0.2),
        ("Medium update (ok)", -1.5, 0.5),
        ("Large update (risky!)", -0.5, 1.5),
        ("Huge update (DANGER!)", 1.0, 3.0),
    ]

    for name, new_log_prob, expected_ratio in scenarios:
        log_ratio = new_log_prob - old_log_prob
        ratio = math.exp(log_ratio)

        print(f"\n{name}:")
        print(f"  Old policy: log P = {old_log_prob:.1f}")
        print(f"  New policy: log P = {new_log_prob:.1f}")
        print(f"  Log ratio: {log_ratio:.1f}")
        print(f"  Ratio: {ratio:.2f}x")

        if ratio > 2.0:
            print(f"  ⚠️  WARNING: Policy changed by {ratio:.1f}x - very risky!")
        elif ratio > 1.5:
            print(f"  ⚠️  Ratio > 1.5x - getting risky")
        else:
            print(f"  ✓ Safe update")

def explain_clipping():
    """Explain how clipping works."""

    print("\n\n" + "=" * 80)
    print("PART 2: PPO CLIPPING MECHANISM")
    print("=" * 80)

    print("""
STANDARD POLICY GRADIENT (no clipping):
  loss = -ratio * advantage

PROBLEM:
  If ratio is huge (e.g., 10x), gradient is HUGE!
  One step could destroy your policy!

PPO SOLUTION:
  Clip the ratio to [1-ε, 1+ε] where ε = 0.2 (typically)

  unclipped_loss = -ratio * advantage
  clipped_loss = -clip(ratio, 0.8, 1.2) * advantage

  final_loss = max(unclipped_loss, clipped_loss)

WHY MAX?
  We want to be CONSERVATIVE (pessimistic).
  Taking max gives us the LESS NEGATIVE loss (smaller gradient).
    """)

    print("\n" + "-" * 80)
    print("VISUAL DEMONSTRATION")
    print("-" * 80)

    eps = 0.2
    advantage = 1.0  # Positive advantage (good action)

    test_ratios = [0.5, 0.8, 1.0, 1.2, 1.5, 3.0, 10.0]

    print(f"\nAdvantage = {advantage} (positive - we want to reinforce this)")
    print(f"Clip range: [{1-eps}, {1+eps}] = [0.8, 1.2]")
    print()

    print(f"{'Ratio':<8} {'Clipped':<10} {'Unclipped':<12} {'Clipped':<12} {'Final':<12} {'Clipping?':<10}")
    print(f"{'r':<8} {'clip(r)':<10} {'-r*A':<12} {'-clip(r)*A':<12} {'max':<12} {'':<10}")
    print("-" * 80)

    for ratio in test_ratios:
        clipped_ratio = max(1-eps, min(ratio, 1+eps))
        unclipped_loss = -ratio * advantage
        clipped_loss = -clipped_ratio * advantage
        final_loss = max(unclipped_loss, clipped_loss)
        is_clipped = (ratio < 1-eps or ratio > 1+eps)

        print(f"{ratio:<8.2f} {clipped_ratio:<10.2f} {unclipped_loss:<12.3f} "
              f"{clipped_loss:<12.3f} {final_loss:<12.3f} {'✗ CLIPPED' if is_clipped else '✓ not clipped'}")

    print("\nOBSERVATIONS:")
    print("  • When ratio ∈ [0.8, 1.2]: No clipping, use unclipped loss")
    print("  • When ratio > 1.2: Clipped! Final loss = -1.2 (capped)")
    print("  • When ratio = 10.0: Would be -10.0, but clipped to -1.2")
    print("  • Gradient is proportional to loss, so clipping limits gradient magnitude")

def explain_different_advantage_signs():
    """Show how clipping behaves with positive/negative advantages."""

    print("\n\n" + "=" * 80)
    print("PART 3: CLIPPING WITH DIFFERENT ADVANTAGE SIGNS")
    print("=" * 80)

    eps = 0.2

    print("\n" + "-" * 80)
    print("SCENARIO A: Positive Advantage (Good Action)")
    print("-" * 80)
    print("We want to INCREASE probability of this action\n")

    advantage = 1.0
    test_ratios = [0.5, 1.0, 1.5, 3.0]

    for ratio in test_ratios:
        clipped_ratio = max(1-eps, min(ratio, 1+eps))
        unclipped_loss = -ratio * advantage
        clipped_loss = -clipped_ratio * advantage
        final_loss = max(unclipped_loss, clipped_loss)

        print(f"Ratio = {ratio:.1f}:")
        print(f"  Unclipped: {unclipped_loss:.2f}, Clipped: {clipped_loss:.2f}, Final: {final_loss:.2f}")

        if ratio > 1 + eps:
            print(f"  → Policy already increased too much! Clip to {1+eps}")
        elif ratio < 1 - eps:
            print(f"  → Policy decreased too much! Clip to {1-eps}")
        else:
            print(f"  → Safe update range, no clipping")
        print()

    print("\n" + "-" * 80)
    print("SCENARIO B: Negative Advantage (Bad Action)")
    print("-" * 80)
    print("We want to DECREASE probability of this action\n")

    advantage = -1.0

    for ratio in test_ratios:
        clipped_ratio = max(1-eps, min(ratio, 1+eps))
        unclipped_loss = -ratio * advantage
        clipped_loss = -clipped_ratio * advantage
        final_loss = max(unclipped_loss, clipped_loss)

        print(f"Ratio = {ratio:.1f}:")
        print(f"  Unclipped: {unclipped_loss:.2f}, Clipped: {clipped_loss:.2f}, Final: {final_loss:.2f}")

        if ratio < 1 - eps:
            print(f"  → Policy already decreased too much! Clip to {1-eps}")
        elif ratio > 1 + eps:
            print(f"  → Policy increased too much (bad action!)! Clip to {1+eps}")
        else:
            print(f"  → Safe update range, no clipping")
        print()

def demonstrate_policy_collapse():
    """Show what happens without clipping."""

    print("\n\n" + "=" * 80)
    print("PART 4: WHY CLIPPING PREVENTS POLICY COLLAPSE")
    print("=" * 80)

    print("""
SCENARIO: Model generates correct answer to "What is 2+2?"

WITHOUT CLIPPING:
  Iteration 1:
    Generated: "The answer is 4"
    Log prob: -2.0, Reward: 1.0, Advantage: 0.8
    Loss: -exp(-2.0 - -2.0) * 0.8 = -1.0 * 0.8 = -0.8
    Gradient: moderate

  Iteration 2:
    Log prob increased to: -1.0 (due to gradient)
    Ratio: exp(-1.0 - -2.0) = exp(1.0) = 2.7x
    Loss: -2.7 * 0.8 = -2.16
    Gradient: LARGE

  Iteration 3:
    Log prob increased to: 0.5
    Ratio: exp(0.5 - -2.0) = exp(2.5) = 12.2x
    Loss: -12.2 * 0.8 = -9.76
    Gradient: HUGE! Policy explodes!

  Iteration 4:
    Model outputs: "4 4 4 4 4 4 4 4 4 4 4 4 4 4..."
    Model forgot how to generate coherent text!
    POLICY COLLAPSE!

WITH CLIPPING (ε = 0.2):
  Iteration 1:
    Same as above: Loss = -0.8

  Iteration 2:
    Log prob: -1.0, Ratio: 2.7x
    Unclipped loss: -2.16
    Clipped loss: -1.2 * 0.8 = -0.96
    Final loss: max(-2.16, -0.96) = -0.96
    Gradient: MODERATE (capped at 1.2x)

  Iteration 3:
    Log prob: -0.5, Ratio: 4.5x
    Still clipped to: -1.2 * 0.8 = -0.96
    Gradient: STILL MODERATE

  Iteration 10:
    Gradual improvement, policy stays stable!
    ✓ No collapse!

KEY INSIGHT:
  Clipping trades convergence speed for stability.
  Better to take many small safe steps than one catastrophic leap!
    """)

def explain_code_connection():
    """Connect to actual Miles code."""

    print("\n\n" + "=" * 80)
    print("PART 5: CONNECTION TO MILES CODE")
    print("=" * 80)

    print("""
In miles/utils/ppo_utils.py:125-148:

    def compute_policy_loss(ppo_kl, advantages, eps_clip, eps_clip_high):
        # Line 132: Compute importance ratio
        ratio = (-ppo_kl).exp()
        # Note: ppo_kl = old_log_prob - new_log_prob
        # So -ppo_kl = new_log_prob - old_log_prob
        # ratio = exp(new_log_prob - old_log_prob) = π_new/π_old ✓

        # Line 133-134: Compute both losses
        pg_losses1 = -ratio * advantages  # Unclipped
        pg_losses2 = -ratio.clamp(1 - eps_clip, 1 + eps_clip_high) * advantages  # Clipped

        # Line 135: Take maximum (conservative/pessimistic)
        clip_pg_losses1 = torch.maximum(pg_losses1, pg_losses2)

        return clip_pg_losses1

PARAMETERS:
  • eps_clip = 0.2 (typical value)
  • eps_clip_high = 0.2 (symmetric clipping)
  • Clip range: [0.8, 1.2]

FLOW:
  1. Rollout generates text with old policy
     → Save old_log_probs in sample.rollout_log_probs

  2. Training forward pass with new policy
     → Compute new_log_probs

  3. Compute ppo_kl = old_log_probs - new_log_probs
     (This is stored for KL divergence monitoring)

  4. Compute ratio = exp(-ppo_kl) = exp(new - old)

  5. Compute clipped loss

  6. Backpropagate and update policy

  7. Repeat!
    """)

def summary():
    """Final summary."""

    print("\n\n" + "=" * 80)
    print("SUMMARY: PPO IN 5 KEY POINTS")
    print("=" * 80)

    print("""
1. PROBLEM: Policy gradients can be unstable (policy collapse)

2. SOLUTION: Clip the importance sampling ratio to safe range [0.8, 1.2]

3. RATIO: r = π_new/π_old measures how much policy changed

4. CLIPPING: loss = max(-r*A, -clip(r)*A)
   - Takes maximum → less negative → smaller gradient → safer update

5. TRADEOFF: Slower convergence for much better stability

FORMULAS:
  ratio = exp(new_log_prob - old_log_prob)
  unclipped = -ratio * advantage
  clipped = -clip(ratio, 0.8, 1.2) * advantage
  final = max(unclipped, clipped)  ← Conservative!

WHY IT WORKS:
  • Limits how much policy can change per update
  • Prevents catastrophic gradient steps
  • Keeps policy improving gradually
  • Much more stable than vanilla policy gradient
    """)

if __name__ == "__main__":
    explain_importance_sampling()
    explain_clipping()
    explain_different_advantage_signs()
    demonstrate_policy_collapse()
    explain_code_connection()
    summary()

    print("\n" + "=" * 80)
    print("Next: Read miles/utils/ppo_utils.py:125-148")
    print("You'll now understand exactly what every line does!")
    print("=" * 80)
