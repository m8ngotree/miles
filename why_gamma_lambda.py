#!/usr/bin/env python3
"""
Why do we multiply γ and λ in GAE?

This shows the different roles of gamma and lambda.
"""

print("=" * 80)
print("WHY γ AND λ MULTIPLY IN GAE")
print("=" * 80)

# ============================================================================
# Setup: Simple 4-token sequence
# ============================================================================
tokens = ["The", "answer", "is", "4"]
rewards = [0.0, 0.0, 0.0, 1.0]
values = [0.6, 0.7, 0.8, 0.9]

print("\nScenario:")
print(f"  Tokens: {tokens}")
print(f"  Rewards: {rewards}")
print(f"  Values: {values}")

# ============================================================================
# Compute deltas (same regardless of γ, λ)
# ============================================================================
gamma = 1.0  # For simplicity, use γ=1 first
deltas = []
for t in range(len(rewards)):
    next_value = values[t + 1] if t < len(rewards) - 1 else 0.0
    delta = rewards[t] + gamma * next_value - values[t]
    deltas.append(delta)

print(f"\nTD errors: δ = {[f'{d:.3f}' for d in deltas]}")

# ============================================================================
# EXPERIMENT 1: Different λ values (γ = 1.0)
# ============================================================================
print("\n" + "=" * 80)
print("EXPERIMENT 1: How λ affects advantage (with γ=1.0)")
print("=" * 80)

gamma = 1.0

for lambd in [0.0, 0.5, 0.95, 1.0]:
    advantages = [0] * len(rewards)
    lastgaelam = 0

    for t in reversed(range(len(rewards))):
        lastgaelam = deltas[t] + gamma * lambd * lastgaelam
        advantages[t] = lastgaelam

    print(f"\nλ = {lambd:.2f}:")
    print(f"  A = {[f'{a:.3f}' for a in advantages]}")

    # Show the expansion for A[0]
    print(f"  A[0] expansion:")
    if lambd == 0:
        print(f"    A[0] = δ[0] = {deltas[0]:.3f}")
        print(f"    (Only immediate TD error, ignore future)")
    elif lambd == 1.0:
        print(f"    A[0] = δ[0] + δ[1] + δ[2] + δ[3]")
        total = sum(deltas)
        print(f"         = {' + '.join([f'{d:.3f}' for d in deltas])}")
        print(f"         = {total:.3f}")
        print(f"    (Full sum, no discounting of future TDs)")
    else:
        print(f"    A[0] = δ[0] + {lambd}*δ[1] + {lambd}²*δ[2] + {lambd}³*δ[3]")
        print(f"         = {deltas[0]:.3f} + {lambd * deltas[1]:.3f} + {lambd**2 * deltas[2]:.3f} + {lambd**3 * deltas[3]:.3f}")
        print(f"         = {advantages[0]:.3f}")
        print(f"    (Exponential decay: closer TDs have more weight)")

print("\n" + "-" * 80)
print("INTERPRETATION:")
print("  • λ controls how quickly we decay the weighting of future TD errors")
print("  • λ=0: Only use immediate TD (trust value estimates)")
print("  • λ=1: Use all TDs equally (trust actual rewards)")
print("  • λ=0.95: Exponential weighting (balanced)")

# ============================================================================
# EXPERIMENT 2: Different γ values (λ = 0.95)
# ============================================================================
print("\n\n" + "=" * 80)
print("EXPERIMENT 2: How γ affects advantage (with λ=0.95)")
print("=" * 80)

lambd = 0.95

print("\nNOTE: γ appears in BOTH delta computation AND GAE recursion!")
print()

for gamma in [0.9, 0.99, 1.0]:
    # Recompute deltas with this gamma
    deltas_gamma = []
    for t in range(len(rewards)):
        next_value = values[t + 1] if t < len(rewards) - 1 else 0.0
        delta = rewards[t] + gamma * next_value - values[t]
        deltas_gamma.append(delta)

    # Compute advantages
    advantages = [0] * len(rewards)
    lastgaelam = 0

    for t in reversed(range(len(rewards))):
        lastgaelam = deltas_gamma[t] + gamma * lambd * lastgaelam
        advantages[t] = lastgaelam

    print(f"\nγ = {gamma:.2f}:")
    print(f"  δ (recomputed) = {[f'{d:.3f}' for d in deltas_gamma]}")
    print(f"  A = {[f'{a:.3f}' for a in advantages]}")
    print(f"  A[0] breakdown:")
    print(f"    = δ[0] + (γλ)*δ[1] + (γλ)²*δ[2] + (γλ)³*δ[3]")
    print(f"    = {deltas_gamma[0]:.3f} + ({gamma}*{lambd})*{deltas_gamma[1]:.3f} + ({gamma}*{lambd})²*{deltas_gamma[2]:.3f} + ({gamma}*{lambd})³*{deltas_gamma[3]:.3f}")

    gl = gamma * lambd
    term1 = deltas_gamma[0]
    term2 = gl * deltas_gamma[1]
    term3 = gl**2 * deltas_gamma[2]
    term4 = gl**3 * deltas_gamma[3]
    print(f"    = {term1:.3f} + {term2:.3f} + {term3:.3f} + {term4:.3f}")
    print(f"    = {advantages[0]:.3f}")

print("\n" + "-" * 80)
print("INTERPRETATION:")
print("  • γ discounts future rewards (appears in δ computation)")
print("  • γ also discounts future advantages (appears in GAE recursion)")
print("  • Lower γ → care less about distant future")

# ============================================================================
# EXPERIMENT 3: The compound effect of γλ
# ============================================================================
print("\n\n" + "=" * 80)
print("EXPERIMENT 3: The Compound Effect of γλ")
print("=" * 80)

print("""
In the GAE formula:
    A[t] = δ[t] + γλ * A[t+1]

Both γ and λ serve as discount factors, but for DIFFERENT reasons:

γ (gamma): Temporal discount
  • "How much do we care about future rewards?"
  • Standard in all RL algorithms
  • Accounts for uncertainty increasing with time
  • γ = 0.99 means: reward 10 steps away is worth 0.99^10 = 90.4%

λ (lambda): Bootstrapping discount
  • "How much do we trust value estimates vs. actual rewards?"
  • Specific to GAE
  • Controls bias-variance tradeoff
  • λ = 0.95 means: exponentially decay mixing weights

They MULTIPLY because both discounting effects compound:

  Weight of δ[t+k] in A[t] = (γλ)^k

Example with γ=0.99, λ=0.95:
  γλ = 0.99 * 0.95 = 0.9405

  δ[t]:   weight = (0.9405)^0 = 1.000  (current TD error)
  δ[t+1]: weight = (0.9405)^1 = 0.940  (1 step future)
  δ[t+2]: weight = (0.9405)^2 = 0.885  (2 steps future)
  δ[t+3]: weight = (0.9405)^3 = 0.832  (3 steps future)
  δ[t+10]: weight = (0.9405)^10 = 0.544 (10 steps future)

Both effects compound:
  • Temporally distant (γ discount)
  • AND relies more on bootstrapping (λ discount)
  → DOUBLE discounting!
""")

# ============================================================================
# Visual comparison
# ============================================================================
print("\n" + "=" * 80)
print("VISUAL COMPARISON: Weights at Different Distances")
print("=" * 80)

import math

configs = [
    ("γ=1.0, λ=1.0", 1.0, 1.0, "No discounting at all"),
    ("γ=0.99, λ=1.0", 0.99, 1.0, "Only temporal discount"),
    ("γ=1.0, λ=0.95", 1.0, 0.95, "Only bootstrapping discount"),
    ("γ=0.99, λ=0.95", 0.99, 0.95, "Both discounts (typical)"),
]

print("\nWeight of δ[t+k] in A[t]:")
print(f"\n{'Config':<20} {'k=0':<8} {'k=1':<8} {'k=2':<8} {'k=5':<8} {'k=10':<8}")
print("-" * 80)

for name, gamma, lambd, desc in configs:
    weights = [(gamma * lambd)**k for k in [0, 1, 2, 5, 10]]
    print(f"{name:<20} {weights[0]:.3f}   {weights[1]:.3f}   {weights[2]:.3f}   {weights[3]:.3f}   {weights[4]:.3f}   {desc}")

print("\n" + "-" * 80)
print("KEY INSIGHT:")
print("  With γ=0.99, λ=0.95 (typical PPO settings):")
print("  • δ[t+10] has weight 0.544 (about half)")
print("  • γλ = 0.9405 causes exponential decay")
print("  • This balances:")
print("    - Considering future consequences (credit assignment)")
print("    - Not over-weighting distant, uncertain estimates")

# ============================================================================
# Summary
# ============================================================================
print("\n\n" + "=" * 80)
print("SUMMARY: WHY γλ TOGETHER?")
print("=" * 80)

print("""
1. THEY SERVE DIFFERENT PURPOSES:
   γ: Temporal discounting (standard RL)
   λ: Bias-variance tradeoff (GAE-specific)

2. THEY MULTIPLY BECAUSE EFFECTS COMPOUND:
   A[t] = δ[t] + γλ·δ[t+1] + (γλ)²·δ[t+2] + ...

   Each step forward:
   • Gets discounted by γ (time distance)
   • Gets discounted by λ (bootstrapping)
   • Total discount: (γλ)^k for δ[t+k]

3. IN CODE (miles/utils/ppo_utils.py:355):
   lastgaelam = delta + gamma * lambd * lastgaelam
                        ^^^^^^^^^^^^^^
                        Both multiply!

4. INTUITION:
   "How much should δ[t+k] contribute to A[t]?"
   • Discount by γ^k because it's k steps in future
   • Discount by λ^k because it involves k bootstrapped values
   • Total: (γλ)^k

5. TYPICAL VALUES:
   γ = 0.99 to 1.0   (near-greedy to greedy)
   λ = 0.95 to 0.99  (balanced to low-bias)
   γλ ≈ 0.94 to 0.99 (effective decay rate)
""")

print("=" * 80)
