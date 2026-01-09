#!/usr/bin/env python3
"""
GAE (Generalized Advantage Estimation) Explained Step-by-Step

This demonstrates why GAE works and why it must be computed backward.
"""

def compute_gae_verbose(rewards, values, gamma=1.0, lambd=0.95):
    """Compute GAE with detailed explanations."""

    print("=" * 80)
    print("GAE COMPUTATION: STEP-BY-STEP")
    print("=" * 80)

    T = len(rewards)

    # Step 1: Compute TD errors (deltas)
    print("\nSTEP 1: Compute TD Errors (δ)")
    print("-" * 80)
    deltas = []
    for t in range(T):
        next_value = values[t + 1] if t < T - 1 else 0.0
        delta = rewards[t] + gamma * next_value - values[t]
        deltas.append(delta)
        print(f"δ[{t}] = r[{t}] + γ*V[{t+1}] - V[{t}]")
        print(f"     = {rewards[t]:.2f} + {gamma}*{next_value:.2f} - {values[t]:.2f}")
        print(f"     = {delta:.3f}")

    # Step 2: Compute advantages (BACKWARD!)
    print("\nSTEP 2: Compute Advantages (BACKWARD ITERATION)")
    print("-" * 80)
    print(f"Formula: A[t] = δ[t] + γλ * A[t+1]  (where γ={gamma}, λ={lambd})")
    print()

    advantages = [0] * T
    lastgaelam = 0

    for t in reversed(range(T)):
        # GAE recursive formula
        lastgaelam = deltas[t] + gamma * lambd * lastgaelam
        advantages[t] = lastgaelam

        print(f"t={t} (working backward):")
        if t == T - 1:
            print(f"  A[{t}] = δ[{t}] + γλ * 0  (last token, no future)")
            print(f"       = {deltas[t]:.3f} + 0")
        else:
            print(f"  A[{t}] = δ[{t}] + γλ * A[{t+1}]")
            print(f"       = {deltas[t]:.3f} + {gamma}*{lambd}*{advantages[t+1]:.3f}")
        print(f"       = {advantages[t]:.3f}")
        print()

    return advantages

def demonstrate_why_backward():
    """Show why forward iteration doesn't work."""

    print("\n" + "=" * 80)
    print("WHY BACKWARD ITERATION IS REQUIRED")
    print("=" * 80)

    print("""
The GAE formula is:
    A[t] = δ[t] + γλ * A[t+1]

This means:
    • To compute A[0], you need A[1]
    • To compute A[1], you need A[2]
    • To compute A[2], you need A[3]
    • ...
    • To compute A[T-1], you need A[T] (which is 0)

DEPENDENCY CHAIN:
    A[T-1] ← A[T-2] ← A[T-3] ← ... ← A[1] ← A[0]

Start here! ────────────────────────────────────→ End here

If you tried FORWARD iteration:
    for t in range(T):  # 0, 1, 2, ..., T-1
        A[t] = δ[t] + γλ * A[t+1]  # ERROR! A[t+1] doesn't exist yet!

You'd be trying to use A[1] before computing it!

CORRECT (backward iteration):
    for t in reversed(range(T)):  # T-1, T-2, ..., 1, 0
        A[t] = δ[t] + γλ * A[t+1]  # ✓ A[t+1] already computed!
    """)

def demonstrate_credit_assignment():
    """Show how GAE assigns credit backward through time."""

    print("\n" + "=" * 80)
    print("CREDIT ASSIGNMENT: HOW GAE PROPAGATES REWARDS BACKWARD")
    print("=" * 80)

    # Example: Only final token gets reward
    tokens = ["The", "answer", "is", "4"]
    rewards = [0.0, 0.0, 0.0, 1.0]
    values = [0.6, 0.7, 0.8, 0.9]

    print(f"\nTokens:  {tokens}")
    print(f"Rewards: {rewards}")
    print(f"Values:  {values}")
    print()
    print("Notice: Only the final token '4' gets reward=1.0")
    print()

    advantages = compute_gae_verbose(rewards, values, gamma=1.0, lambd=0.95)

    print("\n" + "=" * 80)
    print("RESULTS:")
    print("-" * 80)
    for i, (token, adv) in enumerate(zip(tokens, advantages)):
        print(f"  Token '{token}' (t={i}): A[{i}] = {adv:.3f}")

    print("\n" + "=" * 80)
    print("KEY INSIGHTS:")
    print("=" * 80)
    print(f"""
1. Advantage DECREASES as we go forward in time:
   A[0]={advantages[0]:.3f} > A[1]={advantages[1]:.3f} > A[2]={advantages[2]:.3f} > A[3]={advantages[3]:.3f}

2. Earlier tokens get HIGHER advantages even though their reward=0!
   This is GAE's credit assignment mechanism.

3. Each advantage includes λ-weighted future advantages:
   A[0] = δ[0] + 0.95 * A[1]
        = δ[0] + 0.95 * (δ[1] + 0.95 * A[2])
        = δ[0] + 0.95 * δ[1] + 0.95² * A[2]
        = δ[0] + 0.95 * δ[1] + 0.95² * δ[2] + 0.95³ * δ[3]

   So A[0] is a weighted sum of ALL future TD errors!

4. λ = {0.95} controls exponential decay:
   - Recent rewards: weight = 0.95¹ = 0.950
   - 2 steps away:   weight = 0.95² = 0.903
   - 3 steps away:   weight = 0.95³ = 0.857

   Closer rewards get more weight (better credit assignment).
""")

def compare_lambda_values():
    """Show how λ affects credit assignment."""

    print("\n" + "=" * 80)
    print("HOW λ (LAMBDA) AFFECTS CREDIT ASSIGNMENT")
    print("=" * 80)

    rewards = [0.0, 0.0, 0.0, 1.0]
    values = [0.6, 0.7, 0.8, 0.9]

    print("\nSame scenario, different λ values:")
    print(f"Rewards: {rewards}")
    print(f"Values:  {values}")
    print()

    for lambd in [0.0, 0.5, 0.95, 1.0]:
        deltas = []
        for t in range(len(rewards)):
            next_value = values[t + 1] if t < len(rewards) - 1 else 0.0
            delta = rewards[t] + 1.0 * next_value - values[t]
            deltas.append(delta)

        advantages = [0] * len(rewards)
        lastgaelam = 0
        for t in reversed(range(len(rewards))):
            lastgaelam = deltas[t] + 1.0 * lambd * lastgaelam
            advantages[t] = lastgaelam

        print(f"λ = {lambd:.2f}: A = {[f'{a:.3f}' for a in advantages]}")

    print("""

INTERPRETATION:

λ = 0.00: Only immediate TD errors (no credit propagation)
          A[0] is small because r[0]=0 and we ignore future

λ = 0.50: Moderate credit propagation (50% decay)
          A[0] gets some credit from A[1], but heavily discounted

λ = 0.95: Strong credit propagation (typical choice)
          A[0] gets substantial credit from future rewards

λ = 1.00: Full credit propagation (Monte Carlo)
          A[0] gets maximum credit, but also maximum variance
          (depends entirely on noisy reward signal)

TRADEOFF:
    Low λ → Low variance, high bias (trust value estimates)
    High λ → High variance, low bias (trust actual rewards)
    """)

if __name__ == "__main__":
    demonstrate_credit_assignment()
    demonstrate_why_backward()
    compare_lambda_values()

    print("\n" + "=" * 80)
    print("SUMMARY: GAE IN 4 KEY POINTS")
    print("=" * 80)
    print("""
1. WHAT: GAE computes how much better each action was than expected

2. WHY: Solves credit assignment - earlier tokens get credit for final reward

3. HOW: Recursive formula A[t] = δ[t] + γλ * A[t+1]
        where δ[t] = r[t] + γ*V[t+1] - V[t]

4. BACKWARD: Must iterate backward because A[t] depends on A[t+1]

IN CODE (miles/utils/ppo_utils.py:352-356):

    lastgaelam = 0
    for t in reversed(range(response_len)):
        nextvalues = values[t + 1] if t < response_len - 1 else 0.0
        delta = rewards[t] + gamma * nextvalues - values[t]
        lastgaelam = delta + gamma * lambd * lastgaelam
        advantages[t] = lastgaelam
    """)
    print("=" * 80)
