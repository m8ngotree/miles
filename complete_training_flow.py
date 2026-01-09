#!/usr/bin/env python3
"""
COMPLETE PPO TRAINING FLOW: From Generation to Gradients

This demonstrates the entire pipeline with concrete numbers.
"""

import math

class Model:
    """Simplified model for demonstration."""

    def __init__(self, name, initial_log_probs):
        self.name = name
        self.log_probs = initial_log_probs.copy()

    def forward(self, tokens):
        """Return log probabilities for tokens."""
        return self.log_probs

    def update(self, gradients):
        """Update log probs based on gradients."""
        # Simple gradient descent: log_prob -= learning_rate * gradient
        lr = 0.1
        self.log_probs = [lp - lr * g for lp, g in zip(self.log_probs, gradients)]

def complete_training_iteration():
    """Demonstrate one complete training iteration."""

    print("=" * 80)
    print("COMPLETE PPO TRAINING ITERATION")
    print("=" * 80)

    # ===========================================================================
    # STEP 1: ROLLOUT (Generation with old policy)
    # ===========================================================================
    print("\n" + "=" * 80)
    print("STEP 1: ROLLOUT - Generate text with old policy")
    print("=" * 80)

    tokens = ["The", "answer", "is", "4"]
    old_policy = Model("old_policy", [-2.0, -1.8, -1.5, -1.0])

    old_log_probs = old_policy.forward(tokens)

    print(f"\nPrompt: 'What is 2+2?'")
    print(f"Generated tokens: {tokens}")
    print(f"Old policy log probs: {old_log_probs}")
    print(f"\n→ Save these to sample.rollout_log_probs")

    # ===========================================================================
    # STEP 2: REWARD EVALUATION
    # ===========================================================================
    print("\n" + "=" * 80)
    print("STEP 2: REWARD EVALUATION")
    print("=" * 80)

    # Sparse reward: only final token gets reward
    rewards = [0.0, 0.0, 0.0, 1.0]

    print(f"\nReward model evaluates answer...")
    print(f"Rewards: {rewards}")
    print(f"Interpretation: Only final token '4' gets reward (correct answer!)")

    # ===========================================================================
    # STEP 3: CRITIC FORWARD PASS (Value prediction)
    # ===========================================================================
    print("\n" + "=" * 80)
    print("STEP 3: CRITIC FORWARD PASS - Predict future rewards")
    print("=" * 80)

    # Critic tries to predict expected future reward from each state
    critic_values = [0.6, 0.7, 0.8, 0.9]

    print(f"\nCritic predictions (values): {critic_values}")
    print(f"Interpretation:")
    print(f"  V[0] = 0.6: From 'The', expect total reward ~0.6")
    print(f"  V[1] = 0.7: From 'answer', expect total reward ~0.7")
    print(f"  V[2] = 0.8: From 'is', expect total reward ~0.8")
    print(f"  V[3] = 0.9: From '4', expect total reward ~0.9")

    # ===========================================================================
    # STEP 4: GAE - Compute advantages
    # ===========================================================================
    print("\n" + "=" * 80)
    print("STEP 4: GAE - Compute advantages (CREDIT ASSIGNMENT)")
    print("=" * 80)

    gamma = 1.0
    lambd = 0.95

    print(f"\nHyperparameters: γ={gamma}, λ={lambd}")
    print(f"\nComputing TD errors (δ):")

    # Compute deltas
    deltas = []
    for t in range(len(rewards)):
        next_value = critic_values[t + 1] if t < len(rewards) - 1 else 0.0
        delta = rewards[t] + gamma * next_value - critic_values[t]
        deltas.append(delta)
        print(f"  δ[{t}] = {rewards[t]} + {gamma}*{next_value} - {critic_values[t]} = {delta:.3f}")

    print(f"\nComputing advantages (BACKWARD):")

    # Compute advantages backward
    advantages = [0] * len(rewards)
    lastgaelam = 0

    for t in reversed(range(len(rewards))):
        lastgaelam = deltas[t] + gamma * lambd * lastgaelam
        advantages[t] = lastgaelam

        if t == len(rewards) - 1:
            print(f"  A[{t}] = δ[{t}] = {deltas[t]:.3f}")
        else:
            print(f"  A[{t}] = δ[{t}] + γλ*A[{t+1}] = {deltas[t]:.3f} + {gamma*lambd:.2f}*{advantages[t+1]:.3f} = {advantages[t]:.3f}")

    print(f"\nFinal advantages: {[f'{a:.3f}' for a in advantages]}")
    print(f"\n✓ Credit assignment: Earlier tokens get higher advantages!")

    # Compute returns for critic training
    returns = [a + v for a, v in zip(advantages, critic_values)]
    print(f"Returns (for critic): {[f'{r:.3f}' for r in returns]}")

    # ===========================================================================
    # STEP 5: ACTOR FORWARD PASS (New policy)
    # ===========================================================================
    print("\n" + "=" * 80)
    print("STEP 5: ACTOR FORWARD PASS - New policy (after some updates)")
    print("=" * 80)

    # New policy has changed slightly from old policy
    new_policy = Model("new_policy", [-1.5, -1.5, -1.2, -0.8])

    new_log_probs = new_policy.forward(tokens)

    print(f"\nOld policy log probs: {old_log_probs}")
    print(f"New policy log probs: {new_log_probs}")

    # ===========================================================================
    # STEP 6: COMPUTE PPO LOSS
    # ===========================================================================
    print("\n" + "=" * 80)
    print("STEP 6: COMPUTE PPO CLIPPED LOSS")
    print("=" * 80)

    eps_clip = 0.2

    print(f"\nClip range: [{1-eps_clip}, {1+eps_clip}] = [0.8, 1.2]")
    print()

    ppo_losses = []
    unclipped_losses = []
    clipped_losses = []
    ratios = []

    for t in range(len(tokens)):
        # Compute importance ratio
        log_ratio = new_log_probs[t] - old_log_probs[t]
        ratio = math.exp(log_ratio)
        ratios.append(ratio)

        # Compute losses
        unclipped_loss = -ratio * advantages[t]
        clipped_ratio = max(1-eps_clip, min(ratio, 1+eps_clip))
        clipped_loss = -clipped_ratio * advantages[t]
        final_loss = max(unclipped_loss, clipped_loss)

        ppo_losses.append(final_loss)
        unclipped_losses.append(unclipped_loss)
        clipped_losses.append(clipped_loss)

        clipping_status = "CLIPPED" if abs(ratio - clipped_ratio) > 1e-6 else "not clipped"

        print(f"Token '{tokens[t]}' (t={t}):")
        print(f"  Ratio: {ratio:.3f}, Advantage: {advantages[t]:.3f}")
        print(f"  Unclipped: {unclipped_loss:.3f}, Clipped: {clipped_loss:.3f}")
        print(f"  Final: {final_loss:.3f} [{clipping_status}]")

    total_ppo_loss = sum(ppo_losses) / len(ppo_losses)
    print(f"\nTotal PPO loss (mean): {total_ppo_loss:.3f}")

    # ===========================================================================
    # STEP 7: COMPUTE VALUE LOSS
    # ===========================================================================
    print("\n" + "=" * 80)
    print("STEP 7: COMPUTE VALUE LOSS (Critic training)")
    print("=" * 80)

    # MSE loss between predictions and returns
    value_loss = sum((v - r)**2 for v, r in zip(critic_values, returns)) / len(critic_values)

    print(f"\nCritic predictions: {critic_values}")
    print(f"Target returns:     {[f'{r:.3f}' for r in returns]}")
    print(f"Value loss (MSE):   {value_loss:.3f}")
    print(f"\n→ Critic learns to better predict future rewards")

    # ===========================================================================
    # STEP 8: BACKPROPAGATION (Conceptual)
    # ===========================================================================
    print("\n" + "=" * 80)
    print("STEP 8: BACKPROPAGATION & WEIGHT UPDATE")
    print("=" * 80)

    print("""
Gradients flow backward through:

ACTOR:
  PPO Loss → Model logits → Model weights
  • Increases probability of tokens with positive advantages
  • Decreases probability of tokens with negative advantages
  • Clipping prevents too-large updates

CRITIC:
  Value Loss → Value predictions → Critic weights
  • Learns to predict actual returns
  • Better value estimates → better advantages next iteration

OPTIMIZER STEP:
  weights ← weights - learning_rate * gradients
    """)

    # ===========================================================================
    # STEP 9: WEIGHT SYNC
    # ===========================================================================
    print("\n" + "=" * 80)
    print("STEP 9: SYNC WEIGHTS TO ROLLOUT")
    print("=" * 80)

    print("""
After training update:
  1. Updated actor weights → Rollout workers (SGLang engines)
  2. Now generation uses NEW policy
  3. Generate next batch of samples
  4. Repeat from STEP 1!

This is the ON-POLICY requirement:
  • Samples generated with policy π_old
  • Used to train π_new
  • π_new becomes π_old for next iteration
  • Must sync weights frequently!
    """)

    # ===========================================================================
    # SUMMARY
    # ===========================================================================
    print("\n" + "=" * 80)
    print("COMPLETE FLOW SUMMARY")
    print("=" * 80)

    print(f"""
ITERATION SUMMARY:

1. ROLLOUT: Generated "{' '.join(tokens)}" with old policy
   → old_log_probs = {old_log_probs}

2. REWARD: Evaluated answer → rewards = {rewards}

3. CRITIC: Predicted future rewards → values = {critic_values}

4. GAE: Computed advantages (credit assignment)
   → advantages = {[f'{a:.3f}' for a in advantages]}
   → Earlier tokens get more credit!

5. ACTOR: Forward pass with new policy
   → new_log_probs = {new_log_probs}

6. PPO LOSS: Computed clipped loss
   → Ratio = π_new/π_old = {[f'{r:.3f}' for r in ratios]}
   → Loss = {total_ppo_loss:.3f}

7. VALUE LOSS: Critic training target
   → Loss = {value_loss:.3f}

8. BACKPROP: Gradients update weights

9. SYNC: New weights → Rollout workers

10. REPEAT: Generate next batch!

KEY INSIGHTS:
  • GAE assigns credit backward through time
  • PPO clipping prevents policy collapse
  • Value function helps estimate advantages
  • On-policy: must sync weights frequently
    """)

def explain_key_variables():
    """Explain what each key variable means."""

    print("\n\n" + "=" * 80)
    print("KEY VARIABLES EXPLAINED")
    print("=" * 80)

    print("""
SAVED DURING ROLLOUT (in Sample object):
  • rollout_log_probs: Log P(token | old_policy)
    → Used to compute importance ratio
    → Saved in miles/utils/types.py:25

  • rewards: Scalar reward from reward model
    → Used in GAE computation
    → Saved in miles/utils/types.py:22

  • loss_mask: Which tokens contribute to loss
    → Binary mask [0, 0, 1, 1] for selective training
    → Saved in miles/utils/types.py:23

COMPUTED DURING TRAINING:
  • values: Critic predictions V(state)
    → Used in GAE: δ = r + γV[t+1] - V[t]
    → Computed in actor.py forward pass

  • deltas (δ): TD errors
    → δ[t] = r[t] + γ*V[t+1] - V[t]
    → Used to build advantages
    → Computed in ppo_utils.py:354

  • advantages (A): How good was this action?
    → A[t] = δ[t] + γλ*A[t+1]
    → Used in PPO loss: -ratio * A
    → Computed in ppo_utils.py:352-356

  • returns (G): Targets for critic
    → G[t] = A[t] + V[t]
    → Used in value loss: MSE(V, G)
    → Computed in ppo_utils.py:358

  • new_log_probs: Log P(token | new_policy)
    → Computed during training forward pass
    → Used to compute ratio

  • ratio: Importance sampling
    → ratio = exp(new_log_probs - old_log_probs)
    → Measures policy change
    → Computed in ppo_utils.py:132

  • ppo_loss: Clipped policy gradient loss
    → loss = max(-r*A, -clip(r)*A)
    → Used for actor backprop
    → Computed in ppo_utils.py:125-148

  • value_loss: Critic training loss
    → loss = MSE(V, returns)
    → Used for critic backprop
    → Computed in actor training code
    """)

def explain_where_in_code():
    """Map concepts to actual code locations."""

    print("\n\n" + "=" * 80)
    print("WHERE IN THE CODE")
    print("=" * 80)

    print("""
SAMPLE CREATION (miles/rollout/sglang_rollout.py):
  • Line 90-119: generate() - Generate tokens with old policy
  • Line 145-199: Process response, save rollout_log_probs
  • Line 202-244: generate_and_rm() - Compute rewards

GAE COMPUTATION (miles/utils/ppo_utils.py):
  • Line 352-356: GAE backward loop
  • Line 354: delta = rewards[t] + gamma * nextvalues - values[t]
  • Line 355: lastgaelam = delta + gamma * lambd * lastgaelam
  • Line 358: returns = advantages + values

PPO LOSS (miles/utils/ppo_utils.py):
  • Line 132: ratio = (-ppo_kl).exp()
  • Line 133-134: Unclipped and clipped losses
  • Line 135: final_loss = max(unclipped, clipped)

TRAINING (miles/backends/fsdp_utils/actor.py or megatron_utils/):
  • Actor forward pass: Get new_log_probs
  • Call ppo_utils functions to compute losses
  • Backprop and optimizer.step()
  • Sync weights to rollout workers

COMPLETE LOOP (train.py or train_async.py):
  • Rollout: Generate samples
  • Buffer: Store samples
  • Training: Compute losses and update
  • Sync: Send weights to rollout
  • Repeat!
    """)

if __name__ == "__main__":
    complete_training_iteration()
    explain_key_variables()
    explain_where_in_code()

    print("\n" + "=" * 80)
    print("🎓 YOU NOW UNDERSTAND THE COMPLETE FLOW!")
    print("=" * 80)
    print("""
Next steps:
  1. Re-read miles/utils/ppo_utils.py with this understanding
  2. Trace through miles/backends/fsdp_utils/actor.py
  3. See how train.py orchestrates everything

You're ready to dive deep into the actual implementation!
    """)
