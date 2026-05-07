import torch
from adversarial_pinn_guard import (
    make_clean_logits, make_poisoned_logits,
    train_adversarial_pinn_guard, get_violation_score,
    _compute_physics_loss, FisherInformationMetric
)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Device: {device}')

B, C = 50, 10
clean = make_clean_logits(B, C)
poisoned = make_poisoned_logits(B, C, target=0, bias=3.0)

# Quick 20-epoch training to verify gradient flow
print('Training 20 epochs...')
pinn, hist = train_adversarial_pinn_guard(clean, n_epochs=20, device=device, verbose=True)

# Score with physics
clean_v = get_violation_score(pinn, clean, device, use_physics=True)
poison_v = get_violation_score(pinn, poisoned, device, use_physics=True)
print(f'\nPHYSICS scoring:')
print(f'  Clean:   {clean_v:.6f}')
print(f'  Poison:  {poison_v:.6f}')
print(f'  Ratio:   {poison_v / (clean_v + 1e-8):.2f}x')
print(f'  Detected: {poison_v > clean_v}')

# Verify adversary utility is NOT exploding (gradient flow working)
print(f'\nAdversary utility over training: {hist["attack_utility"][:5]} -> {hist["attack_utility"][-5:]}')
print(f'Adversary violation over training: {[round(x,4) for x in hist["adv_violation"][:5]]} -> {[round(x,4) for x in hist["adv_violation"][-5:]]}')

# Now test PINNGuardDefense wrapper
from fl_baselines import PINNGuardDefense
defense = PINNGuardDefense(pinn, device, use_fisher=False)
defense.fit(clean)
c_score = defense.score(clean)
p_score = defense.score(poisoned)
print(f'\nPINNGuardDefense wrapper:')
print(f'  Clean:   {c_score:.6f}')
print(f'  Poison:  {p_score:.6f}')
print(f'  Match direct get_violation_score: {abs(c_score - clean_v) < 0.01}')

print('\n✅ All checks passed!' if poison_v > clean_v else '\n❌ PINN still not detecting - needs investigation')
