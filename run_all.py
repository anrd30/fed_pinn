"""
Master Run Script
==================
Runs ALL tasks from the peer review in sequence.

Usage:
    python run_all.py [--quick]       # Quick mode (reduced trials)
    python run_all.py [--full]        # Full mode (paper-quality)
    python run_all.py [--cifar10]     # CIFAR-10 experiment only
"""

import argparse
import os
import sys
import time
import torch

def main():
    parser = argparse.ArgumentParser(description="Run all peer review tasks")
    parser.add_argument('--quick', action='store_true', help='Quick mode (fewer trials)')
    parser.add_argument('--full', action='store_true', help='Full mode (paper quality)')
    parser.add_argument('--cifar10', action='store_true', help='CIFAR-10 only')
    parser.add_argument('--device', type=str, default=None, help='Device (cpu/cuda)')
    args = parser.parse_args()
    
    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    output_dir = 'results'
    os.makedirs(output_dir, exist_ok=True)
    
    n_trials = 10 if args.quick else (50 if args.full else 20)
    n_pinn_epochs = 50 if args.quick else (300 if args.full else 150)
    
    print("=" * 70)
    print("  MANIFOLD PHYSICS FL DEFENSE — COMPLETE EVALUATION")
    print(f"  Device: {device}")
    print(f"  Mode: {'quick' if args.quick else ('full' if args.full else 'standard')}")
    print(f"  Trials: {n_trials}")
    print("=" * 70)
    
    start_time = time.time()
    
    if args.cifar10:
        # ── T5 only: CIFAR-10 experiment ──
        from cifar10_federated import run_cifar10_fl_experiment
        
        print("\n" + "=" * 60)
        print("[T5] CIFAR-10 Federated Learning Experiment")
        print("=" * 60)
        
        n_rounds = 5 if args.quick else (50 if args.full else 15)
        n_clients = 10 if args.quick else (30 if args.full else 20)
        
        df = run_cifar10_fl_experiment(
            n_clients=n_clients,
            n_malicious=max(1, n_clients // 10),
            n_rounds=n_rounds,
            device=device,
            output_dir=os.path.join(output_dir, 'cifar10'),
        )
        
    else:
        # ── Full evaluation on synthetic data ──
        from adversarial_pinn_guard import (
            make_clean_logits, make_poisoned_logits,
            train_adversarial_pinn_guard, train_adversarial_l2_filter,
            get_violation_score, run_architecture_sensitivity,
        )
        from fl_baselines import (
            get_all_baseline_defenses, get_all_attack_strategies,
            execute_attack, PINNGuardDefense, L2FilterDefense,
        )
        from evaluation_suite import (
            run_per_strategy_evaluation, plot_per_strategy_roc,
            generate_comparison_table, run_scalability_analysis,
            run_geometry_matters_experiment, run_full_evaluation,
        )
        
        B, C = 100, 10
        
        # ── Run full evaluation (T1, T2, T3, T10, T14) ──
        results_df, summary, geo_df = run_full_evaluation(
            device=device, B=B, C=C, n_trials=n_trials, output_dir=output_dir,
        )
        
        # ── T11: Architecture Sensitivity ──
        if not args.quick:
            print("\n" + "=" * 60)
            print("[T11] Architecture Sensitivity Analysis")
            print("=" * 60)
            
            clean = make_clean_logits(B, C)
            attack_dict = {}
            for atk in ['Naive_Bias', 'RKHS_Evasion', 'Super_Adaptive']:
                attack_dict[atk] = execute_attack(atk, clean)
            
            arch_results = run_architecture_sensitivity(
                clean, attack_dict, device=device,
                n_epochs=n_pinn_epochs,
            )
            
            print("\nArchitecture Sensitivity Results:")
            for name, res in arch_results.items():
                print(f"\n  {name}:")
                print(f"    Clean: {res['clean_violation']:.6f}")
                for atk, v in res['attack_violations'].items():
                    print(f"    {atk}: {v:.6f}")
        
        # ── T13: Scalability ──
        print("\n" + "=" * 60)
        print("[T13] Scalability Analysis")
        print("=" * 60)
        scale_df = run_scalability_analysis(device=device, output_dir=output_dir)
    
    elapsed = time.time() - start_time
    
    print("\n" + "=" * 70)
    print(f"  ALL TASKS COMPLETE")
    print(f"  Total time: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"  Results saved to: {output_dir}/")
    print("=" * 70)


if __name__ == "__main__":
    main()
