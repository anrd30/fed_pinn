"""
Triple-Threat Server Implementation for Federated Learning.
=========================================================
Integrates Geometric (PINN), Structural (TDA), and Physical (Hardware) defenses.
"""

import torch
import time
import numpy as np
import os
from typing import List, Tuple, Dict, Any, Optional
from logging import INFO, WARNING

from backfed.servers.fedavg_server import UnweightedFedAvgServer
from backfed.utils.logging_utils import log
from backfed.const import ModelUpdate, client_id, num_examples, Metrics

# Import Triple-Threat Defenses
from backfed.defenses.pinn_guard import PINNGuard, train_adversarial_pinn_guard, get_violation_score
from backfed.defenses.topology_guard import TopologyGuard
from backfed.defenses.hardware_defense import ComputeAsymmetryTrap, ClientTelemetry

class TripleThreatServer(UnweightedFedAvgServer):
    """
    A unified defense server that implements the Triple-Domain Defense Framework:
    1. Geometric (PINN Guard): Logit Manifold Smoothness.
    2. Structural (Topology Guard): Persistent Homology (H1) on Logit Space.
    3. Physical (Hardware Trap): Silicon Timing & Memory Footprint validation.
    """

    def __init__(self, server_config, server_type="triple_threat", **kwargs):
        super(TripleThreatServer, self).__init__(server_config, server_type)
        
        # Initialize PINN Guard
        self.pinn_guard = None
        self.pinn_history = []
        
        # Initialize Topology Guard
        self.topology_guard = TopologyGuard()
        
        # Initialize Hardware Trap
        self.hardware_trap = ComputeAsymmetryTrap()
        
        # Store logits collected during training
        self.current_round_logits = {}
        
        log(INFO, "Initialized Unified Triple-Threat Defense Server.")
        log(INFO, "Domains Active: [Geometric (PINN), Structural (TDA), Physical (Silicon)]")

    def fit_round(self, clients_mapping: Dict[Any, List[int]]) -> Metrics:
        """Override fit_round to collect logits during training."""
        train_time_start = time.time()
        
        # Use a custom trainer method that collects logits
        client_packages = self._train_with_logits_collection(clients_mapping)
        
        train_time_end = time.time()
        train_time = train_time_end - train_time_start
        log(INFO, f"Clients training time: {train_time:.2f} seconds")

        client_metrics = []
        client_updates = []

        for client_id, package in client_packages.items():
            num_examples, model_updates, metrics = package
            client_metrics.append((client_id, num_examples, metrics))
            client_updates.append((client_id, num_examples, model_updates))

        aggregate_time_start = time.time()

        if self.aggregate_client_updates(client_updates):
            aggregated_metrics = self.aggregate_client_metrics(client_metrics)
        else:
            log(WARNING, "No client updates to aggregate. Global model parameters are not updated.")
            aggregated_metrics = {}

        aggregate_time_end = time.time()
        aggregate_time = aggregate_time_end - aggregate_time_start
        log(INFO, f"Server aggregate time: {aggregate_time:.2f} seconds")

        return aggregated_metrics

    def _train_with_logits_collection(self, clients_mapping: Dict[Any, List[int]]) -> Dict[int, Tuple[int, ModelUpdate, Metrics]]:
        """Train clients and collect logits from their trained models."""
        client_packages = {}
        self.current_round_logits = {}
        
        if self.config.training_mode == "parallel":
            raise NotImplementedError("Parallel mode with logit collection not yet implemented. Use sequential mode.")
        else:
            # Sequential mode
            for client_cls in clients_mapping.keys():
                init_args, train_package = self.train_package(client_cls)
                for client_id in clients_mapping[client_cls]:
                    # Train the client
                    train_time, client_package = self.trainer.worker.train(
                        client_cls=client_cls,
                        client_id=client_id,
                        init_args=init_args,
                        train_package=train_package,
                        timeout=self.config.client_timeout
                    )
                    
                    if isinstance(client_package, dict) and client_package.get("status") == "failure":
                        continue
                    
                    # Collect logits before cleanup
                    logits = self.trainer.worker.get_logits(self.mirror_images, self.normalization)
                    self.current_round_logits[client_id] = logits
                    
                    client_packages[client_id] = client_package
        
        return client_packages

    def collect_client_logits(self, selected_clients: List[int]) -> Dict[int, torch.Tensor]:
        """
        Return the logits collected during training.
        """
        return {cid: self.current_round_logits.get(cid, torch.zeros(100, 10)) for cid in selected_clients}

    def _update_pinn_guard(self, benign_logits: torch.Tensor):
        """Pre-train or update the PINN on confirmed benign logits."""
        log(INFO, "Updating PINN Guard on confirmed benign logit manifold...")
        self.pinn_guard, _ = train_adversarial_pinn_guard(
            benign_logits, 
            n_epochs=50, 
            device=self.device, 
            verbose=False
        )

    def aggregate_client_updates(self, client_updates: List[Tuple[client_id, num_examples, ModelUpdate]]) -> bool:
        """
        Overridden aggregation method that filters updates via the Triple-Threat pipeline.
        """
        if len(client_updates) == 0:
            return False

        # 1. Step: Collect Logits (LDFL adaptation)
        client_ids = [cid for cid, _, _ in client_updates]
        log(INFO, f"Round {self.current_round}: Running Triple-Threat Defenses on {len(client_ids)} clients...")
        
        # Collect logits using our LDFL bridge
        all_logits = self.collect_client_logits(client_ids)
        
        # Initialize score dictionaries
        geometric_scores = {cid: 0.0 for cid in client_ids}
        topological_scores = {cid: 0.0 for cid in client_ids}
        
        # 2. Step: Physical Domain (Hardware Defense)
        # In this simulation, we check execution times vs architecture bounds
        # (Assuming client_packages in fit_round would provide timing, 
        # for now we simulate the check score).
        hardware_passed = []
        for cid in client_ids:
            # Create a simulated telemetry packet
            # In a real deployment, these values come from the client's attested hardware
            tel = ClientTelemetry(
                client_id=str(cid),
                declared_device="jetson_nano",
                round_id=self.current_round,
                exec_time_s=8.25 + np.random.normal(0, 0.5), # Simulated honest time
                peak_mem_mb=650.0 + np.random.normal(0, 20),
                logit_vector=all_logits[cid]
            )
            
            pinn_score = geometric_scores.get(cid, 0.0)
            res = self.hardware_trap.validate(tel, pinn_laplacian_score=pinn_score)
            
            if res.recommendation != "REJECT":
                hardware_passed.append(cid)
        
        log(INFO, f"Physical Filter: {len(hardware_passed)}/{len(client_ids)} clients passed hardware bounds.")

        # 3. Step: Geometric Domain (PINN Guard)
        # We need a trained PINN. If not yet trained, we use the first round to bootstrap.
        if self.pinn_guard is not None:
            for cid, logits in all_logits.items():
                score = get_violation_score(self.pinn_guard, logits, device=self.device)
                geometric_scores[cid] = score
        else:
            # Bootstrap logic: In round 1, we might assume first-round is clean or use a baseline
            for cid in client_ids: geometric_scores[cid] = 0.0

        # 4. Step: Structural Domain (Topology Guard)
        for cid, logits in all_logits.items():
            # For CIFAR-10, we expect TDA to see H1 persistence cavities in poisoned updates
            res = self.topology_guard.compute_persistence(logits.numpy())
            # Simple heuristic: max persistence of H1
            if res and 'H1' in res and len(res['H1']) > 0:
                topological_scores[cid] = np.max(res['H1'][:, 1] - res['H1'][:, 0])
            else:
                topological_scores[cid] = 0.0

        # 5. Combined Scoring & Pruning
        # We prune clients that are outliers in ANY of the three domains (Triple-Lock)
        pruned_updates = []
        benign_logits_for_next_pinn = []
        
        # Dynamic Thresholding (Top 80% or similar)
        pinn_vals = np.array(list(geometric_scores.values()))
        tda_vals = np.array(list(topological_scores.values()))
        
        p_threshold = np.percentile(pinn_vals, 80) if len(pinn_vals) > 0 else 1e9
        t_threshold = np.percentile(tda_vals, 80) if len(tda_vals) > 0 else 1e9

        # Log domain statistics
        log(INFO, f"Geometric (PINN) Domain -> Mean: {np.mean(pinn_vals):.4f}, Max: {np.max(pinn_vals):.4f}, Threshold: {p_threshold:.4f}")
        log(INFO, f"Structural (TDA) Domain -> Mean: {np.mean(tda_vals):.4f}, Max: {np.max(tda_vals):.4f}, Threshold: {t_threshold:.4f}")

        for i, (cid, num, updates) in enumerate(client_updates):
            is_benign = True
            
            # Check Hardware
            if cid not in hardware_passed: 
                is_benign = False
                log(WARNING, f"Client {cid} failed Physical Domain (Hardware).")
            
            # Check Geometric
            g_score = geometric_scores.get(cid, 0.0)
            if g_score > p_threshold and self.pinn_guard is not None: 
                is_benign = False
                log(WARNING, f"Client {cid} flagged by PINN Guard (Geom). Score: {g_score:.4f}")
            
            # Check Topological
            t_score = topological_scores.get(cid, 0.0)
            if t_score > t_threshold and t_score > 0.1: # Added min noise floor
                is_benign = False
                log(WARNING, f"Client {cid} flagged by Topology Guard (Struct). Score: {t_score:.4f}")
            
            if is_benign:
                pruned_updates.append((cid, num, updates))
                benign_logits_for_next_pinn.append(all_logits[cid])

        log(INFO, f"Combined Filter: {len(pruned_updates)}/{len(client_ids)} updates accepted.")

        # 6. Update PINN for next round using confirmed benign data
        if benign_logits_for_next_pinn:
            all_benign = torch.cat(benign_logits_for_next_pinn, dim=0)
            self._update_pinn_guard(all_benign[:500]) # Sample for speed

        # 7. Final Aggregation
        return super().aggregate_client_updates(pruned_updates)
