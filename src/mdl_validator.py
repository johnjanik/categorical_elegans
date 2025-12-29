#!/usr/bin/env python3
"""
MDL (Minimum Description Length) Validator
===========================================

Implements the MDL scoring framework from minimal_model_theory.tex
for comparing neural network models against real behavioral data.

The MDL score is:
    L(M) = L_struct(M) + L_param(M|D)

Where:
    L_struct = structural complexity (neurons, synapses)
    L_param = parameter/data fit error

A minimal model minimizes L(M) while maintaining behavioral accuracy.
"""

import math
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple, Union
import numpy as np

from .c_elegans_simulation import ObservableVector, CElegansSimulation
from .c_elegans_connectome_loader import ConnectomeData
from .behavioral_data_loader import RealBehaviorDataset, N2_REFERENCE_OBSERVABLES


# =============================================================================
# MDL Score Components
# =============================================================================

@dataclass
class MDLScore:
    """
    Complete MDL score for a model.

    L(M) = L_struct + L_param

    Lower is better (more parsimonious model with good fit).
    """
    # Structural complexity
    L_struct: float = 0.0

    # Breakdown of structural complexity
    L_neurons: float = 0.0      # log(|neurons|)
    L_synapses: float = 0.0     # log(|synapses|)
    L_gap_junctions: float = 0.0

    # Parameter/fit complexity
    L_param: float = 0.0

    # Observable-wise errors (for diagnostics)
    observable_errors: Dict[str, float] = field(default_factory=dict)

    # Behavioral equivalence check
    is_behaviorally_equivalent: bool = False
    equivalence_epsilon: float = 0.1

    @property
    def total(self) -> float:
        """Total MDL score."""
        return self.L_struct + self.L_param

    @property
    def normalized(self) -> float:
        """Normalized score (0-1 range, lower is better)."""
        # Normalize by typical ranges
        struct_norm = self.L_struct / 20.0  # log(302) + log(7000) ≈ 15
        param_norm = self.L_param / 100.0   # Typical error range
        return (struct_norm + param_norm) / 2.0

    def __repr__(self) -> str:
        return (
            f"MDLScore(\n"
            f"  L_struct={self.L_struct:.3f} "
            f"(neurons={self.L_neurons:.2f}, synapses={self.L_synapses:.2f})\n"
            f"  L_param={self.L_param:.3f}\n"
            f"  Total={self.total:.3f}\n"
            f"  Behaviorally equivalent: {self.is_behaviorally_equivalent}\n"
            f")"
        )


# =============================================================================
# MDL Validator
# =============================================================================

class MDLValidator:
    """
    Validate neural network models using MDL scoring against real data.

    Implements the theoretical framework from Section 5 of minimal_model_theory.tex.
    """

    def __init__(self,
                 real_data: Optional[RealBehaviorDataset] = None,
                 reference_observable: Optional[ObservableVector] = None,
                 epsilon: float = 0.1):
        """
        Initialize MDL validator.

        Args:
            real_data: Real behavioral dataset for comparison
            reference_observable: Reference observable (if no real data)
            epsilon: Threshold for behavioral equivalence
        """
        self.real_data = real_data
        self.epsilon = epsilon

        # Get reference observable
        if real_data is not None:
            self.reference_observable = real_data.get_observable_vector()
            self.reference_std = real_data.get_observable_std()
        elif reference_observable is not None:
            self.reference_observable = reference_observable
            self.reference_std = None
        else:
            # Use literature values for N2
            self.reference_observable = N2_REFERENCE_OBSERVABLES
            self.reference_std = None

        # Weights for different observables (importance)
        self.observable_weights = self._default_weights()

    def _default_weights(self) -> Dict[str, float]:
        """Default weights for observables in MDL scoring."""
        return {
            'velocity': 1.0,
            'angular_velocity': 0.5,
            'reversal_rate': 2.0,          # Important for touch response
            'run_length': 1.0,
            'speed_mean': 1.0,
            'speed_variance': 0.5,
            'turn_angle_mean': 0.5,
            'turn_angle_variance': 0.3,
            'omega_turn_rate': 1.5,
            'chemotaxis_index': 1.0,
            'anterior_touch_prob': 2.0,    # Key behavioral output
            'posterior_touch_prob': 2.0,   # Key behavioral output
            'response_latency': 1.0,
            'pharyngeal_pumping': 0.5,
        }

    def compute_structural_complexity(self,
                                       connectome: ConnectomeData) -> Tuple[float, float, float]:
        """
        Compute L_struct from connectome structure.

        Uses log coding: L_struct = log(|N|) + log(|S|) + log(|G|)

        Args:
            connectome: Connectome data

        Returns:
            (L_neurons, L_synapses, L_gap_junctions)
        """
        n_neurons = len(connectome.neurons)
        n_synapses = len(connectome.synapses)
        n_gap_junctions = len(connectome.gap_junctions)

        # Use log2 for information-theoretic interpretation
        L_neurons = math.log2(max(1, n_neurons))
        L_synapses = math.log2(max(1, n_synapses))
        L_gap = math.log2(max(1, n_gap_junctions))

        return L_neurons, L_synapses, L_gap

    def compute_parameter_error(self,
                                 simulated: ObservableVector,
                                 normalize: bool = True) -> Tuple[float, Dict[str, float]]:
        """
        Compute L_param as weighted sum of squared errors.

        L_param = Σ_i w_i * (sim_i - ref_i)² / σ_i²

        Args:
            simulated: Observable vector from simulation
            normalize: Whether to normalize by reference variance

        Returns:
            (total_error, per_observable_errors)
        """
        sim_array = simulated.to_array()
        ref_array = self.reference_observable.to_array()

        # Get variances for normalization
        if self.reference_std is not None:
            std_array = self.reference_std.to_array()
            var_array = std_array ** 2 + 0.01  # Add small constant to avoid division by zero
        else:
            # Use reference values for scaling
            var_array = ref_array ** 2 + 0.01

        # Observable names
        obs_names = [
            'velocity', 'angular_velocity', 'reversal_rate', 'run_length',
            'speed_mean', 'speed_variance', 'turn_angle_mean', 'turn_angle_variance',
            'omega_turn_rate', 'chemotaxis_index', 'anterior_touch_prob',
            'posterior_touch_prob', 'response_latency', 'pharyngeal_pumping',
            'reserved'
        ]

        # Compute weighted errors
        errors = {}
        total_error = 0.0

        for i, name in enumerate(obs_names[:-1]):  # Skip reserved
            diff = sim_array[i] - ref_array[i]
            weight = self.observable_weights.get(name, 1.0)

            if normalize:
                normalized_error = (diff ** 2) / var_array[i]
            else:
                normalized_error = diff ** 2

            weighted_error = weight * normalized_error
            errors[name] = float(weighted_error)
            total_error += weighted_error

        return total_error, errors

    def check_behavioral_equivalence(self,
                                      simulated: ObservableVector) -> bool:
        """
        Check if simulated observables are within epsilon of reference.

        ||M(σ) - D||₂ < ε for all stimuli σ

        Args:
            simulated: Observable from simulation

        Returns:
            True if behaviorally equivalent
        """
        distance = simulated.distance(self.reference_observable)

        # Normalize by number of observables
        normalized_distance = distance / math.sqrt(14)  # 14 observables

        return normalized_distance < self.epsilon

    def score_connectome(self,
                          connectome: ConnectomeData,
                          simulated_observable: ObservableVector) -> MDLScore:
        """
        Compute full MDL score for a connectome model.

        Args:
            connectome: The connectome data
            simulated_observable: Observable from running simulation

        Returns:
            Complete MDL score
        """
        # Structural complexity
        L_neurons, L_synapses, L_gap = self.compute_structural_complexity(connectome)
        L_struct = L_neurons + L_synapses + L_gap

        # Parameter error
        L_param, obs_errors = self.compute_parameter_error(simulated_observable)

        # Behavioral equivalence
        is_equivalent = self.check_behavioral_equivalence(simulated_observable)

        return MDLScore(
            L_struct=L_struct,
            L_neurons=L_neurons,
            L_synapses=L_synapses,
            L_gap_junctions=L_gap,
            L_param=L_param,
            observable_errors=obs_errors,
            is_behaviorally_equivalent=is_equivalent,
            equivalence_epsilon=self.epsilon,
        )

    def score_simulation(self,
                          simulation: CElegansSimulation,
                          connectome: ConnectomeData,
                          n_steps: int = 1000) -> MDLScore:
        """
        Run simulation and compute MDL score.

        Args:
            simulation: Initialized simulation
            connectome: Connectome data
            n_steps: Number of simulation steps

        Returns:
            MDL score
        """
        # Run simulation to accumulate history
        simulation.step(n_steps)

        # Get observables
        simulated = simulation.get_observables(force_update=True)

        # Compute score
        return self.score_connectome(connectome, simulated)


# =============================================================================
# Model Comparison
# =============================================================================

@dataclass
class ModelComparison:
    """Results of comparing multiple models."""
    model_names: List[str] = field(default_factory=list)
    scores: List[MDLScore] = field(default_factory=list)
    observables: List[ObservableVector] = field(default_factory=list)

    def best_model(self) -> str:
        """Return name of model with lowest MDL score."""
        if not self.scores:
            return ""
        idx = min(range(len(self.scores)), key=lambda i: self.scores[i].total)
        return self.model_names[idx]

    def ranking(self) -> List[Tuple[str, float]]:
        """Return models ranked by MDL score (best first)."""
        pairs = list(zip(self.model_names, [s.total for s in self.scores]))
        return sorted(pairs, key=lambda x: x[1])

    def summary_table(self) -> str:
        """Generate summary table of comparison."""
        lines = [
            "=" * 70,
            "MODEL COMPARISON (MDL Scoring)",
            "=" * 70,
            f"{'Model':<25} {'L_struct':>10} {'L_param':>10} {'Total':>10} {'Equiv':>8}",
            "-" * 70,
        ]

        for name, score in zip(self.model_names, self.scores):
            equiv = "Yes" if score.is_behaviorally_equivalent else "No"
            lines.append(
                f"{name:<25} {score.L_struct:>10.2f} {score.L_param:>10.2f} "
                f"{score.total:>10.2f} {equiv:>8}"
            )

        lines.append("-" * 70)
        lines.append(f"Best model: {self.best_model()}")
        lines.append("=" * 70)

        return "\n".join(lines)


def compare_models_mdl(
    models: Dict[str, Tuple[ConnectomeData, CElegansSimulation]],
    real_data: Optional[RealBehaviorDataset] = None,
    n_steps: int = 1000,
    epsilon: float = 0.1,
) -> ModelComparison:
    """
    Compare multiple models using MDL scoring.

    Args:
        models: Dict mapping model names to (connectome, simulation) tuples
        real_data: Real behavioral data for validation
        n_steps: Simulation steps for each model
        epsilon: Behavioral equivalence threshold

    Returns:
        ModelComparison with results
    """
    validator = MDLValidator(real_data=real_data, epsilon=epsilon)
    comparison = ModelComparison()

    for name, (connectome, simulation) in models.items():
        print(f"Evaluating {name}...")

        # Reset and run simulation
        simulation.reset()
        score = validator.score_simulation(simulation, connectome, n_steps)

        comparison.model_names.append(name)
        comparison.scores.append(score)
        comparison.observables.append(simulation.get_observables())

    return comparison


# =============================================================================
# Quotient Category Analysis
# =============================================================================

class BehavioralContext(Enum):
    """Different behavioral contexts for validation."""
    TOUCH_ESCAPE = auto()
    CHEMOTAXIS = auto()
    THERMOTAXIS = auto()
    FORAGING = auto()
    BASELINE = auto()


def get_context_weights(context: BehavioralContext) -> Dict[str, float]:
    """
    Get observable weights specialized for a behavioral context.

    Different contexts prioritize different observables.
    """
    if context == BehavioralContext.TOUCH_ESCAPE:
        return {
            'anterior_touch_prob': 5.0,
            'posterior_touch_prob': 5.0,
            'reversal_rate': 3.0,
            'response_latency': 3.0,
            'velocity': 1.0,
            'speed_mean': 1.0,
            'omega_turn_rate': 2.0,
        }

    elif context == BehavioralContext.CHEMOTAXIS:
        return {
            'chemotaxis_index': 5.0,
            'turn_angle_mean': 3.0,
            'turn_angle_variance': 2.0,
            'velocity': 2.0,
            'omega_turn_rate': 2.0,
            'reversal_rate': 1.0,
        }

    elif context == BehavioralContext.THERMOTAXIS:
        return {
            'velocity': 2.0,
            'turn_angle_mean': 3.0,
            'run_length': 2.0,
            'reversal_rate': 1.0,
        }

    elif context == BehavioralContext.FORAGING:
        return {
            'pharyngeal_pumping': 5.0,
            'velocity': 2.0,
            'reversal_rate': 1.0,
            'chemotaxis_index': 2.0,
        }

    else:  # BASELINE
        return {}  # Use default weights


class ContextualMDLValidator(MDLValidator):
    """MDL validator specialized for a behavioral context."""

    def __init__(self,
                 context: BehavioralContext,
                 real_data: Optional[RealBehaviorDataset] = None,
                 **kwargs):
        super().__init__(real_data=real_data, **kwargs)
        self.context = context

        # Override weights with context-specific weights
        context_weights = get_context_weights(context)
        if context_weights:
            # Keep defaults for unspecified observables but scale down
            for key in self.observable_weights:
                if key not in context_weights:
                    self.observable_weights[key] *= 0.1
            self.observable_weights.update(context_weights)


# =============================================================================
# Main / Testing
# =============================================================================

if __name__ == "__main__":
    print("MDL Validator")
    print("=" * 50)

    # Test with reference observable
    validator = MDLValidator()

    print("\nReference observable:")
    print(validator.reference_observable)

    # Create a simulated observable (slightly different from reference)
    simulated = ObservableVector(
        velocity=0.18,
        angular_velocity=0.35,
        reversal_rate=2.5,
        run_length=8.0,
        speed_mean=0.19,
        speed_variance=0.012,
        turn_angle_mean=28.0,
        turn_angle_variance=420.0,
        omega_turn_rate=0.6,
        chemotaxis_index=0.25,
        anterior_touch_prob=0.75,
        posterior_touch_prob=0.65,
        response_latency=220.0,
        pharyngeal_pumping=190.0,
    )

    print("\nSimulated observable:")
    print(simulated)

    # Compute parameter error
    L_param, errors = validator.compute_parameter_error(simulated)
    print(f"\nL_param = {L_param:.3f}")
    print("\nPer-observable errors:")
    for name, err in sorted(errors.items(), key=lambda x: x[1], reverse=True):
        print(f"  {name}: {err:.4f}")

    # Check equivalence
    is_equiv = validator.check_behavioral_equivalence(simulated)
    print(f"\nBehaviorally equivalent: {is_equiv}")

    # Test context-specific weights
    print("\n" + "=" * 50)
    print("Context-specific weights:")
    for ctx in BehavioralContext:
        weights = get_context_weights(ctx)
        print(f"\n{ctx.name}:")
        for k, v in weights.items():
            print(f"  {k}: {v}")
