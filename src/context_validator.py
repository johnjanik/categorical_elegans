#!/usr/bin/env python3
"""
Context-Specific Validation
============================

Implements behavioral context-specific validation for C. elegans models.

Different behavioral contexts (touch escape, chemotaxis, thermotaxis, etc.)
involve different subsets of neurons and observables. This module:

1. Defines required neurons for each context
2. Validates that models preserve essential circuit structure
3. Computes context-specific MDL scores
4. Discovers minimal models for each context

From minimal_model_theory.tex Section 6:
"Different behavioral contexts may require different minimal models,
as the quotient category collapses different neurons for different tasks."
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Set, Tuple
import numpy as np

from .c_elegans_simulation import (
    ObservableVector,
    CElegansSimulation,
    SensoryInput,
)
from .c_elegans_connectome_loader import ConnectomeData, NeuronType
from .mdl_validator import MDLValidator, MDLScore, BehavioralContext, get_context_weights


# =============================================================================
# Context-Specific Neuron Requirements
# =============================================================================

# Neurons essential for each behavioral context
CONTEXT_REQUIRED_NEURONS: Dict[BehavioralContext, Set[str]] = {
    BehavioralContext.TOUCH_ESCAPE: {
        # Sensory
        'ALML', 'ALMR', 'AVM',          # Anterior touch receptors
        'PLML', 'PLMR', 'PVM',          # Posterior touch receptors

        # Command interneurons
        'AVAL', 'AVAR',                  # Backward command
        'AVDL', 'AVDR', 'AVEL', 'AVER',  # Backward command
        'AVBL', 'AVBR',                  # Forward command
        'PVCL', 'PVCR',                  # Forward command

        # Key motor neurons (representative)
        'DA1', 'DA2', 'DA3',             # Backward motor
        'DB1', 'DB2', 'DB3',             # Forward motor
        'VA1', 'VA2', 'VA3',             # Backward motor
        'VB1', 'VB2', 'VB3',             # Forward motor
    },

    BehavioralContext.CHEMOTAXIS: {
        # Sensory
        'AWAL', 'AWAR',                  # Attractive odorants
        'AWBL', 'AWBR',                  # Repulsive odorants
        'AWCL', 'AWCR',                  # Olfaction
        'ASEL', 'ASER',                  # Chemosensation

        # Processing interneurons
        'AIAL', 'AIAR',                  # Integration
        'AIBL', 'AIBR',                  # Integration
        'AIYL', 'AIYR',                  # Thermotaxis/chemotaxis

        # Head movement
        'RIAL', 'RIAR',                  # Head turns
        'RIVL', 'RIVR',                  # Head turns
        'SMDDL', 'SMDDR', 'SMDVL', 'SMDVR',  # Head motor

        # Command
        'AVBL', 'AVBR',                  # Forward
    },

    BehavioralContext.THERMOTAXIS: {
        # Sensory
        'AFDL', 'AFDR',                  # Primary thermosensor
        'AWCL', 'AWCR',                  # Also thermosensitive

        # Processing
        'AIYL', 'AIYR',                  # Key thermotaxis interneuron
        'AIZL', 'AIZR',                  # Thermotaxis processing
        'AIBL', 'AIBR',                  # Integration

        # Head movement
        'RIAL', 'RIAR',
        'RIML', 'RIMR',

        # Command
        'AVBL', 'AVBR',
    },

    BehavioralContext.FORAGING: {
        # Pharyngeal sensory/motor
        'MCL', 'MCR',                    # Marginal cells
        'M1', 'M2L', 'M2R', 'M3L', 'M3R', 'M4', 'M5',  # Pharyngeal motor
        'I1L', 'I1R', 'I2L', 'I2R',     # Pharyngeal interneurons
        'NSML', 'NSMR',                  # Serotonergic, feeding

        # Dopaminergic (food detection)
        'CEPDL', 'CEPDR', 'CEPVL', 'CEPVR',
        'ADEL', 'ADER',

        # Locomotion (reduced)
        'AVBL', 'AVBR',
        'DB1', 'DB2', 'VB1', 'VB2',
    },

    BehavioralContext.BASELINE: set(),  # All neurons are relevant
}


# Neurons that can potentially be removed without affecting behavior
CONTEXT_OPTIONAL_NEURONS: Dict[BehavioralContext, Set[str]] = {
    BehavioralContext.TOUCH_ESCAPE: {
        # Pharyngeal neurons not needed for touch
        'MCL', 'MCR', 'M1', 'M2L', 'M2R', 'M3L', 'M3R', 'M4', 'M5',
        'I1L', 'I1R', 'I2L', 'I2R', 'I3', 'I4', 'I5', 'I6',
        'NSML', 'NSMR', 'MI',

        # Chemosensory not needed
        'AWAL', 'AWAR', 'AWBL', 'AWBR',
        'ASEL', 'ASER', 'ASGL', 'ASGR',
    },

    BehavioralContext.CHEMOTAXIS: {
        # Touch receptors less relevant
        'ALML', 'ALMR', 'AVM', 'PLML', 'PLMR', 'PVM',

        # Pharyngeal not needed
        'MCL', 'MCR', 'M1', 'M4', 'M5',
        'NSML', 'NSMR',
    },

    BehavioralContext.THERMOTAXIS: {
        # Similar to chemotaxis
        'ALML', 'ALMR', 'AVM', 'PLML', 'PLMR', 'PVM',
        'MCL', 'MCR', 'NSML', 'NSMR',
    },

    BehavioralContext.FORAGING: {
        # Touch less relevant during feeding
        'ALML', 'ALMR', 'AVM',
        # Most of locomotion command can be simplified
        'AVAL', 'AVAR', 'AVDL', 'AVDR',
    },

    BehavioralContext.BASELINE: set(),
}


# =============================================================================
# Context-Specific Stimuli
# =============================================================================

def get_context_stimulus(context: BehavioralContext) -> SensoryInput:
    """
    Get the appropriate sensory stimulus for a behavioral context.

    Args:
        context: Behavioral context

    Returns:
        SensoryInput configured for the context
    """
    stimulus = SensoryInput()

    if context == BehavioralContext.TOUCH_ESCAPE:
        stimulus.anterior_touch = 0.8

    elif context == BehavioralContext.CHEMOTAXIS:
        stimulus.attractive_odor = 0.7

    elif context == BehavioralContext.THERMOTAXIS:
        stimulus.temperature = 25.0  # Above cultivation temp
        stimulus.temp_gradient = 0.5

    elif context == BehavioralContext.FORAGING:
        stimulus.food_present = True
        stimulus.attractive_odor = 0.3

    return stimulus


# =============================================================================
# Context Validator
# =============================================================================

@dataclass
class ContextValidationResult:
    """Result of validating a model for a specific context."""
    context: BehavioralContext
    model_name: str

    # Circuit integrity
    has_required_neurons: bool = False
    missing_neurons: Set[str] = field(default_factory=set)
    coverage_fraction: float = 0.0

    # Behavioral performance
    mdl_score: Optional[MDLScore] = None
    observable: Optional[ObservableVector] = None

    # Context-specific metrics
    primary_response_correct: bool = False
    response_magnitude: float = 0.0

    def __repr__(self) -> str:
        return (
            f"ContextValidationResult({self.context.name}, {self.model_name})\n"
            f"  Required neurons: {self.coverage_fraction:.1%} "
            f"({'PASS' if self.has_required_neurons else 'FAIL'})\n"
            f"  Primary response: {'PASS' if self.primary_response_correct else 'FAIL'} "
            f"(magnitude={self.response_magnitude:.3f})\n"
            f"  MDL score: {self.mdl_score.total:.3f if self.mdl_score else 'N/A'}\n"
        )


class ContextValidator:
    """
    Validates models for specific behavioral contexts.

    For each context:
    1. Check that required neurons are present
    2. Apply context-specific stimulus
    3. Measure behavioral response
    4. Compute context-specific MDL score
    """

    def __init__(self, context: BehavioralContext):
        """
        Initialize context validator.

        Args:
            context: The behavioral context to validate
        """
        self.context = context
        self.required_neurons = CONTEXT_REQUIRED_NEURONS.get(context, set())
        self.optional_neurons = CONTEXT_OPTIONAL_NEURONS.get(context, set())

        # Create context-specific MDL validator
        self.mdl_validator = MDLValidator(epsilon=0.15)
        self.mdl_validator.observable_weights = get_context_weights(context)

        # Fill in missing weights with low values
        default_weights = {
            'velocity': 0.1, 'angular_velocity': 0.1, 'reversal_rate': 0.1,
            'run_length': 0.1, 'speed_mean': 0.1, 'speed_variance': 0.1,
            'turn_angle_mean': 0.1, 'turn_angle_variance': 0.1, 'omega_turn_rate': 0.1,
            'chemotaxis_index': 0.1, 'anterior_touch_prob': 0.1,
            'posterior_touch_prob': 0.1, 'response_latency': 0.1,
            'pharyngeal_pumping': 0.1,
        }
        for key, val in default_weights.items():
            if key not in self.mdl_validator.observable_weights:
                self.mdl_validator.observable_weights[key] = val

    def check_circuit_integrity(self, connectome: ConnectomeData) -> Tuple[bool, Set[str], float]:
        """
        Check if connectome has required neurons for this context.

        Args:
            connectome: Connectome to check

        Returns:
            (all_present, missing_set, coverage_fraction)
        """
        if not self.required_neurons:
            return True, set(), 1.0

        present = set(connectome.neurons.keys())
        missing = self.required_neurons - present
        coverage = len(self.required_neurons - missing) / len(self.required_neurons)

        return len(missing) == 0, missing, coverage

    def check_primary_response(self,
                                simulation: CElegansSimulation,
                                baseline_behavior: 'BehavioralState',
                                stimulated_behavior: 'BehavioralState') -> Tuple[bool, float]:
        """
        Check if the model produces the correct primary response for this context.

        Args:
            simulation: The simulation
            baseline_behavior: Behavior before stimulus
            stimulated_behavior: Behavior after stimulus

        Returns:
            (is_correct, response_magnitude)
        """
        if self.context == BehavioralContext.TOUCH_ESCAPE:
            # Should show backward drive increase on anterior touch
            delta = stimulated_behavior.backward_drive - baseline_behavior.backward_drive
            is_correct = delta > 0.1
            return is_correct, delta

        elif self.context == BehavioralContext.CHEMOTAXIS:
            # Should show forward movement toward attractant
            delta = stimulated_behavior.forward_drive - baseline_behavior.forward_drive
            is_correct = delta > 0.05 or stimulated_behavior.forward_drive > 0.3
            return is_correct, delta

        elif self.context == BehavioralContext.THERMOTAXIS:
            # Should show modulated movement
            speed_change = abs(stimulated_behavior.speed - baseline_behavior.speed)
            is_correct = speed_change > 0.02
            return is_correct, speed_change

        elif self.context == BehavioralContext.FORAGING:
            # Should show pharyngeal activity
            is_correct = stimulated_behavior.pharyngeal_pumping > 0.1
            return is_correct, stimulated_behavior.pharyngeal_pumping

        return True, 0.0

    def validate(self,
                 connectome: ConnectomeData,
                 simulation: CElegansSimulation,
                 model_name: str = "Model",
                 n_baseline_steps: int = 200,
                 n_stimulus_steps: int = 500) -> ContextValidationResult:
        """
        Validate a model for this behavioral context.

        Args:
            connectome: Connectome data
            simulation: Simulation instance
            model_name: Name for reporting
            n_baseline_steps: Steps to run baseline
            n_stimulus_steps: Steps to run with stimulus

        Returns:
            ContextValidationResult
        """
        result = ContextValidationResult(
            context=self.context,
            model_name=model_name,
        )

        # Check circuit integrity
        result.has_required_neurons, result.missing_neurons, result.coverage_fraction = \
            self.check_circuit_integrity(connectome)

        # Run baseline
        simulation.reset()
        simulation.step(n_baseline_steps)
        baseline_behavior = simulation.behavior

        # Apply context-specific stimulus
        stimulus = get_context_stimulus(self.context)
        simulation.sensory = stimulus
        simulation.step(n_stimulus_steps)
        stimulated_behavior = simulation.behavior

        # Check primary response
        result.primary_response_correct, result.response_magnitude = \
            self.check_primary_response(simulation, baseline_behavior, stimulated_behavior)

        # Compute MDL score
        result.observable = simulation.get_observables(force_update=True)
        result.mdl_score = self.mdl_validator.score_connectome(connectome, result.observable)

        return result


# =============================================================================
# Multi-Context Validation
# =============================================================================

@dataclass
class MultiContextValidation:
    """Results of validating a model across multiple contexts."""
    model_name: str
    results: Dict[BehavioralContext, ContextValidationResult] = field(default_factory=dict)

    def summary(self) -> str:
        """Generate summary table."""
        lines = [
            f"Multi-Context Validation: {self.model_name}",
            "=" * 60,
            f"{'Context':<20} {'Circuit':>10} {'Response':>10} {'MDL':>10}",
            "-" * 60,
        ]

        for ctx, result in self.results.items():
            circuit = "PASS" if result.has_required_neurons else "FAIL"
            response = "PASS" if result.primary_response_correct else "FAIL"
            mdl = f"{result.mdl_score.total:.2f}" if result.mdl_score else "N/A"

            lines.append(f"{ctx.name:<20} {circuit:>10} {response:>10} {mdl:>10}")

        lines.append("=" * 60)
        return "\n".join(lines)

    def passing_contexts(self) -> List[BehavioralContext]:
        """Return list of contexts where model passes all checks."""
        return [
            ctx for ctx, result in self.results.items()
            if result.has_required_neurons and result.primary_response_correct
        ]


def validate_all_contexts(
    connectome: ConnectomeData,
    simulation: CElegansSimulation,
    model_name: str = "Model",
    contexts: Optional[List[BehavioralContext]] = None,
) -> MultiContextValidation:
    """
    Validate a model across all (or specified) behavioral contexts.

    Args:
        connectome: Connectome data
        simulation: Simulation instance
        model_name: Name for reporting
        contexts: Specific contexts to test (None = all)

    Returns:
        MultiContextValidation with all results
    """
    if contexts is None:
        contexts = list(BehavioralContext)

    multi_result = MultiContextValidation(model_name=model_name)

    for ctx in contexts:
        validator = ContextValidator(ctx)
        result = validator.validate(connectome, simulation, model_name)
        multi_result.results[ctx] = result

    return multi_result


# =============================================================================
# Minimal Model Discovery per Context
# =============================================================================

def find_essential_neurons(
    context: BehavioralContext,
    full_connectome: ConnectomeData,
    target_coverage: float = 0.95,
) -> Set[str]:
    """
    Find the minimal set of neurons essential for a behavioral context.

    Uses the context's required neurons as a starting point, then
    adds neurons connected to them until coverage target is met.

    Args:
        context: Behavioral context
        full_connectome: Complete connectome
        target_coverage: Target fraction of pathway activity

    Returns:
        Set of essential neuron names
    """
    essential = set(CONTEXT_REQUIRED_NEURONS.get(context, set()))

    # Add neurons directly connected to required neurons
    for syn in full_connectome.synapses:
        if syn.pre in essential or syn.post in essential:
            if syn.weight > 5:  # Only strong connections
                essential.add(syn.pre)
                essential.add(syn.post)

    # Add gap junction partners
    for gj in full_connectome.gap_junctions:
        if gj.neuron_a in essential or gj.neuron_b in essential:
            if gj.weight > 3:
                essential.add(gj.neuron_a)
                essential.add(gj.neuron_b)

    return essential


# =============================================================================
# Main / Testing
# =============================================================================

if __name__ == "__main__":
    print("Context-Specific Validation")
    print("=" * 60)

    print("\nRequired neurons by context:")
    for ctx in BehavioralContext:
        neurons = CONTEXT_REQUIRED_NEURONS.get(ctx, set())
        print(f"\n{ctx.name}: {len(neurons)} neurons")
        if neurons:
            print(f"  {sorted(neurons)[:10]}...")

    print("\n" + "=" * 60)
    print("Context-specific stimuli:")
    for ctx in BehavioralContext:
        stim = get_context_stimulus(ctx)
        active = []
        if stim.anterior_touch > 0:
            active.append(f"anterior_touch={stim.anterior_touch}")
        if stim.posterior_touch > 0:
            active.append(f"posterior_touch={stim.posterior_touch}")
        if stim.attractive_odor > 0:
            active.append(f"attractive_odor={stim.attractive_odor}")
        if stim.repulsive_odor > 0:
            active.append(f"repulsive_odor={stim.repulsive_odor}")
        if stim.temperature != 20.0:
            active.append(f"temperature={stim.temperature}")

        print(f"{ctx.name}: {', '.join(active) if active else 'baseline'}")
