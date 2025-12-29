#!/usr/bin/env python3
"""
C. elegans Multi-Model Simulation & Comparison
===============================================
Runs simulations using different connectome data sources and compares
behavioral outputs to understand how model completeness affects behavior.

Models available:
- Categorical Elegans: Manual categorical model (121 neurons, focused circuits)
- OpenWorm: Full hermaphrodite connectome (448 neurons, 7379 connections)
"""

import numpy as np
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Tuple, Optional
from collections import deque
import time

from .c_elegans_connectome_loader import (
    load_connectome, get_available_loaders, ConnectomeData,
    NeuronType, Neurotransmitter, NeuronInfo, SynapseInfo, GapJunctionInfo
)


# =============================================================================
# Simulation Classes
# =============================================================================

@dataclass
class NeurochemicalState:
    """Global neurochemical modulation levels (0.0 to 2.0)."""
    ACh: float = 1.0      # Acetylcholine
    Glu: float = 1.0      # Glutamate
    GABA: float = 1.0     # GABA
    DA: float = 1.0       # Dopamine
    serotonin: float = 1.0  # 5-HT
    Oct: float = 1.0      # Octopamine
    Tyr: float = 1.0      # Tyramine


@dataclass
class SensoryInput:
    """Sensory inputs to the network."""
    anterior_touch: float = 0.0
    posterior_touch: float = 0.0
    harsh_touch: float = 0.0
    attractive_odor: float = 0.0
    repulsive_odor: float = 0.0
    thermotaxis_gradient: float = 0.0


@dataclass
class BehavioralOutput:
    """Behavioral state outputs."""
    forward_drive: float = 0.0
    backward_drive: float = 0.0
    turn_left: float = 0.0
    turn_right: float = 0.0
    speed: float = 0.0
    head_angle: float = 0.0


@dataclass
class NeuronState:
    """Dynamic state of a single neuron."""
    membrane_potential: float = -70.0
    calcium: float = 0.0
    activity: float = 0.0


# =============================================================================
# Multi-Model Simulation
# =============================================================================

class MultiModelSimulation:
    """
    Simulation engine that can use any connectome data source.
    Allows comparison of behavior across different models.
    """

    def __init__(self, connectome: ConnectomeData):
        self.connectome = connectome
        self.dt = 1.0  # ms

        # Scale parameters based on network density
        n_neurons = len(connectome.neurons)
        n_synapses = len(connectome.synapses)
        self.density = n_synapses / max(n_neurons, 1)

        # Adjust gain based on network density (denser = lower gain)
        self.synaptic_gain = 20.0 / (1.0 + self.density / 10.0)
        self.gap_gain = 0.01 / (1.0 + self.density / 20.0)

        # Initialize neuron states
        self.states: Dict[str, NeuronState] = {}
        for name in connectome.neurons:
            self.states[name] = NeuronState()

        # Build lookup structures for efficiency
        self._build_connection_maps()

        # Input/Output
        self.neurochemistry = NeurochemicalState()
        self.sensory = SensoryInput()
        self.behavior = BehavioralOutput()

        # History for analysis
        self.history_len = 500
        self.activity_history: Dict[str, deque] = {
            name: deque(maxlen=self.history_len)
            for name in connectome.neurons
        }
        self.behavior_history = {
            'forward': deque(maxlen=self.history_len),
            'backward': deque(maxlen=self.history_len),
            'speed': deque(maxlen=self.history_len),
        }

        # Identify key neuron classes for behavior computation
        self._identify_key_neurons()

    def _build_connection_maps(self):
        """Build efficient lookup maps for synapses and gap junctions."""
        # Presynaptic -> list of (postsynaptic, weight)
        self.chemical_pre_to_post: Dict[str, List[Tuple[str, float]]] = {}
        for syn in self.connectome.synapses:
            if syn.pre not in self.chemical_pre_to_post:
                self.chemical_pre_to_post[syn.pre] = []
            strength = np.log1p(syn.weight) / 3.0
            self.chemical_pre_to_post[syn.pre].append((syn.post, strength))

        # Gap junction partners
        self.gap_partners: Dict[str, List[Tuple[str, float]]] = {}
        for gj in self.connectome.gap_junctions:
            conductance = np.log1p(gj.weight) / 2.0
            if gj.neuron_a not in self.gap_partners:
                self.gap_partners[gj.neuron_a] = []
            if gj.neuron_b not in self.gap_partners:
                self.gap_partners[gj.neuron_b] = []
            self.gap_partners[gj.neuron_a].append((gj.neuron_b, conductance))
            self.gap_partners[gj.neuron_b].append((gj.neuron_a, conductance))

    def _identify_key_neurons(self):
        """Identify neurons by class for behavior computation."""
        self.neuron_classes: Dict[str, List[str]] = {}

        for name, info in self.connectome.neurons.items():
            nclass = info.neuron_class
            if nclass not in self.neuron_classes:
                self.neuron_classes[nclass] = []
            self.neuron_classes[nclass].append(name)

        # Key circuit neurons
        self.touch_sensors = (
            self.neuron_classes.get('ALM', []) +
            self.neuron_classes.get('PLM', []) +
            self.neuron_classes.get('AVM', []) +
            self.neuron_classes.get('PVM', [])
        )

        self.backward_command = (
            self.neuron_classes.get('AVA', []) +
            self.neuron_classes.get('AVD', []) +
            self.neuron_classes.get('AVE', [])
        )

        self.forward_command = (
            self.neuron_classes.get('AVB', []) +
            self.neuron_classes.get('PVC', [])
        )

        self.a_type_motors = (
            self.neuron_classes.get('DA', []) +
            self.neuron_classes.get('VA', [])
        )

        self.b_type_motors = (
            self.neuron_classes.get('DB', []) +
            self.neuron_classes.get('VB', [])
        )

        self.d_type_motors = (
            self.neuron_classes.get('DD', []) +
            self.neuron_classes.get('VD', [])
        )

        self.head_motors = self.neuron_classes.get('SMD', [])

        self.chemosensory = (
            self.neuron_classes.get('AWA', []) +
            self.neuron_classes.get('AWB', []) +
            self.neuron_classes.get('AWC', [])
        )

        self.thermosensory = self.neuron_classes.get('AFD', [])

    def _get_nt_modulation(self, nt: Neurotransmitter) -> float:
        """Get modulation factor for a neurotransmitter."""
        if nt == Neurotransmitter.ACH:
            return self.neurochemistry.ACh
        elif nt == Neurotransmitter.GLU:
            return self.neurochemistry.Glu
        elif nt == Neurotransmitter.GABA:
            return self.neurochemistry.GABA
        elif nt == Neurotransmitter.DA:
            return self.neurochemistry.DA
        elif nt == Neurotransmitter.SEROTONIN:
            return self.neurochemistry.serotonin
        elif nt == Neurotransmitter.OCT:
            return self.neurochemistry.Oct
        elif nt == Neurotransmitter.TYR:
            return self.neurochemistry.Tyr
        return 1.0

    def _is_inhibitory(self, nt: Neurotransmitter) -> bool:
        """Check if neurotransmitter is inhibitory."""
        return nt == Neurotransmitter.GABA

    def step(self, n: int = 1):
        """Run n simulation steps."""
        for _ in range(n):
            self._step_once()

    def _step_once(self):
        """Single simulation step."""
        # Apply sensory inputs
        self._apply_sensory_input()

        # Compute synaptic currents
        currents = {name: 0.0 for name in self.states}

        # Chemical synapses
        for pre_name, targets in self.chemical_pre_to_post.items():
            if pre_name not in self.states:
                continue
            pre_state = self.states[pre_name]
            pre_info = self.connectome.neurons.get(pre_name)
            if pre_info is None:
                continue

            nt = pre_info.neurotransmitter
            nt_mod = self._get_nt_modulation(nt)
            sign = -1.0 if self._is_inhibitory(nt) else 1.0

            for post_name, strength in targets:
                if post_name not in currents:
                    continue
                current = sign * pre_state.activity * strength * nt_mod
                currents[post_name] += current

        # Gap junctions
        for name, partners in self.gap_partners.items():
            if name not in self.states:
                continue
            state = self.states[name]
            for partner_name, conductance in partners:
                if partner_name not in self.states:
                    continue
                partner_state = self.states[partner_name]
                # Current proportional to voltage difference
                gap_current = conductance * (partner_state.membrane_potential - state.membrane_potential) * self.gap_gain
                currents[name] += gap_current

        # Update membrane potentials and activities
        for name, state in self.states.items():
            # Membrane dynamics
            tau = 10.0  # ms
            rest = -70.0
            I = currents.get(name, 0.0)

            dV = (rest - state.membrane_potential + I * self.synaptic_gain) * (self.dt / tau)
            state.membrane_potential += dV
            # Clamp to physiological range
            state.membrane_potential = np.clip(state.membrane_potential, -90.0, 0.0)

            # Activity (sigmoid of membrane potential)
            state.activity = 1.0 / (1.0 + np.exp(-(state.membrane_potential + 55.0) / 5.0))

            # Record history
            self.activity_history[name].append(state.activity)

        # Compute behavioral output
        self._compute_behavior()

    def _apply_sensory_input(self):
        """Apply sensory inputs to appropriate neurons."""
        # Anterior touch -> ALM, AVM
        for name in self.neuron_classes.get('ALM', []) + self.neuron_classes.get('AVM', []):
            if name in self.states:
                self.states[name].membrane_potential += self.sensory.anterior_touch * 30.0

        # Posterior touch -> PLM, PVM
        for name in self.neuron_classes.get('PLM', []) + self.neuron_classes.get('PVM', []):
            if name in self.states:
                self.states[name].membrane_potential += self.sensory.posterior_touch * 30.0

        # Attractive odor -> AWA
        for name in self.neuron_classes.get('AWA', []):
            if name in self.states:
                self.states[name].membrane_potential += self.sensory.attractive_odor * 20.0

        # Repulsive odor -> AWB
        for name in self.neuron_classes.get('AWB', []):
            if name in self.states:
                self.states[name].membrane_potential += self.sensory.repulsive_odor * 20.0

    def _compute_behavior(self):
        """Compute behavioral outputs from motor neuron activities."""
        # Forward drive from B-type motors
        if self.b_type_motors:
            self.behavior.forward_drive = np.mean([
                self.states[n].activity for n in self.b_type_motors if n in self.states
            ]) if self.b_type_motors else 0.0
        else:
            self.behavior.forward_drive = 0.0

        # Backward drive from A-type motors
        if self.a_type_motors:
            self.behavior.backward_drive = np.mean([
                self.states[n].activity for n in self.a_type_motors if n in self.states
            ]) if self.a_type_motors else 0.0
        else:
            self.behavior.backward_drive = 0.0

        # Overall speed
        self.behavior.speed = max(self.behavior.forward_drive, self.behavior.backward_drive)

        # Head angle from SMD neurons (if present)
        smd_left = [n for n in self.head_motors if 'L' in n]
        smd_right = [n for n in self.head_motors if 'R' in n]

        if smd_left and smd_right:
            left_activity = np.mean([self.states[n].activity for n in smd_left if n in self.states])
            right_activity = np.mean([self.states[n].activity for n in smd_right if n in self.states])
            self.behavior.head_angle = (right_activity - left_activity) * 30.0  # degrees
        else:
            self.behavior.head_angle = 0.0

        # Record history
        self.behavior_history['forward'].append(self.behavior.forward_drive)
        self.behavior_history['backward'].append(self.behavior.backward_drive)
        self.behavior_history['speed'].append(self.behavior.speed)

    def reset(self):
        """Reset all states to initial conditions."""
        for state in self.states.values():
            state.membrane_potential = -70.0
            state.calcium = 0.0
            state.activity = 0.0

        for hist in self.activity_history.values():
            hist.clear()
        for hist in self.behavior_history.values():
            hist.clear()

        self.sensory = SensoryInput()

    def get_mean_activity_by_type(self) -> Dict[str, float]:
        """Get mean activity for each neuron type."""
        type_activities = {}
        for ntype in NeuronType:
            neurons = [n for n, info in self.connectome.neurons.items()
                      if info.neuron_type == ntype]
            if neurons:
                type_activities[ntype.name] = np.mean([
                    self.states[n].activity for n in neurons if n in self.states
                ])
        return type_activities

    def get_mean_activity_by_class(self, class_name: str) -> float:
        """Get mean activity for a specific neuron class."""
        neurons = self.neuron_classes.get(class_name, [])
        if not neurons:
            return 0.0
        activities = [self.states[n].activity for n in neurons if n in self.states]
        return np.mean(activities) if activities else 0.0


# =============================================================================
# Model Comparison
# =============================================================================

def compare_models(stimulus_type: str = 'anterior_touch',
                   stimulus_strength: float = 0.8,
                   duration_ms: int = 200) -> Dict:
    """
    Run the same stimulus on multiple models and compare responses.

    Args:
        stimulus_type: Type of sensory stimulus
        stimulus_strength: Strength of stimulus (0.0 to 1.0)
        duration_ms: How long to run simulation

    Returns:
        Dictionary with comparison results
    """
    results = {}
    loaders = get_available_loaders()

    for model_name, loader in loaders.items():
        try:
            connectome = loader.load()
        except Exception as e:
            print(f"Skipping {model_name}: {e}")
            continue

        sim = MultiModelSimulation(connectome)

        # Record baseline (100ms)
        for _ in range(100):
            sim.step()

        baseline = {
            'forward': sim.behavior.forward_drive,
            'backward': sim.behavior.backward_drive,
            'speed': sim.behavior.speed,
        }

        # Apply stimulus
        if stimulus_type == 'anterior_touch':
            sim.sensory.anterior_touch = stimulus_strength
        elif stimulus_type == 'posterior_touch':
            sim.sensory.posterior_touch = stimulus_strength
        elif stimulus_type == 'attractive_odor':
            sim.sensory.attractive_odor = stimulus_strength
        elif stimulus_type == 'repulsive_odor':
            sim.sensory.repulsive_odor = stimulus_strength

        # Run with stimulus (record peaks)
        forward_peak = 0.0
        backward_peak = 0.0
        speed_peak = 0.0

        for i in range(duration_ms):
            sim.step()
            forward_peak = max(forward_peak, sim.behavior.forward_drive)
            backward_peak = max(backward_peak, sim.behavior.backward_drive)
            speed_peak = max(speed_peak, sim.behavior.speed)

        results[model_name] = {
            'model_name': connectome.name,
            'neurons': len(connectome.neurons),
            'synapses': len(connectome.synapses),
            'gap_junctions': len(connectome.gap_junctions),
            'baseline': baseline,
            'response': {
                'forward_peak': forward_peak,
                'backward_peak': backward_peak,
                'speed_peak': speed_peak,
                'forward_delta': forward_peak - baseline['forward'],
                'backward_delta': backward_peak - baseline['backward'],
            },
            'type_activities': sim.get_mean_activity_by_type(),
        }

    return results


def print_comparison(results: Dict):
    """Pretty-print comparison results."""
    print("\n" + "=" * 70)
    print("MODEL COMPARISON RESULTS")
    print("=" * 70)

    for model_key, data in results.items():
        print(f"\n{data['model_name']}")
        print("-" * 50)
        print(f"  Network: {data['neurons']} neurons, {data['synapses']} synapses, "
              f"{data['gap_junctions']} gap junctions")
        print(f"  Baseline: fwd={data['baseline']['forward']:.3f}, "
              f"back={data['baseline']['backward']:.3f}")
        print(f"  Peak Response:")
        print(f"    Forward:  {data['response']['forward_peak']:.3f} "
              f"(+{data['response']['forward_delta']:.3f})")
        print(f"    Backward: {data['response']['backward_peak']:.3f} "
              f"(+{data['response']['backward_delta']:.3f})")
        print(f"  Activity by type: ", end="")
        for ntype, act in data['type_activities'].items():
            print(f"{ntype}={act:.3f} ", end="")
        print()

    print("\n" + "=" * 70)


# =============================================================================
# Interactive Comparison Mode
# =============================================================================

def run_interactive_comparison():
    """Interactive mode to compare models with different stimuli."""
    print("\nC. elegans Multi-Model Comparison")
    print("=" * 50)

    # Load all available models
    models = {}
    loaders = get_available_loaders()

    for name, loader in loaders.items():
        try:
            connectome = loader.load()
            models[name] = MultiModelSimulation(connectome)
            print(f"Loaded {name}: {len(connectome.neurons)} neurons")
        except Exception as e:
            print(f"Could not load {name}: {e}")

    if not models:
        print("No models loaded!")
        return

    print("\nCommands:")
    print("  touch_a     - Apply anterior touch")
    print("  touch_p     - Apply posterior touch")
    print("  odor_good   - Apply attractive odor")
    print("  odor_bad    - Apply repulsive odor")
    print("  ach <val>   - Set ACh modulation (0.0-2.0)")
    print("  glu <val>   - Set Glu modulation (0.0-2.0)")
    print("  gaba <val>  - Set GABA modulation (0.0-2.0)")
    print("  run <n>     - Run n steps")
    print("  compare     - Run comparison with current settings")
    print("  reset       - Reset all models")
    print("  quit        - Exit")
    print()

    while True:
        try:
            cmd = input("> ").strip().lower().split()
            if not cmd:
                continue

            if cmd[0] == 'quit':
                break

            elif cmd[0] == 'reset':
                for sim in models.values():
                    sim.reset()
                print("All models reset")

            elif cmd[0] == 'touch_a':
                for sim in models.values():
                    sim.sensory.anterior_touch = 0.8
                print("Applied anterior touch")

            elif cmd[0] == 'touch_p':
                for sim in models.values():
                    sim.sensory.posterior_touch = 0.8
                print("Applied posterior touch")

            elif cmd[0] == 'odor_good':
                for sim in models.values():
                    sim.sensory.attractive_odor = 0.8
                print("Applied attractive odor")

            elif cmd[0] == 'odor_bad':
                for sim in models.values():
                    sim.sensory.repulsive_odor = 0.8
                print("Applied repulsive odor")

            elif cmd[0] in ['ach', 'glu', 'gaba']:
                if len(cmd) < 2:
                    print("Need value")
                    continue
                val = float(cmd[1])
                for sim in models.values():
                    if cmd[0] == 'ach':
                        sim.neurochemistry.ACh = val
                    elif cmd[0] == 'glu':
                        sim.neurochemistry.Glu = val
                    elif cmd[0] == 'gaba':
                        sim.neurochemistry.GABA = val
                print(f"Set {cmd[0].upper()} = {val}")

            elif cmd[0] == 'run':
                n = int(cmd[1]) if len(cmd) > 1 else 100
                for name, sim in models.items():
                    sim.step(n)
                print(f"\nAfter {n} steps:")
                for name, sim in models.items():
                    print(f"  {name}: fwd={sim.behavior.forward_drive:.3f}, "
                          f"back={sim.behavior.backward_drive:.3f}, "
                          f"speed={sim.behavior.speed:.3f}")

            elif cmd[0] == 'compare':
                print("\nRunning comparison with current neurochemical settings...")
                # Reset stimuli for fair comparison
                for sim in models.values():
                    sim.sensory = SensoryInput()
                    sim.reset()

                # Run comparison
                results = compare_models('anterior_touch', 0.8, 200)
                print_comparison(results)

        except KeyboardInterrupt:
            print("\nInterrupted")
            break
        except Exception as e:
            print(f"Error: {e}")


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == '--interactive':
        run_interactive_comparison()
    else:
        # Run default comparison
        print("Running model comparison with anterior touch stimulus...")
        results = compare_models('anterior_touch', 0.8, 200)
        print_comparison(results)

        print("\nRunning model comparison with posterior touch stimulus...")
        results = compare_models('posterior_touch', 0.8, 200)
        print_comparison(results)

        print("\nUse --interactive for interactive mode")
