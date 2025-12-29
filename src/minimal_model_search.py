#!/usr/bin/env python3
"""
Minimal Model Search
====================

Implements algorithms to discover minimal neural network models
that preserve behavioral equivalence.

From minimal_model_theory.tex:
"A model M is minimal for behavior B if removing any neuron
causes ||M(σ) - D||₂ > ε for some stimulus σ."

This module implements:
1. Greedy neuron removal (remove least important first)
2. Eigenvector centrality ranking
3. Context-specific pruning
4. Behavioral equivalence verification
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple
from copy import deepcopy
import numpy as np

from .c_elegans_simulation import ObservableVector
from .c_elegans_connectome_loader import ConnectomeData, NeuronType
from .mdl_validator import MDLValidator, MDLScore, BehavioralContext
from .context_validator import (
    ContextValidator,
    CONTEXT_REQUIRED_NEURONS,
    CONTEXT_OPTIONAL_NEURONS,
)


# =============================================================================
# Neuron Importance Ranking
# =============================================================================

def compute_adjacency_matrix(connectome: ConnectomeData) -> Tuple[np.ndarray, List[str]]:
    """
    Build adjacency matrix from connectome.

    Args:
        connectome: Connectome data

    Returns:
        (adjacency_matrix, neuron_names)
    """
    neurons = list(connectome.neurons.keys())
    n = len(neurons)
    name_to_idx = {name: i for i, name in enumerate(neurons)}

    adj = np.zeros((n, n))

    # Add chemical synapses
    for syn in connectome.synapses:
        if syn.pre in name_to_idx and syn.post in name_to_idx:
            i, j = name_to_idx[syn.pre], name_to_idx[syn.post]
            adj[i, j] += np.log1p(syn.weight)

    # Add gap junctions (symmetric)
    for gj in connectome.gap_junctions:
        if gj.neuron_a in name_to_idx and gj.neuron_b in name_to_idx:
            i, j = name_to_idx[gj.neuron_a], name_to_idx[gj.neuron_b]
            adj[i, j] += np.log1p(gj.weight) * 0.5
            adj[j, i] += np.log1p(gj.weight) * 0.5

    return adj, neurons


def compute_eigenvector_centrality(connectome: ConnectomeData,
                                    max_iter: int = 100,
                                    tol: float = 1e-6) -> Dict[str, float]:
    """
    Compute eigenvector centrality for each neuron.

    High centrality = more important in network structure.

    Args:
        connectome: Connectome data
        max_iter: Maximum iterations
        tol: Convergence tolerance

    Returns:
        Dict mapping neuron name to centrality score
    """
    adj, neurons = compute_adjacency_matrix(connectome)
    n = len(neurons)

    if n == 0:
        return {}

    # Power iteration for principal eigenvector
    x = np.ones(n) / np.sqrt(n)

    for _ in range(max_iter):
        # Sum of incoming and outgoing weights
        y = (adj @ x + adj.T @ x) / 2
        norm = np.linalg.norm(y)
        if norm < 1e-10:
            break
        y = y / norm

        if np.linalg.norm(y - x) < tol:
            break
        x = y

    return {neurons[i]: float(x[i]) for i in range(n)}


def compute_degree_centrality(connectome: ConnectomeData) -> Dict[str, float]:
    """
    Compute degree centrality (in + out degree) for each neuron.

    Args:
        connectome: Connectome data

    Returns:
        Dict mapping neuron name to degree
    """
    degree = {name: 0.0 for name in connectome.neurons}

    for syn in connectome.synapses:
        if syn.pre in degree:
            degree[syn.pre] += syn.weight
        if syn.post in degree:
            degree[syn.post] += syn.weight

    for gj in connectome.gap_junctions:
        if gj.neuron_a in degree:
            degree[gj.neuron_a] += gj.weight
        if gj.neuron_b in degree:
            degree[gj.neuron_b] += gj.weight

    # Normalize
    max_degree = max(degree.values()) if degree else 1
    return {k: v / max_degree for k, v in degree.items()}


def rank_neurons_by_importance(connectome: ConnectomeData,
                                context: Optional[BehavioralContext] = None) -> List[Tuple[str, float]]:
    """
    Rank neurons by importance (least important first).

    Combines eigenvector centrality with context-specific requirements.

    Args:
        connectome: Connectome data
        context: Optional behavioral context

    Returns:
        List of (neuron_name, importance_score) sorted ascending
    """
    # Compute centralities
    eigen = compute_eigenvector_centrality(connectome)
    degree = compute_degree_centrality(connectome)

    # Combine scores
    importance = {}
    for name in connectome.neurons:
        score = eigen.get(name, 0) * 0.6 + degree.get(name, 0) * 0.4
        importance[name] = score

    # Boost required neurons for context
    if context and context in CONTEXT_REQUIRED_NEURONS:
        required = CONTEXT_REQUIRED_NEURONS[context]
        for name in required:
            if name in importance:
                importance[name] += 10.0  # Make them very important

    # Penalize optional neurons for context
    if context and context in CONTEXT_OPTIONAL_NEURONS:
        optional = CONTEXT_OPTIONAL_NEURONS[context]
        for name in optional:
            if name in importance:
                importance[name] *= 0.5  # Reduce importance

    # Sort by importance (ascending = least important first)
    ranked = sorted(importance.items(), key=lambda x: x[1])

    return ranked


# =============================================================================
# Connectome Pruning
# =============================================================================

def remove_neuron(connectome: ConnectomeData, neuron_name: str) -> ConnectomeData:
    """
    Create a new connectome with a neuron removed.

    Args:
        connectome: Original connectome
        neuron_name: Neuron to remove

    Returns:
        New ConnectomeData without the neuron
    """
    # Create shallow copy of connectome
    new_connectome = ConnectomeData(name=connectome.name)

    # Copy neurons except the removed one
    new_connectome.neurons = {
        k: v for k, v in connectome.neurons.items()
        if k != neuron_name
    }

    # Copy synapses not involving the removed neuron
    new_connectome.synapses = [
        syn for syn in connectome.synapses
        if syn.pre != neuron_name and syn.post != neuron_name
    ]

    # Copy gap junctions not involving the removed neuron
    new_connectome.gap_junctions = [
        gj for gj in connectome.gap_junctions
        if gj.neuron_a != neuron_name and gj.neuron_b != neuron_name
    ]

    return new_connectome


def remove_neurons(connectome: ConnectomeData, neuron_names: Set[str]) -> ConnectomeData:
    """
    Create a new connectome with multiple neurons removed.

    Args:
        connectome: Original connectome
        neuron_names: Set of neurons to remove

    Returns:
        New ConnectomeData without the neurons
    """
    new_connectome = ConnectomeData(name=connectome.name)

    new_connectome.neurons = {
        k: v for k, v in connectome.neurons.items()
        if k not in neuron_names
    }

    new_connectome.synapses = [
        syn for syn in connectome.synapses
        if syn.pre not in neuron_names and syn.post not in neuron_names
    ]

    new_connectome.gap_junctions = [
        gj for gj in connectome.gap_junctions
        if gj.neuron_a not in neuron_names and gj.neuron_b not in neuron_names
    ]

    return new_connectome


# =============================================================================
# Minimal Model Search Algorithm
# =============================================================================

@dataclass
class MinimalModelResult:
    """Result of minimal model search."""
    context: Optional[BehavioralContext]
    original_neurons: int
    original_synapses: int
    minimal_neurons: int
    minimal_synapses: int

    # The minimal model
    minimal_connectome: Optional[ConnectomeData] = None

    # Neurons removed
    removed_neurons: Set[str] = field(default_factory=set)

    # Search statistics
    neurons_tested: int = 0
    neurons_removable: int = 0

    # Quality metrics
    final_mdl_score: Optional[MDLScore] = None
    behavioral_distance: float = 0.0

    @property
    def neuron_reduction(self) -> float:
        """Fraction of neurons removed."""
        return 1 - (self.minimal_neurons / self.original_neurons)

    @property
    def synapse_reduction(self) -> float:
        """Fraction of synapses removed."""
        return 1 - (self.minimal_synapses / self.original_synapses)

    def summary(self) -> str:
        ctx_name = self.context.name if self.context else "ALL"
        return (
            f"Minimal Model Search Result ({ctx_name})\n"
            f"{'=' * 50}\n"
            f"Original:  {self.original_neurons} neurons, {self.original_synapses} synapses\n"
            f"Minimal:   {self.minimal_neurons} neurons, {self.minimal_synapses} synapses\n"
            f"Reduction: {self.neuron_reduction:.1%} neurons, {self.synapse_reduction:.1%} synapses\n"
            f"Removed:   {self.neurons_removable} neurons\n"
            f"Behavioral distance: {self.behavioral_distance:.4f}\n"
        )


class MinimalModelSearcher:
    """
    Search for minimal models using greedy neuron removal.

    Algorithm:
    1. Rank neurons by importance (least important first)
    2. For each neuron in order:
       a. Try removing the neuron
       b. Run simulation and compute observable
       c. If behavioral distance < epsilon, keep removal
       d. Else restore neuron
    3. Return minimal connectome
    """

    def __init__(self,
                 reference_observable: Optional[ObservableVector] = None,
                 epsilon: float = 0.1,
                 context: Optional[BehavioralContext] = None):
        """
        Initialize minimal model searcher.

        Args:
            reference_observable: Target behavior to preserve
            epsilon: Maximum allowed behavioral distance
            context: Optional behavioral context
        """
        self.reference_observable = reference_observable
        self.epsilon = epsilon
        self.context = context

        # Create validators
        self.mdl_validator = MDLValidator(
            reference_observable=reference_observable,
            epsilon=epsilon
        )

        if context:
            self.context_validator = ContextValidator(context)
        else:
            self.context_validator = None

    def search(self,
               connectome: ConnectomeData,
               simulation_factory,  # Callable that creates simulation from connectome
               max_neurons_to_try: int = 500,
               verbose: bool = True) -> MinimalModelResult:
        """
        Search for minimal model by greedy neuron removal.

        Args:
            connectome: Starting connectome
            simulation_factory: Function that creates CElegansSimulation from ConnectomeData
            max_neurons_to_try: Maximum neurons to attempt removing
            verbose: Print progress

        Returns:
            MinimalModelResult
        """
        result = MinimalModelResult(
            context=self.context,
            original_neurons=len(connectome.neurons),
            original_synapses=len(connectome.synapses),
            minimal_neurons=len(connectome.neurons),
            minimal_synapses=len(connectome.synapses),
        )

        # Get reference observable if not provided
        if self.reference_observable is None:
            sim = simulation_factory(connectome)
            sim.step(1000)
            self.reference_observable = sim.get_observables(force_update=True)
            self.mdl_validator.reference_observable = self.reference_observable

        # Rank neurons by importance
        ranked = rank_neurons_by_importance(connectome, self.context)

        if verbose:
            print(f"Starting minimal model search")
            print(f"  Original: {result.original_neurons} neurons, {result.original_synapses} synapses")
            print(f"  Epsilon: {self.epsilon}")
            print(f"  Context: {self.context.name if self.context else 'ALL'}")

        # Current working connectome
        current = connectome
        removed = set()

        # Try removing neurons in order of least importance
        for i, (neuron_name, importance) in enumerate(ranked):
            if i >= max_neurons_to_try:
                break

            result.neurons_tested += 1

            # Skip required neurons
            if self.context and neuron_name in CONTEXT_REQUIRED_NEURONS.get(self.context, set()):
                continue

            # Try removing this neuron
            candidate = remove_neuron(current, neuron_name)

            # Skip if this would leave too few neurons
            if len(candidate.neurons) < 10:
                continue

            try:
                # Create simulation and run
                sim = simulation_factory(candidate)
                sim.step(1000)
                obs = sim.get_observables(force_update=True)

                # Check behavioral equivalence
                distance = obs.distance(self.reference_observable)
                normalized_distance = distance / np.sqrt(14)

                if normalized_distance < self.epsilon:
                    # Removal is acceptable
                    current = candidate
                    removed.add(neuron_name)
                    result.neurons_removable += 1

                    if verbose and result.neurons_removable % 10 == 0:
                        print(f"  Removed {result.neurons_removable} neurons "
                              f"({len(current.neurons)} remaining)")

            except Exception as e:
                # Simulation failed, skip this neuron
                if verbose:
                    print(f"  Error removing {neuron_name}: {e}")
                continue

        # Store result
        result.minimal_connectome = current
        result.minimal_neurons = len(current.neurons)
        result.minimal_synapses = len(current.synapses)
        result.removed_neurons = removed

        # Compute final metrics
        try:
            sim = simulation_factory(current)
            sim.step(1000)
            final_obs = sim.get_observables(force_update=True)
            result.behavioral_distance = final_obs.distance(self.reference_observable)
            result.final_mdl_score = self.mdl_validator.score_connectome(current, final_obs)
        except Exception:
            pass

        if verbose:
            print(f"\nSearch complete:")
            print(result.summary())

        return result


# =============================================================================
# Context-Specific Minimal Models
# =============================================================================

def find_minimal_model_for_context(
    full_connectome: ConnectomeData,
    context: BehavioralContext,
    simulation_factory,
    epsilon: float = 0.15,
    verbose: bool = True,
) -> MinimalModelResult:
    """
    Find minimal model for a specific behavioral context.

    Args:
        full_connectome: Complete connectome
        context: Behavioral context
        simulation_factory: Function to create simulation
        epsilon: Behavioral equivalence threshold
        verbose: Print progress

    Returns:
        MinimalModelResult for this context
    """
    searcher = MinimalModelSearcher(
        epsilon=epsilon,
        context=context,
    )

    return searcher.search(
        full_connectome,
        simulation_factory,
        verbose=verbose,
    )


def find_all_minimal_models(
    full_connectome: ConnectomeData,
    simulation_factory,
    contexts: Optional[List[BehavioralContext]] = None,
    epsilon: float = 0.15,
    verbose: bool = True,
) -> Dict[BehavioralContext, MinimalModelResult]:
    """
    Find minimal models for all behavioral contexts.

    Args:
        full_connectome: Complete connectome
        simulation_factory: Function to create simulation
        contexts: Specific contexts (None = all)
        epsilon: Behavioral equivalence threshold
        verbose: Print progress

    Returns:
        Dict mapping context to minimal model result
    """
    if contexts is None:
        contexts = [
            BehavioralContext.TOUCH_ESCAPE,
            BehavioralContext.CHEMOTAXIS,
            BehavioralContext.THERMOTAXIS,
            BehavioralContext.FORAGING,
        ]

    results = {}

    for ctx in contexts:
        if verbose:
            print(f"\n{'=' * 60}")
            print(f"Finding minimal model for {ctx.name}")
            print('=' * 60)

        results[ctx] = find_minimal_model_for_context(
            full_connectome,
            ctx,
            simulation_factory,
            epsilon=epsilon,
            verbose=verbose,
        )

    # Summary table
    if verbose:
        print(f"\n{'=' * 60}")
        print("MINIMAL MODEL COMPARISON")
        print('=' * 60)
        print(f"{'Context':<20} {'Original':>12} {'Minimal':>12} {'Reduction':>12}")
        print('-' * 60)

        for ctx, result in results.items():
            print(f"{ctx.name:<20} {result.original_neurons:>12} "
                  f"{result.minimal_neurons:>12} {result.neuron_reduction:>11.1%}")

    return results


# =============================================================================
# Main / Testing
# =============================================================================

if __name__ == "__main__":
    print("Minimal Model Search")
    print("=" * 60)

    # Test centrality computation with mock data
    from .c_elegans_connectome_loader import load_connectome

    print("\nLoading Categorical Elegans connectome...")
    try:
        connectome = load_connectome('categorical')
        print(f"Loaded: {len(connectome.neurons)} neurons, "
              f"{len(connectome.chemical_synapses)} synapses")

        print("\nComputing neuron importance ranking...")
        ranked = rank_neurons_by_importance(connectome)

        print("\nLeast important neurons (candidates for removal):")
        for name, score in ranked[:10]:
            print(f"  {name}: {score:.4f}")

        print("\nMost important neurons (should preserve):")
        for name, score in ranked[-10:]:
            print(f"  {name}: {score:.4f}")

        print("\nContext-specific ranking (Touch Escape):")
        ranked_touch = rank_neurons_by_importance(
            connectome,
            context=BehavioralContext.TOUCH_ESCAPE
        )
        print("Least important for touch:")
        for name, score in ranked_touch[:10]:
            print(f"  {name}: {score:.4f}")

    except Exception as e:
        print(f"Could not load connectome: {e}")
        print("This is expected if running standalone.")
