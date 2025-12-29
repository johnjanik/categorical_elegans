"""
Categorical Elegans - A Minimal Model of C. elegans Behavior

This package provides tools for simulating the C. elegans nervous system
using multiple connectome data sources.

Modules:
    c_elegans_simulation: Core simulation engine (Categorical Elegans model)
    c_elegans_visualizer: Real-time visualization with matplotlib
    c_elegans_connectome_loader: Unified data loading from multiple sources
    c_elegans_multi_model: Model comparison framework
"""

from .c_elegans_connectome_loader import (
    load_connectome,
    get_available_loaders,
    ConnectomeData,
    NeuronType,
    Neurotransmitter,
    CategoricalConnectomeLoader,
    OpenWormConnectomeLoader,
)

from .c_elegans_multi_model import (
    MultiModelSimulation,
    compare_models,
)

__version__ = "1.0.0"
__author__ = "John Janik"
