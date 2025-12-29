"""
Categorical Elegans - A Minimal Model of C. elegans Behavior

This package provides tools for simulating the C. elegans nervous system
using multiple connectome data sources.

Modules:
    c_elegans_simulation: Core simulation engine (Categorical Elegans model)
    c_elegans_visualizer: Real-time visualization with matplotlib
    c_elegans_connectome_loader: Unified data loading from multiple sources
    c_elegans_multi_model: Model comparison framework
    behavioral_data_loader: Real behavioral data ingestion
    mdl_validator: MDL scoring for model validation
    context_validator: Context-specific behavioral validation
    minimal_model_search: Minimal model discovery algorithms
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

from .c_elegans_simulation import (
    ObservableVector,
    BehaviorHistory,
    CElegansSimulation,
)

from .behavioral_data_loader import (
    RealBehaviorDataset,
    N2_REFERENCE_OBSERVABLES,
    get_reference_observables,
    list_available_datasets,
)

from .mdl_validator import (
    MDLScore,
    MDLValidator,
    BehavioralContext,
    compare_models_mdl,
)

from .context_validator import (
    ContextValidator,
    ContextValidationResult,
    validate_all_contexts,
)

from .minimal_model_search import (
    MinimalModelSearcher,
    MinimalModelResult,
    find_minimal_model_for_context,
    find_all_minimal_models,
)

__version__ = "1.2.0"
__author__ = "John Janik"
