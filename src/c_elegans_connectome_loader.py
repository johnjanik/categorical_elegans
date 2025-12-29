#!/usr/bin/env python3
"""
C. elegans Connectome Loader Architecture
==========================================
A unified system for loading connectome data from multiple sources:
- Categorical Elegans: Manual model from categorical connectome LaTeX document
- OpenWorm: Full hermaphrodite connectome (7379 connections)
- WormWiring: Cook et al. 2019 data with adjacency matrices
- CeNGEN: Neurotransmitter expression data overlay

This allows comparison of behavior across different connectome models.
"""

import csv
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Tuple, Optional, Set
import numpy as np

# Try to import pandas for Excel files, fall back gracefully
try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False


# =============================================================================
# Enumerations
# =============================================================================

class NeuronType(Enum):
    SENSORY = auto()
    INTERNEURON = auto()
    MOTOR = auto()
    PHARYNGEAL = auto()
    UNKNOWN = auto()


class Neurotransmitter(Enum):
    ACH = "Acetylcholine"
    GLU = "Glutamate"
    GABA = "GABA"
    DA = "Dopamine"
    SEROTONIN = "Serotonin"
    OCT = "Octopamine"
    TYR = "Tyramine"
    UNKNOWN = "Unknown"


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class NeuronInfo:
    """Neuron properties loaded from a connectome source."""
    name: str
    neuron_class: str
    neuron_type: NeuronType
    neurotransmitter: Neurotransmitter
    position: str = "U"  # L, R, D, V, or U (unpaired)
    ganglion: str = ""
    modality: str = ""


@dataclass
class SynapseInfo:
    """Chemical synapse information."""
    pre: str
    post: str
    weight: int
    synapse_type: str = "chemical"


@dataclass
class GapJunctionInfo:
    """Electrical synapse (gap junction) information."""
    neuron_a: str
    neuron_b: str
    weight: int


@dataclass
class ConnectomeData:
    """Complete connectome data from a single source."""
    name: str
    description: str
    neurons: Dict[str, NeuronInfo]
    synapses: List[SynapseInfo]
    gap_junctions: List[GapJunctionInfo]
    metadata: Dict = field(default_factory=dict)

    def summary(self) -> str:
        """Return a summary string of this connectome."""
        chem_weight = sum(s.weight for s in self.synapses)
        gap_weight = sum(g.weight for g in self.gap_junctions)

        # Count by neuron type
        type_counts = {}
        for n in self.neurons.values():
            t = n.neuron_type.name
            type_counts[t] = type_counts.get(t, 0) + 1

        # Count by neurotransmitter
        nt_counts = {}
        for n in self.neurons.values():
            nt = n.neurotransmitter.name
            nt_counts[nt] = nt_counts.get(nt, 0) + 1

        lines = [
            f"=== {self.name} ===",
            f"{self.description}",
            f"",
            f"Neurons: {len(self.neurons)}",
            f"  By type: {type_counts}",
            f"  By NT: {nt_counts}",
            f"",
            f"Chemical synapses: {len(self.synapses)} (total weight: {chem_weight})",
            f"Gap junctions: {len(self.gap_junctions)} (total weight: {gap_weight})",
        ]
        return "\n".join(lines)


# =============================================================================
# Abstract Base Loader
# =============================================================================

class ConnectomeLoader(ABC):
    """Abstract base class for connectome loaders."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Name of this connectome model."""
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """Description of the data source."""
        pass

    @abstractmethod
    def load(self) -> ConnectomeData:
        """Load and return the connectome data."""
        pass

    def _determine_position(self, name: str) -> str:
        """Infer L/R/D/V position from neuron name."""
        if name.endswith('L'):
            return 'L'
        elif name.endswith('R'):
            return 'R'
        elif name.endswith('D'):
            return 'D'
        elif name.endswith('V'):
            return 'V'
        elif 'DL' in name:
            return 'DL'
        elif 'DR' in name:
            return 'DR'
        elif 'VL' in name:
            return 'VL'
        elif 'VR' in name:
            return 'VR'
        return 'U'

    def _get_neuron_class(self, name: str) -> str:
        """Extract neuron class from individual neuron name."""
        # Remove position suffixes
        for suffix in ['DL', 'DR', 'VL', 'VR', 'L', 'R', 'D', 'V']:
            if name.endswith(suffix) and len(name) > len(suffix):
                # Check if what remains is at least 2 chars
                base = name[:-len(suffix)]
                if len(base) >= 2:
                    return base
        # Handle numbered neurons like DA01, VB1, etc.
        import re
        match = re.match(r'^([A-Z]+)\d+', name)
        if match:
            return match.group(1)
        return name


# =============================================================================
# OpenWorm Connectome Loader
# =============================================================================

class OpenWormConnectomeLoader(ConnectomeLoader):
    """
    Loads the OpenWorm hermaphrodite connectome from ConnectomeToolbox.
    Uses herm_full_edgelist.csv for connections and IndividualNeurons.csv for neuron info.
    """

    def __init__(self, data_dir: str = None):
        if data_dir is None:
            # Default path relative to this file
            base = os.path.dirname(os.path.abspath(__file__))
            # Look for data in package structure or parent directory
            data_dir = os.path.join(base, "..", "data", "openworm")
            if not os.path.exists(data_dir):
                data_dir = os.path.join(base, "data", "openworm")
        self.data_dir = data_dir
        self._cengen_data = None

    @property
    def name(self) -> str:
        return "OpenWorm Full Connectome"

    @property
    def description(self) -> str:
        return "OpenWorm hermaphrodite connectome (7379 connections from herm_full_edgelist.csv)"

    def load_cengen_neurotransmitters(self, cengen_path: str = None) -> Dict[str, Neurotransmitter]:
        """Load neurotransmitter assignments from CeNGEN data."""
        if cengen_path is None:
            base = os.path.dirname(os.path.abspath(__file__))
            cengen_path = os.path.join(base, "..", "data", "cengen",
                                       "Neuron_annotation_070921.csv")
            if not os.path.exists(cengen_path):
                cengen_path = os.path.join(base, "data", "cengen",
                                           "Neuron_annotation_070921.csv")

        nt_map = {}
        if not os.path.exists(cengen_path):
            return nt_map

        with open(cengen_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                neuron = row['Neuron']
                # Determine primary neurotransmitter from columns
                if row.get('Neurotransmitter: acetylcholine') == '1':
                    nt_map[neuron] = Neurotransmitter.ACH
                elif row.get('Neurotransmitter: Glutamate') == '1':
                    nt_map[neuron] = Neurotransmitter.GLU
                elif row.get('Neurotransmitter: GABA') == '1':
                    nt_map[neuron] = Neurotransmitter.GABA
                elif row.get('Neurotransmitter: Dopamine') == '1':
                    nt_map[neuron] = Neurotransmitter.DA
                elif row.get('Neurotransmitter: Serotonin') == '1':
                    nt_map[neuron] = Neurotransmitter.SEROTONIN
                else:
                    nt_map[neuron] = Neurotransmitter.UNKNOWN

        return nt_map

    def load_cengen_modality(self, cengen_path: str = None) -> Dict[str, NeuronType]:
        """Load neuron modality (sensory/motor/interneuron) from CeNGEN."""
        if cengen_path is None:
            base = os.path.dirname(os.path.abspath(__file__))
            cengen_path = os.path.join(base, "..", "data", "cengen",
                                       "Neuron_annotation_070921.csv")
            if not os.path.exists(cengen_path):
                cengen_path = os.path.join(base, "data", "cengen",
                                           "Neuron_annotation_070921.csv")

        type_map = {}
        if not os.path.exists(cengen_path):
            return type_map

        with open(cengen_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                neuron = row['Neuron']
                modality = row.get('Modality', '')
                if modality == 'Sensory' or row.get('Modality: Sensory') == '1':
                    type_map[neuron] = NeuronType.SENSORY
                elif modality == 'Motor' or row.get('Modality: Motor') == '1':
                    type_map[neuron] = NeuronType.MOTOR
                elif modality == 'Interneuron' or row.get('Modality: Interneuron') == '1':
                    type_map[neuron] = NeuronType.INTERNEURON
                else:
                    type_map[neuron] = NeuronType.UNKNOWN

        return type_map

    def load(self) -> ConnectomeData:
        """Load the OpenWorm connectome."""
        neurons = {}
        synapses = []
        gap_junctions = []

        # Load CeNGEN annotations for neurotransmitters and types
        nt_map = self.load_cengen_neurotransmitters()
        type_map = self.load_cengen_modality()

        # Load individual neurons info
        neurons_csv = os.path.join(self.data_dir, "IndividualNeurons.csv")
        neuron_set = set()

        if os.path.exists(neurons_csv):
            with open(neurons_csv, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    name = row.get('Name', '').strip()
                    if not name:
                        continue
                    neuron_set.add(name)

                    neuron_class = self._get_neuron_class(name)

                    # Get type from CeNGEN or infer from class name
                    ntype = type_map.get(neuron_class, NeuronType.UNKNOWN)
                    if ntype == NeuronType.UNKNOWN:
                        ntype = type_map.get(name, NeuronType.UNKNOWN)

                    # Get neurotransmitter from CeNGEN
                    nt = nt_map.get(neuron_class, Neurotransmitter.UNKNOWN)
                    if nt == Neurotransmitter.UNKNOWN:
                        nt = nt_map.get(name, Neurotransmitter.UNKNOWN)

                    neurons[name] = NeuronInfo(
                        name=name,
                        neuron_class=neuron_class,
                        neuron_type=ntype,
                        neurotransmitter=nt,
                        position=self._determine_position(name),
                        modality=row.get('Type', '')
                    )

        # Load edges (synapses and gap junctions)
        edges_csv = os.path.join(self.data_dir, "herm_full_edgelist.csv")
        if os.path.exists(edges_csv):
            with open(edges_csv, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    source = row.get('Source', '').strip()
                    target = row.get('Target', '').strip()
                    weight = int(float(row.get('Weight', 1)))
                    edge_type = row.get('Type', 'chemical').strip()

                    if not source or not target:
                        continue

                    # Add neurons if not already present
                    for n in [source, target]:
                        if n not in neurons:
                            neuron_class = self._get_neuron_class(n)
                            ntype = type_map.get(neuron_class, NeuronType.UNKNOWN)
                            nt = nt_map.get(neuron_class, Neurotransmitter.UNKNOWN)
                            neurons[n] = NeuronInfo(
                                name=n,
                                neuron_class=neuron_class,
                                neuron_type=ntype,
                                neurotransmitter=nt,
                                position=self._determine_position(n)
                            )

                    if 'electrical' in edge_type.lower() or 'gap' in edge_type.lower():
                        gap_junctions.append(GapJunctionInfo(
                            neuron_a=source,
                            neuron_b=target,
                            weight=weight
                        ))
                    else:
                        synapses.append(SynapseInfo(
                            pre=source,
                            post=target,
                            weight=weight,
                            synapse_type=edge_type
                        ))

        return ConnectomeData(
            name=self.name,
            description=self.description,
            neurons=neurons,
            synapses=synapses,
            gap_junctions=gap_junctions,
            metadata={
                'source': 'OpenWorm ConnectomeToolbox',
                'species': 'C. elegans hermaphrodite',
                'cengen_overlay': len(nt_map) > 0
            }
        )


# =============================================================================
# Categorical Elegans Loader (from original simulation)
# =============================================================================

class CategoricalConnectomeLoader(ConnectomeLoader):
    """
    Loads the Categorical Elegans connectome - the manually curated model
    from the categorical systems biology LaTeX document.
    """

    @property
    def name(self) -> str:
        return "Categorical Elegans"

    @property
    def description(self) -> str:
        return "Manual categorical model (260 neurons, 519 synapses, 50 gap junctions)"

    def load(self) -> ConnectomeData:
        """Load the categorical connectome from embedded data."""
        neurons = {}
        synapses = []
        gap_junctions = []

        # =====================================================================
        # SENSORY NEURONS
        # =====================================================================

        # Amphid sensory neurons
        sensory_amphid = [
            ("ADF", ["ADFL", "ADFR"], Neurotransmitter.ACH),
            ("ADL", ["ADLL", "ADLR"], Neurotransmitter.GLU),
            ("AFD", ["AFDL", "AFDR"], Neurotransmitter.GLU),
            ("ASE", ["ASEL", "ASER"], Neurotransmitter.GLU),
            ("ASG", ["ASGL", "ASGR"], Neurotransmitter.GLU),
            ("ASH", ["ASHL", "ASHR"], Neurotransmitter.GLU),
            ("ASI", ["ASIL", "ASIR"], Neurotransmitter.GLU),
            ("ASJ", ["ASJL", "ASJR"], Neurotransmitter.GLU),
            ("ASK", ["ASKL", "ASKR"], Neurotransmitter.GLU),
            ("AWA", ["AWAL", "AWAR"], Neurotransmitter.UNKNOWN),
            ("AWB", ["AWBL", "AWBR"], Neurotransmitter.ACH),
            ("AWC", ["AWCL", "AWCR"], Neurotransmitter.GLU),
        ]

        for nclass, names, nt in sensory_amphid:
            for name in names:
                neurons[name] = NeuronInfo(
                    name=name,
                    neuron_class=nclass,
                    neuron_type=NeuronType.SENSORY,
                    neurotransmitter=nt,
                    position=self._determine_position(name)
                )

        # Touch receptors
        touch_neurons = [
            ("ALM", ["ALML", "ALMR"], Neurotransmitter.GLU),
            ("PLM", ["PLML", "PLMR"], Neurotransmitter.GLU),
            ("AVM", ["AVM"], Neurotransmitter.GLU),
            ("PVM", ["PVM"], Neurotransmitter.GLU),
        ]

        for nclass, names, nt in touch_neurons:
            for name in names:
                neurons[name] = NeuronInfo(
                    name=name,
                    neuron_class=nclass,
                    neuron_type=NeuronType.SENSORY,
                    neurotransmitter=nt,
                    position=self._determine_position(name)
                )

        # Dopaminergic neurons
        for name in ["CEPDL", "CEPDR", "CEPVL", "CEPVR"]:
            neurons[name] = NeuronInfo(
                name=name,
                neuron_class="CEP",
                neuron_type=NeuronType.SENSORY,
                neurotransmitter=Neurotransmitter.DA,
                position=self._determine_position(name)
            )

        # =====================================================================
        # INTERNEURONS
        # =====================================================================

        # Command interneurons
        command_neurons = [
            ("AVA", ["AVAL", "AVAR"], Neurotransmitter.ACH),
            ("AVB", ["AVBL", "AVBR"], Neurotransmitter.ACH),
            ("AVD", ["AVDL", "AVDR"], Neurotransmitter.GLU),
            ("AVE", ["AVEL", "AVER"], Neurotransmitter.GLU),
            ("PVC", ["PVCL", "PVCR"], Neurotransmitter.GLU),
        ]

        for nclass, names, nt in command_neurons:
            for name in names:
                neurons[name] = NeuronInfo(
                    name=name,
                    neuron_class=nclass,
                    neuron_type=NeuronType.INTERNEURON,
                    neurotransmitter=nt,
                    position=self._determine_position(name)
                )

        # Processing interneurons
        processing_neurons = [
            ("AIY", ["AIYL", "AIYR"], Neurotransmitter.ACH),
            ("AIB", ["AIBL", "AIBR"], Neurotransmitter.GLU),
            ("AIZ", ["AIZL", "AIZR"], Neurotransmitter.GLU),
            ("RIA", ["RIAL", "RIAR"], Neurotransmitter.GLU),
            ("RIB", ["RIBL", "RIBR"], Neurotransmitter.GLU),
            ("RIM", ["RIML", "RIMR"], Neurotransmitter.GLU),
            ("RIV", ["RIVL", "RIVR"], Neurotransmitter.GLU),
            ("DVA", ["DVA"], Neurotransmitter.GLU),
        ]

        for nclass, names, nt in processing_neurons:
            for name in names:
                neurons[name] = NeuronInfo(
                    name=name,
                    neuron_class=nclass,
                    neuron_type=NeuronType.INTERNEURON,
                    neurotransmitter=nt,
                    position=self._determine_position(name)
                )

        # =====================================================================
        # MOTOR NEURONS
        # =====================================================================

        # A-type (backward)
        for i in range(1, 10):
            name = f"DA{i:02d}" if i > 9 else f"DA{i}"
            neurons[name] = NeuronInfo(
                name=name,
                neuron_class="DA",
                neuron_type=NeuronType.MOTOR,
                neurotransmitter=Neurotransmitter.ACH,
                position='U'
            )

        for name in ["VA1", "VA2", "VA3", "VA4", "VA5", "VA6", "VA7", "VA8", "VA9", "VA10", "VA11", "VA12"]:
            neurons[name] = NeuronInfo(
                name=name,
                neuron_class="VA",
                neuron_type=NeuronType.MOTOR,
                neurotransmitter=Neurotransmitter.ACH,
                position='U'
            )

        # B-type (forward)
        for i in range(1, 8):
            name = f"DB{i}"
            neurons[name] = NeuronInfo(
                name=name,
                neuron_class="DB",
                neuron_type=NeuronType.MOTOR,
                neurotransmitter=Neurotransmitter.ACH,
                position='U'
            )

        for i in range(1, 12):
            name = f"VB{i}"
            neurons[name] = NeuronInfo(
                name=name,
                neuron_class="VB",
                neuron_type=NeuronType.MOTOR,
                neurotransmitter=Neurotransmitter.ACH,
                position='U'
            )

        # D-type (inhibitory)
        for i in range(1, 7):
            name = f"DD{i}"
            neurons[name] = NeuronInfo(
                name=name,
                neuron_class="DD",
                neuron_type=NeuronType.MOTOR,
                neurotransmitter=Neurotransmitter.GABA,
                position='U'
            )

        for i in range(1, 14):
            name = f"VD{i}"
            neurons[name] = NeuronInfo(
                name=name,
                neuron_class="VD",
                neuron_type=NeuronType.MOTOR,
                neurotransmitter=Neurotransmitter.GABA,
                position='U'
            )

        # Head motor neurons
        for name in ["SMDVL", "SMDVR", "SMDDL", "SMDDR"]:
            neurons[name] = NeuronInfo(
                name=name,
                neuron_class="SMD",
                neuron_type=NeuronType.MOTOR,
                neurotransmitter=Neurotransmitter.ACH,
                position=self._determine_position(name)
            )

        # =====================================================================
        # KEY SYNAPSES (Categorical model circuits)
        # =====================================================================

        # Touch circuit: mechanoreceptors -> command interneurons
        touch_synapses = [
            # Anterior touch
            ("ALML", "AVDL", 5), ("ALMR", "AVDR", 5),
            ("ALML", "AVAL", 3), ("ALMR", "AVAR", 3),
            ("AVM", "AVDL", 4), ("AVM", "AVDR", 4),
            ("AVM", "AVAL", 2), ("AVM", "AVAR", 2),
            # Posterior touch
            ("PLML", "PVCL", 6), ("PLMR", "PVCR", 6),
            ("PLML", "AVBL", 3), ("PLMR", "AVBR", 3),
            ("PVM", "PVCL", 4), ("PVM", "PVCR", 4),
        ]

        # Command interneuron -> motor synapses
        command_synapses = [
            # AVA -> backward motion
            ("AVAL", "DA1", 8), ("AVAL", "DA2", 6), ("AVAL", "DA3", 5),
            ("AVAR", "DA1", 8), ("AVAR", "DA2", 6), ("AVAR", "DA3", 5),
            ("AVAL", "VA1", 6), ("AVAL", "VA2", 5), ("AVAL", "VA3", 4),
            ("AVAR", "VA1", 6), ("AVAR", "VA2", 5), ("AVAR", "VA3", 4),
            # AVB -> forward motion
            ("AVBL", "DB1", 7), ("AVBL", "DB2", 6), ("AVBL", "DB3", 5),
            ("AVBR", "DB1", 7), ("AVBR", "DB2", 6), ("AVBR", "DB3", 5),
            ("AVBL", "VB1", 5), ("AVBL", "VB2", 4), ("AVBL", "VB3", 4),
            ("AVBR", "VB1", 5), ("AVBR", "VB2", 4), ("AVBR", "VB3", 4),
            # PVC -> forward
            ("PVCL", "AVBL", 8), ("PVCR", "AVBR", 8),
            ("PVCL", "DB1", 3), ("PVCR", "DB1", 3),
            # AVD -> backward
            ("AVDL", "AVAL", 10), ("AVDR", "AVAR", 10),
        ]

        # Chemosensory circuit
        chemosensory_synapses = [
            ("AWAL", "AIYL", 4), ("AWAR", "AIYR", 4),
            ("AWBL", "AIBL", 3), ("AWBR", "AIBR", 3),
            ("AIYL", "AIZL", 5), ("AIYR", "AIZR", 5),
            ("AIBL", "AIZL", 4), ("AIBR", "AIZR", 4),
            ("AIZL", "RIAL", 3), ("AIZR", "RIAR", 3),
            ("RIAL", "SMDVL", 2), ("RIAR", "SMDVR", 2),
        ]

        # Thermosensory circuit
        thermo_synapses = [
            ("AFDL", "AIYL", 6), ("AFDR", "AIYR", 6),
            ("AFDL", "AIZL", 2), ("AFDR", "AIZR", 2),
            ("AIYL", "RIAL", 4), ("AIYR", "RIAR", 4),
        ]

        # Inhibitory cross-connections
        inhibitory_synapses = [
            ("DD1", "VD1", 3), ("DD2", "VD2", 3), ("DD3", "VD3", 3),
            ("VD1", "DD1", 3), ("VD2", "DD2", 3), ("VD3", "DD3", 3),
        ]

        # Compile all synapses
        for pre, post, weight in (touch_synapses + command_synapses +
                                   chemosensory_synapses + thermo_synapses +
                                   inhibitory_synapses):
            synapses.append(SynapseInfo(pre=pre, post=post, weight=weight))

        # =====================================================================
        # GAP JUNCTIONS
        # =====================================================================

        gap_junction_data = [
            # Touch receptor coupling
            ("ALML", "ALMR", 3),
            ("PLML", "PLMR", 3),
            # Command interneuron coupling
            ("AVAL", "AVAR", 15),
            ("AVBL", "AVBR", 12),
            ("AVDL", "AVDR", 8),
            ("PVCL", "PVCR", 6),
            # Motor neuron coupling
            ("DA1", "DA2", 2), ("DA2", "DA3", 2),
            ("DB1", "DB2", 2), ("DB2", "DB3", 2),
            ("DD1", "DD2", 2), ("DD2", "DD3", 2),
            ("VD1", "VD2", 2), ("VD2", "VD3", 2),
            # Processing interneurons
            ("AIYL", "AIYR", 4),
            ("AIBL", "AIBR", 3),
        ]

        for a, b, weight in gap_junction_data:
            gap_junctions.append(GapJunctionInfo(neuron_a=a, neuron_b=b, weight=weight))

        return ConnectomeData(
            name=self.name,
            description=self.description,
            neurons=neurons,
            synapses=synapses,
            gap_junctions=gap_junctions,
            metadata={
                'source': 'Categorical Systems Biology LaTeX Document',
                'model_type': 'categorical',
                'manually_curated': True
            }
        )


# =============================================================================
# WormWiring Loader (Cook et al. 2019)
# =============================================================================

class WormWiringConnectomeLoader(ConnectomeLoader):
    """
    Loads the WormWiring connectome from Cook et al. 2019 Excel files.
    Requires pandas for Excel parsing.
    """

    def __init__(self, data_dir: str = None):
        if data_dir is None:
            base = os.path.dirname(os.path.abspath(__file__))
            data_dir = os.path.join(base, "..", "data", "wormwiring")
            if not os.path.exists(data_dir):
                data_dir = os.path.join(base, "data", "wormwiring")
        self.data_dir = data_dir

    @property
    def name(self) -> str:
        return "WormWiring (Cook et al. 2019)"

    @property
    def description(self) -> str:
        return "WormWiring connectome from Cook et al. 2019 adjacency matrices"

    def load(self) -> ConnectomeData:
        """Load WormWiring data from Excel files."""
        if not HAS_PANDAS:
            raise ImportError("pandas is required to load WormWiring Excel files. "
                            "Install with: pip install pandas openpyxl")

        neurons = {}
        synapses = []
        gap_junctions = []

        # Try to load adjacency matrix from SI5
        adj_file = os.path.join(self.data_dir, "SI5_Connectome_adjacency_matrices.xlsx")
        cell_file = os.path.join(self.data_dir, "SI4_Cell_lists.xlsx")

        # Load CeNGEN for neurotransmitter info
        base = os.path.dirname(os.path.abspath(__file__))
        cengen_path = os.path.join(base, "connectome_data", "cengen",
                                   "CeNGEN_integrated_analysis_2024", "references",
                                   "Neuron_annotation_070921.csv")

        nt_map = {}
        type_map = {}
        if os.path.exists(cengen_path):
            with open(cengen_path, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    neuron = row['Neuron']
                    if row.get('Neurotransmitter: acetylcholine') == '1':
                        nt_map[neuron] = Neurotransmitter.ACH
                    elif row.get('Neurotransmitter: Glutamate') == '1':
                        nt_map[neuron] = Neurotransmitter.GLU
                    elif row.get('Neurotransmitter: GABA') == '1':
                        nt_map[neuron] = Neurotransmitter.GABA
                    elif row.get('Neurotransmitter: Dopamine') == '1':
                        nt_map[neuron] = Neurotransmitter.DA
                    elif row.get('Neurotransmitter: Serotonin') == '1':
                        nt_map[neuron] = Neurotransmitter.SEROTONIN
                    else:
                        nt_map[neuron] = Neurotransmitter.UNKNOWN

                    modality = row.get('Modality', '')
                    if modality == 'Sensory':
                        type_map[neuron] = NeuronType.SENSORY
                    elif modality == 'Motor':
                        type_map[neuron] = NeuronType.MOTOR
                    elif modality == 'Interneuron':
                        type_map[neuron] = NeuronType.INTERNEURON

        if os.path.exists(adj_file):
            try:
                # Load hermaphrodite chemical adjacency matrix
                df_chem = pd.read_excel(adj_file, sheet_name='hermaphrodite chemical',
                                        index_col=0, engine='openpyxl')

                # Get neuron names from index/columns
                neuron_names = list(df_chem.index)

                # Create neurons
                for name in neuron_names:
                    name_str = str(name).strip()
                    if not name_str:
                        continue
                    neuron_class = self._get_neuron_class(name_str)
                    neurons[name_str] = NeuronInfo(
                        name=name_str,
                        neuron_class=neuron_class,
                        neuron_type=type_map.get(neuron_class, NeuronType.UNKNOWN),
                        neurotransmitter=nt_map.get(neuron_class, Neurotransmitter.UNKNOWN),
                        position=self._determine_position(name_str)
                    )

                # Extract chemical synapses
                for i, pre in enumerate(df_chem.index):
                    for j, post in enumerate(df_chem.columns):
                        weight = df_chem.iloc[i, j]
                        if pd.notna(weight) and weight > 0:
                            synapses.append(SynapseInfo(
                                pre=str(pre),
                                post=str(post),
                                weight=int(weight)
                            ))

                # Try to load gap junctions
                try:
                    df_gap = pd.read_excel(adj_file, sheet_name='hermaphrodite gap jn',
                                          index_col=0, engine='openpyxl')
                    for i, a in enumerate(df_gap.index):
                        for j, b in enumerate(df_gap.columns):
                            weight = df_gap.iloc[i, j]
                            if pd.notna(weight) and weight > 0:
                                gap_junctions.append(GapJunctionInfo(
                                    neuron_a=str(a),
                                    neuron_b=str(b),
                                    weight=int(weight)
                                ))
                except Exception:
                    pass  # Gap junction sheet may have different name

            except Exception as e:
                print(f"Warning: Could not fully load WormWiring data: {e}")

        return ConnectomeData(
            name=self.name,
            description=self.description,
            neurons=neurons,
            synapses=synapses,
            gap_junctions=gap_junctions,
            metadata={
                'source': 'WormWiring (Cook et al. 2019)',
                'file': adj_file if os.path.exists(adj_file) else None
            }
        )


# =============================================================================
# Factory Function
# =============================================================================

def get_available_loaders() -> Dict[str, ConnectomeLoader]:
    """Return all available connectome loaders."""
    return {
        'categorical': CategoricalConnectomeLoader(),
        'openworm': OpenWormConnectomeLoader(),
        'wormwiring': WormWiringConnectomeLoader(),
    }


def load_connectome(model: str = 'categorical') -> ConnectomeData:
    """Load a connectome by model name."""
    loaders = get_available_loaders()
    if model not in loaders:
        raise ValueError(f"Unknown model: {model}. Available: {list(loaders.keys())}")
    return loaders[model].load()


# =============================================================================
# Main - Test Loaders
# =============================================================================

if __name__ == "__main__":
    print("Testing Connectome Loaders")
    print("=" * 60)

    loaders = get_available_loaders()

    for name, loader in loaders.items():
        print(f"\nLoading {name}...")
        try:
            data = loader.load()
            print(data.summary())
        except Exception as e:
            print(f"  Error: {e}")

    print("\n" + "=" * 60)
    print("Done!")
