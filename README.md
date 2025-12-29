# Categorical Elegans

**A Minimal Model of the *C. elegans* Nervous System**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains the code and data for reproducing the results in:

> **Minimal Model Theory for *C. elegans* Behavior: Three Formal Approaches**
>
> We demonstrate that a manually curated "Categorical Elegans" model with 121 neurons
> achieves 100% accuracy on touch-escape behavior while the full OpenWorm connectome
> (448 neurons) saturates and fails to differentiate stimuli.

## Quick Start

```bash
# Clone the repository
git clone https://github.com/johnjanik/categorical_elegans.git
cd categorical_elegans

# Install dependencies
pip install -r requirements.txt

# Run the multi-model comparison (reproduces manuscript results)
python c_elegans_multi_model.py

# Launch interactive visualization
python c_elegans_visualizer.py
```

## Key Results

| Model | Neurons | Synapses | Anterior Touch | Posterior Touch | Status |
|-------|---------|----------|----------------|-----------------|--------|
| Categorical Elegans | 121 | 68 | Backward (+0.21) | Forward (+0.19) | ✓ Works |
| OpenWorm Full | 448 | 4,681 | Saturated | Saturated | ✗ Fails |

The Categorical model achieves **60% neuron reduction** and **99% synapse reduction** while maintaining biologically correct behavior.

## Repository Structure

```
categorical_elegans/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── LICENSE                            # MIT License
│
├── src/                               # Source code
│   ├── c_elegans_simulation.py        # Core simulation engine
│   ├── c_elegans_visualizer.py        # Real-time visualization
│   ├── c_elegans_connectome_loader.py # Multi-source data loader
│   └── c_elegans_multi_model.py       # Model comparison framework
│
├── manuscripts/                       # LaTeX documents
│   ├── minimal_model_theory.tex       # Main manuscript
│   └── c_elegans_categorical_connectome.tex  # Connectome specification
│
└── data/                              # Connectome data
    ├── openworm/                      # OpenWorm ConnectomeToolbox
    ├── wormwiring/                    # Cook et al. 2019 data
    └── cengen/                        # CeNGEN neurotransmitter data
```

## Installation

### Requirements

- Python 3.8 or higher
- NumPy
- Matplotlib (for visualization)
- Optional: pandas, openpyxl (for WormWiring Excel files)

### Install via pip

```bash
pip install -r requirements.txt
```

### Install from source

```bash
git clone https://github.com/johnjanik/categorical_elegans.git
cd categorical_elegans
pip install -e .
```

## Usage

### 1. Reproduce Manuscript Results

Run the multi-model comparison to reproduce Table 1 from the manuscript:

```bash
python src/c_elegans_multi_model.py
```

Expected output:
```
Running model comparison with anterior touch stimulus...

======================================================================
MODEL COMPARISON RESULTS
======================================================================

Categorical Elegans
--------------------------------------------------
  Network: 121 neurons, 68 synapses, 16 gap junctions
  Baseline: fwd=0.053, back=0.052
  Peak Response:
    Forward:  0.053 (+0.000)
    Backward: 0.260 (+0.209)  <-- Correct backward escape!

OpenWorm Full Connectome
--------------------------------------------------
  Network: 448 neurons, 4681 synapses, 2698 gap junctions
  Baseline: fwd=0.737, back=0.957
  Peak Response:
    Forward:  0.739 (+0.003)
    Backward: 0.957 (+0.000)  <-- Saturated, no differentiation
```

### 2. Interactive Visualization

Launch the real-time neural simulation with on-screen controls:

```bash
python src/c_elegans_visualizer.py
```

Features:
- **Worm body animation** with neural activity overlays
- **Neural heatmap** showing all 121 neurons by type
- **Circuit diagrams** for touch, locomotion, and chemotaxis
- **Neurochemical sliders** (ACh, Glu, GABA, DA, 5-HT, Oct, Tyr)
- **Sensory buttons** to apply touch and odor stimuli

### 3. Interactive Command-Line Mode

Explore model behavior interactively:

```bash
python src/c_elegans_multi_model.py --interactive
```

Commands:
```
> touch_a      # Apply anterior touch
> touch_p      # Apply posterior touch
> ach 1.5      # Increase acetylcholine to 1.5x
> gaba 0.5     # Decrease GABA to 0.5x
> run 100      # Run 100 simulation steps
> compare      # Run full model comparison
> reset        # Reset all models
> quit         # Exit
```

### 4. Load Different Connectome Models

```python
from c_elegans_connectome_loader import load_connectome, get_available_loaders

# See available models
print(get_available_loaders().keys())
# -> dict_keys(['categorical', 'openworm', 'wormwiring'])

# Load the Categorical Elegans model
cat_connectome = load_connectome('categorical')
print(cat_connectome.summary())

# Load the full OpenWorm connectome
ow_connectome = load_connectome('openworm')
print(ow_connectome.summary())
```

### 5. Run Custom Simulations

```python
from c_elegans_multi_model import MultiModelSimulation
from c_elegans_connectome_loader import load_connectome

# Load a connectome
connectome = load_connectome('categorical')

# Create simulation
sim = MultiModelSimulation(connectome)

# Set neurochemical modulation
sim.neurochemistry.GABA = 0.5  # Reduce inhibition

# Apply sensory input
sim.sensory.anterior_touch = 0.8

# Run simulation
for _ in range(200):
    sim.step()
    print(f"Forward: {sim.behavior.forward_drive:.3f}, "
          f"Backward: {sim.behavior.backward_drive:.3f}")
```

## Data Sources

### Categorical Elegans (121 neurons)
Manually curated from the categorical connectome specification, containing:
- 34 sensory neurons (touch receptors, amphid chemosensory)
- 25 interneurons (command interneurons, processing)
- 62 motor neurons (complete ventral nerve cord)

### OpenWorm (448 neurons, 7,379 connections)
Downloaded from [OpenWorm ConnectomeToolbox](https://github.com/openworm/ConnectomeToolbox):
- `herm_full_edgelist.csv`: Complete hermaphrodite connectome
- `IndividualNeurons.csv`: Neuron metadata

### CeNGEN (Neurotransmitter annotations)
Downloaded from [CeNGEN](https://cengen.org):
- Neurotransmitter assignments for 161 neuron classes
- Used to annotate both Categorical and OpenWorm models

### WormWiring (Cook et al. 2019)
Excel files from [WormWiring](https://wormwiring.org):
- `SI5_Connectome_adjacency_matrices.xlsx`
- Requires pandas and openpyxl to load

## Compiling the Manuscript

```bash
cd manuscripts
pdflatex minimal_model_theory.tex
pdflatex minimal_model_theory.tex  # Run twice for cross-references
```

## Citation

If you use this code in your research, please cite:

```bibtex
@software{categorical_elegans,
  author = {Janik, John},
  title = {Categorical Elegans: A Minimal Model of C. elegans Behavior},
  year = {2024},
  url = {https://github.com/johnjanik/categorical_elegans}
}
```

## Key References

1. **White et al. (1986)** - Original C. elegans connectome
2. **Cook et al. (2019)** - Updated whole-animal connectomes (WormWiring)
3. **Taylor et al. (2021)** - CeNGEN neurotransmitter atlas
4. **Chalfie et al. (1985)** - Touch sensitivity circuit
5. **Towlson et al. (2013)** - Rich club structure

## License

MIT License - see [LICENSE](LICENSE) for details.

## Contributing

Contributions welcome! Please open an issue or submit a pull request.

## Acknowledgments

- [OpenWorm](https://openworm.org) for connectome data
- [CeNGEN](https://cengen.org) for neurotransmitter expression data
- [WormWiring](https://wormwiring.org) for Cook et al. 2019 data
