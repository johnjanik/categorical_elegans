#!/usr/bin/env python3
"""
C. elegans Real-Time Visualization
===================================
Real-time visualization of the 302-neuron C. elegans nervous system
with on-screen controls for neurochemical modulation and sensory input.

Usage:
    python c_elegans_visualizer.py

Controls:
    - Neurochemical sliders: Adjust ACh, Glu, GABA, DA, 5-HT, Oct, Tyr levels
    - Sensory buttons: Toggle touch, odor, and other stimuli
    - Speed slider: Control simulation speed (0.5x to 5x)
    - Pause/Reset buttons: Control simulation state
"""

import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # Use TkAgg backend for widgets
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider, Button
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Circle, FancyArrowPatch, Polygon
from matplotlib.collections import PatchCollection, LineCollection
from collections import deque
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from c_elegans_simulation import (
    CElegansSimulation,
    NeuronType,
    Neurotransmitter,
    NeurochemicalState,
    SensoryInput,
    BehavioralState
)


# =============================================================================
# Color Schemes
# =============================================================================

COLORS = {
    # Neuron types
    'SENSORY': '#4CAF50',       # Green
    'INTERNEURON': '#2196F3',   # Blue
    'MOTOR': '#FF9800',         # Orange
    'PHARYNGEAL': '#9C27B0',    # Purple

    # Neurotransmitters
    'ACH': '#4CAF50',           # Green (excitatory)
    'GLU': '#8BC34A',           # Light green (excitatory)
    'GABA': '#F44336',          # Red (inhibitory)
    'DA': '#FFC107',            # Amber (modulatory)
    'SEROTONIN': '#E91E63',     # Pink (modulatory)
    'OCT': '#FF5722',           # Deep orange (arousal)
    'TYR': '#795548',           # Brown (reversal)

    # Circuit colors
    'EXCITATORY': '#4CAF50',    # Green
    'INHIBITORY': '#F44336',    # Red
    'MODULATORY': '#FFC107',    # Amber

    # Worm body
    'BODY_FILL': '#D7CCC8',     # Light brown
    'BODY_OUTLINE': '#5D4037',  # Dark brown

    # Background
    'BG_DARK': '#1a1a2e',
    'BG_PANEL': '#16213e',
    'TEXT': '#e8e8e8',
}

ACTIVITY_CMAP = 'plasma'

# Circuit definitions with associated neurons and neurotransmitters
CIRCUITS = {
    'Touch Response': {
        'neurons': ['ALM', 'PLM', 'AVM', 'PVM', 'AVD', 'PVC'],
        'color': '#FF5722',  # Deep orange
        'affected_by': ['GLU'],  # Primarily glutamatergic
        'description': 'Mechanosensory escape',
    },
    'Forward Command': {
        'neurons': ['AVB', 'PVC', 'DB', 'VB'],
        'color': '#2196F3',  # Blue
        'affected_by': ['ACH'],  # Cholinergic
        'description': 'Forward locomotion',
    },
    'Backward Command': {
        'neurons': ['AVA', 'AVD', 'AVE', 'DA', 'VA'],
        'color': '#F44336',  # Red
        'affected_by': ['GLU'],  # Glutamatergic
        'description': 'Backward locomotion',
    },
    'Inhibitory': {
        'neurons': ['DD', 'VD', 'RIB', 'RIS'],
        'color': '#9C27B0',  # Purple
        'affected_by': ['GABA'],  # GABAergic
        'description': 'Cross-inhibition',
    },
    'Chemotaxis': {
        'neurons': ['AWA', 'AWB', 'ASE', 'AIY', 'AIB', 'AIZ'],
        'color': '#4CAF50',  # Green
        'affected_by': ['GLU', 'ACH'],
        'description': 'Chemical navigation',
    },
    'Modulatory': {
        'neurons': ['CEP', 'ADE', 'PDE', 'NSM', 'RIC', 'RIM', 'RIH'],
        'color': '#FFC107',  # Amber
        'affected_by': ['DA', 'SEROTONIN', 'OCT', 'TYR'],
        'description': 'Neuromodulation',
    },
}

# Map neurotransmitter names to attribute names
NT_TO_ATTR = {
    'ACH': 'acetylcholine',
    'GLU': 'glutamate',
    'GABA': 'gaba',
    'DA': 'dopamine',
    'SEROTONIN': 'serotonin',
    'OCT': 'octopamine',
    'TYR': 'tyramine',
}


# =============================================================================
# Worm Body Renderer
# =============================================================================

class WormBodyRenderer:
    """Renders the animated worm body with neural activity overlays."""

    def __init__(self, ax, simulation):
        self.ax = ax
        self.sim = simulation
        self.n_segments = 24

        # Worm position in world
        self.x_pos = 0.5
        self.y_pos = 0.5
        self.heading = 0.0  # radians

        # Body artists
        self.body_line = None
        self.body_fill = None
        self.neural_circles = []

        # Key neurons to display on body
        self.display_neurons = {
            # Head (0.0-0.15)
            'AVAL': (0.08, 0.02), 'AVAR': (0.08, -0.02),
            'AVBL': (0.10, 0.02), 'AVBR': (0.10, -0.02),
            # Touch sensors
            'ALML': (0.15, 0.03), 'ALMR': (0.15, -0.03),
            'PLML': (0.85, 0.02), 'PLMR': (0.85, -0.02),
            # Pharynx
            'NSML': (0.05, 0.01), 'NSMR': (0.05, -0.01),
        }

        self.setup()

    def setup(self):
        """Initialize body visualization."""
        self.ax.set_xlim(-0.1, 1.1)
        self.ax.set_ylim(-0.4, 0.4)
        self.ax.set_aspect('equal')
        self.ax.set_facecolor(COLORS['BG_PANEL'])
        self.ax.set_title('Worm Body', color=COLORS['TEXT'], fontsize=10)
        self.ax.axis('off')

        # Create initial body shape
        x, y = self._compute_body_curve()

        # Body fill (polygon)
        self.body_line, = self.ax.plot(x, y, color=COLORS['BODY_OUTLINE'],
                                        linewidth=8, solid_capstyle='round')
        self.body_inner, = self.ax.plot(x, y, color=COLORS['BODY_FILL'],
                                         linewidth=6, solid_capstyle='round')

        # Neural activity circles
        for name, (rel_x, rel_y) in self.display_neurons.items():
            circle = Circle((rel_x, rel_y), 0.02,
                           facecolor='gray', edgecolor='white',
                           linewidth=0.5, alpha=0.8)
            self.ax.add_patch(circle)
            self.neural_circles.append((name, circle))

        # Direction indicator
        self.direction_arrow = self.ax.annotate('', xy=(0.0, 0.0), xytext=(-0.05, 0.0),
                                                 arrowprops=dict(arrowstyle='->',
                                                                color=COLORS['TEXT'],
                                                                lw=1.5),
                                                 annotation_clip=False)

        # Behavior label
        self.behavior_text = self.ax.text(0.5, -0.3, '', ha='center',
                                          color=COLORS['TEXT'], fontsize=9)

    def _get_class_mean(self, prefix):
        """Get mean activity of neurons with given prefix."""
        activities = []
        for name in self.sim.neuron_names:
            if name.startswith(prefix):
                idx = self.sim.neuron_index[name]
                activities.append(self.sim.activity[idx])
        return np.mean(activities) if activities else 0.0

    def _compute_body_curve(self):
        """Generate body curve from motor neuron activity."""
        t = np.linspace(0, 1, self.n_segments)

        # Base sinusoidal body wave
        phase = self.sim.time * 0.005

        # Wave direction based on locomotion
        forward = self.sim.behavior.forward_drive
        backward = self.sim.behavior.backward_drive

        if forward > backward:
            wave_dir = 1
            amplitude = 0.08 * (0.3 + forward * 0.7)
        else:
            wave_dir = -1
            amplitude = 0.08 * (0.3 + backward * 0.7)

        # Speed affects wave frequency
        speed = self.sim.behavior.speed
        freq = 1.5 + speed * 1.5

        # Body wave
        y = amplitude * np.sin(2 * np.pi * freq * t + phase * wave_dir * 10)

        # Dorsal/ventral bias from D-type neurons
        dd_activity = self._get_class_mean('DD')
        vd_activity = self._get_class_mean('VD')
        dv_bias = (dd_activity - vd_activity) * 0.05
        y += dv_bias * np.sin(np.pi * t)

        # Head angle from behavior
        head_angle = self.sim.behavior.head_angle * 0.1
        head_region = np.exp(-t * 8)
        y += head_angle * head_region

        # Turn bias affects head
        turn = self.sim.behavior.turn_bias * 0.05
        y[:6] += turn * np.linspace(1, 0, 6)

        # Taper the body (thinner at head and tail)
        # x positions
        x = t

        return x, y

    def update(self):
        """Update body visualization."""
        x, y = self._compute_body_curve()

        self.body_line.set_data(x, y)
        self.body_inner.set_data(x, y)

        # Update neural activity circles
        cmap = plt.cm.get_cmap(ACTIVITY_CMAP)
        for name, circle in self.neural_circles:
            if name in self.sim.neuron_index:
                activity = self.sim.activity[self.sim.neuron_index[name]]
                color = cmap(activity)
                circle.set_facecolor(color)
                # Pulse size with activity
                circle.set_radius(0.015 + activity * 0.01)

        # Update direction arrow
        if self.sim.behavior.forward_drive > self.sim.behavior.backward_drive:
            self.direction_arrow.xy = (0.0, 0.0)
            self.direction_arrow.set_visible(True)
        elif self.sim.behavior.backward_drive > 0.3:
            self.direction_arrow.xy = (1.0, 0.0)
            self.direction_arrow.xyann = (1.05, 0.0)
            self.direction_arrow.set_visible(True)
        else:
            self.direction_arrow.set_visible(False)

        # Update behavior text
        behavior = self.sim.behavior.primary_behavior.value
        speed = self.sim.behavior.speed
        self.behavior_text.set_text(f'{behavior}\nSpeed: {speed:.2f}')

        return [self.body_line, self.body_inner, self.behavior_text]


# =============================================================================
# Neural Heatmap Panel
# =============================================================================

class NeuralHeatmapPanel:
    """Displays 302 neurons as a heatmap organized by type."""

    def __init__(self, ax, simulation):
        self.ax = ax
        self.sim = simulation

        # Organize neurons by type
        self.type_order = [NeuronType.SENSORY, NeuronType.INTERNEURON,
                          NeuronType.MOTOR, NeuronType.PHARYNGEAL]
        self.organized_indices = self._organize_by_type()

        # Grid dimensions
        self.n_cols = 15
        self.n_rows = (len(self.sim.neuron_names) + self.n_cols - 1) // self.n_cols

        self.setup()

    def _organize_by_type(self):
        """Group neuron indices by type."""
        indices = []
        type_boundaries = []

        for ntype in self.type_order:
            start = len(indices)
            for name in self.sim.neuron_names:
                if self.sim.neurons[name].neuron_type == ntype:
                    indices.append(self.sim.neuron_index[name])
            type_boundaries.append((ntype, start, len(indices)))

        self.type_boundaries = type_boundaries
        return indices

    def setup(self):
        """Initialize heatmap."""
        self.ax.set_facecolor(COLORS['BG_PANEL'])
        self.ax.set_title('Neural Activity', color=COLORS['TEXT'], fontsize=10)

        # Create heatmap data
        n_neurons = len(self.organized_indices)
        self.grid_data = np.zeros((self.n_rows, self.n_cols))

        # Initial display
        self.heatmap = self.ax.imshow(self.grid_data, cmap=ACTIVITY_CMAP,
                                       vmin=0, vmax=1, aspect='auto',
                                       interpolation='nearest')

        # Add type labels
        y_positions = []
        for ntype, start, end in self.type_boundaries:
            row_start = start // self.n_cols
            row_end = (end - 1) // self.n_cols
            y_mid = (row_start + row_end) / 2
            y_positions.append((ntype.name[:4], y_mid))

        self.ax.set_yticks([yp[1] for yp in y_positions])
        self.ax.set_yticklabels([yp[0] for yp in y_positions],
                                color=COLORS['TEXT'], fontsize=7)
        self.ax.set_xticks([])

        # Add horizontal lines between types
        for ntype, start, end in self.type_boundaries[1:]:
            row = start // self.n_cols
            self.ax.axhline(y=row - 0.5, color='white', linewidth=0.5, alpha=0.5)

    def update(self):
        """Update heatmap with current activity."""
        # Fill grid with organized activity
        self.grid_data.fill(0)
        for i, idx in enumerate(self.organized_indices):
            row = i // self.n_cols
            col = i % self.n_cols
            if row < self.n_rows:
                self.grid_data[row, col] = self.sim.activity[idx]

        self.heatmap.set_array(self.grid_data)
        return [self.heatmap]


# =============================================================================
# Circuit Diagram Panel
# =============================================================================

class CircuitDiagramPanel:
    """Displays key neural circuits with activity-based visualization."""

    def __init__(self, ax, simulation):
        self.ax = ax
        self.sim = simulation

        # Define circuit layout
        self.nodes = {
            # Touch circuit (top)
            'ALM': (0.15, 0.85), 'PLM': (0.15, 0.70),
            'AVD': (0.40, 0.85), 'PVC': (0.40, 0.70),

            # Command interneurons (middle)
            'AVA': (0.50, 0.50), 'AVB': (0.70, 0.50),

            # Motor neurons (bottom)
            'DA': (0.30, 0.25), 'VA': (0.45, 0.25),
            'DB': (0.70, 0.25), 'VB': (0.85, 0.25),

            # Chemotaxis (right side)
            'AWA': (0.85, 0.85), 'AIY': (0.85, 0.65),
            'AIB': (0.70, 0.65),
        }

        # Define edges (pre, post, type)
        self.edges = [
            # Touch -> Command
            ('ALM', 'AVD', 'exc'), ('PLM', 'PVC', 'exc'),
            ('AVD', 'AVA', 'exc'), ('PVC', 'AVB', 'exc'),

            # Command -> Motor
            ('AVA', 'DA', 'exc'), ('AVA', 'VA', 'exc'),
            ('AVB', 'DB', 'exc'), ('AVB', 'VB', 'exc'),

            # Chemotaxis
            ('AWA', 'AIY', 'exc'), ('AIY', 'AIB', 'exc'),
            ('AIB', 'AVA', 'mod'),
        ]

        self.node_circles = {}
        self.edge_lines = []

        self.setup()

    def _get_neuron_activity(self, name):
        """Get activity for a neuron or class average."""
        # Try direct lookup
        if name in self.sim.neuron_index:
            return self.sim.activity[self.sim.neuron_index[name]]

        # Try bilateral average (e.g., AVA -> average of AVAL, AVAR)
        left = name + 'L'
        right = name + 'R'
        activities = []
        if left in self.sim.neuron_index:
            activities.append(self.sim.activity[self.sim.neuron_index[left]])
        if right in self.sim.neuron_index:
            activities.append(self.sim.activity[self.sim.neuron_index[right]])

        if activities:
            return np.mean(activities)

        # Try class average (e.g., DA -> average of DA1-DA9)
        class_activities = []
        for neuron_name in self.sim.neuron_names:
            if self.sim.neurons[neuron_name].neuron_class == name:
                class_activities.append(self.sim.activity[self.sim.neuron_index[neuron_name]])

        if class_activities:
            return np.mean(class_activities)

        return 0.0

    def setup(self):
        """Initialize circuit diagram."""
        self.ax.set_xlim(0, 1)
        self.ax.set_ylim(0, 1)
        self.ax.set_facecolor(COLORS['BG_PANEL'])
        self.ax.set_title('Circuit Diagram', color=COLORS['TEXT'], fontsize=10)
        self.ax.axis('off')

        # Draw edges first (so nodes are on top)
        for pre, post, etype in self.edges:
            x1, y1 = self.nodes[pre]
            x2, y2 = self.nodes[post]

            color = COLORS['EXCITATORY'] if etype == 'exc' else COLORS['MODULATORY']

            line, = self.ax.plot([x1, x2], [y1, y2], color=color,
                                 linewidth=1, alpha=0.3)
            self.edge_lines.append((pre, post, line, etype))

        # Draw nodes
        for name, (x, y) in self.nodes.items():
            circle = Circle((x, y), 0.04, facecolor='gray',
                           edgecolor='white', linewidth=1)
            self.ax.add_patch(circle)
            self.node_circles[name] = circle

            # Label
            self.ax.text(x, y - 0.08, name, ha='center', va='top',
                        color=COLORS['TEXT'], fontsize=7)

        # Add legend
        self.ax.text(0.05, 0.15, 'Touch', color=COLORS['TEXT'], fontsize=7)
        self.ax.text(0.05, 0.08, 'Command', color=COLORS['TEXT'], fontsize=7)
        self.ax.text(0.05, 0.01, 'Motor', color=COLORS['TEXT'], fontsize=7)

    def update(self):
        """Update circuit visualization."""
        cmap = plt.cm.get_cmap(ACTIVITY_CMAP)

        # Update nodes
        for name, circle in self.node_circles.items():
            activity = self._get_neuron_activity(name)
            circle.set_facecolor(cmap(activity))
            circle.set_radius(0.03 + activity * 0.02)

        # Update edges
        for pre, post, line, etype in self.edge_lines:
            pre_activity = self._get_neuron_activity(pre)
            # Edge alpha based on presynaptic activity
            line.set_alpha(0.2 + pre_activity * 0.6)
            line.set_linewidth(1 + pre_activity * 2)

        return list(self.node_circles.values())


# =============================================================================
# Circuit Activity Panel - Shows which circuits are activated
# =============================================================================

class CircuitActivityPanel:
    """
    Displays horizontal bar chart showing activation level of each circuit.
    Bars glow/pulse when circuits are highly active.
    Shows which neurotransmitter levers affect each circuit.
    """

    def __init__(self, ax, simulation):
        self.ax = ax
        self.sim = simulation
        self.bars = {}
        self.nt_indicators = {}
        self.glow_patches = {}
        self.setup()

    def _get_neuron_activity(self, name):
        """Get activity for a neuron or class average."""
        # Try bilateral pair
        left = name + 'L'
        right = name + 'R'
        activities = []
        if left in self.sim.neuron_index:
            activities.append(self.sim.activity[self.sim.neuron_index[left]])
        if right in self.sim.neuron_index:
            activities.append(self.sim.activity[self.sim.neuron_index[right]])
        if activities:
            return np.mean(activities)

        # Try class average
        for neuron_name in self.sim.neuron_names:
            if self.sim.neurons[neuron_name].neuron_class == name:
                activities.append(self.sim.activity[self.sim.neuron_index[neuron_name]])
        if activities:
            return np.mean(activities)

        return 0.0

    def _get_circuit_activity(self, circuit_name):
        """Calculate average activity of neurons in a circuit."""
        circuit = CIRCUITS[circuit_name]
        activities = [self._get_neuron_activity(n) for n in circuit['neurons']]
        return np.mean(activities) if activities else 0.0

    def _get_nt_modulation(self, circuit_name):
        """Get the current modulation level affecting this circuit."""
        circuit = CIRCUITS[circuit_name]
        modulations = []
        for nt in circuit['affected_by']:
            if nt in NT_TO_ATTR:
                attr = NT_TO_ATTR[nt]
                val = getattr(self.sim.neurochemistry, attr, 1.0)
                modulations.append(val)
        return np.mean(modulations) if modulations else 1.0

    def setup(self):
        """Initialize the circuit activity bars."""
        self.ax.set_facecolor(COLORS['BG_PANEL'])
        self.ax.set_xlim(0, 1.2)
        self.ax.set_ylim(-0.5, len(CIRCUITS) - 0.5)
        self.ax.set_title('Circuit Activation', color=COLORS['TEXT'], fontsize=10)
        self.ax.axis('off')

        y_positions = list(range(len(CIRCUITS)))
        circuit_names = list(CIRCUITS.keys())

        for i, name in enumerate(circuit_names):
            circuit = CIRCUITS[name]
            y = len(CIRCUITS) - 1 - i  # Reverse order (top to bottom)

            # Background bar (max level)
            self.ax.barh(y, 1.0, height=0.6, color='#333355', alpha=0.5)

            # Active bar
            bar = self.ax.barh(y, 0.1, height=0.6, color=circuit['color'], alpha=0.8)
            self.bars[name] = bar[0]

            # Glow effect (wider, semi-transparent bar behind)
            glow = self.ax.barh(y, 0.1, height=0.75, color=circuit['color'], alpha=0.0)
            self.glow_patches[name] = glow[0]

            # Circuit label
            self.ax.text(-0.02, y, name, ha='right', va='center',
                        color=COLORS['TEXT'], fontsize=7, fontweight='bold')

            # NT indicators (small colored dots showing which NTs affect this circuit)
            nt_x = 1.05
            for j, nt in enumerate(circuit['affected_by']):
                nt_color = COLORS.get(nt, '#888888')
                indicator = Circle((nt_x + j * 0.05, y), 0.12,
                                  facecolor=nt_color, alpha=0.3,
                                  edgecolor='white', linewidth=0.5)
                self.ax.add_patch(indicator)
                if name not in self.nt_indicators:
                    self.nt_indicators[name] = []
                self.nt_indicators[name].append((nt, indicator))

        # Add legend for NT indicators
        self.ax.text(1.05, len(CIRCUITS) + 0.3, 'NT', ha='left',
                    color=COLORS['TEXT'], fontsize=6)

    def update(self):
        """Update circuit activity bars."""
        for name in CIRCUITS:
            # Get circuit activity
            activity = self._get_circuit_activity(name)

            # Get NT modulation effect
            modulation = self._get_nt_modulation(name)

            # Update bar width
            self.bars[name].set_width(activity)

            # Glow effect when highly active (activity > 0.5)
            if activity > 0.5:
                glow_alpha = (activity - 0.5) * 0.6  # 0 to 0.3
                self.glow_patches[name].set_width(activity * 1.1)
                self.glow_patches[name].set_alpha(glow_alpha)
            else:
                self.glow_patches[name].set_alpha(0)

            # Update NT indicator brightness based on current modulation
            if name in self.nt_indicators:
                for nt, indicator in self.nt_indicators[name]:
                    if nt in NT_TO_ATTR:
                        attr = NT_TO_ATTR[nt]
                        val = getattr(self.sim.neurochemistry, attr, 1.0)
                        # Brighter when modulated above baseline
                        alpha = 0.3 + (val - 1.0) * 0.4  # 0.3 at baseline, up to 0.7 at 2.0
                        alpha = np.clip(alpha, 0.1, 0.9)
                        indicator.set_alpha(alpha)
                        # Pulse size when high
                        radius = 0.12 + (val - 1.0) * 0.04
                        indicator.set_radius(np.clip(radius, 0.08, 0.18))

        return list(self.bars.values())


# =============================================================================
# Activity Graph Panel
# =============================================================================

class ActivityGraphPanel:
    """Real-time line plots of neural activity."""

    def __init__(self, axes, simulation, history_length=200):
        self.axes = axes  # Dict: 'command', 'motor', 'behavior'
        self.sim = simulation
        self.history = history_length

        # History buffers
        self.traces = {
            'command': {
                'AVA': deque(maxlen=history_length),
                'AVB': deque(maxlen=history_length),
                'PVC': deque(maxlen=history_length),
            },
            'motor': {
                'DA': deque(maxlen=history_length),
                'DB': deque(maxlen=history_length),
                'DD': deque(maxlen=history_length),
            },
            'behavior': {
                'forward': deque(maxlen=history_length),
                'backward': deque(maxlen=history_length),
                'speed': deque(maxlen=history_length),
            }
        }

        self.lines = {}
        self.setup()

    def _get_activity(self, name):
        """Get activity for neuron or class."""
        # Try bilateral pair
        left = name + 'L'
        right = name + 'R'
        if left in self.sim.neuron_index and right in self.sim.neuron_index:
            return (self.sim.activity[self.sim.neuron_index[left]] +
                    self.sim.activity[self.sim.neuron_index[right]]) / 2

        # Try class average
        activities = []
        for n in self.sim.neuron_names:
            if self.sim.neurons[n].neuron_class == name:
                activities.append(self.sim.activity[self.sim.neuron_index[n]])
        if activities:
            return np.mean(activities)

        return 0.0

    def setup(self):
        """Initialize graph panels."""
        colors = {
            'AVA': '#F44336', 'AVB': '#2196F3', 'PVC': '#4CAF50',
            'DA': '#FF9800', 'DB': '#00BCD4', 'DD': '#9C27B0',
            'forward': '#4CAF50', 'backward': '#F44336', 'speed': '#FFC107'
        }

        for panel_name, ax in self.axes.items():
            ax.set_facecolor(COLORS['BG_PANEL'])
            ax.set_ylim(-0.05, 1.05)
            ax.set_xlim(0, self.history)
            ax.set_xticks([])
            ax.tick_params(colors=COLORS['TEXT'], labelsize=7)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_color(COLORS['TEXT'])
            ax.spines['left'].set_color(COLORS['TEXT'])

            # Create lines for this panel
            self.lines[panel_name] = {}
            for trace_name in self.traces[panel_name]:
                line, = ax.plot([], [], color=colors[trace_name],
                               linewidth=1.5, label=trace_name)
                self.lines[panel_name][trace_name] = line

            ax.legend(loc='upper left', fontsize=6,
                     facecolor=COLORS['BG_PANEL'],
                     labelcolor=COLORS['TEXT'],
                     framealpha=0.5)

        self.axes['command'].set_title('Command', color=COLORS['TEXT'], fontsize=8)
        self.axes['motor'].set_title('Motor', color=COLORS['TEXT'], fontsize=8)
        self.axes['behavior'].set_title('Behavior', color=COLORS['TEXT'], fontsize=8)

    def update(self):
        """Update traces with current values."""
        # Record current values
        for name in ['AVA', 'AVB', 'PVC']:
            self.traces['command'][name].append(self._get_activity(name))

        for name in ['DA', 'DB', 'DD']:
            self.traces['motor'][name].append(self._get_activity(name))

        self.traces['behavior']['forward'].append(self.sim.behavior.forward_drive)
        self.traces['behavior']['backward'].append(self.sim.behavior.backward_drive)
        self.traces['behavior']['speed'].append(self.sim.behavior.speed)

        # Update line data
        artists = []
        for panel_name in self.traces:
            for trace_name, buffer in self.traces[panel_name].items():
                if len(buffer) > 0:
                    x = np.arange(len(buffer))
                    y = np.array(buffer)
                    self.lines[panel_name][trace_name].set_data(x, y)
                    artists.append(self.lines[panel_name][trace_name])

        return artists


# =============================================================================
# Control Panel
# =============================================================================

class ControlPanel:
    """On-screen controls for neurochemistry and sensory input."""

    def __init__(self, fig, simulation):
        self.fig = fig
        self.sim = simulation
        self.sliders = {}
        self.buttons = {}
        self.sensory_states = {}  # Track toggle states

    def setup(self):
        """Create all widgets."""
        # Neurochemical sliders (horizontal, stacked)
        neurochems = [
            ('ACh', 'acetylcholine', '#4CAF50'),
            ('Glu', 'glutamate', '#8BC34A'),
            ('GABA', 'gaba', '#F44336'),
            ('DA', 'dopamine', '#FFC107'),
            ('5-HT', 'serotonin', '#E91E63'),
            ('Oct', 'octopamine', '#FF5722'),
            ('Tyr', 'tyramine', '#795548'),
        ]

        slider_height = 0.018
        slider_width = 0.08
        start_x = 0.08
        start_y = 0.12

        for i, (label, attr, color) in enumerate(neurochems):
            ax = self.fig.add_axes([start_x + i * 0.11, start_y, slider_width, slider_height])
            slider = Slider(ax, label, 0.0, 2.0, valinit=1.0,
                           color=color, valstep=0.1)
            slider.label.set_color(COLORS['TEXT'])
            slider.label.set_fontsize(7)
            slider.valtext.set_color(COLORS['TEXT'])
            slider.valtext.set_fontsize(6)
            slider.on_changed(self._make_neurochem_callback(attr))
            self.sliders[attr] = slider

        # Sensory buttons
        sensory_buttons = [
            ('Ant\nTouch', 'anterior_touch'),
            ('Post\nTouch', 'posterior_touch'),
            ('Harsh', 'harsh_touch_head'),
            ('Odor+', 'attractive_odor'),
            ('Odor-', 'repulsive_odor'),
        ]

        btn_width = 0.055
        btn_height = 0.035
        btn_start_x = 0.08
        btn_y = 0.05

        for i, (label, attr) in enumerate(sensory_buttons):
            ax = self.fig.add_axes([btn_start_x + i * 0.065, btn_y, btn_width, btn_height])
            btn = Button(ax, label, color=COLORS['BG_PANEL'], hovercolor='#3a3a5e')
            btn.label.set_color(COLORS['TEXT'])
            btn.label.set_fontsize(6)
            btn.on_clicked(self._make_sensory_callback(attr))
            self.buttons[attr] = (btn, ax)
            self.sensory_states[attr] = False

        # Speed slider
        ax_speed = self.fig.add_axes([0.55, 0.05, 0.12, 0.018])
        self.speed_slider = Slider(ax_speed, 'Speed', 0.5, 5.0, valinit=1.0,
                                   color='#9E9E9E', valstep=0.5)
        self.speed_slider.label.set_color(COLORS['TEXT'])
        self.speed_slider.label.set_fontsize(7)
        self.speed_slider.valtext.set_color(COLORS['TEXT'])
        self.speed_slider.valtext.set_fontsize(6)

        # Control buttons
        ax_pause = self.fig.add_axes([0.72, 0.05, 0.06, 0.035])
        self.pause_btn = Button(ax_pause, 'Pause', color=COLORS['BG_PANEL'],
                                hovercolor='#3a3a5e')
        self.pause_btn.label.set_color(COLORS['TEXT'])
        self.pause_btn.label.set_fontsize(7)

        ax_reset = self.fig.add_axes([0.79, 0.05, 0.06, 0.035])
        self.reset_btn = Button(ax_reset, 'Reset', color=COLORS['BG_PANEL'],
                                hovercolor='#3a3a5e')
        self.reset_btn.label.set_color(COLORS['TEXT'])
        self.reset_btn.label.set_fontsize(7)

        # Info text
        self.info_ax = self.fig.add_axes([0.87, 0.02, 0.12, 0.08])
        self.info_ax.set_facecolor(COLORS['BG_PANEL'])
        self.info_ax.axis('off')
        self.info_text = self.info_ax.text(0.5, 0.5, '', ha='center', va='center',
                                           color=COLORS['TEXT'], fontsize=7,
                                           family='monospace')

    def _make_neurochem_callback(self, attr):
        """Create callback for neurochemical slider."""
        def callback(val):
            setattr(self.sim.neurochemistry, attr, val)
        return callback

    def _make_sensory_callback(self, attr):
        """Create callback for sensory button (toggle)."""
        def callback(event):
            self.sensory_states[attr] = not self.sensory_states[attr]
            new_val = 0.8 if self.sensory_states[attr] else 0.0
            setattr(self.sim.sensory, attr, new_val)

            # Update button appearance
            btn, ax = self.buttons[attr]
            if self.sensory_states[attr]:
                ax.set_facecolor('#4a4a6e')
            else:
                ax.set_facecolor(COLORS['BG_PANEL'])
        return callback

    def get_speed(self):
        """Get current speed setting."""
        return self.speed_slider.val

    def update_info(self, time_ms, n_neurons):
        """Update info display."""
        self.info_text.set_text(f'Time: {time_ms:.0f} ms\nNeurons: {n_neurons}')


# =============================================================================
# Main Visualizer
# =============================================================================

class CElegansVisualizer:
    """Main orchestrator for the C. elegans visualization."""

    def __init__(self):
        print("Initializing C. elegans Visualizer...")

        # Create simulation
        self.sim = CElegansSimulation()

        # Animation state
        self.running = True
        self.steps_per_frame = 5

        # Set up the figure
        self.setup_layout()

        print(f"Ready! Neurons: {self.sim.n_neurons}")

    def setup_layout(self):
        """Set up the figure layout with all panels."""
        # Create figure
        self.fig = plt.figure(figsize=(18, 10), facecolor=COLORS['BG_DARK'])
        self.fig.suptitle('C. elegans Neural Simulation - Circuit Activation Monitor',
                         color=COLORS['TEXT'], fontsize=14, y=0.98)

        # Create grid layout (4 columns now to include circuit activity panel)
        gs = GridSpec(3, 4, figure=self.fig,
                     height_ratios=[0.55, 0.25, 0.20],
                     width_ratios=[0.28, 0.22, 0.28, 0.22],
                     hspace=0.3, wspace=0.2,
                     left=0.08, right=0.98, top=0.93, bottom=0.18)

        # Main visualization panels (top row)
        ax_worm = self.fig.add_subplot(gs[0, 0])
        ax_heatmap = self.fig.add_subplot(gs[0, 1])
        ax_circuit = self.fig.add_subplot(gs[0, 2])
        ax_circuit_activity = self.fig.add_subplot(gs[0, 3])  # NEW: Circuit activity bars

        # Activity graphs (middle row)
        ax_command = self.fig.add_subplot(gs[1, 0])
        ax_motor = self.fig.add_subplot(gs[1, 1])
        ax_behavior = self.fig.add_subplot(gs[1, 2])
        ax_nt_effects = self.fig.add_subplot(gs[1, 3])  # NEW: NT effects display

        # Create panel objects
        self.worm_renderer = WormBodyRenderer(ax_worm, self.sim)
        self.heatmap_panel = NeuralHeatmapPanel(ax_heatmap, self.sim)
        self.circuit_panel = CircuitDiagramPanel(ax_circuit, self.sim)
        self.circuit_activity_panel = CircuitActivityPanel(ax_circuit_activity, self.sim)  # NEW
        self.activity_panel = ActivityGraphPanel(
            {'command': ax_command, 'motor': ax_motor, 'behavior': ax_behavior},
            self.sim
        )

        # Setup NT effects panel (shows which NT is currently elevated/suppressed)
        self._setup_nt_effects_panel(ax_nt_effects)

        # Control panel
        self.control_panel = ControlPanel(self.fig, self.sim)
        self.control_panel.setup()

        # Connect control buttons
        self.control_panel.pause_btn.on_clicked(self._pause_callback)
        self.control_panel.reset_btn.on_clicked(self._reset_callback)

    def _setup_nt_effects_panel(self, ax):
        """Setup the neurotransmitter effects display panel."""
        self.nt_ax = ax
        ax.set_facecolor(COLORS['BG_PANEL'])
        ax.set_title('NT Modulation', color=COLORS['TEXT'], fontsize=10)
        ax.axis('off')

        # Create text displays for each NT showing current level and affected circuits
        self.nt_texts = {}
        self.nt_bars = {}

        nt_info = [
            ('ACh', 'acetylcholine', '#4CAF50', 'Forward Cmd'),
            ('Glu', 'glutamate', '#8BC34A', 'Touch, Backward'),
            ('GABA', 'gaba', '#F44336', 'Inhibitory'),
            ('DA', 'dopamine', '#FFC107', 'Modulatory'),
            ('5-HT', 'serotonin', '#E91E63', 'Modulatory'),
            ('Oct', 'octopamine', '#FF5722', 'Modulatory'),
            ('Tyr', 'tyramine', '#795548', 'Modulatory'),
        ]

        ax.set_xlim(0, 1)
        ax.set_ylim(0, len(nt_info) + 0.5)

        for i, (label, attr, color, affects) in enumerate(nt_info):
            y = len(nt_info) - i - 0.5

            # NT label
            ax.text(0.02, y, f'{label}:', ha='left', va='center',
                   color=color, fontsize=8, fontweight='bold')

            # Value bar background
            ax.barh(y, 1.0, left=0.25, height=0.5, color='#333355', alpha=0.3)

            # Value bar (will be updated)
            bar = ax.barh(y, 0.5, left=0.25, height=0.5, color=color, alpha=0.7)
            self.nt_bars[attr] = bar[0]

            # Affected circuits text
            ax.text(0.98, y, affects, ha='right', va='center',
                   color='#888888', fontsize=6)

        # Baseline indicator
        ax.axvline(x=0.75, color='white', linewidth=1, linestyle='--', alpha=0.3)
        ax.text(0.75, len(nt_info) + 0.2, '1.0', ha='center', va='bottom',
               color='#666666', fontsize=6)

    def _update_nt_effects(self):
        """Update the NT effects display."""
        for attr, bar in self.nt_bars.items():
            val = getattr(self.sim.neurochemistry, attr, 1.0)
            # Map 0-2 range to 0-1 bar width (0.5 is baseline at center)
            bar.set_width(val / 2.0)
            # Brighten when above baseline
            alpha = 0.5 + (val - 1.0) * 0.25
            bar.set_alpha(np.clip(alpha, 0.3, 0.9))

    def _pause_callback(self, event):
        """Toggle pause state."""
        self.running = not self.running
        self.control_panel.pause_btn.label.set_text('Resume' if not self.running else 'Pause')

    def _reset_callback(self, event):
        """Reset simulation."""
        self.sim.reset()
        self.sim.neurochemistry = NeurochemicalState()
        self.sim.sensory = SensoryInput()

        # Reset sliders
        for attr, slider in self.control_panel.sliders.items():
            slider.set_val(1.0)

        # Reset sensory buttons
        for attr in self.control_panel.sensory_states:
            self.control_panel.sensory_states[attr] = False
            btn, ax = self.control_panel.buttons[attr]
            ax.set_facecolor(COLORS['BG_PANEL'])

        # Clear activity history
        for panel in self.activity_panel.traces.values():
            for buffer in panel.values():
                buffer.clear()

    def animate(self, frame):
        """Animation update function."""
        if not self.running:
            return []

        # Get speed multiplier
        speed = self.control_panel.get_speed()
        n_steps = int(self.steps_per_frame * speed)

        # Run simulation
        self.sim.step(n_steps)

        # Update all panels
        artists = []
        artists.extend(self.worm_renderer.update() or [])
        artists.extend(self.heatmap_panel.update() or [])
        artists.extend(self.circuit_panel.update() or [])
        artists.extend(self.circuit_activity_panel.update() or [])  # NEW
        artists.extend(self.activity_panel.update() or [])
        self._update_nt_effects()  # NEW

        # Update info
        self.control_panel.update_info(self.sim.time, self.sim.n_neurons)

        return artists

    def run(self):
        """Start the visualization."""
        print("Starting animation... Close window to exit.")

        self.anim = FuncAnimation(
            self.fig,
            self.animate,
            frames=None,
            interval=33,  # ~30 FPS
            blit=False,   # Full redraw for widgets
            cache_frame_data=False
        )

        plt.show()


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    """Main entry point."""
    print("=" * 60)
    print("  C. elegans Real-Time Neural Visualization")
    print("  302 Neurons | 7 Neurochemical Levers | Live Control")
    print("=" * 60)
    print()

    viz = CElegansVisualizer()
    viz.run()


if __name__ == "__main__":
    main()
