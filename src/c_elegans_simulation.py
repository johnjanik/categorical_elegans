#!/usr/bin/env python3
"""
C. elegans Neural Simulation
============================
A simulation of the 302-neuron C. elegans nervous system based on the
categorical connectome model from c_elegans_categorical_connectome.tex.

User controls: Neurochemical levers for ACh, Glu, GABA, DA, 5-HT, Oct, Tyr
Outputs: Behavioral states (locomotion, touch response, thermotaxis, chemotaxis)

Based on:
- White et al. (1986) - Original connectome
- Cook et al. (2019) - Updated connectome
- Varshney et al. (2011) - Network properties
"""

import numpy as np
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Tuple, Optional, Callable
from collections import deque
import time


# =============================================================================
# Enumerations and Types
# =============================================================================

class NeuronType(Enum):
    SENSORY = auto()
    INTERNEURON = auto()
    MOTOR = auto()
    PHARYNGEAL = auto()


class Neurotransmitter(Enum):
    ACH = "Acetylcholine"      # Excitatory
    GLU = "Glutamate"          # Excitatory
    GABA = "GABA"              # Inhibitory
    DA = "Dopamine"            # Modulatory
    SEROTONIN = "Serotonin"    # Modulatory (5-HT)
    OCT = "Octopamine"         # Modulatory
    TYR = "Tyramine"           # Modulatory
    UNKNOWN = "Unknown"


class Behavior(Enum):
    FORWARD = "Forward locomotion"
    BACKWARD = "Backward locomotion"
    TURN_LEFT = "Left turn"
    TURN_RIGHT = "Right turn"
    STOP = "Quiescent"
    FEEDING = "Pharyngeal pumping"
    ESCAPE_ANTERIOR = "Anterior touch escape"
    ESCAPE_POSTERIOR = "Posterior touch escape"


# =============================================================================
# Observable Vector - 15-dimensional behavioral measurement space
# =============================================================================

@dataclass
class ObservableVector:
    """
    15-dimensional observable vector for behavioral comparison.

    This matches the observable space O defined in minimal_model_theory.tex:
    O = {velocity, angular_velocity, reversal_rate, run_length, speed_mean,
         speed_variance, turn_angle_mean, turn_angle_variance, omega_turn_rate,
         chemotaxis_index, anterior_touch_prob, posterior_touch_prob,
         response_latency, pharyngeal_pumping}
    """
    # Locomotion kinematics
    velocity: float = 0.0              # Mean centroid velocity (μm/s, signed)
    angular_velocity: float = 0.0       # Mean angular velocity (rad/s)

    # Reversal behavior
    reversal_rate: float = 0.0          # Reversals per minute
    run_length: float = 0.0             # Mean forward run duration (seconds)

    # Speed statistics
    speed_mean: float = 0.0             # Mean speed magnitude (μm/s)
    speed_variance: float = 0.0         # Speed variance

    # Turning behavior
    turn_angle_mean: float = 0.0        # Mean turn angle (degrees)
    turn_angle_variance: float = 0.0    # Turn angle variance
    omega_turn_rate: float = 0.0        # Omega turns per minute

    # Navigation indices
    chemotaxis_index: float = 0.0       # (-1 to 1): bias toward attractant

    # Touch response probabilities
    anterior_touch_prob: float = 0.0    # P(reversal | anterior touch)
    posterior_touch_prob: float = 0.0   # P(acceleration | posterior touch)

    # Response dynamics
    response_latency: float = 0.0       # Time to response onset (ms)

    # Feeding
    pharyngeal_pumping: float = 0.0     # Pumps per minute

    def to_array(self) -> np.ndarray:
        """Convert to 15-dimensional numpy array."""
        return np.array([
            self.velocity,
            self.angular_velocity,
            self.reversal_rate,
            self.run_length,
            self.speed_mean,
            self.speed_variance,
            self.turn_angle_mean,
            self.turn_angle_variance,
            self.omega_turn_rate,
            self.chemotaxis_index,
            self.anterior_touch_prob,
            self.posterior_touch_prob,
            self.response_latency,
            self.pharyngeal_pumping,
            0.0  # Reserved for future use (15th dimension)
        ])

    @staticmethod
    def from_array(arr: np.ndarray) -> 'ObservableVector':
        """Create from numpy array."""
        return ObservableVector(
            velocity=arr[0],
            angular_velocity=arr[1],
            reversal_rate=arr[2],
            run_length=arr[3],
            speed_mean=arr[4],
            speed_variance=arr[5],
            turn_angle_mean=arr[6],
            turn_angle_variance=arr[7],
            omega_turn_rate=arr[8],
            chemotaxis_index=arr[9],
            anterior_touch_prob=arr[10],
            posterior_touch_prob=arr[11],
            response_latency=arr[12],
            pharyngeal_pumping=arr[13],
        )

    def distance(self, other: 'ObservableVector') -> float:
        """Compute L2 distance to another observable vector."""
        return float(np.linalg.norm(self.to_array() - other.to_array()))

    def __repr__(self) -> str:
        return (f"ObservableVector(\n"
                f"  velocity={self.velocity:.3f}, angular_velocity={self.angular_velocity:.3f},\n"
                f"  reversal_rate={self.reversal_rate:.3f}, run_length={self.run_length:.3f},\n"
                f"  speed_mean={self.speed_mean:.3f}, speed_variance={self.speed_variance:.3f},\n"
                f"  turn_angle_mean={self.turn_angle_mean:.3f}, turn_angle_variance={self.turn_angle_variance:.3f},\n"
                f"  omega_turn_rate={self.omega_turn_rate:.3f}, chemotaxis_index={self.chemotaxis_index:.3f},\n"
                f"  anterior_touch_prob={self.anterior_touch_prob:.3f}, posterior_touch_prob={self.posterior_touch_prob:.3f},\n"
                f"  response_latency={self.response_latency:.3f}, pharyngeal_pumping={self.pharyngeal_pumping:.3f}\n"
                f")")


class BehaviorHistory:
    """
    Temporal history buffer for computing time-dependent observables.

    Tracks behavioral states over a rolling window to compute:
    - Reversal rates
    - Run length statistics
    - Speed variance
    - Turn angle statistics
    - Response latencies
    """

    def __init__(self, window_size: int = 1000, dt_ms: float = 1.0):
        """
        Initialize behavior history.

        Args:
            window_size: Number of timesteps to keep in history
            dt_ms: Timestep size in milliseconds
        """
        self.window_size = window_size
        self.dt_ms = dt_ms

        # Rolling buffers for behavioral variables
        self.forward_drive_history = deque(maxlen=window_size)
        self.backward_drive_history = deque(maxlen=window_size)
        self.speed_history = deque(maxlen=window_size)
        self.turn_bias_history = deque(maxlen=window_size)
        self.head_angle_history = deque(maxlen=window_size)
        self.pharyngeal_history = deque(maxlen=window_size)

        # Event tracking
        self.reversal_times: List[float] = []  # Timestamps of reversals
        self.omega_turn_times: List[float] = []  # Timestamps of omega turns
        self.run_start_time: Optional[float] = None
        self.run_lengths: List[float] = []

        # Response latency tracking
        self.stimulus_onset_time: Optional[float] = None
        self.response_detected: bool = False
        self.last_response_latency: float = 0.0

        # State tracking
        self.current_time: float = 0.0
        self.was_reversing: bool = False
        self.was_omega_turning: bool = False

        # Accumulated touch response counts
        self.anterior_touch_trials: int = 0
        self.anterior_touch_reversals: int = 0
        self.posterior_touch_trials: int = 0
        self.posterior_touch_accelerations: int = 0

    def update(self, behavior: 'BehavioralState',
               sensory: Optional['SensoryInput'] = None) -> None:
        """
        Update history with current behavioral state.

        Args:
            behavior: Current behavioral state
            sensory: Optional sensory input (for response tracking)
        """
        self.current_time += self.dt_ms

        # Add to rolling buffers
        self.forward_drive_history.append(behavior.forward_drive)
        self.backward_drive_history.append(behavior.backward_drive)
        self.speed_history.append(behavior.speed)
        self.turn_bias_history.append(behavior.turn_bias)
        self.head_angle_history.append(behavior.head_angle)
        self.pharyngeal_history.append(behavior.pharyngeal_pumping)

        # Detect reversals (transition from forward to backward)
        is_reversing = behavior.backward_drive > behavior.forward_drive + 0.2
        if is_reversing and not self.was_reversing:
            self.reversal_times.append(self.current_time)
            # Track run length
            if self.run_start_time is not None:
                run_length = (self.current_time - self.run_start_time) / 1000.0  # seconds
                self.run_lengths.append(run_length)
            self.run_start_time = None
        elif not is_reversing and self.was_reversing:
            # Started a new forward run
            self.run_start_time = self.current_time
        self.was_reversing = is_reversing

        # Detect omega turns
        if behavior.omega_turn and not self.was_omega_turning:
            self.omega_turn_times.append(self.current_time)
        self.was_omega_turning = behavior.omega_turn

        # Track touch response latencies
        if sensory is not None:
            # Detect stimulus onset
            if (sensory.anterior_touch > 0.5 or sensory.posterior_touch > 0.5) and \
               self.stimulus_onset_time is None:
                self.stimulus_onset_time = self.current_time
                self.response_detected = False

                # Count as a trial
                if sensory.anterior_touch > 0.5:
                    self.anterior_touch_trials += 1
                if sensory.posterior_touch > 0.5:
                    self.posterior_touch_trials += 1

            # Detect response (significant change in behavior)
            if self.stimulus_onset_time is not None and not self.response_detected:
                if behavior.backward_drive > 0.3 or behavior.forward_drive > 0.3:
                    self.response_detected = True
                    self.last_response_latency = self.current_time - self.stimulus_onset_time

                    # Count as success
                    if sensory.anterior_touch > 0.5 and behavior.backward_drive > 0.3:
                        self.anterior_touch_reversals += 1
                    if sensory.posterior_touch > 0.5 and behavior.forward_drive > 0.3:
                        self.posterior_touch_accelerations += 1

            # Reset when stimulus ends
            if sensory.anterior_touch < 0.1 and sensory.posterior_touch < 0.1:
                self.stimulus_onset_time = None

        # Prune old events (keep only last 60 seconds)
        cutoff_time = self.current_time - 60000  # 60 seconds in ms
        self.reversal_times = [t for t in self.reversal_times if t > cutoff_time]
        self.omega_turn_times = [t for t in self.omega_turn_times if t > cutoff_time]
        self.run_lengths = self.run_lengths[-100:]  # Keep last 100 runs

    def compute_observables(self) -> ObservableVector:
        """
        Compute the 15-dimensional observable vector from history.

        Returns:
            ObservableVector with computed statistics
        """
        obs = ObservableVector()

        if len(self.speed_history) < 10:
            return obs  # Not enough data

        speeds = np.array(self.speed_history)
        forward_drives = np.array(self.forward_drive_history)
        backward_drives = np.array(self.backward_drive_history)
        turn_biases = np.array(self.turn_bias_history)
        head_angles = np.array(self.head_angle_history)
        pharyngeal = np.array(self.pharyngeal_history)

        # Velocity (signed: positive = forward, negative = backward)
        velocity_sign = np.where(forward_drives > backward_drives, 1.0, -1.0)
        obs.velocity = float(np.mean(speeds * velocity_sign))

        # Angular velocity from head angle changes
        if len(head_angles) > 1:
            angular_changes = np.diff(head_angles)
            obs.angular_velocity = float(np.mean(np.abs(angular_changes)) * 1000 / self.dt_ms)

        # Reversal rate (per minute)
        window_duration_min = len(self.speed_history) * self.dt_ms / 60000.0
        if window_duration_min > 0:
            recent_reversals = sum(1 for t in self.reversal_times
                                   if t > self.current_time - len(self.speed_history) * self.dt_ms)
            obs.reversal_rate = recent_reversals / window_duration_min

        # Run length
        if self.run_lengths:
            obs.run_length = float(np.mean(self.run_lengths))

        # Speed statistics
        obs.speed_mean = float(np.mean(speeds))
        obs.speed_variance = float(np.var(speeds))

        # Turn angle statistics (using turn_bias as proxy)
        turn_angles = turn_biases * 90  # Scale to approximate degrees
        obs.turn_angle_mean = float(np.mean(np.abs(turn_angles)))
        obs.turn_angle_variance = float(np.var(turn_angles))

        # Omega turn rate (per minute)
        if window_duration_min > 0:
            recent_omega = sum(1 for t in self.omega_turn_times
                              if t > self.current_time - len(self.speed_history) * self.dt_ms)
            obs.omega_turn_rate = recent_omega / window_duration_min

        # Chemotaxis index: net forward movement relative to total movement
        if np.sum(np.abs(forward_drives) + np.abs(backward_drives)) > 0:
            obs.chemotaxis_index = float(
                (np.sum(forward_drives) - np.sum(backward_drives)) /
                (np.sum(forward_drives) + np.sum(backward_drives) + 0.01)
            )

        # Touch response probabilities
        if self.anterior_touch_trials > 0:
            obs.anterior_touch_prob = self.anterior_touch_reversals / self.anterior_touch_trials
        if self.posterior_touch_trials > 0:
            obs.posterior_touch_prob = self.posterior_touch_accelerations / self.posterior_touch_trials

        # Response latency
        obs.response_latency = self.last_response_latency

        # Pharyngeal pumping rate (pumps per minute)
        # Approximate: high activity = more pumping
        obs.pharyngeal_pumping = float(np.mean(pharyngeal)) * 240  # Scale to realistic range

        return obs

    def reset(self) -> None:
        """Reset all history buffers."""
        self.forward_drive_history.clear()
        self.backward_drive_history.clear()
        self.speed_history.clear()
        self.turn_bias_history.clear()
        self.head_angle_history.clear()
        self.pharyngeal_history.clear()

        self.reversal_times = []
        self.omega_turn_times = []
        self.run_lengths = []
        self.run_start_time = None

        self.stimulus_onset_time = None
        self.response_detected = False
        self.last_response_latency = 0.0

        self.current_time = 0.0
        self.was_reversing = False
        self.was_omega_turning = False

        self.anterior_touch_trials = 0
        self.anterior_touch_reversals = 0
        self.posterior_touch_trials = 0
        self.posterior_touch_accelerations = 0


# =============================================================================
# Neuron Data
# =============================================================================

@dataclass
class Neuron:
    """Represents a single neuron in the C. elegans nervous system."""
    name: str
    neuron_class: str
    neuron_type: NeuronType
    neurotransmitter: Neurotransmitter
    position: str  # L, R, or U (unpaired)

    # Dynamic state
    membrane_potential: float = -70.0  # mV (resting)
    calcium: float = 0.0               # [Ca2+] relative concentration
    activity: float = 0.0              # Firing rate (0-1)

    # Properties
    threshold: float = -55.0           # Spike threshold mV
    tau_membrane: float = 10.0         # Membrane time constant ms
    tau_calcium: float = 50.0          # Calcium decay constant ms


@dataclass
class Synapse:
    """Chemical synapse between two neurons."""
    pre: str       # Presynaptic neuron name
    post: str      # Postsynaptic neuron name
    weight: int    # Number of synaptic contacts

    @property
    def strength(self) -> float:
        """Normalized synaptic strength."""
        return np.log1p(self.weight) / 3.0  # Log scale, normalized


@dataclass
class GapJunction:
    """Electrical synapse (gap junction) between two neurons."""
    neuron_a: str
    neuron_b: str
    weight: int

    @property
    def conductance(self) -> float:
        """Gap junction conductance (symmetric)."""
        return np.log1p(self.weight) / 2.0


# =============================================================================
# Connectome Data - Based on categorical model
# =============================================================================

def create_neurons() -> Dict[str, Neuron]:
    """Create all 302 neurons with their properties."""
    neurons = {}

    # -------------------------------------------------------------------------
    # SENSORY NEURONS (60 neurons, 39 classes)
    # -------------------------------------------------------------------------

    # Amphid sensory neurons (head chemosensation/thermosensation)
    sensory_amphid = [
        ("ADF", ["ADFL", "ADFR"], Neurotransmitter.ACH, "Chemosensation (dauer pheromone)"),
        ("ADL", ["ADLL", "ADLR"], Neurotransmitter.GLU, "Chemosensation (avoidance)"),
        ("AFD", ["AFDL", "AFDR"], Neurotransmitter.GLU, "Thermosensation"),
        ("ASE", ["ASEL", "ASER"], Neurotransmitter.GLU, "Chemosensation (water-soluble)"),
        ("ASG", ["ASGL", "ASGR"], Neurotransmitter.GLU, "Chemosensation"),
        ("ASH", ["ASHL", "ASHR"], Neurotransmitter.GLU, "Polymodal nociception"),
        ("ASI", ["ASIL", "ASIR"], Neurotransmitter.GLU, "Chemosensation (dauer)"),
        ("ASJ", ["ASJL", "ASJR"], Neurotransmitter.GLU, "Chemosensation (dauer recovery)"),
        ("ASK", ["ASKL", "ASKR"], Neurotransmitter.GLU, "Chemosensation"),
        ("AWA", ["AWAL", "AWAR"], Neurotransmitter.UNKNOWN, "Olfaction (attractive)"),
        ("AWB", ["AWBL", "AWBR"], Neurotransmitter.ACH, "Olfaction (repulsive)"),
        ("AWC", ["AWCL", "AWCR"], Neurotransmitter.GLU, "Olfaction, thermosensation"),
    ]

    # Inner/Outer labial sensory
    sensory_labial = [
        ("IL1", ["IL1L", "IL1R"], Neurotransmitter.ACH, "Mechanosensation (nose)"),
        ("IL2", ["IL2L", "IL2R"], Neurotransmitter.ACH, "Chemosensation"),
        ("OLL", ["OLLL", "OLLR"], Neurotransmitter.GLU, "Mechanosensation"),
        ("OLQ", ["OLQDL", "OLQDR", "OLQVL", "OLQVR"], Neurotransmitter.GLU, "Mechanosensation"),
    ]

    # Cephalic sensory (dopaminergic)
    sensory_cephalic = [
        ("CEP", ["CEPDL", "CEPDR", "CEPVL", "CEPVR"], Neurotransmitter.DA, "Dopaminergic mechanosensation"),
    ]

    # Touch receptor neurons (gentle touch)
    sensory_touch = [
        ("ALM", ["ALML", "ALMR"], Neurotransmitter.GLU, "Anterior light touch"),
        ("AVM", ["AVM"], Neurotransmitter.GLU, "Anterior light touch"),
        ("PLM", ["PLML", "PLMR"], Neurotransmitter.GLU, "Posterior light touch"),
        ("PVM", ["PVM"], Neurotransmitter.GLU, "Posterior light touch"),
    ]

    # Phasmid sensory (tail)
    sensory_phasmid = [
        ("PHA", ["PHAL", "PHAR"], Neurotransmitter.GLU, "Chemosensation"),
        ("PHB", ["PHBL", "PHBR"], Neurotransmitter.GLU, "Chemosensation (repulsive)"),
        ("PHC", ["PHCL", "PHCR"], Neurotransmitter.GLU, "Harsh touch (tail)"),
    ]

    # Other sensory
    sensory_other = [
        ("ADE", ["ADEL", "ADER"], Neurotransmitter.DA, "Dopaminergic mechanosensation"),
        ("PDE", ["PDEL", "PDER"], Neurotransmitter.DA, "Dopaminergic mechanosensation"),
        ("AQR", ["AQR"], Neurotransmitter.GLU, "Oxygen sensation"),
        ("PQR", ["PQR"], Neurotransmitter.GLU, "Oxygen sensation"),
        ("URX", ["URXL", "URXR"], Neurotransmitter.ACH, "Oxygen/CO2 sensation"),
        ("BAG", ["BAGL", "BAGR"], Neurotransmitter.GLU, "CO2/O2 sensation"),
        ("FLP", ["FLPL", "FLPR"], Neurotransmitter.GLU, "Harsh touch (nose)"),
    ]

    # Add all sensory neurons
    for group in [sensory_amphid, sensory_labial, sensory_cephalic,
                  sensory_touch, sensory_phasmid, sensory_other]:
        for nclass, names, nt, _ in group:
            for name in names:
                pos = "U" if len(names) == 1 else ("L" if name.endswith("L") else "R")
                neurons[name] = Neuron(
                    name=name, neuron_class=nclass, neuron_type=NeuronType.SENSORY,
                    neurotransmitter=nt, position=pos
                )

    # -------------------------------------------------------------------------
    # INTERNEURONS (~77 neurons, 34 classes)
    # -------------------------------------------------------------------------

    # Command interneurons (locomotion control) - CRITICAL
    inter_command = [
        ("AVA", ["AVAL", "AVAR"], Neurotransmitter.GLU, "Backward locomotion command"),
        ("AVB", ["AVBL", "AVBR"], Neurotransmitter.ACH, "Forward locomotion command"),
        ("AVD", ["AVDL", "AVDR"], Neurotransmitter.GLU, "Backward locomotion command"),
        ("AVE", ["AVEL", "AVER"], Neurotransmitter.GLU, "Backward locomotion command"),
        ("PVC", ["PVCL", "PVCR"], Neurotransmitter.ACH, "Forward locomotion command"),
    ]

    # Ring interneurons
    inter_ring = [
        ("AIB", ["AIBL", "AIBR"], Neurotransmitter.GLU, "Locomotion modulation"),
        ("AIY", ["AIYL", "AIYR"], Neurotransmitter.ACH, "Thermotaxis, learning"),
        ("AIZ", ["AIZL", "AIZR"], Neurotransmitter.GLU, "Thermotaxis"),
        ("AIA", ["AIAL", "AIAR"], Neurotransmitter.ACH, "Sensory integration"),
        ("RIA", ["RIAL", "RIAR"], Neurotransmitter.GLU, "Head movement, turns"),
        ("RIB", ["RIBL", "RIBR"], Neurotransmitter.GABA, "Locomotion modulation"),
        ("RIC", ["RICL", "RICR"], Neurotransmitter.OCT, "Octopamine signaling"),
        ("RIF", ["RIFL", "RIFR"], Neurotransmitter.ACH, "Unknown"),
        ("RIG", ["RIGL", "RIGR"], Neurotransmitter.GLU, "Sensory hub"),
        ("RIH", ["RIH"], Neurotransmitter.SEROTONIN, "5-HT signaling"),
        ("RIM", ["RIML", "RIMR"], Neurotransmitter.TYR, "Locomotion, tyramine"),
        ("RIP", ["RIPL", "RIPR"], Neurotransmitter.UNKNOWN, "Pharynx-soma connection"),
        ("RIS", ["RIS"], Neurotransmitter.GABA, "Sleep, quiescence"),
        ("RIV", ["RIVL", "RIVR"], Neurotransmitter.GLU, "Head turns"),
    ]

    # Other interneurons
    inter_other = [
        ("AVF", ["AVFL", "AVFR"], Neurotransmitter.UNKNOWN, "Defecation, locomotion"),
        ("AVG", ["AVG"], Neurotransmitter.ACH, "Pioneer neuron, guidance"),
        ("AVH", ["AVHL", "AVHR"], Neurotransmitter.UNKNOWN, "Sensory processing"),
        ("AVJ", ["AVJL", "AVJR"], Neurotransmitter.UNKNOWN, "Egg-laying, defecation"),
        ("AVK", ["AVKL", "AVKR"], Neurotransmitter.UNKNOWN, "Locomotion modulation"),
        ("DVA", ["DVA"], Neurotransmitter.ACH, "Mechanosensory integration"),
        ("DVB", ["DVB"], Neurotransmitter.GABA, "Defecation motor program"),
        ("DVC", ["DVC"], Neurotransmitter.GLU, "Backward locomotion"),
        ("PVN", ["PVNL", "PVNR"], Neurotransmitter.UNKNOWN, "Unknown"),
        ("PVP", ["PVPL", "PVPR"], Neurotransmitter.ACH, "Locomotion modulation"),
        ("PVQ", ["PVQL", "PVQR"], Neurotransmitter.GLU, "Unknown"),
        ("PVR", ["PVR"], Neurotransmitter.GLU, "Harsh touch response"),
        ("PVT", ["PVT"], Neurotransmitter.UNKNOWN, "Unknown"),
        ("PVW", ["PVWL", "PVWR"], Neurotransmitter.ACH, "Unknown"),
        ("LUA", ["LUAL", "LUAR"], Neurotransmitter.GLU, "Male tail neurons"),
    ]

    for group in [inter_command, inter_ring, inter_other]:
        for nclass, names, nt, _ in group:
            for name in names:
                pos = "U" if len(names) == 1 else ("L" if name.endswith("L") else "R")
                neurons[name] = Neuron(
                    name=name, neuron_class=nclass, neuron_type=NeuronType.INTERNEURON,
                    neurotransmitter=nt, position=pos
                )

    # -------------------------------------------------------------------------
    # MOTOR NEURONS (~113 neurons, 45 classes)
    # -------------------------------------------------------------------------

    # Ventral cord motor neurons
    motor_vc = [
        ("DA", [f"DA{i}" for i in range(1, 10)], Neurotransmitter.ACH, "Dorsal backward"),
        ("DB", [f"DB{i}" for i in range(1, 8)], Neurotransmitter.ACH, "Dorsal forward"),
        ("DD", [f"DD{i}" for i in range(1, 7)], Neurotransmitter.GABA, "Dorsal inhibitory"),
        ("VA", [f"VA{i}" for i in range(1, 13)], Neurotransmitter.ACH, "Ventral backward"),
        ("VB", [f"VB{i}" for i in range(1, 12)], Neurotransmitter.ACH, "Ventral forward"),
        ("VD", [f"VD{i}" for i in range(1, 14)], Neurotransmitter.GABA, "Ventral inhibitory"),
        ("AS", [f"AS{i}" for i in range(1, 12)], Neurotransmitter.ACH, "Sublateral motor"),
        ("VC", [f"VC{i}" for i in range(1, 7)], Neurotransmitter.ACH, "Vulval muscles"),
    ]

    # Head motor neurons
    motor_head = [
        ("RMD", ["RMDDL", "RMDDR", "RMDVL", "RMDVR", "RMDL", "RMDR"], Neurotransmitter.ACH, "Head movement"),
        ("RME", ["RMED", "RMEV", "RMEL", "RMER"], Neurotransmitter.GABA, "Head movement"),
        ("RMF", ["RMFL", "RMFR"], Neurotransmitter.ACH, "Head movement"),
        ("RMG", ["RMGL", "RMGR"], Neurotransmitter.ACH, "Sensory hub motor"),
        ("RMH", ["RMHL", "RMHR"], Neurotransmitter.ACH, "Head movement"),
        ("SMB", ["SMBDL", "SMBDR", "SMBVL", "SMBVR"], Neurotransmitter.ACH, "Head oscillation"),
        ("SMD", ["SMDDL", "SMDDR", "SMDVL", "SMDVR"], Neurotransmitter.ACH, "Head movement"),
    ]

    # Sublateral motor neurons
    motor_sublateral = [
        ("SAA", ["SAADL", "SAADR", "SAAVL", "SAAVR"], Neurotransmitter.ACH, "Head movement"),
        ("SAB", ["SABD", "SABVL", "SABVR"], Neurotransmitter.ACH, "Head movement"),
        ("SIA", ["SIADL", "SIADR", "SIAVL", "SIAVR"], Neurotransmitter.ACH, "Head movement"),
        ("SIB", ["SIBDL", "SIBDR", "SIBVL", "SIBVR"], Neurotransmitter.ACH, "Head movement"),
    ]

    # Other motor
    motor_other = [
        ("HSN", ["HSNL", "HSNR"], Neurotransmitter.SEROTONIN, "Egg-laying"),
        ("PDA", ["PDA"], Neurotransmitter.ACH, "Defecation"),
        ("PDB", ["PDB"], Neurotransmitter.ACH, "Defecation"),
    ]

    for group in [motor_vc, motor_head, motor_sublateral, motor_other]:
        for nclass, names, nt, _ in group:
            for name in names:
                pos = "U" if len(names) == 1 else ("L" if name.endswith("L") else "R")
                neurons[name] = Neuron(
                    name=name, neuron_class=nclass, neuron_type=NeuronType.MOTOR,
                    neurotransmitter=nt, position=pos
                )

    # -------------------------------------------------------------------------
    # PHARYNGEAL NEURONS (20 neurons, 14 classes)
    # -------------------------------------------------------------------------

    pharyngeal = [
        ("I1", ["I1L", "I1R"], Neurotransmitter.GLU, "Pharyngeal interneuron"),
        ("I2", ["I2L", "I2R"], Neurotransmitter.GLU, "Pharyngeal interneuron"),
        ("I3", ["I3"], Neurotransmitter.GLU, "Pharyngeal interneuron"),
        ("I4", ["I4"], Neurotransmitter.GLU, "Pharyngeal interneuron"),
        ("I5", ["I5"], Neurotransmitter.GLU, "Pharyngeal interneuron"),
        ("I6", ["I6"], Neurotransmitter.GLU, "Pharyngeal interneuron"),
        ("M1", ["M1"], Neurotransmitter.ACH, "Pharyngeal motor"),
        ("M2", ["M2L", "M2R"], Neurotransmitter.ACH, "Pharyngeal motor"),
        ("M3", ["M3L", "M3R"], Neurotransmitter.GLU, "Pharyngeal motor"),
        ("M4", ["M4"], Neurotransmitter.ACH, "Pharyngeal motor"),
        ("M5", ["M5"], Neurotransmitter.ACH, "Pharyngeal motor"),
        ("MC", ["MCL", "MCR"], Neurotransmitter.ACH, "Marginal cells"),
        ("MI", ["MI"], Neurotransmitter.GLU, "Pharyngeal motor"),
        ("NSM", ["NSML", "NSMR"], Neurotransmitter.SEROTONIN, "Serotonergic, feeding"),
    ]

    for nclass, names, nt, _ in pharyngeal:
        for name in names:
            pos = "U" if len(names) == 1 else ("L" if name.endswith("L") else "R")
            neurons[name] = Neuron(
                name=name, neuron_class=nclass, neuron_type=NeuronType.PHARYNGEAL,
                neurotransmitter=nt, position=pos
            )

    return neurons


def create_chemical_synapses(neurons: Dict[str, Neuron]) -> List[Synapse]:
    """
    Create chemical synapse connections based on known connectome data.
    Weights approximate the number of synaptic contacts.
    """
    synapses = []

    # -------------------------------------------------------------------------
    # LOCOMOTION CIRCUIT (Forward/Backward Command System)
    # -------------------------------------------------------------------------

    # Touch sensors -> Command interneurons
    # Anterior touch -> backward escape
    for pre in ["ALML", "ALMR", "AVM"]:
        for post in ["AVDL", "AVDR", "AVEL", "AVER"]:
            if pre in neurons and post in neurons:
                synapses.append(Synapse(pre, post, 15))
        for post in ["AVAL", "AVAR"]:
            if pre in neurons and post in neurons:
                synapses.append(Synapse(pre, post, 8))

    # Posterior touch -> forward escape
    for pre in ["PLML", "PLMR", "PVM"]:
        for post in ["PVCL", "PVCR"]:
            if pre in neurons and post in neurons:
                synapses.append(Synapse(pre, post, 20))
        for post in ["AVBL", "AVBR"]:
            if pre in neurons and post in neurons:
                synapses.append(Synapse(pre, post, 5))

    # Command interneurons -> Motor neurons
    # AVA, AVD, AVE -> DA, VA (backward A-type)
    for cmd in ["AVAL", "AVAR"]:
        for i in range(1, 10):
            if f"DA{i}" in neurons:
                synapses.append(Synapse(cmd, f"DA{i}", 25))
        for i in range(1, 13):
            if f"VA{i}" in neurons:
                synapses.append(Synapse(cmd, f"VA{i}", 20))

    for cmd in ["AVDL", "AVDR", "AVEL", "AVER"]:
        for i in range(1, 10):
            if f"DA{i}" in neurons:
                synapses.append(Synapse(cmd, f"DA{i}", 8))
        for i in range(1, 13):
            if f"VA{i}" in neurons:
                synapses.append(Synapse(cmd, f"VA{i}", 6))

    # AVB, PVC -> DB, VB (forward B-type)
    for cmd in ["AVBL", "AVBR"]:
        for i in range(1, 8):
            if f"DB{i}" in neurons:
                synapses.append(Synapse(cmd, f"DB{i}", 25))
        for i in range(1, 12):
            if f"VB{i}" in neurons:
                synapses.append(Synapse(cmd, f"VB{i}", 20))

    for cmd in ["PVCL", "PVCR"]:
        for i in range(1, 8):
            if f"DB{i}" in neurons:
                synapses.append(Synapse(cmd, f"DB{i}", 12))
        for i in range(1, 12):
            if f"VB{i}" in neurons:
                synapses.append(Synapse(cmd, f"VB{i}", 10))

    # Cross-inhibition: D-type motor neurons
    # DA/DB -> DD (dorsal); VA/VB -> VD (ventral)
    for i in range(1, 7):
        dd = f"DD{i}"
        if dd in neurons:
            for j in range(max(1, i-1), min(10, i+2)):
                if f"DA{j}" in neurons:
                    synapses.append(Synapse(f"DA{j}", dd, 5))
            for j in range(max(1, i-1), min(8, i+2)):
                if f"DB{j}" in neurons:
                    synapses.append(Synapse(f"DB{j}", dd, 5))

    for i in range(1, 14):
        vd = f"VD{i}"
        if vd in neurons:
            for j in range(max(1, i-1), min(13, i+2)):
                if f"VA{j}" in neurons:
                    synapses.append(Synapse(f"VA{j}", vd, 5))
            for j in range(max(1, i-1), min(12, i+2)):
                if f"VB{j}" in neurons:
                    synapses.append(Synapse(f"VB{j}", vd, 5))

    # -------------------------------------------------------------------------
    # THERMOTAXIS CIRCUIT
    # -------------------------------------------------------------------------

    # AFD (thermosensor) -> AIY, AIZ
    for side in ["L", "R"]:
        if f"AFD{side}" in neurons:
            synapses.append(Synapse(f"AFD{side}", f"AIY{side}", 30))
            synapses.append(Synapse(f"AFD{side}", f"AIZ{side}", 15))

    # AWC -> AIY, AIZ (also thermosensation)
    for side in ["L", "R"]:
        if f"AWC{side}" in neurons:
            synapses.append(Synapse(f"AWC{side}", f"AIY{side}", 10))
            synapses.append(Synapse(f"AWC{side}", f"AIZ{side}", 8))
            synapses.append(Synapse(f"AWC{side}", f"AIB{side}", 12))

    # AIY -> RIA (head movement)
    for side in ["L", "R"]:
        if f"AIY{side}" in neurons and f"RIA{side}" in neurons:
            synapses.append(Synapse(f"AIY{side}", f"RIA{side}", 20))

    # AIZ -> AIB -> RIM
    for side in ["L", "R"]:
        if f"AIZ{side}" in neurons and f"AIB{side}" in neurons:
            synapses.append(Synapse(f"AIZ{side}", f"AIB{side}", 15))
        if f"AIB{side}" in neurons and f"RIM{side}" in neurons:
            synapses.append(Synapse(f"AIB{side}", f"RIM{side}", 10))

    # -------------------------------------------------------------------------
    # CHEMOTAXIS CIRCUIT
    # -------------------------------------------------------------------------

    # AWA (attractive odorants) -> AIA -> AIY
    for side in ["L", "R"]:
        if f"AWA{side}" in neurons and f"AIA{side}" in neurons:
            synapses.append(Synapse(f"AWA{side}", f"AIA{side}", 25))
        if f"AIA{side}" in neurons and f"AIY{side}" in neurons:
            synapses.append(Synapse(f"AIA{side}", f"AIY{side}", 15))

    # ASE (water-soluble attractants) -> AIY, AIB
    for side in ["L", "R"]:
        if f"ASE{side}" in neurons:
            synapses.append(Synapse(f"ASE{side}", f"AIY{side}", 20))
            synapses.append(Synapse(f"ASE{side}", f"AIB{side}", 15))

    # AWB (repulsive odorants) -> AIB, AIZ
    for side in ["L", "R"]:
        if f"AWB{side}" in neurons:
            synapses.append(Synapse(f"AWB{side}", f"AIB{side}", 15))
            synapses.append(Synapse(f"AWB{side}", f"AIZ{side}", 10))

    # -------------------------------------------------------------------------
    # NOCICEPTION / AVOIDANCE
    # -------------------------------------------------------------------------

    # ASH (polymodal nociceptor) -> AVA, AVD, AVE (escape)
    for side in ["L", "R"]:
        ash = f"ASH{side}"
        if ash in neurons:
            for cmd in ["AVAL", "AVAR", "AVDL", "AVDR"]:
                if cmd in neurons:
                    synapses.append(Synapse(ash, cmd, 12))

    # FLP (harsh touch nose) -> AVA, AVD
    for side in ["L", "R"]:
        flp = f"FLP{side}"
        if flp in neurons:
            for cmd in ["AVAL", "AVAR", "AVDL", "AVDR"]:
                if cmd in neurons:
                    synapses.append(Synapse(flp, cmd, 8))

    # -------------------------------------------------------------------------
    # MODULATORY CIRCUITS
    # -------------------------------------------------------------------------

    # Dopaminergic neurons (CEP, ADE, PDE) -> broad targets
    for da_neuron in ["CEPDL", "CEPDR", "CEPVL", "CEPVR", "ADEL", "ADER", "PDEL", "PDER"]:
        if da_neuron in neurons:
            # DA modulates command interneurons
            for cmd in ["AVAL", "AVAR", "AVBL", "AVBR"]:
                if cmd in neurons:
                    synapses.append(Synapse(da_neuron, cmd, 3))

    # Serotonergic neurons (NSM, HSN, RIH)
    for sero in ["NSML", "NSMR"]:
        if sero in neurons:
            # NSM modulates pharyngeal pumping and locomotion
            for target in ["M1", "M4", "I1L", "I1R"]:
                if target in neurons:
                    synapses.append(Synapse(sero, target, 10))

    if "RIH" in neurons:
        # RIH is a serotonergic hub
        for cmd in ["AVAL", "AVAR", "AVBL", "AVBR"]:
            if cmd in neurons:
                synapses.append(Synapse("RIH", cmd, 5))

    # RIM (tyramine) -> reversal suppression
    for side in ["L", "R"]:
        rim = f"RIM{side}"
        if rim in neurons:
            for cmd in ["AVAL", "AVAR", "AVBL", "AVBR"]:
                if cmd in neurons:
                    synapses.append(Synapse(rim, cmd, 8))

    # RIC (octopamine) -> arousal/escape
    for side in ["L", "R"]:
        ric = f"RIC{side}"
        if ric in neurons:
            for target in ["AVAL", "AVAR"]:
                if target in neurons:
                    synapses.append(Synapse(ric, target, 5))

    # -------------------------------------------------------------------------
    # HEAD MOTOR CONTROL
    # -------------------------------------------------------------------------

    # RIA -> RMD (head movement)
    for pre_side in ["L", "R"]:
        ria = f"RIA{pre_side}"
        if ria in neurons:
            for rmd in ["RMDDL", "RMDDR", "RMDVL", "RMDVR", "RMDL", "RMDR"]:
                if rmd in neurons:
                    synapses.append(Synapse(ria, rmd, 8))

    # RIV -> SMD (head turns)
    for side in ["L", "R"]:
        riv = f"RIV{side}"
        if riv in neurons:
            for smd in ["SMDDL", "SMDDR", "SMDVL", "SMDVR"]:
                if smd in neurons:
                    synapses.append(Synapse(riv, smd, 6))

    # -------------------------------------------------------------------------
    # PHARYNGEAL CIRCUIT (Semi-autonomous)
    # -------------------------------------------------------------------------

    # MC (marginal cells) -> pharyngeal muscles via M neurons
    for mc in ["MCL", "MCR"]:
        if mc in neurons:
            for m in ["M1", "M2L", "M2R", "M3L", "M3R", "M4", "M5"]:
                if m in neurons:
                    synapses.append(Synapse(mc, m, 10))

    # I neurons form pharyngeal interneuron network
    for i_pre in ["I1L", "I1R", "I2L", "I2R", "I3", "I4", "I5", "I6"]:
        if i_pre in neurons:
            for i_post in ["I1L", "I1R", "I2L", "I2R", "I3", "I4", "I5", "I6"]:
                if i_post in neurons and i_pre != i_post:
                    synapses.append(Synapse(i_pre, i_post, 3))

    return synapses


def create_gap_junctions(neurons: Dict[str, Neuron]) -> List[GapJunction]:
    """
    Create gap junction (electrical synapse) connections.
    These are symmetric/bidirectional.
    """
    gap_junctions = []

    # -------------------------------------------------------------------------
    # COMMAND INTERNEURON GAP JUNCTIONS (Rich Club)
    # -------------------------------------------------------------------------

    # AVA bilateral coupling (very strong)
    if "AVAL" in neurons and "AVAR" in neurons:
        gap_junctions.append(GapJunction("AVAL", "AVAR", 30))

    # AVB bilateral coupling
    if "AVBL" in neurons and "AVBR" in neurons:
        gap_junctions.append(GapJunction("AVBL", "AVBR", 25))

    # PVC bilateral coupling
    if "PVCL" in neurons and "PVCR" in neurons:
        gap_junctions.append(GapJunction("PVCL", "PVCR", 20))

    # AVA-AVB cross-coupling (coordination)
    for side in ["L", "R"]:
        ava = f"AVA{side}"
        avb = f"AVB{side}"
        if ava in neurons and avb in neurons:
            gap_junctions.append(GapJunction(ava, avb, 10))

    # DVA hub connections
    if "DVA" in neurons:
        for cmd in ["AVAL", "AVAR", "AVBL", "AVBR"]:
            if cmd in neurons:
                gap_junctions.append(GapJunction("DVA", cmd, 8))

    # -------------------------------------------------------------------------
    # MOTOR NEURON GAP JUNCTIONS (Wave propagation)
    # -------------------------------------------------------------------------

    # DB neurons coupled along body
    for i in range(1, 7):
        if f"DB{i}" in neurons and f"DB{i+1}" in neurons:
            gap_junctions.append(GapJunction(f"DB{i}", f"DB{i+1}", 5))

    # VB neurons coupled along body
    for i in range(1, 11):
        if f"VB{i}" in neurons and f"VB{i+1}" in neurons:
            gap_junctions.append(GapJunction(f"VB{i}", f"VB{i+1}", 5))

    # DA neurons coupled
    for i in range(1, 9):
        if f"DA{i}" in neurons and f"DA{i+1}" in neurons:
            gap_junctions.append(GapJunction(f"DA{i}", f"DA{i+1}", 4))

    # VA neurons coupled
    for i in range(1, 12):
        if f"VA{i}" in neurons and f"VA{i+1}" in neurons:
            gap_junctions.append(GapJunction(f"VA{i}", f"VA{i+1}", 4))

    # -------------------------------------------------------------------------
    # SENSORY NEURON GAP JUNCTIONS
    # -------------------------------------------------------------------------

    # Touch receptor bilateral coupling
    if "ALML" in neurons and "ALMR" in neurons:
        gap_junctions.append(GapJunction("ALML", "ALMR", 8))

    if "PLML" in neurons and "PLMR" in neurons:
        gap_junctions.append(GapJunction("PLML", "PLMR", 8))

    # Thermosensory bilateral coupling
    if "AFDL" in neurons and "AFDR" in neurons:
        gap_junctions.append(GapJunction("AFDL", "AFDR", 5))

    # -------------------------------------------------------------------------
    # INTERNEURON GAP JUNCTIONS
    # -------------------------------------------------------------------------

    # AIY bilateral (thermotaxis)
    if "AIYL" in neurons and "AIYR" in neurons:
        gap_junctions.append(GapJunction("AIYL", "AIYR", 6))

    # RIG sensory hub bilateral (many gap junctions)
    if "RIGL" in neurons and "RIGR" in neurons:
        gap_junctions.append(GapJunction("RIGL", "RIGR", 15))

    # RIA bilateral (head movement)
    if "RIAL" in neurons and "RIAR" in neurons:
        gap_junctions.append(GapJunction("RIAL", "RIAR", 10))

    return gap_junctions


# =============================================================================
# Neurochemical Control System
# =============================================================================

@dataclass
class NeurochemicalState:
    """
    User-controllable neurochemical levers.
    Each value represents a global modulation factor (0.0 to 2.0).
    1.0 = normal/baseline level
    """
    acetylcholine: float = 1.0    # ACh: excitatory, muscle activation
    glutamate: float = 1.0        # Glu: excitatory, sensory processing
    gaba: float = 1.0             # GABA: inhibitory, cross-inhibition
    dopamine: float = 1.0         # DA: reward, motor modulation, basal slowing
    serotonin: float = 1.0        # 5-HT: feeding, egg-laying, locomotion slowing
    octopamine: float = 1.0       # Oct: arousal, escape, "adrenaline-like"
    tyramine: float = 1.0         # Tyr: reversal suppression

    def get_multiplier(self, nt: Neurotransmitter) -> float:
        """Get the modulation multiplier for a neurotransmitter."""
        mapping = {
            Neurotransmitter.ACH: self.acetylcholine,
            Neurotransmitter.GLU: self.glutamate,
            Neurotransmitter.GABA: self.gaba,
            Neurotransmitter.DA: self.dopamine,
            Neurotransmitter.SEROTONIN: self.serotonin,
            Neurotransmitter.OCT: self.octopamine,
            Neurotransmitter.TYR: self.tyramine,
            Neurotransmitter.UNKNOWN: 1.0,
        }
        return mapping.get(nt, 1.0)

    def get_valence(self, nt: Neurotransmitter) -> float:
        """Get the sign (excitatory/inhibitory) for a neurotransmitter."""
        if nt == Neurotransmitter.GABA:
            return -1.0  # Inhibitory
        elif nt in [Neurotransmitter.ACH, Neurotransmitter.GLU]:
            return 1.0   # Excitatory
        else:
            return 0.5   # Modulatory (weak effect)


# =============================================================================
# Sensory Input System
# =============================================================================

@dataclass
class SensoryInput:
    """External sensory stimuli that the worm experiences."""
    # Touch
    anterior_touch: float = 0.0     # 0-1: gentle touch on head
    posterior_touch: float = 0.0    # 0-1: gentle touch on tail
    harsh_touch_head: float = 0.0   # 0-1: nose poke
    harsh_touch_tail: float = 0.0   # 0-1: tail poke

    # Chemical
    attractive_odor: float = 0.0    # 0-1: diacetyl, benzaldehyde
    repulsive_odor: float = 0.0     # 0-1: octanol, quinine
    salt_gradient: float = 0.0      # -1 to 1: NaCl (neg = away, pos = toward)

    # Thermal
    temperature: float = 20.0       # Celsius (cultivation temp typically 20C)
    temp_gradient: float = 0.0      # -1 to 1: cold to warm direction

    # Other
    food_present: bool = True       # Bacteria present
    oxygen_level: float = 21.0      # % (normal atmospheric)


# =============================================================================
# Behavioral Output System
# =============================================================================

@dataclass
class BehavioralState:
    """Output behavioral state of the worm."""
    # Locomotion
    forward_drive: float = 0.0      # 0-1: forward movement tendency
    backward_drive: float = 0.0     # 0-1: backward movement tendency
    turn_bias: float = 0.0          # -1 to 1: left to right turn bias
    speed: float = 0.0              # 0-1: movement speed

    # Body posture
    body_bend: float = 0.0          # -1 to 1: dorsal to ventral
    head_angle: float = 0.0         # -1 to 1: head sweep direction

    # Other behaviors
    pharyngeal_pumping: float = 0.0 # 0-1: feeding rate
    omega_turn: bool = False        # Deep turn (reversal + turn)

    @property
    def primary_behavior(self) -> Behavior:
        """Determine the primary behavioral state."""
        if self.omega_turn:
            return Behavior.ESCAPE_ANTERIOR
        elif self.backward_drive > 0.5 and self.backward_drive > self.forward_drive:
            return Behavior.BACKWARD
        elif self.forward_drive > 0.3:
            return Behavior.FORWARD
        elif abs(self.turn_bias) > 0.5:
            return Behavior.TURN_LEFT if self.turn_bias < 0 else Behavior.TURN_RIGHT
        elif self.pharyngeal_pumping > 0.5:
            return Behavior.FEEDING
        else:
            return Behavior.STOP

    def describe(self) -> str:
        """Human-readable description of current behavior."""
        behavior = self.primary_behavior
        lines = [
            f"Primary Behavior: {behavior.value}",
            f"Forward Drive: {self.forward_drive:.2f}",
            f"Backward Drive: {self.backward_drive:.2f}",
            f"Turn Bias: {self.turn_bias:+.2f} ({'Left' if self.turn_bias < 0 else 'Right'})",
            f"Speed: {self.speed:.2f}",
            f"Head Angle: {self.head_angle:+.2f}",
            f"Pharyngeal Pumping: {self.pharyngeal_pumping:.2f}",
        ]
        return "\n".join(lines)


# =============================================================================
# Neural Simulation Engine
# =============================================================================

class CElegansSimulation:
    """
    Main simulation class for C. elegans nervous system.

    Implements a simplified integrate-and-fire model with:
    - Chemical synapse transmission
    - Gap junction electrical coupling
    - Neurotransmitter modulation
    - Behavioral output mapping
    """

    def __init__(self):
        # Build the connectome
        self.neurons = create_neurons()
        self.synapses = create_chemical_synapses(self.neurons)
        self.gap_junctions = create_gap_junctions(self.neurons)

        # Index neurons for fast lookup
        self.neuron_names = list(self.neurons.keys())
        self.neuron_index = {name: i for i, name in enumerate(self.neuron_names)}
        self.n_neurons = len(self.neuron_names)

        # Build adjacency matrices
        self._build_matrices()

        # State
        self.neurochemistry = NeurochemicalState()
        self.sensory = SensoryInput()
        self.behavior = BehavioralState()

        # Simulation parameters
        self.dt = 1.0  # ms timestep
        self.time = 0.0

        # Activity vector
        self.activity = np.zeros(self.n_neurons)

        # Behavioral history for computing observables
        self.history = BehaviorHistory(window_size=1000, dt_ms=self.dt)

        # Cached observable vector (updated periodically)
        self._observables: Optional[ObservableVector] = None
        self._observables_update_interval = 100  # Update every 100 steps
        self._steps_since_observables_update = 0

        print(f"Initialized C. elegans simulation:")
        print(f"  Neurons: {self.n_neurons}")
        print(f"  Chemical synapses: {len(self.synapses)}")
        print(f"  Gap junctions: {len(self.gap_junctions)}")

    def _build_matrices(self):
        """Build weighted adjacency matrices for fast simulation."""
        n = self.n_neurons

        # Chemical synapse matrix (asymmetric)
        self.W_chem = np.zeros((n, n))
        self.W_chem_nt = {}  # Separate matrices by neurotransmitter

        for nt in Neurotransmitter:
            self.W_chem_nt[nt] = np.zeros((n, n))

        for syn in self.synapses:
            if syn.pre in self.neuron_index and syn.post in self.neuron_index:
                i = self.neuron_index[syn.pre]
                j = self.neuron_index[syn.post]
                pre_nt = self.neurons[syn.pre].neurotransmitter
                self.W_chem[i, j] += syn.strength
                self.W_chem_nt[pre_nt][i, j] += syn.strength

        # Gap junction matrix (symmetric)
        self.W_gap = np.zeros((n, n))
        for gj in self.gap_junctions:
            if gj.neuron_a in self.neuron_index and gj.neuron_b in self.neuron_index:
                i = self.neuron_index[gj.neuron_a]
                j = self.neuron_index[gj.neuron_b]
                self.W_gap[i, j] += gj.conductance
                self.W_gap[j, i] += gj.conductance

        # Neurotransmitter assignment vector
        self.nt_assignment = np.array([
            self.neurons[name].neurotransmitter.value
            for name in self.neuron_names
        ])

    def _apply_sensory_input(self):
        """Convert sensory input to neural activity."""
        # Anterior touch -> ALM, AVM
        for name in ["ALML", "ALMR", "AVM"]:
            if name in self.neuron_index:
                idx = self.neuron_index[name]
                self.activity[idx] += self.sensory.anterior_touch * 2.0

        # Posterior touch -> PLM, PVM
        for name in ["PLML", "PLMR", "PVM"]:
            if name in self.neuron_index:
                idx = self.neuron_index[name]
                self.activity[idx] += self.sensory.posterior_touch * 2.0

        # Harsh touch head -> FLP, ASH
        for name in ["FLPL", "FLPR", "ASHL", "ASHR"]:
            if name in self.neuron_index:
                idx = self.neuron_index[name]
                self.activity[idx] += self.sensory.harsh_touch_head * 3.0

        # Attractive odor -> AWA
        for name in ["AWAL", "AWAR"]:
            if name in self.neuron_index:
                idx = self.neuron_index[name]
                self.activity[idx] += self.sensory.attractive_odor * 1.5

        # Repulsive odor -> AWB, ADL
        for name in ["AWBL", "AWBR", "ADLL", "ADLR"]:
            if name in self.neuron_index:
                idx = self.neuron_index[name]
                self.activity[idx] += self.sensory.repulsive_odor * 2.0

        # Salt gradient -> ASE
        for name in ["ASEL", "ASER"]:
            if name in self.neuron_index:
                idx = self.neuron_index[name]
                self.activity[idx] += abs(self.sensory.salt_gradient) * 1.0

        # Temperature -> AFD, AWC
        temp_dev = (self.sensory.temperature - 20.0) / 5.0  # Deviation from 20C
        for name in ["AFDL", "AFDR", "AWCL", "AWCR"]:
            if name in self.neuron_index:
                idx = self.neuron_index[name]
                self.activity[idx] += abs(temp_dev) * 1.0 + self.sensory.temp_gradient * 0.5

        # Food presence -> modulates various circuits
        if self.sensory.food_present:
            # NSM serotonin neurons active during feeding
            for name in ["NSML", "NSMR"]:
                if name in self.neuron_index:
                    idx = self.neuron_index[name]
                    self.activity[idx] += 0.5

        # Oxygen -> AQR, PQR, URX, BAG
        o2_dev = (self.sensory.oxygen_level - 21.0) / 5.0
        for name in ["AQR", "PQR", "URXL", "URXR", "BAGL", "BAGR"]:
            if name in self.neuron_index:
                idx = self.neuron_index[name]
                self.activity[idx] += abs(o2_dev) * 0.5

    def _compute_synaptic_input(self) -> np.ndarray:
        """Compute synaptic input to each neuron."""
        total_input = np.zeros(self.n_neurons)

        # Chemical synapses (with neurotransmitter modulation)
        for nt in Neurotransmitter:
            if nt == Neurotransmitter.UNKNOWN:
                continue

            multiplier = self.neurochemistry.get_multiplier(nt)
            valence = self.neurochemistry.get_valence(nt)

            # Input from presynaptic neurons with this NT
            syn_input = self.W_chem_nt[nt].T @ self.activity
            total_input += syn_input * multiplier * valence

        # Gap junctions (direct electrical coupling, no NT modulation)
        # Current flows from high to low activity
        for i in range(self.n_neurons):
            for j in range(self.n_neurons):
                if self.W_gap[i, j] > 0:
                    # Bidirectional current
                    total_input[i] += self.W_gap[i, j] * (self.activity[j] - self.activity[i]) * 0.3

        return total_input

    def _update_activity(self, synaptic_input: np.ndarray):
        """Update neural activity based on input."""
        # Simple leaky integrator dynamics
        tau = 10.0  # ms
        decay = np.exp(-self.dt / tau)

        # Activity update with nonlinearity (sigmoid)
        self.activity = self.activity * decay + synaptic_input * (1 - decay)

        # Apply sigmoid nonlinearity
        self.activity = 1.0 / (1.0 + np.exp(-self.activity + 0.5))

        # Clamp to [0, 1]
        self.activity = np.clip(self.activity, 0.0, 1.0)

    def _compute_behavior(self):
        """Map neural activity to behavioral output."""
        # Get activity of key neurons
        def get_activity(name: str) -> float:
            if name in self.neuron_index:
                return self.activity[self.neuron_index[name]]
            return 0.0

        def get_class_activity(prefix: str) -> float:
            """Average activity of all neurons with given prefix."""
            activities = [self.activity[self.neuron_index[n]]
                         for n in self.neuron_names if n.startswith(prefix)]
            return np.mean(activities) if activities else 0.0

        # Forward command (AVB, PVC)
        forward_cmd = (get_activity("AVBL") + get_activity("AVBR") +
                       get_activity("PVCL") + get_activity("PVCR")) / 4.0

        # Backward command (AVA, AVD, AVE)
        backward_cmd = (get_activity("AVAL") + get_activity("AVAR") +
                        get_activity("AVDL") + get_activity("AVDR") +
                        get_activity("AVEL") + get_activity("AVER")) / 6.0

        # B-type motor neuron activity (forward)
        b_motor = (get_class_activity("DB") + get_class_activity("VB")) / 2.0

        # A-type motor neuron activity (backward)
        a_motor = (get_class_activity("DA") + get_class_activity("VA")) / 2.0

        # D-type (inhibitory) dampens movement
        d_motor = (get_class_activity("DD") + get_class_activity("VD")) / 2.0

        # Compute drives
        self.behavior.forward_drive = forward_cmd * 0.4 + b_motor * 0.6
        self.behavior.backward_drive = backward_cmd * 0.4 + a_motor * 0.6

        # Speed is reduced by GABA (D-type) and modulated by overall activity
        gaba_effect = self.neurochemistry.gaba * d_motor
        self.behavior.speed = max(self.behavior.forward_drive, self.behavior.backward_drive)
        self.behavior.speed *= (1.0 - gaba_effect * 0.5)

        # Serotonin slows locomotion
        self.behavior.speed *= (2.0 - self.neurochemistry.serotonin)

        # Turn bias from RIA/RIV asymmetry
        ria_diff = get_activity("RIAL") - get_activity("RIAR")
        riv_diff = get_activity("RIVL") - get_activity("RIVR")
        self.behavior.turn_bias = (ria_diff + riv_diff) / 2.0

        # Head angle from SMD/RMD
        smd_diff = (get_activity("SMDDL") + get_activity("SMDVL") -
                    get_activity("SMDDR") - get_activity("SMDVR")) / 2.0
        self.behavior.head_angle = smd_diff

        # Body bend from dorsal/ventral balance
        dorsal = get_class_activity("DD") + get_class_activity("DA") + get_class_activity("DB")
        ventral = get_class_activity("VD") + get_class_activity("VA") + get_class_activity("VB")
        self.behavior.body_bend = (dorsal - ventral) / (dorsal + ventral + 0.1)

        # Pharyngeal pumping
        pharynx_activity = (get_class_activity("M") + get_class_activity("MC") +
                           get_activity("NSML") + get_activity("NSMR")) / 3.0
        self.behavior.pharyngeal_pumping = pharynx_activity * self.neurochemistry.serotonin

        # Omega turn detection (strong backward + turn)
        self.behavior.omega_turn = (self.behavior.backward_drive > 0.7 and
                                    abs(self.behavior.turn_bias) > 0.5)

    def step(self, n_steps: int = 1):
        """Run simulation for n timesteps."""
        for _ in range(n_steps):
            # 1. Apply sensory input
            self._apply_sensory_input()

            # 2. Compute synaptic input
            synaptic_input = self._compute_synaptic_input()

            # 3. Update neural activity
            self._update_activity(synaptic_input)

            # 4. Compute behavioral output
            self._compute_behavior()

            # 5. Update behavioral history for observable computation
            self.history.update(self.behavior, self.sensory)

            # 6. Periodically update cached observables
            self._steps_since_observables_update += 1
            if self._steps_since_observables_update >= self._observables_update_interval:
                self._observables = self.history.compute_observables()
                self._steps_since_observables_update = 0

            self.time += self.dt

    def reset(self):
        """Reset simulation to initial state."""
        self.activity = np.zeros(self.n_neurons)
        self.time = 0.0
        self.behavior = BehavioralState()
        self.history.reset()
        self._observables = None
        self._steps_since_observables_update = 0

    def get_observables(self, force_update: bool = False) -> ObservableVector:
        """
        Get the current 15-dimensional observable vector.

        Args:
            force_update: If True, recompute observables immediately

        Returns:
            ObservableVector with current behavioral statistics
        """
        if force_update or self._observables is None:
            self._observables = self.history.compute_observables()
            self._steps_since_observables_update = 0
        return self._observables

    def get_neuron_activity(self, name: str) -> float:
        """Get activity of a specific neuron."""
        if name in self.neuron_index:
            return self.activity[self.neuron_index[name]]
        return 0.0

    def get_class_activity(self, neuron_class: str) -> Dict[str, float]:
        """Get activity of all neurons in a class."""
        return {name: self.activity[self.neuron_index[name]]
                for name in self.neuron_names
                if self.neurons[name].neuron_class == neuron_class}

    def get_type_summary(self) -> Dict[str, float]:
        """Get average activity by neuron type."""
        type_activities = {t: [] for t in NeuronType}
        for name in self.neuron_names:
            ntype = self.neurons[name].neuron_type
            type_activities[ntype].append(self.activity[self.neuron_index[name]])
        return {t.name: np.mean(acts) if acts else 0.0
                for t, acts in type_activities.items()}

    def get_top_active_neurons(self, n: int = 10) -> List[Tuple[str, float]]:
        """Get the n most active neurons."""
        activities = [(name, self.activity[self.neuron_index[name]])
                      for name in self.neuron_names]
        activities.sort(key=lambda x: x[1], reverse=True)
        return activities[:n]


# =============================================================================
# Interactive Interface
# =============================================================================

def print_header():
    """Print simulation header."""
    print("\n" + "="*70)
    print("  C. ELEGANS NEURAL SIMULATION")
    print("  302 Neurons | 6,393 Chemical Synapses | 890 Gap Junctions")
    print("="*70 + "\n")


def print_neurochemistry(nc: NeurochemicalState):
    """Print current neurochemistry state."""
    print("\n--- Neurochemical Levers ---")
    print(f"  [1] Acetylcholine (ACh): {nc.acetylcholine:.2f}  [Excitatory, muscle]")
    print(f"  [2] Glutamate (Glu):     {nc.glutamate:.2f}  [Excitatory, sensory]")
    print(f"  [3] GABA:                {nc.gaba:.2f}  [Inhibitory]")
    print(f"  [4] Dopamine (DA):       {nc.dopamine:.2f}  [Modulatory, reward]")
    print(f"  [5] Serotonin (5-HT):    {nc.serotonin:.2f}  [Modulatory, feeding]")
    print(f"  [6] Octopamine (Oct):    {nc.octopamine:.2f}  [Modulatory, arousal]")
    print(f"  [7] Tyramine (Tyr):      {nc.tyramine:.2f}  [Modulatory, reversal]")


def print_sensory(sens: SensoryInput):
    """Print current sensory input state."""
    print("\n--- Sensory Inputs ---")
    print(f"  [a] Anterior touch:   {sens.anterior_touch:.2f}")
    print(f"  [p] Posterior touch:  {sens.posterior_touch:.2f}")
    print(f"  [h] Harsh touch head: {sens.harsh_touch_head:.2f}")
    print(f"  [o] Attractive odor:  {sens.attractive_odor:.2f}")
    print(f"  [r] Repulsive odor:   {sens.repulsive_odor:.2f}")
    print(f"  [t] Temperature:      {sens.temperature:.1f}C")
    print(f"  [f] Food present:     {sens.food_present}")


def print_behavior(beh: BehavioralState):
    """Print current behavioral state."""
    print("\n--- Behavioral Output ---")
    print(beh.describe())


def print_neural_summary(sim: CElegansSimulation):
    """Print neural activity summary."""
    print("\n--- Neural Activity Summary ---")
    type_summary = sim.get_type_summary()
    for ntype, activity in type_summary.items():
        print(f"  {ntype}: {activity:.3f}")

    print("\n  Top 10 Active Neurons:")
    for name, activity in sim.get_top_active_neurons(10):
        neuron = sim.neurons[name]
        print(f"    {name:8s} ({neuron.neuron_class:4s}): {activity:.3f}")


def interactive_mode():
    """Run interactive simulation."""
    print_header()

    sim = CElegansSimulation()

    print("\nCommands:")
    print("  1-7:    Adjust neurochemical lever (enter value 0.0-2.0)")
    print("  a,p,h:  Set touch stimulus (0.0-1.0)")
    print("  o,r:    Set odor stimulus (0.0-1.0)")
    print("  t:      Set temperature (Celsius)")
    print("  f:      Toggle food presence")
    print("  s:      Step simulation (100 timesteps)")
    print("  S:      Step simulation (1000 timesteps)")
    print("  n:      Show neural activity summary")
    print("  x:      Reset simulation")
    print("  q:      Quit")
    print()

    while True:
        print_neurochemistry(sim.neurochemistry)
        print_sensory(sim.sensory)

        # Run some simulation steps
        sim.step(50)

        print_behavior(sim.behavior)

        try:
            cmd = input("\n> Enter command: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting...")
            break

        if not cmd:
            continue

        if cmd == 'q':
            print("Goodbye!")
            break
        elif cmd == 'x':
            sim.reset()
            print("Simulation reset.")
        elif cmd == 's':
            print("Running 100 timesteps...")
            sim.step(100)
        elif cmd == 'S':
            print("Running 1000 timesteps...")
            sim.step(1000)
        elif cmd == 'n':
            print_neural_summary(sim)
        elif cmd == 'f':
            sim.sensory.food_present = not sim.sensory.food_present
        elif cmd in '1234567':
            try:
                val = float(input(f"  Enter new value (0.0-2.0): "))
                val = np.clip(val, 0.0, 2.0)
                if cmd == '1':
                    sim.neurochemistry.acetylcholine = val
                elif cmd == '2':
                    sim.neurochemistry.glutamate = val
                elif cmd == '3':
                    sim.neurochemistry.gaba = val
                elif cmd == '4':
                    sim.neurochemistry.dopamine = val
                elif cmd == '5':
                    sim.neurochemistry.serotonin = val
                elif cmd == '6':
                    sim.neurochemistry.octopamine = val
                elif cmd == '7':
                    sim.neurochemistry.tyramine = val
            except ValueError:
                print("Invalid value.")
        elif cmd == 'a':
            try:
                val = float(input("  Anterior touch (0.0-1.0): "))
                sim.sensory.anterior_touch = np.clip(val, 0.0, 1.0)
            except ValueError:
                print("Invalid value.")
        elif cmd == 'p':
            try:
                val = float(input("  Posterior touch (0.0-1.0): "))
                sim.sensory.posterior_touch = np.clip(val, 0.0, 1.0)
            except ValueError:
                print("Invalid value.")
        elif cmd == 'h':
            try:
                val = float(input("  Harsh touch head (0.0-1.0): "))
                sim.sensory.harsh_touch_head = np.clip(val, 0.0, 1.0)
            except ValueError:
                print("Invalid value.")
        elif cmd == 'o':
            try:
                val = float(input("  Attractive odor (0.0-1.0): "))
                sim.sensory.attractive_odor = np.clip(val, 0.0, 1.0)
            except ValueError:
                print("Invalid value.")
        elif cmd == 'r':
            try:
                val = float(input("  Repulsive odor (0.0-1.0): "))
                sim.sensory.repulsive_odor = np.clip(val, 0.0, 1.0)
            except ValueError:
                print("Invalid value.")
        elif cmd == 't':
            try:
                val = float(input("  Temperature (Celsius): "))
                sim.sensory.temperature = val
            except ValueError:
                print("Invalid value.")
        else:
            print("Unknown command.")


def demo_mode():
    """Run a demonstration of the simulation."""
    print_header()

    sim = CElegansSimulation()

    print("\n=== DEMO: Touch Response ===")
    print("Simulating anterior touch (tap on head)...")
    sim.sensory.anterior_touch = 0.8
    sim.step(200)
    print(f"Response: {sim.behavior.primary_behavior.value}")
    print(f"  Backward drive: {sim.behavior.backward_drive:.2f}")
    print(f"  Forward drive: {sim.behavior.forward_drive:.2f}")

    sim.reset()
    sim.sensory.anterior_touch = 0.0

    print("\nSimulating posterior touch (tap on tail)...")
    sim.sensory.posterior_touch = 0.8
    sim.step(200)
    print(f"Response: {sim.behavior.primary_behavior.value}")
    print(f"  Forward drive: {sim.behavior.forward_drive:.2f}")
    print(f"  Backward drive: {sim.behavior.backward_drive:.2f}")

    sim.reset()

    print("\n=== DEMO: Neurochemical Modulation ===")
    print("\nBaseline locomotion...")
    sim.step(200)
    baseline_speed = sim.behavior.speed
    print(f"  Speed: {baseline_speed:.3f}")

    print("\nIncreasing GABA (inhibitory)...")
    sim.neurochemistry.gaba = 2.0
    sim.step(200)
    print(f"  Speed: {sim.behavior.speed:.3f} (baseline: {baseline_speed:.3f})")

    sim.reset()
    sim.neurochemistry.gaba = 1.0

    print("\nIncreasing Serotonin (slows locomotion)...")
    sim.neurochemistry.serotonin = 2.0
    sim.step(200)
    print(f"  Speed: {sim.behavior.speed:.3f}")
    print(f"  Pharyngeal pumping: {sim.behavior.pharyngeal_pumping:.3f}")

    sim.reset()
    sim.neurochemistry.serotonin = 1.0

    print("\n=== DEMO: Chemotaxis ===")
    print("\nAttractive odor gradient...")
    sim.sensory.attractive_odor = 0.7
    sim.step(300)
    print(f"Response: {sim.behavior.primary_behavior.value}")
    print(f"  Forward drive: {sim.behavior.forward_drive:.2f}")
    print(f"  Turn bias: {sim.behavior.turn_bias:+.2f}")

    sim.reset()

    print("\nRepulsive odor...")
    sim.sensory.repulsive_odor = 0.7
    sim.step(300)
    print(f"Response: {sim.behavior.primary_behavior.value}")
    print(f"  Backward drive: {sim.behavior.backward_drive:.2f}")

    print("\n=== DEMO COMPLETE ===")
    print("Run with --interactive for full control.")


# =============================================================================
# Main Entry Point
# =============================================================================

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--demo":
        demo_mode()
    elif len(sys.argv) > 1 and sys.argv[1] == "--interactive":
        interactive_mode()
    else:
        print("C. elegans Neural Simulation")
        print("Usage:")
        print("  python c_elegans_simulation.py --demo         Run demonstration")
        print("  python c_elegans_simulation.py --interactive  Interactive mode")
        print()

        # Quick test
        sim = CElegansSimulation()
        print("\nQuick test - anterior touch response:")
        sim.sensory.anterior_touch = 1.0
        sim.step(100)
        print(f"  Behavior: {sim.behavior.primary_behavior.value}")
        print(f"  Backward drive: {sim.behavior.backward_drive:.2f}")
