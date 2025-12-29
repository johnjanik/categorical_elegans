#!/usr/bin/env python3
"""
Behavioral Data Loader
======================

Load real C. elegans behavioral data from published datasets for
model validation against observed worm behavior.

Supported formats:
- WCON (Worm tracking Common Object Notation) from OpenWorm Movement Database
- Tierpsy HDF5 feature files
- CSV summary statistics

Data sources:
- OpenWorm Movement Database: https://zenodo.org/communities/open-worm-movement-database
- Tierpsy Tracker: https://github.com/ver228/tierpsy-tracker
- WormBehavior Database: http://wormbehavior.mrc-lmb.cam.ac.uk/
"""

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np

try:
    import h5py
    HAS_H5PY = True
except ImportError:
    HAS_H5PY = False
    h5py = None

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False
    requests = None

from .c_elegans_simulation import ObservableVector


# =============================================================================
# Data Source Registry
# =============================================================================

# Zenodo record IDs for OpenWorm Movement Database datasets
ZENODO_DATASETS = {
    'N2_on_food': '4074963',      # Wild-type N2 on food
    'N2_off_food': '4075007',     # Wild-type N2 off food
    'unc_mutants': '4075033',     # Movement mutants
    'touch_response': '4074990',  # Touch stimulation data
}

# Direct download URLs for sample datasets
SAMPLE_DATA_URLS = {
    'tierpsy_sample': 'https://zenodo.org/record/5551854/files/sample_features.hdf5',
}


# =============================================================================
# WCON Format Parser
# =============================================================================

@dataclass
class WCONTrack:
    """A single worm track from WCON data."""
    worm_id: str
    timestamps: np.ndarray  # Time points (seconds)
    x_positions: np.ndarray  # X coordinates (μm)
    y_positions: np.ndarray  # Y coordinates (μm)

    # Derived quantities (computed on load)
    velocities: Optional[np.ndarray] = None
    angular_velocities: Optional[np.ndarray] = None

    def compute_kinematics(self) -> None:
        """Compute velocity and angular velocity from position data."""
        if len(self.timestamps) < 2:
            return

        dt = np.diff(self.timestamps)
        dx = np.diff(self.x_positions)
        dy = np.diff(self.y_positions)

        # Speed
        self.velocities = np.sqrt(dx**2 + dy**2) / dt

        # Direction angles
        angles = np.arctan2(dy, dx)

        # Angular velocity (handle wraparound)
        dangle = np.diff(angles)
        dangle = np.where(dangle > np.pi, dangle - 2*np.pi, dangle)
        dangle = np.where(dangle < -np.pi, dangle + 2*np.pi, dangle)
        self.angular_velocities = np.zeros(len(angles))
        self.angular_velocities[:-1] = dangle / dt[:-1]


def parse_wcon(filepath: Union[str, Path]) -> List[WCONTrack]:
    """
    Parse a WCON (Worm Common Object Notation) file.

    WCON is a JSON-based format for worm tracking data.
    See: https://github.com/openworm/tracker-commons

    Args:
        filepath: Path to WCON file (.wcon or .json)

    Returns:
        List of WCONTrack objects
    """
    filepath = Path(filepath)

    with open(filepath, 'r') as f:
        data = json.load(f)

    tracks = []

    # WCON format stores data in 'data' array
    worm_data = data.get('data', [])

    for entry in worm_data:
        worm_id = str(entry.get('id', len(tracks)))

        # Time can be in 't' field
        t = np.array(entry.get('t', []))

        # Position in 'x' and 'y' (can be arrays of arrays for skeleton)
        x = entry.get('x', [])
        y = entry.get('y', [])

        # If skeleton data, take centroid (middle point)
        if isinstance(x[0], list):
            x = np.array([np.mean(pts) for pts in x])
            y = np.array([np.mean(pts) for pts in y])
        else:
            x = np.array(x)
            y = np.array(y)

        track = WCONTrack(
            worm_id=worm_id,
            timestamps=t,
            x_positions=x,
            y_positions=y
        )
        track.compute_kinematics()
        tracks.append(track)

    return tracks


# =============================================================================
# Tierpsy HDF5 Parser
# =============================================================================

@dataclass
class TierpsyFeatures:
    """Behavioral features extracted by Tierpsy Tracker."""

    # Summary statistics (726 features in full Tierpsy)
    feature_names: List[str] = field(default_factory=list)
    feature_values: np.ndarray = field(default_factory=lambda: np.array([]))

    # Time series (if available)
    timestamps: Optional[np.ndarray] = None
    velocities: Optional[np.ndarray] = None
    angular_velocities: Optional[np.ndarray] = None

    # Specific features we care about
    speed_mean: float = 0.0
    speed_std: float = 0.0
    angular_velocity_mean: float = 0.0
    path_curvature_mean: float = 0.0

    def to_observable_vector(self) -> ObservableVector:
        """Convert Tierpsy features to ObservableVector format."""
        obs = ObservableVector()

        # Map available features
        obs.speed_mean = self.speed_mean
        obs.speed_variance = self.speed_std ** 2
        obs.angular_velocity = self.angular_velocity_mean
        obs.turn_angle_mean = abs(self.angular_velocity_mean) * 180 / np.pi

        return obs


def parse_tierpsy_hdf5(filepath: Union[str, Path]) -> TierpsyFeatures:
    """
    Parse a Tierpsy Tracker HDF5 file.

    Tierpsy stores features in specific HDF5 groups:
    - /features_summary: Summary statistics
    - /timeseries_data: Frame-by-frame data
    - /trajectories_data: Worm tracks

    Args:
        filepath: Path to Tierpsy HDF5 file

    Returns:
        TierpsyFeatures object
    """
    if not HAS_H5PY:
        raise ImportError("h5py is required to parse Tierpsy files. "
                         "Install with: pip install h5py")

    filepath = Path(filepath)
    features = TierpsyFeatures()

    with h5py.File(filepath, 'r') as f:
        # Try to find feature summary
        if 'features_summary' in f:
            summary = f['features_summary']
            if 'feature_names' in summary.attrs:
                features.feature_names = list(summary.attrs['feature_names'])

            # Load feature values
            if 'values' in summary:
                features.feature_values = summary['values'][:]

        # Try to find timeseries data
        if 'timeseries_data' in f:
            ts = f['timeseries_data']

            if 'timestamp' in ts:
                features.timestamps = ts['timestamp'][:]

            if 'speed' in ts:
                speed_data = ts['speed'][:]
                features.velocities = speed_data
                features.speed_mean = float(np.nanmean(speed_data))
                features.speed_std = float(np.nanstd(speed_data))

            if 'angular_velocity' in ts:
                ang_vel = ts['angular_velocity'][:]
                features.angular_velocities = ang_vel
                features.angular_velocity_mean = float(np.nanmean(ang_vel))

        # Look for features in other common locations
        for path in ['features_stats', 'blob_features', 'features']:
            if path in f:
                grp = f[path]

                # Speed features
                for speed_key in ['speed_mean', 'speed_midbody_mean', 'velocity']:
                    if speed_key in grp:
                        features.speed_mean = float(grp[speed_key][()])
                        break

                # Angular velocity features
                for ang_key in ['angular_velocity_mean', 'ang_vel_mean']:
                    if ang_key in grp:
                        features.angular_velocity_mean = float(grp[ang_key][()])
                        break

    return features


# =============================================================================
# Real Behavioral Dataset
# =============================================================================

@dataclass
class RealBehaviorDataset:
    """
    Container for real C. elegans behavioral data.

    Can be loaded from WCON, Tierpsy HDF5, or CSV files.
    Provides methods to extract ObservableVectors for model comparison.
    """

    name: str
    source: str  # 'wcon', 'tierpsy', 'csv'

    # Raw data
    tracks: List[WCONTrack] = field(default_factory=list)
    tierpsy_features: Optional[TierpsyFeatures] = None

    # Aggregated statistics
    n_worms: int = 0
    total_duration_sec: float = 0.0

    # Precomputed observable statistics
    _mean_observable: Optional[ObservableVector] = None
    _std_observable: Optional[ObservableVector] = None

    @classmethod
    def from_wcon(cls, filepath: Union[str, Path],
                  name: Optional[str] = None) -> 'RealBehaviorDataset':
        """Load from WCON file."""
        filepath = Path(filepath)
        tracks = parse_wcon(filepath)

        dataset = cls(
            name=name or filepath.stem,
            source='wcon',
            tracks=tracks,
            n_worms=len(tracks),
        )

        if tracks:
            dataset.total_duration_sec = sum(
                t.timestamps[-1] - t.timestamps[0]
                for t in tracks if len(t.timestamps) > 0
            )

        dataset._compute_observable_statistics()
        return dataset

    @classmethod
    def from_tierpsy(cls, filepath: Union[str, Path],
                     name: Optional[str] = None) -> 'RealBehaviorDataset':
        """Load from Tierpsy HDF5 file."""
        filepath = Path(filepath)
        features = parse_tierpsy_hdf5(filepath)

        dataset = cls(
            name=name or filepath.stem,
            source='tierpsy',
            tierpsy_features=features,
            n_worms=1,  # Tierpsy typically summarizes across worms
        )

        if features.timestamps is not None and len(features.timestamps) > 0:
            dataset.total_duration_sec = float(
                features.timestamps[-1] - features.timestamps[0]
            )

        dataset._compute_observable_statistics()
        return dataset

    @classmethod
    def from_csv(cls, filepath: Union[str, Path],
                 name: Optional[str] = None) -> 'RealBehaviorDataset':
        """
        Load from CSV file with behavioral summary statistics.

        Expected columns: speed_mean, speed_std, reversal_rate, etc.
        """
        filepath = Path(filepath)

        # Read CSV manually (avoid pandas dependency)
        with open(filepath, 'r') as f:
            lines = f.readlines()

        if len(lines) < 2:
            raise ValueError(f"CSV file {filepath} has no data rows")

        header = [col.strip() for col in lines[0].split(',')]
        values = [float(v.strip()) for v in lines[1].split(',')]

        data = dict(zip(header, values))

        # Create observable from CSV data
        obs = ObservableVector(
            velocity=data.get('velocity', data.get('speed_mean', 0)),
            angular_velocity=data.get('angular_velocity', 0),
            reversal_rate=data.get('reversal_rate', 0),
            run_length=data.get('run_length', 0),
            speed_mean=data.get('speed_mean', 0),
            speed_variance=data.get('speed_variance', data.get('speed_std', 0)**2),
            turn_angle_mean=data.get('turn_angle_mean', 0),
            turn_angle_variance=data.get('turn_angle_variance', 0),
            omega_turn_rate=data.get('omega_turn_rate', 0),
            chemotaxis_index=data.get('chemotaxis_index', 0),
            anterior_touch_prob=data.get('anterior_touch_prob', 0),
            posterior_touch_prob=data.get('posterior_touch_prob', 0),
            response_latency=data.get('response_latency', 0),
            pharyngeal_pumping=data.get('pharyngeal_pumping', 0),
        )

        dataset = cls(
            name=name or filepath.stem,
            source='csv',
            n_worms=int(data.get('n_worms', 1)),
        )
        dataset._mean_observable = obs

        return dataset

    def _compute_observable_statistics(self) -> None:
        """Compute mean and std of observables across all data."""
        if self.source == 'wcon' and self.tracks:
            observables = []

            for track in self.tracks:
                obs = ObservableVector()

                if track.velocities is not None and len(track.velocities) > 0:
                    obs.velocity = float(np.nanmean(track.velocities))
                    obs.speed_mean = float(np.nanmean(np.abs(track.velocities)))
                    obs.speed_variance = float(np.nanvar(track.velocities))

                if track.angular_velocities is not None:
                    obs.angular_velocity = float(np.nanmean(
                        np.abs(track.angular_velocities)
                    ))

                observables.append(obs)

            if observables:
                # Compute mean
                arrays = np.array([o.to_array() for o in observables])
                mean_array = np.nanmean(arrays, axis=0)
                std_array = np.nanstd(arrays, axis=0)

                self._mean_observable = ObservableVector.from_array(mean_array)
                self._std_observable = ObservableVector.from_array(std_array)

        elif self.source == 'tierpsy' and self.tierpsy_features:
            self._mean_observable = self.tierpsy_features.to_observable_vector()

    def get_observable_vector(self) -> ObservableVector:
        """Get the mean observable vector for this dataset."""
        if self._mean_observable is None:
            self._compute_observable_statistics()

        return self._mean_observable or ObservableVector()

    def get_observable_std(self) -> ObservableVector:
        """Get the std of observable vector for this dataset."""
        return self._std_observable or ObservableVector()

    def summary(self) -> str:
        """Return a summary of the dataset."""
        obs = self.get_observable_vector()
        return (
            f"RealBehaviorDataset: {self.name}\n"
            f"  Source: {self.source}\n"
            f"  N worms: {self.n_worms}\n"
            f"  Duration: {self.total_duration_sec:.1f} sec\n"
            f"  Mean speed: {obs.speed_mean:.3f} μm/s\n"
            f"  Mean angular velocity: {obs.angular_velocity:.3f} rad/s\n"
        )


# =============================================================================
# Dataset Download Utilities
# =============================================================================

def download_dataset(dataset_name: str,
                     cache_dir: Union[str, Path] = 'data/behavior') -> Path:
    """
    Download a behavioral dataset from Zenodo.

    Args:
        dataset_name: Name of dataset (see ZENODO_DATASETS)
        cache_dir: Directory to cache downloaded files

    Returns:
        Path to downloaded file
    """
    if not HAS_REQUESTS:
        raise ImportError("requests is required for downloading. "
                         "Install with: pip install requests")

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    if dataset_name in ZENODO_DATASETS:
        record_id = ZENODO_DATASETS[dataset_name]
        # Zenodo API to get file list
        api_url = f"https://zenodo.org/api/records/{record_id}"

        response = requests.get(api_url)
        response.raise_for_status()
        record = response.json()

        # Download first file
        files = record.get('files', [])
        if not files:
            raise ValueError(f"No files found in Zenodo record {record_id}")

        file_info = files[0]
        download_url = file_info['links']['self']
        filename = file_info['key']

        local_path = cache_dir / filename

        if not local_path.exists():
            print(f"Downloading {filename}...")
            response = requests.get(download_url, stream=True)
            response.raise_for_status()

            with open(local_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

            print(f"Downloaded to {local_path}")

        return local_path

    elif dataset_name in SAMPLE_DATA_URLS:
        url = SAMPLE_DATA_URLS[dataset_name]
        filename = url.split('/')[-1]
        local_path = cache_dir / filename

        if not local_path.exists():
            print(f"Downloading {filename}...")
            response = requests.get(url, stream=True)
            response.raise_for_status()

            with open(local_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

            print(f"Downloaded to {local_path}")

        return local_path

    else:
        available = list(ZENODO_DATASETS.keys()) + list(SAMPLE_DATA_URLS.keys())
        raise ValueError(f"Unknown dataset: {dataset_name}. "
                        f"Available: {available}")


def list_available_datasets() -> Dict[str, str]:
    """List all available datasets for download."""
    datasets = {}

    for name, record_id in ZENODO_DATASETS.items():
        datasets[name] = f"Zenodo record {record_id}"

    for name, url in SAMPLE_DATA_URLS.items():
        datasets[name] = url

    return datasets


# =============================================================================
# Reference Behavioral Data
# =============================================================================

# Literature values for wild-type N2 behavior (from various sources)
N2_REFERENCE_OBSERVABLES = ObservableVector(
    velocity=0.2,              # ~200 μm/s forward crawling
    angular_velocity=0.3,       # ~0.3 rad/s typical
    reversal_rate=2.0,          # ~2 reversals per minute on food
    run_length=10.0,            # ~10 seconds between reversals
    speed_mean=0.2,             # 200 μm/s
    speed_variance=0.01,        # Some variability
    turn_angle_mean=30.0,       # ~30 degree turns typical
    turn_angle_variance=400.0,  # High variability
    omega_turn_rate=0.5,        # ~0.5 omega turns per minute
    chemotaxis_index=0.3,       # Moderate bias toward attractant
    anterior_touch_prob=0.8,    # 80% reversal on anterior touch
    posterior_touch_prob=0.7,   # 70% acceleration on posterior touch
    response_latency=200.0,     # ~200ms response time
    pharyngeal_pumping=200.0,   # ~200 pumps per minute on food
)


def get_reference_observables(strain: str = 'N2',
                              condition: str = 'on_food') -> ObservableVector:
    """
    Get reference observable values from literature.

    Args:
        strain: C. elegans strain ('N2', 'unc-31', etc.)
        condition: Experimental condition ('on_food', 'off_food', etc.)

    Returns:
        ObservableVector with reference values
    """
    # Currently only N2 on food is implemented
    if strain == 'N2' and condition == 'on_food':
        return N2_REFERENCE_OBSERVABLES

    # Return scaled version for other conditions
    obs = ObservableVector()
    obs.velocity = N2_REFERENCE_OBSERVABLES.velocity
    obs.speed_mean = N2_REFERENCE_OBSERVABLES.speed_mean

    if condition == 'off_food':
        # Off food: faster movement, more reversals
        obs.velocity *= 1.5
        obs.speed_mean *= 1.5
        obs.reversal_rate = 5.0  # More reversals
        obs.pharyngeal_pumping = 20.0  # Much less pumping

    return obs


# =============================================================================
# Main / Testing
# =============================================================================

if __name__ == "__main__":
    print("Behavioral Data Loader")
    print("=" * 50)

    print("\nAvailable datasets:")
    for name, desc in list_available_datasets().items():
        print(f"  {name}: {desc}")

    print("\nReference N2 observables:")
    print(N2_REFERENCE_OBSERVABLES)

    # Test CSV parsing
    print("\nTesting CSV parser...")
    test_csv = """speed_mean,speed_std,reversal_rate,angular_velocity
0.2,0.05,2.0,0.3"""

    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        f.write(test_csv)
        test_path = f.name

    dataset = RealBehaviorDataset.from_csv(test_path)
    print(dataset.summary())

    os.unlink(test_path)
