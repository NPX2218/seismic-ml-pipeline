"""
visualize.py: Visualize seismic feature extraction pipeline results.

Run this after main.py to generate diagnostic plots for validating
your feature extraction and labeling pipeline.

Usage:
    python visualize.py data/processed/STATION_DATE/

Outputs:
    Saves PNG files to the same directory as the input data.
"""

import sys
from pathlib import Path
import json
from typing import Any, Dict, List

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from numpy.typing import NDArray
from obspy import UTCDateTime

from matplotlib.lines import Line2D
from matplotlib.patches import Patch


def load_results(output_dir: Path) -> Dict[str, Any]:
    """Load all pipeline outputs from a processed data directory."""

    results: Dict[str, Any] = {}

    # Load features
    features_file = output_dir / "features.npy"
    if features_file.exists():
        results['features'] = np.load(features_file)
        print(f"Loaded features: {results['features'].shape}")
    else:
        raise FileNotFoundError(f"Features not found: {features_file}")

    # Load feature names
    names_file = output_dir / "feature_names.txt"
    if names_file.exists():
        with open(names_file, 'r', encoding="utf-8") as file:
            results['feature_names'] = [line.strip() for line in file]
        print(f"Loaded {len(results['feature_names'])} feature names")

    # Load timestamps
    timestamps_file = output_dir / "timestamps.npy"
    if timestamps_file.exists():
        results['timestamps'] = np.load(timestamps_file)
        print(f"Loaded timestamps: {results['timestamps'].shape}")

    # Load labels
    labels_file = output_dir / "labels.npy"
    if labels_file.exists():
        results['labels'] = np.load(labels_file)
        print(f"Loaded labels: {results['labels'].shape}")

    # Load config
    config_file = output_dir / "config.json"
    if config_file.exists():
        with open(config_file, 'r', encoding="utf-8") as file:
            results['config'] = json.load(file)
        print("Loaded config")

    # Load label config
    label_config_file = output_dir / "label_config.json"
    if label_config_file.exists():
        with open(label_config_file, 'r', encoding="utf-8") as file:
            results['label_config'] = json.load(file)
        print("Loaded label config")

    # Load catalog if available
    if 'label_config' in results and 'config' in results:
        catalog_file = Path(results['label_config']['catalog_file'])
        if catalog_file.exists():
            catalog = pd.read_csv(catalog_file)

            # Compute hours_from_start if not present
            if 'hours_from_start' not in catalog.columns and 'data_start_time' in results['config']:
                data_start = UTCDateTime(results['config']['data_start_time'])

                hours: List[float] = []
                for eq_time in catalog['time']:
                    eq_utc = UTCDateTime(eq_time)
                    hours.append((eq_utc - data_start) / 3600)
                catalog['hours_from_start'] = hours

            # Filter by min magnitude
            min_mag = results['label_config'].get('min_magnitude', 0)
            catalog = catalog[catalog['mag'] >= min_mag]

            results['catalog'] = catalog
            print(
                f"Loaded catalog: {len(results['catalog'])} earthquakes (M >= {min_mag})")

    return results


def plot_feature_distributions(
    features: NDArray[np.floating[Any]],
    feature_names: List[str],
    output_dir: Path
) -> None:
    """Plot histogram of each feature to check for reasonable distributions."""

    n_features = features.shape[1]
    n_cols = 3
    n_rows = (n_features + n_cols - 1) // n_cols

    _, axes = plt.subplots(n_rows, n_cols, figsize=(12, 3 * n_rows))
    axes_flat = axes.flatten()

    for i in range(n_features):
        ax = axes_flat[i]
        data = features[:, i]

        ax.hist(data, bins=40, color='steelblue', edgecolor='white', alpha=0.8)
        ax.set_title(feature_names[i], fontsize=10)
        ax.set_xlabel('Value')
        ax.set_ylabel('Count')

        # Add stats
        stats_text = f'u={np.mean(data):.2f}\no={np.std(data):.2f}'
        ax.text(0.95, 0.95, stats_text, transform=ax.transAxes,
                fontsize=8, verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Hide unused subplots
    for i in range(n_features, len(axes_flat)):
        axes_flat[i].set_visible(False)

    plt.suptitle('Feature Distributions', fontsize=14, fontweight='bold')
    plt.tight_layout()

    save_path = output_dir / "viz_feature_distributions.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_feature_timeseries(
    features: NDArray[np.floating[Any]],
    feature_names: List[str],
    timestamps: NDArray[np.floating[Any]],
    labels: NDArray[np.integer[Any]] | None,
    catalog: pd.DataFrame | None,
    config: Dict[str, Any],
    output_dir: Path
) -> None:
    """Plot each feature over time with earthquake markers."""

    n_features = features.shape[1]

    _, axes = plt.subplots(n_features, 1, figsize=(
        14, 2.5 * n_features), sharex=True)

    # Handle single feature case
    if n_features == 1:
        axes_list: List[Any] = [axes]
    else:
        axes_list = np.asarray(axes).flatten().tolist()

    # X-axis: window center times in hours
    window_centers = (timestamps[:, 0] + timestamps[:, 1]) / 2

    for i, ax in enumerate(axes_list):
        # Plot feature values
        if labels is not None:
            colors = ['steelblue' if label ==
                      0 else 'orangered' for label in labels]
            ax.scatter(window_centers,
                       features[:, i], c=colors, s=15, alpha=0.6)
        else:
            ax.plot(window_centers, features[:, i],
                    color='steelblue', linewidth=0.8)

        ax.set_ylabel(feature_names[i], fontsize=9)
        ax.grid(True, alpha=0.3)

        # Add earthquake markers
        if catalog is not None and 'hours_from_start' in catalog.columns:
            duration = config.get('duration_hours', window_centers.max())
            eq_in_range = catalog[
                (catalog['hours_from_start'] >= 0) &
                (catalog['hours_from_start'] <= duration)
            ]
            for _, eq in eq_in_range.iterrows():
                ax.axvline(eq['hours_from_start'], color='red', alpha=0.4,
                           linewidth=1, linestyle='--')

    axes_list[-1].set_xlabel('Time (hours from recording start)')

    # Legend
    if labels is not None:
        legend_elements = [
            Patch(facecolor='steelblue', label='No EQ in lookahead'),
            Patch(facecolor='orangered', label='EQ in lookahead'),
            Line2D([0], [0], color='red', linestyle='--',
                   alpha=0.5, label='Earthquake')
        ]
        axes_list[0].legend(handles=legend_elements,
                            loc='upper right', fontsize=8)

    plt.suptitle('Features Over Time', fontsize=14, fontweight='bold')
    plt.tight_layout()

    save_path = output_dir / "viz_feature_timeseries.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_label_timeline(
    labels: NDArray[np.integer[Any]],
    timestamps: NDArray[np.floating[Any]],
    catalog: pd.DataFrame | None,
    config: Dict[str, Any],
    label_config: Dict[str, Any],
    output_dir: Path
) -> None:
    """Visualize label assignments and earthquake occurrences."""

    _, axes = plt.subplots(2, 1, figsize=(14, 6), sharex=True)

    window_centers = (timestamps[:, 0] + timestamps[:, 1]) / 2
    duration = config.get('duration_hours', window_centers.max())

    # Top: Label timeline
    ax1 = axes[0]
    colors = ['steelblue' if label == 0 else 'orangered' for label in labels]
    ax1.bar(window_centers, labels, width=config.get('step_size_hours', 0.5) * 0.8,
            color=colors, alpha=0.7)
    ax1.set_ylabel('Label')
    ax1.set_yticks([0, 1])
    ax1.set_yticklabels(['No EQ', 'EQ coming'])
    ax1.set_title(
        f"Labels (lookahead = {label_config.get('lookahead_hours', '?')} hours)")
    ax1.grid(True, alpha=0.3, axis='x')

    # Bottom: Earthquake magnitudes over time
    ax2 = axes[1]
    if catalog is not None and 'hours_from_start' in catalog.columns:
        eq_in_range = catalog[
            (catalog['hours_from_start'] >= 0) &
            (catalog['hours_from_start'] <= duration)
        ]

        if len(eq_in_range) > 0:
            ax2.stem(eq_in_range['hours_from_start'], eq_in_range['mag'],
                     linefmt='red', markerfmt='ro', basefmt=' ')

            # Annotate larger earthquakes
            for _, eq in eq_in_range.iterrows():
                if eq['mag'] >= 4.0:
                    ax2.annotate(f"M{eq['mag']:.1f}",
                                 (eq['hours_from_start'], eq['mag']),
                                 textcoords="offset points", xytext=(0, 10),
                                 ha='center', fontsize=8)

    ax2.set_xlabel('Time (hours from recording start)')
    ax2.set_ylabel('Magnitude')
    ax2.set_title('Earthquake Occurrences')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, duration)

    plt.tight_layout()

    save_path = output_dir / "viz_label_timeline.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_features_by_label(
    features: NDArray[np.floating[Any]],
    feature_names: List[str],
    labels: NDArray[np.integer[Any]],
    output_dir: Path
) -> None:
    """Compare feature distributions between positive and negative labels."""

    n_features = features.shape[1]
    n_cols = 3
    n_rows = (n_features + n_cols - 1) // n_cols

    _, axes = plt.subplots(n_rows, n_cols, figsize=(12, 3 * n_rows))
    axes_flat = axes.flatten()

    pos_mask = labels == 1
    neg_mask = labels == 0

    for i in range(n_features):
        ax = axes_flat[i]

        # Plot both distributions
        bins = np.linspace(features[:, i].min(), features[:, i].max(), 30)
        ax.hist(features[neg_mask, i], bins=bins, alpha=0.6,
                label=f'No EQ (n={neg_mask.sum()})', color='steelblue')
        ax.hist(features[pos_mask, i], bins=bins, alpha=0.6,
                label=f'EQ coming (n={pos_mask.sum()})', color='orangered')

        ax.set_title(feature_names[i], fontsize=10)
        ax.set_xlabel('Value')
        ax.set_ylabel('Count')
        ax.legend(fontsize=7)

    # Hide unused subplots
    for i in range(n_features, len(axes_flat)):
        axes_flat[i].set_visible(False)

    plt.suptitle('Feature Distributions by Label',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()

    save_path = output_dir / "viz_features_by_label.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_correlation_matrix(
    features: NDArray[np.floating[Any]],
    feature_names: List[str],
    output_dir: Path
) -> None:
    """Plot correlation matrix between features."""

    corr = np.corrcoef(features.T)

    _, ax = plt.subplots(figsize=(10, 8))

    im = ax.imshow(corr, cmap='RdBu_r', vmin=-1, vmax=1)

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Correlation')

    # Labels
    ax.set_xticks(range(len(feature_names)))
    ax.set_yticks(range(len(feature_names)))
    ax.set_xticklabels(feature_names, rotation=45, ha='right', fontsize=8)
    ax.set_yticklabels(feature_names, fontsize=8)

    # Add correlation values
    for i in range(len(feature_names)):
        for j in range(len(feature_names)):
            color = 'white' if abs(corr[i, j]) > 0.5 else 'black'
            ax.text(j, i, f'{corr[i, j]:.2f}', ha='center', va='center',
                    color=color, fontsize=7)

    plt.title('Feature Correlation Matrix', fontsize=14, fontweight='bold')
    plt.tight_layout()

    save_path = output_dir / "viz_correlation_matrix.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_summary_dashboard(
    features: NDArray[np.floating[Any]],
    labels: NDArray[np.integer[Any]],
    config: Dict[str, Any],
    label_config: Dict[str, Any],
    output_dir: Path
) -> None:
    """Create a single summary dashboard with key stats."""

    fig = plt.figure(figsize=(14, 8))

    # Layout: 2x3 grid
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

    # 1. Label distribution pie chart
    ax1 = fig.add_subplot(gs[0, 0])
    pos_count = (labels == 1).sum()
    neg_count = (labels == 0).sum()
    ax1.pie([neg_count, pos_count], labels=['No EQ', 'EQ coming'],
            colors=['steelblue', 'orangered'], autopct='%1.1f%%',
            explode=[0, 0.05])
    ax1.set_title('Label Distribution')

    # 2. Feature means comparison
    ax2 = fig.add_subplot(gs[0, 1])
    pos_means = features[labels == 1].mean(axis=0)
    neg_means = features[labels == 0].mean(axis=0)
    x = np.arange(features.shape[1])
    width = 0.35
    ax2.bar(x - width/2, neg_means, width,
            label='No EQ', color='steelblue', alpha=0.7)
    ax2.bar(x + width/2, pos_means, width,
            label='EQ coming', color='orangered', alpha=0.7)
    ax2.set_xlabel('Feature Index')
    ax2.set_ylabel('Mean Value')
    ax2.set_title('Feature Means by Label')
    ax2.legend()
    ax2.set_xticks(x)

    # 3. Config summary
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.axis('off')
    config_text = (
        "Pipeline Configuration\n"
        "\n"
        f"Station: {config.get('station', 'N/A')}\n"
        f"Duration: {config.get('duration_hours', 0):.1f} hours\n"
        f"Sampling Rate: {config.get('sampling_rate', 0)} Hz\n"
        f"Window Size: {config.get('window_duration_hours', 0)} hours\n"
        f"Step Size: {config.get('step_size_hours', 0)} hours\n"
        f"N Windows: {config.get('n_windows', 0)}\n"
        f"N Features: {config.get('n_features', 0)}\n"
        "\n"
        "Labeling:\n"
        f"Lookahead: {label_config.get('lookahead_hours', 0)} hours\n"
        f"Min Magnitude: {label_config.get('min_magnitude', 0)}\n"
        f"EQs in Range: {label_config.get('earthquakes_in_range', 0)}"
    )
    ax3.text(0.1, 0.9, config_text, transform=ax3.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # 4-6. First 3 features over window index
    for i in range(min(3, features.shape[1])):
        ax = fig.add_subplot(gs[1, i])
        colors = ['steelblue' if label ==
                  0 else 'orangered' for label in labels]
        ax.scatter(range(len(labels)),
                   features[:, i], c=colors, s=10, alpha=0.5)
        ax.set_xlabel('Window Index')
        ax.set_ylabel(f'Feature {i}')
        ax.set_title(f'Feature {i} Over Time')
        ax.grid(True, alpha=0.3)

    plt.suptitle(f"Pipeline Summary: {config.get('station', 'Unknown')}",
                 fontsize=16, fontweight='bold')

    save_path = output_dir / "viz_summary_dashboard.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def main(output_dir: str) -> None:
    """Run all visualizations for a processed dataset."""

    output_path = Path(output_dir)

    if not output_path.exists():
        print(f"ERROR: Directory not found: {output_path}")
        sys.exit(1)

    print("=" * 60)
    print("Loading pipeline results")
    print("=" * 60)

    try:
        results = load_results(output_path)
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    features = results['features']
    feature_names = results.get(
        'feature_names', [f'Feature_{i}' for i in range(features.shape[1])])
    timestamps = results.get('timestamps')
    labels = results.get('labels')
    catalog = results.get('catalog')
    config = results.get('config', {})
    label_config = results.get('label_config', {})

    print()
    print("=" * 60)
    print("Generating visualizations")
    print("=" * 60)

    # 1. Feature distributions
    print("\n[1/6] Feature distributions...")
    plot_feature_distributions(features, feature_names, output_path)

    # 2. Feature timeseries
    if timestamps is not None:
        print("[2/6] Feature timeseries...")
        plot_feature_timeseries(features, feature_names, timestamps,
                                labels, catalog, config, output_path)
    else:
        print("[2/6] Skipping timeseries (no timestamps)")

    # 3. Label timeline
    if labels is not None and timestamps is not None:
        print("[3/6] Label timeline...")
        plot_label_timeline(labels, timestamps, catalog, config,
                            label_config, output_path)
    else:
        print("[3/6] Skipping label timeline (no labels/timestamps)")

    # 4. Features by label
    if labels is not None:
        print("[4/6] Features by label...")
        plot_features_by_label(features, feature_names, labels, output_path)
    else:
        print("[4/6] Skipping features by label (no labels)")

    # 5. Correlation matrix
    print("[5/6] Correlation matrix...")
    plot_correlation_matrix(features, feature_names, output_path)

    # 6. Summary dashboard
    if labels is not None:
        print("[6/6] Summary dashboard...")
        plot_summary_dashboard(features, labels, config,
                               label_config, output_path)
    else:
        print("[6/6] Skipping dashboard (no labels)")

    print()
    print("=" * 60)
    print("Visualization complete!")
    print("=" * 60)
    print(f"\nAll plots saved to: {output_path}/")
    print("\nGenerated files:")
    for viz_file in sorted(output_path.glob("viz_*.png")):
        print(f"  {viz_file.name}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python visualize.py <output_directory>")
        print("Example: python visualize.py data/processed/...s/")
        sys.exit(1)

    main(sys.argv[1])
