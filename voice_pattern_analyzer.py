"""
Voice Pattern Analysis Interface
Visual tool for discovering voice archetypes from openSMILE data
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import tkinter as tk
from tkinter import ttk, filedialog
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import warnings
import re # Import re for regex in formatting
warnings.filterwarnings('ignore')

class VoicePatternAnalyzer:
    """
    Interactive tool for analyzing voice patterns and finding archetypes
    """

    # NEW: OpenSMILE Feature Name Translation Dictionary
    opensmile_translations = {
        # Energy & Loudness Features
        'audspec_lengthL1norm_sma_mean': 'Average Energy Level',
        'audspec_lengthL1norm_sma_stddev': 'Energy Variation',
        'audspecRasta_lengthL1norm_sma_mean': 'Filtered Energy Level',
        'audspecRasta_lengthL1norm_sma_stddev': 'Filtered Energy Variation',
        'pcm_loudness_sma_mean': 'Average Loudness',
        'pcm_loudness_sma_stddev': 'Loudness Variation',
        'pcm_loudness_sma_range': 'Loudness Range',
        'pcm_loudness_sma_maxPos': 'Peak Loudness Position',
        'pcm_loudness_sma_minPos': 'Quiet Moment Position',
        'pcm_loudness_sma_amean': 'Absolute Loudness',

        # Pitch & F0 Features
        'F0final_sma_mean': 'Average Pitch',
        'F0final_sma_stddev': 'Pitch Variation',
        'F0final_sma_range': 'Pitch Range',
        'F0final_sma_maxPos': 'Highest Pitch Position',
        'F0final_sma_minPos': 'Lowest Pitch Position',
        'F0final_sma_amean': 'Absolute Pitch',
        'logRelF0-F0final_sma_mean': 'Relative Pitch Level',
        'logRelF0-F0final_sma_stddev': 'Pitch Energy Variation',
        'logRelF0-F0final_sma_range': 'Relative Pitch Range',
        'logRelF0-F0final_sma_maxPos': 'Peak Pitch Position',
        'logRelF0-F0final_sma_minPos': 'Low Pitch Position',
        'logRelF0-F0final_sma_amean': 'Pitch Energy Level',

        # Spectral Features (Voice Quality & Timbre)
        'pcm_zcr_sma_mean': 'Voice Crispness',
        'pcm_zcr_sma_stddev': 'Crispness Variation',
        'audspec_centroid_sma_mean': 'Brightness Center',
        'audspec_centroid_sma_stddev': 'Brightness Variation',
        'audspec_flux_sma_mean': 'Voice Change Rate',
        'audspec_flux_sma_stddev': 'Change Rate Variation',
        'audspec_rolloff25.0_sma_mean': 'Low Frequency Focus',
        'audspec_rolloff25.0_sma_stddev': 'Low Frequency Variation',
        'audspec_rolloff50.0_sma_mean': 'Mid Frequency Balance',
        'audspec_rolloff50.0_sma_stddev': 'Mid Frequency Variation',
        'audspec_rolloff75.0_sma_mean': 'High Frequency Presence',
        'audspec_rolloff75.0_sma_stddev': 'High Frequency Variation',
        'audspec_rolloff90.0_sma_mean': 'Extreme High Frequency',
        'audspec_rolloff90.0_sma_stddev': 'Extreme High Variation',
        'audspecRasta_centroid_sma_mean': 'Filtered Brightness',
        'audspecRasta_centroid_sma_stddev': 'Filtered Brightness Variation',
        'audspecRasta_flux_sma_mean': 'Filtered Change Rate',
        'audspecRasta_flux_sma_stddev': 'Filtered Change Variation',
        'audspecRasta_rolloff25.0_sma_mean': 'Filtered Low Frequency',
        'audspecRasta_rolloff25.0_sma_stddev': 'Filtered Low Variation',
        'audspecRasta_rolloff50.0_sma_mean': 'Filtered Mid Frequency',
        'audspecRasta_rolloff50.0_sma_stddev': 'Filtered Mid Variation',
        'audspecRasta_rolloff75.0_sma_mean': 'Filtered High Frequency',
        'audspecRasta_rolloff75.0_sma_stddev': 'Filtered High Variation',
        'audspecRasta_rolloff90.0_sma_mean': 'Filtered Extreme High',
        'audspecRasta_rolloff90.0_sma_stddev': 'Filtered Extreme Variation',

        # Voice Quality Features (Stability)
        'jitterLocal_sma_mean': 'Pitch Steadiness',
        'jitterDDP_sma_mean': 'Pitch Smoothness',
        'shimmerLocal_sma_mean': 'Volume Steadiness',
        'logHNR_sma_mean': 'Voice Clarity',
        'logHNR_sma_stddev': 'Clarity Variation',

        # Temporal Features (Rhythm & Timing)
        'voicingFinalUnclipped_sma_mean': 'Speech Percentage',
        'voicingFinalUnclipped_sma_stddev': 'Speech Consistency',

        # MFCC Features (Detailed Voice Characteristics)
        'mfcc1_sma_mean': 'Vocal Tract Shape 1',
        'mfcc1_sma_stddev': 'Vocal Shape Variation 1',
        'mfcc2_sma_mean': 'Vocal Tract Shape 2',
        'mfcc2_sma_stddev': 'Vocal Shape Variation 2',
        'mfcc3_sma_mean': 'Vocal Tract Shape 3',
        'mfcc3_sma_stddev': 'Vocal Shape Variation 3',
        'mfcc4_sma_mean': 'Vocal Tract Shape 4',
        'mfcc4_sma_stddev': 'Vocal Shape Variation 4',

        # Additional common features that might appear
        'F0final_sma_linregc1': 'Pitch Trend',
        'F0final_sma_linregc2': 'Pitch Curvature',
        'F0final_sma_linregerrA': 'Pitch Prediction Error',
        'F0final_sma_linregerrQ': 'Pitch Variability',
        'F0final_sma_stddevNorm': 'Normalized Pitch Variation',
        'F0final_sma_percentile20.0': 'Low Pitch Boundary',
        'F0final_sma_percentile50.0': 'Median Pitch',
        'F0final_sma_percentile80.0': 'High Pitch Boundary',
        'F0final_sma_pctlrange0-2': 'Pitch Stability Range',
        'F0final_sma_meanRisingSlope': 'Pitch Rise Rate',
        'F0final_sma_stddevRisingSlope': 'Pitch Rise Variation',
        'F0final_sma_meanFallingSlope': 'Pitch Fall Rate',
        'F0final_sma_stddevFallingSlope': 'Pitch Fall Variation',

        # Extended energy features
        'pcm_loudness_sma_linregc1': 'Loudness Trend',
        'pcm_loudness_sma_linregc2': 'Loudness Curvature',
        'pcm_loudness_sma_linregerrA': 'Loudness Prediction Error',
        'pcm_loudness_sma_linregerrQ': 'Loudness Variability',
        'pcm_loudness_sma_stddevNorm': 'Normalized Loudness Variation',
        'pcm_loudness_sma_percentile20.0': 'Low Volume Boundary',
        'pcm_loudness_sma_percentile50.0': 'Median Volume',
        'pcm_loudness_sma_percentile80.0': 'High Volume Boundary',
        'pcm_loudness_sma_pctlrange0-2': 'Volume Stability Range',
        'pcm_loudness_sma_meanRisingSlope': 'Volume Rise Rate',
        'pcm_loudness_sma_stddevRisingSlope': 'Volume Rise Variation',
        'pcm_loudness_sma_meanFallingSlope': 'Volume Fall Rate',
        'pcm_loudness_sma_stddevFallingSlope': 'Volume Fall Variation',

        # Additional spectral features
        'pcm_zcr_sma_range': 'Voice Crispness Range',
        'pcm_zcr_sma_maxPos': 'Peak Crispness Position',
        'pcm_zcr_sma_minPos': 'Smooth Voice Position',
        'pcm_zcr_sma_amean': 'Absolute Voice Crispness',

        # Extended HNR features
        'logHNR_sma_range': 'Voice Clarity Range',
        'logHNR_sma_maxPos': 'Peak Clarity Position',
        'logHNR_sma_minPos': 'Low Clarity Position',
        'logHNR_sma_amean': 'Absolute Voice Clarity'
    }


    def __init__(self, platform_root="C:/Attune-Voice-Platform"):
        self.platform_root = Path(platform_root)
        self.output_dir = self.platform_root / "output"
        self.voice_data = {}  # Store all voice fingerprints
        self.current_comparison = None

        # Load existing analysis data
        self.load_all_voice_data()

        # Setup GUI
        self.setup_gui()

    def load_all_voice_data(self):
        """Load all existing voice analysis data"""
        print("🔍 Loading voice analysis data...")

        analysis_dirs = [d for d in self.output_dir.iterdir() if d.is_dir()]

        for analysis_dir in analysis_dirs:
            # Look for fixed features CSV file first, then original
            fixed_feature_files = list(analysis_dir.glob("*_features_fixed.csv"))
            original_feature_files = list(analysis_dir.glob("*_features.csv"))
            report_files = list(analysis_dir.glob("report.json"))

            feature_file = None
            if fixed_feature_files:
                feature_file = fixed_feature_files[0]
                print(f"🔧 Using fixed features: {feature_file.name}")
            elif original_feature_files:
                feature_file = original_feature_files[0]
                print(f"📄 Using original features: {feature_file.name}")

            if feature_file and report_files:
                try:
                    # Load features - try as regular CSV first
                    try:
                        features_df = pd.read_csv(feature_file)
                        print(f"✅ Loaded CSV: {features_df.shape}")
                    except pd.errors.ParserError:
                        # Fall back to semicolon separator
                        features_df = pd.read_csv(feature_file, sep=';')
                        print(f"✅ Loaded CSV (semicolon): {features_df.shape}")
                    except Exception as csv_e:
                        print(f"⚠️ Failed to load CSV features from {feature_file.name}: {csv_e}")
                        continue

                    # Load metadata
                    with open(report_files[0], 'r') as f:
                        metadata = json.load(f)

                    creator_name = metadata.get('creator', analysis_dir.name)

                    # Store the voice data
                    self.voice_data[creator_name] = {
                        'features': features_df,
                        'metadata': metadata,
                        'analysis_dir': analysis_dir
                    }

                    print(f"✅ Loaded: {creator_name} ({features_df.shape[1]} features)")

                except Exception as e:
                    print(f"⚠️ Failed to load {analysis_dir.name}: {e}")

        print(f"📊 Total voice profiles loaded: {len(self.voice_data)}")

    def setup_gui(self):
        """Setup the main GUI interface"""
        self.root = tk.Tk()
        self.root.title("Attune Voice Pattern Analyzer")
        self.root.geometry("1400x900")

        # Create main frame
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Left panel - Controls
        control_frame = ttk.LabelFrame(main_frame, text="Analysis Controls", padding=10)
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))

        # Voice selection
        ttk.Label(control_frame, text="Select Voices to Compare:").pack(anchor=tk.W)

        self.voice_listbox = tk.Listbox(control_frame, selectmode=tk.MULTIPLE, height=10)
        for creator in self.voice_data.keys():
            self.voice_listbox.insert(tk.END, creator)
        self.voice_listbox.pack(fill=tk.X, pady=(5, 10))

        # Analysis type selection
        ttk.Label(control_frame, text="Analysis Type:").pack(anchor=tk.W)
        self.analysis_type = tk.StringVar(value="overview")

        analysis_options = [
            ("Voice Overview", "overview"),
            ("Feature Comparison", "comparison"),
            ("Archetype Clustering", "clustering"),
            ("Correlation Heatmap", "correlation"),
            ("Feature Distribution", "distribution")
        ]

        for text, value in analysis_options:
            ttk.Radiobutton(control_frame, text=text, variable=self.analysis_type,
                           value=value).pack(anchor=tk.W)

        # Feature category selection
        ttk.Label(control_frame, text="Feature Category:").pack(anchor=tk.W, pady=(10, 0))
        self.feature_category = tk.StringVar(value="all")

        categories = [
            ("All Features", "all"),
            ("Energy/Loudness", "energy"),
            ("Pitch/F0", "pitch"),
            ("Spectral", "spectral"),
            ("Voice Quality", "quality"),
            ("Temporal", "temporal")
        ]

        for text, value in categories:
            ttk.Radiobutton(control_frame, text=text, variable=self.feature_category,
                           value=value).pack(anchor=tk.W)

        # Action buttons
        ttk.Button(control_frame, text="Generate Analysis",
                  command=self.generate_analysis).pack(fill=tk.X, pady=(20, 5))

        ttk.Button(control_frame, text="Export Data",
                  command=self.export_data).pack(fill=tk.X, pady=5)

        ttk.Button(control_frame, text="Refresh Data",
                  command=self.refresh_data).pack(fill=tk.X, pady=5)

        # Right panel - Visualization
        viz_frame = ttk.LabelFrame(main_frame, text="Voice Pattern Visualization", padding=10)
        viz_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # Create matplotlib figure
        self.fig = Figure(figsize=(12, 8))
        self.canvas = FigureCanvasTkAgg(self.fig, viz_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Status bar
        self.status_text = tk.StringVar(value=f"Ready. {len(self.voice_data)} voice profiles loaded.")
        status_bar = ttk.Label(self.root, textvariable=self.status_text, relief=tk.SUNKEN)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)

        # Generate initial overview
        self.generate_analysis()

    def get_selected_voices(self):
        """Get currently selected voice profiles"""
        selected_indices = self.voice_listbox.curselection()
        if not selected_indices:
            return list(self.voice_data.keys())  # Return all if none selected

        return [list(self.voice_data.keys())[i] for i in selected_indices]

    def filter_features_by_category(self, features_df, category):
        """Filter features based on category"""
        if category == "all":
            return features_df

        feature_patterns = {
            "energy": ["loudness", "energy", "rms", "intensity"],
            "pitch": ["F0", "pitch", "fundamental"],
            "spectral": ["spectral", "mfcc", "centroid", "rolloff", "flux"],
            "quality": ["jitter", "shimmer", "hnr", "harmonic"],
            "temporal": ["rate", "duration", "pause", "rhythm"]
        }

        if category not in feature_patterns:
            return features_df

        patterns = feature_patterns[category]
        relevant_cols = []

        for col in features_df.columns:
            if any(pattern.lower() in col.lower() for pattern in patterns):
                relevant_cols.append(col)

        if relevant_cols:
            return features_df[relevant_cols]
        else:
            return features_df

    def _format_feature_label(self, feature_name):
        """
        Formats raw openSMILE feature names into more readable labels.
        Prioritizes direct translations from opensmile_translations.
        """
        # 1. Try direct translation first
        if feature_name in self.opensmile_translations:
            return self.opensmile_translations[feature_name]

        # 2. Fallback to intelligent shortening if no direct translation
        label = feature_name.lower()

        label = label.replace('_sma3nz', '')
        label = label.replace('_sma3', '')
        label = label.replace('_amean', '')
        label = label.replace('_mean', '(M)')
        label = label.replace('_stddev', '(SD)')
        label = label.replace('_range', '(R)')
        label = label.replace('min', '(Min)')
        label = label.replace('max', '(Max)')

        label = label.replace('spectral', 'Spec')
        label = label.replace('loudness', 'Loud')
        label = label.replace('mfcc', 'MFCC')
        label = label.replace('jitter', 'Jit')
        label = label.replace('shimmer', 'Shim')
        label = label.replace('hnr', 'HNR')
        label = label.replace('f0', 'F0')
        label = label.replace('centroid', 'Cent')
        label = label.replace('flux', 'Flux')
        label = label.replace('duration', 'Dur')
        label = label.replace('harmonic', 'Harm')
        label = label.replace('vocal', 'Voc')
        label = label.replace('quality', 'Qual')
        label = label.replace('energy', 'Eng')

        label = re.sub(r'mfcc_(\d+)', r'MFCC\1', label)
        label = re.sub(r'f0_(\d+)', r'F0\1', label)

        label_parts = re.split(r'[-_]', label)
        label = "".join([part.capitalize() if len(part) > 1 else part for part in label_parts])

        label = re.sub(r'(?<!^)(?=[A-Z])', ' ', label)

        label = label.replace(' ', '').replace('(', ' (').replace(')', ')')
        if len(label) > 20:
            label = label[:17] + "..."
        return label.strip()

    def generate_analysis(self):
        """Generate the requested analysis visualization"""
        if not self.voice_data:
            self.status_text.set("No voice data available. Run some analyses first.")
            return

        self.fig.clear()
        analysis_type = self.analysis_type.get()
        selected_voices = self.get_selected_voices()

        self.status_text.set(f"Generating {analysis_type} analysis for {len(selected_voices)} voices...")

        try:
            if analysis_type == "overview":
                self.create_overview_plot(selected_voices)
            elif analysis_type == "comparison":
                self.create_comparison_plot(selected_voices)
            elif analysis_type == "clustering":
                self.create_clustering_plot(selected_voices)
            elif analysis_type == "correlation":
                self.create_correlation_plot(selected_voices)
            elif analysis_type == "distribution":
                self.create_distribution_plot(selected_voices)

            self.canvas.draw()
            self.status_text.set(f"Analysis complete. Showing {len(selected_voices)} voice profiles.")

        except Exception as e:
            self.status_text.set(f"Error generating analysis: {str(e)}")
            print(f"Analysis error: {e}")

    def create_overview_plot(self, selected_voices):
        """Create overview visualization of selected voices"""
        if len(selected_voices) == 0:
            return

        print(f"\n🔍 DEBUG: Analyzing {len(selected_voices)} voices")

        key_features = []
        for voice_name in selected_voices:
            features_df = self.voice_data[voice_name]['features']
            print(f"   {voice_name}: {features_df.shape} shape")

            filtered_features = self.filter_features_by_category(features_df, self.feature_category.get())
            print(f"   {voice_name}: {filtered_features.shape} after filtering")

            if not filtered_features.empty:
                feature_row = filtered_features.iloc[0]
                print(f"   {voice_name}: {len(feature_row)} total features before numeric conversion")

                numeric_features = pd.to_numeric(feature_row, errors='coerce').dropna()
                print(f"   {voice_name}: {len(numeric_features)} numeric features after conversion and dropna")

                if len(numeric_features) > 0:
                    print(f"   {voice_name}: Sample values: {list(numeric_features.head(3).values)}")
                    key_features.append({
                        'voice': voice_name,
                        'data': numeric_features
                    })
                else:
                    print(f"   {voice_name}: ❌ No numeric features found in filtered data")
            else:
                print(f"   {voice_name}: ❌ Filtered features DataFrame is empty")

        if not key_features:
            ax = self.fig.add_subplot(1, 1, 1)
            ax.text(0.5, 0.5, 'No numeric features found\nCheck console for debug info',
                   ha='center', va='center', transform=ax.transAxes, fontsize=12)
            ax.set_title("Data Debug - No Numeric Features")
            return

        n_voices = len(key_features)
        cols = min(3, n_voices)
        rows = (n_voices + cols - 1) // cols

        for i, voice_data in enumerate(key_features):
            ax = self.fig.add_subplot(rows, cols, i + 1)

            feature_values = voice_data['data']

            if len(feature_values) == 0:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f"{voice_data['voice']} - No Data")
                continue

            n_features = min(10, len(feature_values))
            top_features_series = feature_values.abs().nlargest(n_features)

            if len(top_features_series) > 0:
                formatted_labels = [self._format_feature_label(f) for f in top_features_series.index]
                ax.barh(range(len(top_features_series)), top_features_series.values)
                ax.set_yticks(range(len(top_features_series)))
                ax.set_yticklabels(formatted_labels, fontsize=8)
                ax.set_title(f"{voice_data['voice']}", fontsize=10, fontweight='bold')
                ax.grid(True, alpha=0.3)
            else:
                ax.text(0.5, 0.5, 'No features', ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f"{voice_data['voice']} - No Features")

        self.fig.suptitle("Voice Profile Overview - Top Features", fontsize=14, fontweight='bold')
        self.fig.tight_layout()

    def create_comparison_plot(self, selected_voices):
        """Create side-by-side comparison of voice features"""
        if len(selected_voices) < 2:
            self.status_text.set("Select at least 2 voices for comparison")
            return

        comparison_data = {}
        common_features = None

        for voice_name in selected_voices:
            features_df = self.voice_data[voice_name]['features']
            filtered_features = self.filter_features_by_category(features_df, self.feature_category.get())

            if not filtered_features.empty:
                feature_row = filtered_features.iloc[0]
                numeric_features = pd.to_numeric(feature_row, errors='coerce').dropna()

                if len(numeric_features) > 0:
                    comparison_data[voice_name] = numeric_features

                    if common_features is None:
                        common_features = set(numeric_features.index)
                    else:
                        common_features = common_features.intersection(set(numeric_features.index))

        if not comparison_data or not common_features:
            ax = self.fig.add_subplot(1, 1, 1)
            ax.text(0.5, 0.5, 'No common numeric features found', ha='center', va='center', transform=ax.transAxes)
            return

        common_features = list(common_features)
        formatted_common_features = [self._format_feature_label(f) for f in common_features]

        comparison_matrix = []

        for voice_name in selected_voices:
            if voice_name in comparison_data:
                row = [comparison_data[voice_name].get(feature, np.nan) for feature in common_features]
                comparison_matrix.append(row)

        if not comparison_matrix:
            return

        comparison_df = pd.DataFrame(comparison_matrix,
                                   index=[v for v in selected_voices if v in comparison_data],
                                   columns=formatted_common_features)

        ax = self.fig.add_subplot(1, 1, 1)
        sns.heatmap(comparison_df, annot=True, fmt='.2f', cmap='RdBu_r',
                   center=0, ax=ax, cbar_kws={'shrink': 0.8})
        ax.set_title("Voice Feature Comparison Heatmap", fontsize=14, fontweight='bold')
        ax.set_xlabel("Features")
        ax.set_ylabel("Voice Profiles")
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        self.fig.tight_layout()

    def create_clustering_plot(self, selected_voices):
        """Create clustering visualization to identify voice archetypes"""
        if len(selected_voices) < 3:
            self.status_text.set("Select at least 3 voices for clustering")
            return

        try:
            from sklearn.cluster import KMeans
            from sklearn.decomposition import PCA
            from sklearn.preprocessing import StandardScaler
        except ImportError:
            self.status_text.set("sklearn not available for clustering. Install with: pip install scikit-learn")
            return

        feature_matrix = []
        voice_names = []

        all_features_union = set()
        for voice_name in selected_voices:
            features_df = self.voice_data[voice_name]['features']
            filtered_features = self.filter_features_by_category(features_df, self.feature_category.get())
            if not filtered_features.empty:
                numeric_features = pd.to_numeric(filtered_features.iloc[0], errors='coerce').dropna()
                all_features_union.update(numeric_features.index)

        if not all_features_union:
            self.status_text.set("No common features for clustering. Check data.")
            ax = self.fig.add_subplot(1, 1, 1)
            ax.text(0.5, 0.5, 'No common features for clustering', ha='center', va='center', transform=ax.transAxes, fontsize=12)
            ax.set_title("Clustering Debug - No Common Features")
            return

        common_feature_list = sorted(list(all_features_union))

        for voice_name in selected_voices:
            features_df = self.voice_data[voice_name]['features']
            filtered_features = self.filter_features_by_category(features_df, self.feature_category.get())

            if not filtered_features.empty:
                feature_row = filtered_features.iloc[0]
                numeric_features = pd.to_numeric(feature_row, errors='coerce').dropna()

                feature_vector = [numeric_features.get(f, np.nan) for f in common_feature_list]

                if not all(np.isnan(feature_vector)):
                    feature_matrix.append(feature_vector)
                    voice_names.append(voice_name)

        if len(feature_matrix) < 3:
            self.status_text.set("Not enough voices with valid data for clustering (min 3 required).")
            ax = self.fig.add_subplot(1, 1, 1)
            ax.text(0.5, 0.5, 'Not enough voices with valid data for clustering (min 3 required)', ha='center', va='center', transform=ax.transAxes, fontsize=10)
            ax.set_title("Clustering Debug - Insufficient Data")
            return

        feature_df_for_clustering = pd.DataFrame(feature_matrix, columns=common_feature_list, index=voice_names)

        feature_df_for_clustering = feature_df_for_clustering.fillna(feature_df_for_clustering.mean())

        scaler = StandardScaler()
        feature_matrix_scaled = scaler.fit_transform(feature_df_for_clustering)

        n_clusters = min(3, len(voice_names))
        try:
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(feature_matrix_scaled)
        except Exception as kmeans_e:
            self.status_text.set(f"Error during KMeans clustering: {kmeans_e}. Try fewer voices or different features.")
            print(f"KMeans Error: {kmeans_e}")
            ax = self.fig.add_subplot(1, 1, 1)
            ax.text(0.5, 0.5, f"Error during KMeans clustering: {kmeans_e}", ha='center', va='center', transform=ax.transAxes, fontsize=10)
            return

        pca = PCA(n_components=2)
        try:
            features_2d = pca.fit_transform(feature_matrix_scaled)
        except ValueError as pca_e:
            self.status_text.set(f"Error during PCA: {pca_e}. Data might be too uniform or not enough features.")
            print(f"PCA Error: {pca_e}")
            ax = self.fig.add_subplot(1, 1, 1)
            ax.text(0.5, 0.5, f"Error during PCA: {pca_e}", ha='center', va='center', transform=ax.transAxes, fontsize=10)
            return


        ax = self.fig.add_subplot(1, 1, 1)

        colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
        for i in range(n_clusters):
            cluster_mask = cluster_labels == i
            ax.scatter(features_2d[cluster_mask, 0], features_2d[cluster_mask, 1],
                      c=colors[i % len(colors)], label=f'Archetype {i+1}', s=100, alpha=0.7)

        for i, name in enumerate(voice_names):
            ax.annotate(name, (features_2d[i, 0], features_2d[i, 1]),
                       xytext=(5, 5), textcoords='offset points', fontsize=8)

        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
        ax.set_title('Voice Archetype Clustering (PCA Visualization)', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

        self.fig.tight_layout()

    def create_correlation_plot(self, selected_voices):
        """Create correlation heatmap of features"""
        if not selected_voices:
            self.status_text.set("Select at least one voice for correlation plot")
            return

        all_features = []

        for voice_name in selected_voices:
            features_df = self.voice_data[voice_name]['features']
            filtered_features = self.filter_features_by_category(features_df, self.feature_category.get())

            if not filtered_features.empty:
                feature_row = filtered_features.iloc[0]
                numeric_features = pd.to_numeric(feature_row, errors='coerce').dropna()
                all_features.append(numeric_features)

        if not all_features:
            ax = self.fig.add_subplot(1, 1, 1)
            ax.text(0.5, 0.5, 'No numeric features available for correlation', ha='center', va='center', transform=ax.transAxes, fontsize=12)
            ax.set_title("Correlation Debug - No Data")
            return

        combined_df = pd.DataFrame(all_features)

        numeric_cols_df = combined_df.select_dtypes(include=[np.number])

        if numeric_cols_df.empty:
            ax = self.fig.add_subplot(1, 1, 1)
            ax.text(0.5, 0.5, 'No numeric columns to correlate', ha='center', va='center', transform=ax.transAxes, fontsize=12)
            ax.set_title("Correlation Debug - No Numeric Columns")
            return

        correlation_matrix = numeric_cols_df.corr()

        if correlation_matrix.empty:
            ax = self.fig.add_subplot(1, 1, 1)
            ax.text(0.5, 0.5, 'Correlation matrix is empty. Not enough features or variance.', ha='center', va='center', transform=ax.transAxes, fontsize=12)
            ax.set_title("Correlation Debug - Empty Matrix")
            return

        ax = self.fig.add_subplot(1, 1, 1)

        num_features_to_plot = min(20, len(correlation_matrix.columns))
        feature_subset_cols = correlation_matrix.columns[:num_features_to_plot]
        feature_subset = correlation_matrix.loc[feature_subset_cols, feature_subset_cols]

        feature_subset.columns = [self._format_feature_label(f) for f in feature_subset.columns]
        feature_subset.index = [self._format_feature_label(f) for f in feature_subset.index]


        sns.heatmap(feature_subset, annot=True, fmt='.2f', cmap='coolwarm',
                   center=0, ax=ax, square=True, annot_kws={"fontsize":8})
        ax.set_title('Feature Correlation Matrix', fontsize=14, fontweight='bold')
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        plt.setp(ax.get_yticklabels(), rotation=0)

        self.fig.tight_layout()

    def create_distribution_plot(self, selected_voices):
        """Create distribution plots for key features"""
        if not selected_voices:
            self.status_text.set("Select at least one voice for distribution plot")
            return

        feature_data = {}

        for voice_name in selected_voices:
            features_df = self.voice_data[voice_name]['features']
            filtered_features = self.filter_features_by_category(features_df, self.feature_category.get())

            if not filtered_features.empty:
                feature_row = filtered_features.iloc[0]
                numeric_features = pd.to_numeric(feature_row, errors='coerce').dropna()
                feature_data[voice_name] = numeric_features
            else:
                print(f"   {voice_name}: ❌ Filtered features DataFrame is empty for distribution plot")

        if not feature_data:
            ax = self.fig.add_subplot(1, 1, 1)
            ax.text(0.5, 0.5, 'No numeric features available for distribution', ha='center', va='center', transform=ax.transAxes, fontsize=12)
            ax.set_title("Distribution Debug - No Data")
            return

        all_values_df = pd.DataFrame(feature_data).T

        all_values_df = all_values_df.dropna(axis=1, how='all')
        all_values_df = all_values_df.loc[:, all_values_df.nunique() > 1]

        if all_values_df.empty:
            ax = self.fig.add_subplot(1, 1, 1)
            ax.text(0.5, 0.5, 'No variable numeric features for distribution', ha='center', va='center', transform=ax.transAxes, fontsize=12)
            ax.set_title("Distribution Debug - No Variable Features")
            return

        feature_variance = all_values_df.var().nlargest(6)

        if feature_variance.empty:
            ax = self.fig.add_subplot(1, 1, 1)
            ax.text(0.5, 0.5, 'Not enough variable features to plot distributions', ha='center', va='center', transform=ax.transAxes, fontsize=12)
            ax.set_title("Distribution Debug - Not Enough Features")
            return

        fig_rows, fig_cols = 2, 3
        self.fig.clear()

        for i, (feature_raw_name, _) in enumerate(feature_variance.items()):
            if i >= fig_rows * fig_cols:
                break

            ax = self.fig.add_subplot(fig_rows, fig_cols, i + 1)

            values = []
            labels = []
            for voice_name in selected_voices:
                if feature_raw_name in feature_data[voice_name]:
                    values.append(feature_data[voice_name][feature_raw_name])
                    labels.append(voice_name)

            if not values:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
                ax.set_title(self._format_feature_label(feature_raw_name), fontsize=9)
                continue

            ax.bar(range(len(values)), values, color=plt.cm.Set3(np.linspace(0, 1, len(values))))
            ax.set_xticks(range(len(labels)))
            ax.set_xticklabels([label[:8] + "..." if len(label) > 8 else label for label in labels],
                              rotation=45, ha="right", fontsize=8)
            ax.set_title(self._format_feature_label(feature_raw_name), fontsize=9)
            ax.grid(True, alpha=0.3)
            ax.tick_params(axis='y', labelsize=8)

        self.fig.suptitle('Feature Distribution Across Voice Profiles', fontsize=14, fontweight='bold')
        self.fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    def export_data(self):
        """Export current analysis data"""
        if not self.voice_data:
            self.status_text.set("No data to export")
            return

        export_data = {}
        selected_voices = self.get_selected_voices()

        for voice_name in selected_voices:
            voice_info = self.voice_data[voice_name]
            features_df = voice_info['features']

            export_data[voice_name] = {
                'metadata': voice_info['metadata'],
                'features': features_df.to_dict('records')[0] if not features_df.empty else {}
            }

        export_file = self.platform_root / f"voice_analysis_export_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.json"

        with open(export_file, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)

        self.status_text.set(f"Data exported to: {export_file}")

    def refresh_data(self):
        """Refresh voice data from output directory"""
        self.voice_data.clear()
        self.voice_listbox.delete(0, tk.END)

        self.load_all_voice_data()

        for creator in self.voice_data.keys():
            self.voice_listbox.insert(tk.END, creator)

        self.generate_analysis()

    def run(self):
        """Start the GUI application"""
        print("🎨 Starting Voice Pattern Analyzer...")
        print(f"📊 Loaded {len(self.voice_data)} voice profiles")
        print("💡 Use the interface to explore voice patterns and find archetypes!")

        self.root.mainloop()


def main():
    """Main function to run the Voice Pattern Analyzer"""

    try:
        import matplotlib
        import seaborn
        import tkinter
        from sklearn.cluster import KMeans
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler
    except ImportError as e:
        print(f"❌ Missing required dependency: {e}")
        print("Install with: pip install matplotlib seaborn scikit-learn")
        return

    print("🎤 Attune Voice Pattern Analyzer")
    print("="*50)

    analyzer = VoicePatternAnalyzer()
    analyzer.run()


if __name__ == "__main__":
    main()