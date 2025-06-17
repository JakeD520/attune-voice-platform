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
warnings.filterwarnings('ignore')

class VoicePatternAnalyzer:
    """
    Interactive tool for analyzing voice patterns and finding archetypes
    """
    
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
                    except:
                        # Fall back to semicolon separator
                        features_df = pd.read_csv(feature_file, sep=';')
                        print(f"✅ Loaded CSV (semicolon): {features_df.shape}")
                    
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
        
        # Debug: Let's see what we actually have
        print(f"\n🔍 DEBUG: Analyzing {len(selected_voices)} voices")
        
        # Get key features for overview
        key_features = []
        for voice_name in selected_voices:
            features_df = self.voice_data[voice_name]['features']
            print(f"   {voice_name}: {features_df.shape} shape")
            
            filtered_features = self.filter_features_by_category(features_df, self.feature_category.get())
            print(f"   {voice_name}: {filtered_features.shape} after filtering")
            
            if not filtered_features.empty:
                # Take first row and convert to numeric only
                feature_row = filtered_features.iloc[0]
                print(f"   {voice_name}: {len(feature_row)} total features")
                
                # Convert to numeric, errors='coerce' will turn non-numeric to NaN
                numeric_features = pd.to_numeric(feature_row, errors='coerce').dropna()
                print(f"   {voice_name}: {len(numeric_features)} numeric features")
                
                if len(numeric_features) > 0:
                    print(f"   {voice_name}: Sample values: {list(numeric_features.head(3).values)}")
                    key_features.append({
                        'voice': voice_name,
                        'data': numeric_features
                    })
                else:
                    print(f"   {voice_name}: ❌ No numeric features found")
        
        if not key_features:
            ax = self.fig.add_subplot(1, 1, 1)
            ax.text(0.5, 0.5, 'No numeric features found\nCheck console for debug info', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=12)
            ax.set_title("Data Debug - No Numeric Features")
            return
        
        # Create subplots
        n_voices = len(key_features)
        cols = min(3, n_voices)
        rows = (n_voices + cols - 1) // cols
        
        for i, voice_data in enumerate(key_features):
            ax = self.fig.add_subplot(rows, cols, i + 1)
            
            # Plot top 10 features by absolute value
            feature_values = voice_data['data']
            
            # Safety check for empty data
            if len(feature_values) == 0:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f"{voice_data['voice']} - No Data")
                continue
            
            # Get top features, handle case where we have fewer than 10
            n_features = min(10, len(feature_values))
            top_features = feature_values.abs().nlargest(n_features)
            
            if len(top_features) > 0:
                ax.barh(range(len(top_features)), top_features.values)
                ax.set_yticks(range(len(top_features)))
                ax.set_yticklabels([f[:15] + "..." if len(f) > 15 else f for f in top_features.index], fontsize=8)
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
        
        # Collect feature data
        comparison_data = {}
        common_features = None
        
        for voice_name in selected_voices:
            features_df = self.voice_data[voice_name]['features']
            filtered_features = self.filter_features_by_category(features_df, self.feature_category.get())
            
            if not filtered_features.empty:
                feature_row = filtered_features.iloc[0]
                # Convert to numeric, drop non-numeric
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
        
        # Create comparison matrix
        common_features = list(common_features)[:20]  # Limit to top 20 for readability
        comparison_matrix = []
        
        for voice_name in selected_voices:
            if voice_name in comparison_data:
                row = [comparison_data[voice_name][feature] for feature in common_features]
                comparison_matrix.append(row)
        
        if not comparison_matrix:
            return
            
        comparison_df = pd.DataFrame(comparison_matrix, 
                                   index=[v for v in selected_voices if v in comparison_data], 
                                   columns=[f[:10] + "..." if len(f) > 10 else f for f in common_features])
        
        # Create heatmap
        ax = self.fig.add_subplot(1, 1, 1)
        sns.heatmap(comparison_df, annot=True, fmt='.2f', cmap='RdBu_r', 
                   center=0, ax=ax, cbar_kws={'shrink': 0.8})
        ax.set_title("Voice Feature Comparison Heatmap", fontsize=14, fontweight='bold')
        ax.set_xlabel("Features")
        ax.set_ylabel("Voice Profiles")
        
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
            self.status_text.set("sklearn not available for clustering")
            return
        
        # Prepare data for clustering
        feature_matrix = []
        voice_names = []
        
        for voice_name in selected_voices:
            features_df = self.voice_data[voice_name]['features']
            filtered_features = self.filter_features_by_category(features_df, self.feature_category.get())
            
            if not filtered_features.empty:
                feature_row = filtered_features.iloc[0]
                # Convert to numeric and drop NaN
                numeric_features = pd.to_numeric(feature_row, errors='coerce').dropna()
                
                if len(numeric_features) > 0:
                    feature_matrix.append(numeric_features.values)
                    voice_names.append(voice_name)
        
        if len(feature_matrix) < 3:
            return
        
        # Standardize features
        scaler = StandardScaler()
        feature_matrix_scaled = scaler.fit_transform(feature_matrix)
        
        # Perform clustering
        n_clusters = min(3, len(voice_names))
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(feature_matrix_scaled)
        
        # Reduce to 2D for visualization
        pca = PCA(n_components=2)
        features_2d = pca.fit_transform(feature_matrix_scaled)
        
        # Create scatter plot
        ax = self.fig.add_subplot(1, 1, 1)
        
        colors = ['red', 'blue', 'green', 'orange', 'purple']
        for i in range(n_clusters):
            cluster_mask = cluster_labels == i
            ax.scatter(features_2d[cluster_mask, 0], features_2d[cluster_mask, 1], 
                      c=colors[i], label=f'Archetype {i+1}', s=100, alpha=0.7)
        
        # Add voice names as labels
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
            return
        
        # Combine all feature data
        all_features = []
        
        for voice_name in selected_voices:
            features_df = self.voice_data[voice_name]['features']
            filtered_features = self.filter_features_by_category(features_df, self.feature_category.get())
            
            if not filtered_features.empty:
                feature_row = filtered_features.iloc[0]
                # Convert to numeric and drop NaN
                numeric_features = pd.to_numeric(feature_row, errors='coerce').dropna()
                all_features.append(numeric_features)
        
        if not all_features:
            return
        
        # Create correlation matrix
        combined_df = pd.DataFrame(all_features)
        correlation_matrix = combined_df.corr()
        
        # Plot heatmap
        ax = self.fig.add_subplot(1, 1, 1)
        
        # Select top correlated features for readability
        feature_subset = correlation_matrix.iloc[:15, :15]
        
        sns.heatmap(feature_subset, annot=True, fmt='.2f', cmap='coolwarm', 
                   center=0, ax=ax, square=True)
        ax.set_title('Feature Correlation Matrix', fontsize=14, fontweight='bold')
        
        self.fig.tight_layout()
    
    def create_distribution_plot(self, selected_voices):
        """Create distribution plots for key features"""
        if not selected_voices:
            return
        
        # Collect feature data
        feature_data = {}
        
        for voice_name in selected_voices:
            features_df = self.voice_data[voice_name]['features']
            filtered_features = self.filter_features_by_category(features_df, self.feature_category.get())
            
            if not filtered_features.empty:
                feature_row = filtered_features.iloc[0]
                # Convert to numeric and drop NaN
                numeric_features = pd.to_numeric(feature_row, errors='coerce').dropna()
                feature_data[voice_name] = numeric_features
        
        if not feature_data:
            return
        
        # Select top 6 most variable features
        all_values = pd.DataFrame(feature_data).T
        feature_variance = all_values.var().nlargest(6)
        
        # Create subplots
        fig_rows, fig_cols = 2, 3
        
        for i, (feature, _) in enumerate(feature_variance.items()):
            ax = self.fig.add_subplot(fig_rows, fig_cols, i + 1)
            
            # Create distribution plot
            values = [feature_data[voice][feature] for voice in selected_voices if feature in feature_data[voice]]
            labels = [voice for voice in selected_voices if feature in feature_data[voice]]
            
            ax.bar(range(len(values)), values, color=plt.cm.Set3(np.linspace(0, 1, len(values))))
            ax.set_xticks(range(len(labels)))
            ax.set_xticklabels([label[:8] + "..." if len(label) > 8 else label for label in labels], 
                              rotation=45, fontsize=8)
            ax.set_title(feature[:20] + "..." if len(feature) > 20 else feature, fontsize=9)
            ax.grid(True, alpha=0.3)
        
        self.fig.suptitle('Feature Distribution Across Voice Profiles', fontsize=14, fontweight='bold')
        self.fig.tight_layout()
    
    def export_data(self):
        """Export current analysis data"""
        if not self.voice_data:
            self.status_text.set("No data to export")
            return
        
        # Create export data
        export_data = {}
        selected_voices = self.get_selected_voices()
        
        for voice_name in selected_voices:
            voice_info = self.voice_data[voice_name]
            features_df = voice_info['features']
            
            export_data[voice_name] = {
                'metadata': voice_info['metadata'],
                'features': features_df.to_dict('records')[0] if not features_df.empty else {}
            }
        
        # Save to file
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
    
    # Check if we have the required dependencies
    try:
        import matplotlib
        import seaborn
        import tkinter
    except ImportError as e:
        print(f"❌ Missing required dependency: {e}")
        print("Install with: pip install matplotlib seaborn")
        return
    
    print("🎤 Attune Voice Pattern Analyzer")
    print("="*50)
    
    # Create and run analyzer
    analyzer = VoicePatternAnalyzer()
    analyzer.run()


if __name__ == "__main__":
    main()