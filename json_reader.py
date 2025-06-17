"""
JSON Export Reader & Converter
Convert your voice analysis JSON exports to readable formats
"""

import json
import pandas as pd
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

class JSONVoiceDataReader:
    """
    Tool to read and convert voice analysis JSON exports
    """
    
    def __init__(self):
        self.setup_gui()
    
    def setup_gui(self):
        """Setup simple GUI for file selection and conversion"""
        self.root = tk.Tk()
        self.root.title("Voice Analysis JSON Reader")
        self.root.geometry("600x500")
        
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Title
        title_label = ttk.Label(main_frame, text="Voice Analysis JSON Reader", 
                               font=('Arial', 16, 'bold'))
        title_label.pack(pady=(0, 20))
        
        # File selection
        file_frame = ttk.LabelFrame(main_frame, text="Select JSON File", padding="10")
        file_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.file_path_var = tk.StringVar()
        ttk.Label(file_frame, text="JSON File:").pack(anchor=tk.W)
        
        path_frame = ttk.Frame(file_frame)
        path_frame.pack(fill=tk.X, pady=(5, 0))
        
        ttk.Entry(path_frame, textvariable=self.file_path_var, width=50).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(path_frame, text="Browse", command=self.browse_file).pack(side=tk.RIGHT, padx=(5, 0))
        
        # Action buttons
        button_frame = ttk.LabelFrame(main_frame, text="Actions", padding="10")
        button_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Button(button_frame, text="📊 View Data Summary", 
                  command=self.view_summary).pack(fill=tk.X, pady=2)
        ttk.Button(button_frame, text="📋 Export to CSV", 
                  command=self.export_csv).pack(fill=tk.X, pady=2)
        ttk.Button(button_frame, text="📈 Export to Excel", 
                  command=self.export_excel).pack(fill=tk.X, pady=2)
        ttk.Button(button_frame, text="🔍 Show Raw JSON", 
                  command=self.show_raw_json).pack(fill=tk.X, pady=2)
        
        # Output area
        output_frame = ttk.LabelFrame(main_frame, text="Output", padding="10")
        output_frame.pack(fill=tk.BOTH, expand=True)
        
        # Text widget with scrollbar
        text_frame = ttk.Frame(output_frame)
        text_frame.pack(fill=tk.BOTH, expand=True)
        
        self.output_text = tk.Text(text_frame, wrap=tk.WORD, font=('Consolas', 10))
        scrollbar = ttk.Scrollbar(text_frame, orient=tk.VERTICAL, command=self.output_text.yview)
        self.output_text.configure(yscrollcommand=scrollbar.set)
        
        self.output_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Auto-find JSON files
        self.auto_find_json_files()
    
    def auto_find_json_files(self):
        """Automatically find JSON export files"""
        platform_root = Path("C:/Attune-Voice-Platform")
        
        # Look for export files
        json_files = list(platform_root.glob("voice_analysis_export_*.json"))
        
        if json_files:
            # Use the most recent one
            latest_file = max(json_files, key=lambda x: x.stat().st_mtime)
            self.file_path_var.set(str(latest_file))
            self.log(f"Auto-found JSON file: {latest_file.name}")
        else:
            self.log("No JSON export files found. Use 'Browse' to select a file.")
    
    def browse_file(self):
        """Browse for JSON file"""
        filename = filedialog.askopenfilename(
            title="Select Voice Analysis JSON File",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            initialdir="C:/Attune-Voice-Platform"
        )
        
        if filename:
            self.file_path_var.set(filename)
            self.log(f"Selected file: {Path(filename).name}")
    
    def load_json_data(self):
        """Load and parse JSON data"""
        file_path = self.file_path_var.get()
        
        if not file_path:
            messagebox.showerror("Error", "Please select a JSON file first")
            return None
        
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            return data
        except Exception as e:
            messagebox.showerror("Error", f"Could not read JSON file: {str(e)}")
            return None
    
    def view_summary(self):
        """Display summary of voice data"""
        data = self.load_json_data()
        if not data:
            return
        
        self.clear_output()
        self.log("📊 VOICE ANALYSIS DATA SUMMARY")
        self.log("=" * 50)
        
        # Count voices
        voice_count = len(data)
        self.log(f"Total Voice Profiles: {voice_count}")
        self.log("")
        
        # Analyze each voice
        for voice_name, voice_data in data.items():
            self.log(f"🎤 {voice_name}")
            self.log("-" * 30)
            
            # Metadata
            if 'metadata' in voice_data:
                metadata = voice_data['metadata']
                self.log(f"Analysis Date: {metadata.get('timestamp', 'Unknown')}")
                self.log(f"Feature Count: {metadata.get('feature_count', 'Unknown')}")
            
            # Features summary
            if 'features' in voice_data:
                features = voice_data['features']
                feature_count = len(features)
                self.log(f"Available Features: {feature_count}")
                
                # Show top 10 features by value
                if features:
                    numeric_features = {k: v for k, v in features.items() 
                                      if isinstance(v, (int, float)) and v != 0}
                    
                    if numeric_features:
                        top_features = sorted(numeric_features.items(), 
                                            key=lambda x: abs(x[1]), reverse=True)[:10]
                        
                        self.log(f"Top 10 Features by Magnitude:")
                        for feat_name, feat_value in top_features:
                            self.log(f"  {feat_name}: {feat_value:.3f}")
            
            self.log("")
    
    def export_csv(self):
        """Export data to CSV format"""
        data = self.load_json_data()
        if not data:
            return
        
        try:
            # Create DataFrame
            rows = []
            for voice_name, voice_data in data.items():
                row = {'Voice_Name': voice_name}
                
                # Add metadata
                if 'metadata' in voice_data:
                    metadata = voice_data['metadata']
                    row['Analysis_Date'] = metadata.get('timestamp', '')
                    row['Feature_Count'] = metadata.get('feature_count', 0)
                
                # Add features
                if 'features' in voice_data:
                    features = voice_data['features']
                    for feat_name, feat_value in features.items():
                        row[f"Feature_{feat_name}"] = feat_value
                
                rows.append(row)
            
            df = pd.DataFrame(rows)
            
            # Save CSV
            output_file = Path(self.file_path_var.get()).parent / "voice_analysis_export.csv"
            df.to_csv(output_file, index=False)
            
            self.log(f"✅ CSV exported to: {output_file}")
            messagebox.showinfo("Success", f"CSV file saved: {output_file.name}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Could not export CSV: {str(e)}")
    
    def export_excel(self):
        """Export data to Excel format"""
        data = self.load_json_data()
        if not data:
            return
        
        try:
            # Create Excel file with multiple sheets
            output_file = Path(self.file_path_var.get()).parent / "voice_analysis_export.xlsx"
            
            with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
                # Summary sheet
                summary_rows = []
                for voice_name, voice_data in data.items():
                    summary_row = {
                        'Voice_Name': voice_name,
                        'Analysis_Date': voice_data.get('metadata', {}).get('timestamp', ''),
                        'Feature_Count': voice_data.get('metadata', {}).get('feature_count', 0)
                    }
                    summary_rows.append(summary_row)
                
                summary_df = pd.DataFrame(summary_rows)
                summary_df.to_excel(writer, sheet_name='Summary', index=False)
                
                # Individual voice sheets
                for voice_name, voice_data in data.items():
                    if 'features' in voice_data:
                        features = voice_data['features']
                        
                        # Create feature DataFrame
                        feature_rows = []
                        for feat_name, feat_value in features.items():
                            feature_rows.append({
                                'Feature_Name': feat_name,
                                'Value': feat_value,
                                'Category': self.categorize_feature(feat_name)
                            })
                        
                        feat_df = pd.DataFrame(feature_rows)
                        
                        # Clean sheet name
                        sheet_name = voice_name.replace(' ', '_')[:31]  # Excel limit
                        feat_df.to_excel(writer, sheet_name=sheet_name, index=False)
            
            self.log(f"✅ Excel exported to: {output_file}")
            messagebox.showinfo("Success", f"Excel file saved: {output_file.name}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Could not export Excel: {str(e)}")
    
    def categorize_feature(self, feature_name):
        """Categorize feature by name"""
        feature_lower = feature_name.lower()
        
        if 'f0' in feature_lower or 'pitch' in feature_lower:
            return 'Pitch/F0'
        elif 'loudness' in feature_lower:
            return 'Loudness/Energy'
        elif 'spectral' in feature_lower or 'mfcc' in feature_lower:
            return 'Spectral/Quality'
        elif 'jitter' in feature_lower or 'shimmer' in feature_lower or 'hnr' in feature_lower:
            return 'Voice Quality'
        elif 'voiced' in feature_lower or 'segment' in feature_lower:
            return 'Temporal/Rhythm'
        else:
            return 'Other'
    
    def show_raw_json(self):
        """Display raw JSON content"""
        data = self.load_json_data()
        if not data:
            return
        
        self.clear_output()
        self.log("🔍 RAW JSON DATA")
        self.log("=" * 50)
        
        # Pretty print JSON
        json_str = json.dumps(data, indent=2)
        self.log(json_str)
    
    def log(self, message):
        """Add message to output area"""
        self.output_text.insert(tk.END, str(message) + "\n")
        self.output_text.see(tk.END)
        self.root.update()
    
    def clear_output(self):
        """Clear output area"""
        self.output_text.delete(1.0, tk.END)
    
    def run(self):
        """Start the application"""
        self.root.mainloop()


def main():
    """Run the JSON reader application"""
    print("🔍 Starting Voice Analysis JSON Reader...")
    reader = JSONVoiceDataReader()
    reader.run()


if __name__ == "__main__":
    main()
    