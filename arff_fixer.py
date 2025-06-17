"""
ARFF Data Fixer
Reprocess existing ARFF files with the correct parser
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json

def read_arff_file_fixed(arff_file):
    """Fixed ARFF reader that properly extracts numeric features"""
    try:
        with open(arff_file, 'r') as f:
            lines = f.readlines()
        
        # Find the data section and collect attributes
        data_start = -1
        attribute_names = []
        
        for i, line in enumerate(lines):
            line = line.strip()
            if line.startswith('@attribute'):
                # Extract attribute name
                parts = line.split()
                if len(parts) >= 2:
                    attr_name = parts[1]
                    attribute_names.append(attr_name)
            elif line.startswith('@data'):
                data_start = i + 1
                break
        
        if data_start == -1 or not attribute_names:
            print("❌ Could not find data section or attributes")
            return None
        
        # Read the data line(s)
        data_lines = []
        for i in range(data_start, len(lines)):
            line = lines[i].strip()
            if line and not line.startswith('%') and line != '':  # Skip comments and empty lines
                # Split by comma - this should give us all the values
                values = [val.strip() for val in line.split(',')]
                data_lines.append(values)
        
        if not data_lines:
            print("❌ No data lines found")
            return None
        
        print(f"🔍 Found {len(attribute_names)} attributes, {len(data_lines)} data rows")
        print(f"🔍 First data row has {len(data_lines[0])} values")
        
        # Use the data values (skip first column which is the 'name' field)
        if len(data_lines[0]) > 1:
            # Skip the first column (name='unknown') and use numeric columns
            data_values = data_lines[0][1:]  # Skip first column
            feature_names = attribute_names[1:]  # Skip first attribute name
            
            # Make sure we have matching lengths
            min_length = min(len(data_values), len(feature_names))
            data_values = data_values[:min_length]
            feature_names = feature_names[:min_length]
            
            print(f"🔍 Using {len(data_values)} feature values")
            print(f"🔍 Sample feature names: {feature_names[:5]}")
            print(f"🔍 Sample values: {data_values[:5]}")
            
            # Create a single-row DataFrame
            df = pd.DataFrame([data_values], columns=feature_names)
            
            # Convert to numeric, replacing '?' with NaN
            for col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Count numeric columns
            numeric_cols = df.select_dtypes(include=[np.number]).shape[1]
            print(f"🔍 Final DataFrame shape: {df.shape}")
            print(f"🔍 Numeric columns: {numeric_cols}")
            
            return df
        else:
            print("❌ Data row too short")
            return None
        
    except Exception as e:
        print(f"❌ Error reading ARFF file: {e}")
        return None

def fix_existing_data(platform_root="C:/Attune-Voice-Platform"):
    """Fix existing ARFF data files"""
    platform_root = Path(platform_root)
    output_dir = platform_root / "output"
    
    print("🔧 ARFF Data Fixer")
    print("="*50)
    
    # Find analysis directories
    analysis_dirs = [d for d in output_dir.iterdir() if d.is_dir()]
    
    if not analysis_dirs:
        print("❌ No analysis directories found")
        return
    
    print(f"📁 Found {len(analysis_dirs)} analysis directories")
    
    fixed_count = 0
    
    for analysis_dir in analysis_dirs:
        print(f"\n📊 Processing: {analysis_dir.name}")
        
        # Look for ARFF features file
        feature_files = list(analysis_dir.glob("*_features.csv"))
        
        if feature_files:
            arff_file = feature_files[0]
            print(f"✅ Found ARFF file: {arff_file.name}")
            
            # Try to fix the ARFF file
            fixed_df = read_arff_file_fixed(arff_file)
            
            if fixed_df is not None and not fixed_df.empty:
                # Save the fixed data as a proper CSV
                fixed_csv_file = analysis_dir / f"{arff_file.stem}_fixed.csv"
                fixed_df.to_csv(fixed_csv_file, index=False)
                
                print(f"✅ Fixed data saved to: {fixed_csv_file.name}")
                print(f"   Shape: {fixed_df.shape}")
                print(f"   Numeric features: {fixed_df.select_dtypes(include=[np.number]).shape[1]}")
                
                # Update the report.json to indicate fixed data is available
                report_file = analysis_dir / "report.json"
                if report_file.exists():
                    try:
                        with open(report_file, 'r') as f:
                            report = json.load(f)
                        
                        report['fixed_features_available'] = True
                        report['fixed_features_file'] = fixed_csv_file.name
                        report['fixed_feature_count'] = fixed_df.shape[1]
                        
                        with open(report_file, 'w') as f:
                            json.dump(report, f, indent=2)
                        
                        print(f"✅ Updated report.json")
                        
                    except Exception as e:
                        print(f"⚠️ Could not update report: {e}")
                
                fixed_count += 1
            else:
                print(f"❌ Could not fix ARFF file")
        else:
            print(f"❌ No ARFF file found")
    
    print(f"\n🎉 Fixed {fixed_count} analysis files")
    print("Now run the pattern analyzer again to see the voice features!")

if __name__ == "__main__":
    fix_existing_data()
    