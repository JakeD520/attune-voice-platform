"""
Quick OpenSMILE Data Inspector
Let's see what's actually in your voice analysis data
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json

def inspect_voice_data(platform_root="C:/Attune-Voice-Platform"):
    """Inspect the structure of your voice analysis data"""
    
    platform_root = Path(platform_root)
    output_dir = platform_root / "output"
    
    print("🔍 OPENSMILE DATA INSPECTOR")
    print("="*50)
    
    # Find analysis directories
    analysis_dirs = [d for d in output_dir.iterdir() if d.is_dir()]
    
    if not analysis_dirs:
        print("❌ No analysis directories found")
        return
    
    print(f"📁 Found {len(analysis_dirs)} analysis directories")
    
    for analysis_dir in analysis_dirs:
        print(f"\n📊 Analyzing: {analysis_dir.name}")
        print("-" * 30)
        
        # Look for files
        feature_files = list(analysis_dir.glob("*_features.csv"))
        
        if feature_files:
            print(f"✅ Features file: {feature_files[0].name}")
            
            try:
                # First, let's see the raw file content
                print(f"\n📋 Raw file content (first 20 lines):")
                with open(feature_files[0], 'r') as f:
                    lines = f.readlines()
                
                for i, line in enumerate(lines[:20]):
                    print(f"   {i+1:2d}: {line.strip()}")
                
                if len(lines) > 20:
                    print(f"   ... and {len(lines) - 20} more lines")
                
                print(f"\n📋 Total lines in file: {len(lines)}")
                
                # Try to identify the format
                print(f"\n📋 Format analysis:")
                if lines[0].strip().startswith('@relation'):
                    print("   Format: ARFF")
                    
                    # Find attributes and data section
                    attributes = []
                    data_start = -1
                    
                    for i, line in enumerate(lines):
                        line = line.strip()
                        if line.startswith('@attribute'):
                            attributes.append(line)
                        elif line.startswith('@data'):
                            data_start = i
                            break
                    
                    print(f"   Attributes found: {len(attributes)}")
                    print(f"   Data section starts at line: {data_start + 1}")
                    
                    if attributes:
                        print(f"   First few attributes:")
                        for attr in attributes[:5]:
                            print(f"      {attr}")
                    
                    if data_start > 0 and data_start + 1 < len(lines):
                        print(f"   First few data lines:")
                        for i in range(data_start + 1, min(data_start + 6, len(lines))):
                            print(f"      {lines[i].strip()}")
                
                elif ',' in lines[0] or ';' in lines[0]:
                    print("   Format: CSV")
                    # Try reading as CSV
                    df = pd.read_csv(feature_files[0], sep=';')
                    print(f"   Shape: {df.shape}")
                    print(f"   Columns: {list(df.columns[:5])}")
                else:
                    print("   Format: Unknown")
                
            except Exception as e:
                print(f"❌ Error reading features file: {e}")
        
        else:
            print("❌ No features file found")


if __name__ == "__main__":
    inspect_voice_data()