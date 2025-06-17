"""
Attune Voice Platform - Clean Analyzer
Fixed version for c:/attune-voice-platform
"""

import os
import json
import subprocess
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

class AttuneVoiceAnalyzer:
    """Main voice analyzer for Attune Voice Platform"""
    
    def __init__(self, platform_root=None):
        if platform_root is None:
            # Auto-detect current directory
            platform_root = Path.cwd()
        else:
            platform_root = Path(platform_root)
        self.platform_root = Path(platform_root)
        self.setup_structure()
        self.load_tools()
        print("🎤 Attune Voice Platform Analyzer")
        print(f"📁 Platform Root: {self.platform_root}")
        print("="*50)
    
    def setup_structure(self):
        """Create organized subdirectories"""
        directories = ["input", "output", "logs", "temp"]
        
        for directory in directories:
            dir_path = self.platform_root / directory
            dir_path.mkdir(exist_ok=True)
        
        print("✅ Platform structure organized")
    
    def load_tools(self):
        """Find openSMILE and FFmpeg"""
        
        # Look for openSMILE in multiple locations
        possible_opensmile_paths = [
            self.platform_root / "tools" / "opensmile",
            self.platform_root / "tools" / "opensmile" / "bin",
            Path("C:/Users/JakePC/Documents/opensmile-3.0.2-windows-x86_64"),
            Path("C:/opensmile"),
            Path("C:/tools/opensmile"),
        ]
        
        self.opensmile_path = None
        for path in possible_opensmile_paths:
            if (path / "SMILExtract.exe").exists():
                self.opensmile_path = path
                break
        
        # Look for FFmpeg in multiple locations
        self.ffmpeg_path = None
        ffmpeg_locations = [
            self.platform_root / "tools" / "ffmpeg" / "ffmpeg.exe",
            self.platform_root / "tools" / "ffmpeg.exe",
            self.opensmile_path / "ffmpeg.exe" if self.opensmile_path else None,
            Path("C:/ffmpeg/bin/ffmpeg.exe"),
            "ffmpeg",  # Try system PATH
        ]
        
        for path in ffmpeg_locations:
            if path and self.test_ffmpeg(path):
                self.ffmpeg_path = path
                break
        
        print(f"🔧 openSMILE: {'✅ Found at ' + str(self.opensmile_path) if self.opensmile_path else '❌ Not found'}")
        print(f"🎬 FFmpeg: {'✅ Found at ' + str(self.ffmpeg_path) if self.ffmpeg_path else '❌ Not found'}")
    
    def test_ffmpeg(self, path):
        """Test if FFmpeg path works"""
        try:
            result = subprocess.run([str(path), "-version"], 
                                  capture_output=True, text=True, timeout=5)
            return result.returncode == 0
        except:
            return False
    
    def analyze_video(self, video_file, creator_name=None):
        """Analyze a single video file"""
        
        # Handle file path
        video_path = Path(video_file)
        if not video_path.is_absolute():
            video_path = self.platform_root / "input" / video_file
        
        if not video_path.exists():
            print(f"❌ Video file not found: {video_path}")
            return None
        
        # Generate analysis name
        if not creator_name:
            creator_name = video_path.stem
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        analysis_name = f"{creator_name}_{timestamp}"
        
        print(f"\n🎬 Analyzing: {video_path.name}")
        print(f"📝 Creator: {creator_name}")
        
        # Create output directory
        output_dir = self.platform_root / "output" / analysis_name
        output_dir.mkdir(exist_ok=True)
        
        results = {"analysis_name": analysis_name, "creator": creator_name}
        
        try:
            # Extract audio if FFmpeg is available
            if self.ffmpeg_path:
                audio_file = self.extract_audio(video_path, output_dir)
                results["audio_extracted"] = audio_file is not None
                
                # Extract features if openSMILE is available
                if audio_file and self.opensmile_path:
                    features = self.extract_features(audio_file, output_dir)
                    results["features_extracted"] = features is not None
                    results["feature_count"] = len(features.columns) if features is not None else 0
            
            # Generate report
            self.save_report(results, output_dir)
            
            print(f"✅ Analysis complete! Results in: {output_dir}")
            return output_dir
            
        except Exception as e:
            print(f"❌ Analysis failed: {str(e)}")
            return None
    
    def extract_audio(self, video_path, output_dir):
        """Extract audio using FFmpeg"""
        audio_file = output_dir / f"{video_path.stem}_audio.wav"
        
        cmd = [
            str(self.ffmpeg_path),
            "-i", str(video_path),
            "-ac", "1",  # Mono
            "-ar", "16000",  # 16kHz
            "-y",  # Overwrite
            str(audio_file)
        ]
        
        print("🎵 Extracting audio...")
        try:
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                print(f"✅ Audio extracted: {audio_file.name}")
                return audio_file
            else:
                print(f"❌ FFmpeg error: {result.stderr}")
                return None
        except Exception as e:
            print(f"❌ Audio extraction failed: {str(e)}")
            return None
    
    def extract_features(self, audio_file, output_dir):
        """Extract features using openSMILE"""
        features_file = output_dir / f"{audio_file.stem}_features.csv"
        
        smile_exe = self.opensmile_path / "SMILExtract.exe"
        
        # Try multiple config file locations
        config_locations = [
            self.opensmile_path.parent / "config" / "egemaps" / "v02" / "eGeMAPSv02.conf",  # correct path
            self.opensmile_path.parent / "config" / "eGeMAPSv02.conf",  # tools/opensmile/config/
            self.opensmile_path / ".." / "config" / "eGeMAPSv02.conf",  # relative path
            self.opensmile_path / "config" / "eGeMAPSv02.conf",         # bin/config/
        ]
        
        config_file = None
        for location in config_locations:
            if Path(location).exists():
                config_file = location
                break
        
        if not config_file:
            print(f"❌ Config file not found. Checked:")
            for loc in config_locations:
                print(f"   - {loc}")
            return None
        
        print(f"Using config: {config_file}")
        
        cmd = [
            str(smile_exe),
            "-C", str(config_file),
            "-I", str(audio_file),
            "-O", str(features_file),
            "-csvoutput", "1",  # Force CSV output format
            "-headercsv", "1"   # Include headers
        ]
        
        print("🎼 Extracting features...")
        try:
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                # Try to read as CSV first
                try:
                    features = pd.read_csv(features_file, sep=';')
                    print(f"✅ Extracted {len(features.columns)} features (CSV format)")
                    return features
                except:
                    # If CSV fails, try reading as ARFF
                    features = self.read_arff_file(features_file)
                    if features is not None:
                        print(f"✅ Extracted {len(features.columns)} features (ARFF format)")
                        return features
                    else:
                        print("❌ Could not parse features file")
                        return None
            else:
                print(f"❌ openSMILE error: {result.stderr}")
                return None
        except Exception as e:
            print(f"❌ Feature extraction failed: {str(e)}")
            return None
    
    def read_arff_file(self, arff_file):
        """Read ARFF format file and convert to pandas DataFrame"""
        try:
            with open(arff_file, 'r') as f:
                lines = f.readlines()
            
            # Find the data section and collect attributes
            data_start = -1
            attribute_names = []
            
            for i, line in enumerate(lines):
                line = line.strip()
                if line.startswith('@attribute'):
                    # Extract attribute name (skip the first 'name' attribute)
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
            
            print(f"🔍 ARFF Debug: Found {len(attribute_names)} attributes, {len(data_lines)} data rows")
            print(f"🔍 ARFF Debug: First data row has {len(data_lines[0])} values")
            
            # Use the data values (skip first column which is the 'name' field)
            if len(data_lines[0]) > 1:
                # Skip the first column (name='unknown') and use numeric columns
                data_values = data_lines[0][1:]  # Skip first column
                feature_names = attribute_names[1:]  # Skip first attribute name
                
                # Make sure we have matching lengths
                min_length = min(len(data_values), len(feature_names))
                data_values = data_values[:min_length]
                feature_names = feature_names[:min_length]
                
                print(f"🔍 ARFF Debug: Using {len(data_values)} feature values")
                print(f"🔍 ARFF Debug: Sample values: {data_values[:5]}")
                
                # Create a single-row DataFrame
                df = pd.DataFrame([data_values], columns=feature_names)
                
                # Convert to numeric, replacing '?' with NaN
                for col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                
                print(f"🔍 ARFF Debug: Final DataFrame shape: {df.shape}")
                print(f"🔍 ARFF Debug: Numeric columns: {df.select_dtypes(include=[np.number]).shape[1]}")
                
                return df
            else:
                print("❌ Data row too short")
                return None
            
        except Exception as e:
            print(f"❌ Error reading ARFF file: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def save_report(self, results, output_dir):
        """Save analysis report"""
        results["timestamp"] = datetime.now().isoformat()
        
        # Save JSON
        json_file = output_dir / "report.json"
        with open(json_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Save readable report
        report_text = f"""
ATTUNE VOICE ANALYSIS
====================

Analysis: {results['analysis_name']}
Creator: {results['creator']}
Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Results:
- Audio extracted: {'Yes' if results.get('audio_extracted') else 'No'}
- Features extracted: {'Yes' if results.get('features_extracted') else 'No'}
- Feature count: {results.get('feature_count', 0)}

Tools used:
- FFmpeg: {'Available' if self.ffmpeg_path else 'Not found'}
- openSMILE: {'Available' if self.opensmile_path else 'Not found'}
"""
        
        text_file = output_dir / "report.txt"
        with open(text_file, 'w') as f:
            f.write(report_text)
        
        print(f"📄 Report saved: {json_file.name}, {text_file.name}")
    
    def list_videos(self):
        """List available videos in input folder"""
        input_dir = self.platform_root / "input"
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
        
        videos = []
        for ext in video_extensions:
            videos.extend(input_dir.glob(f"*{ext}"))
        
        return videos
    
    def status(self):
        """Show system status"""
        print(f"\n📊 SYSTEM STATUS")
        print(f"Platform: {self.platform_root}")
        print(f"openSMILE: {'✅' if self.opensmile_path else '❌'}")
        print(f"FFmpeg: {'✅' if self.ffmpeg_path else '❌'}")
        
        videos = self.list_videos()
        print(f"Input videos: {len(videos)}")
        
        if videos:
            print("\nAvailable videos:")
            for i, video in enumerate(videos, 1):
                print(f"  {i}. {video.name}")


def main():
    """Main function"""
    print("🎤 Attune Voice Platform")
    print("Voice Analysis System")
    print("="*50)
    
    # Initialize
    analyzer = AttuneVoiceAnalyzer()
    
    # Show status
    analyzer.status()
    
    # Check for videos
    videos = analyzer.list_videos()
    
    if not videos:
        print(f"\n📁 No videos found in input folder")
        print(f"Place video files in: {analyzer.platform_root}/input/")
        print(f"Supported formats: .mp4, .avi, .mov, .mkv")
        return
    
    # Simple menu
    print(f"\n🎯 Select video to analyze:")
    for i, video in enumerate(videos, 1):
        print(f"  {i}. {video.name}")
    
    try:
        choice = input(f"\nEnter number (1-{len(videos)}) or 'q' to quit: ")
        
        if choice.lower() == 'q':
            print("👋 Goodbye!")
            return
        
        video_index = int(choice) - 1
        if 0 <= video_index < len(videos):
            selected_video = videos[video_index]
            
            creator_name = input(f"Creator name (or Enter for '{selected_video.stem}'): ").strip()
            if not creator_name:
                creator_name = selected_video.stem
            
            # Analyze
            result = analyzer.analyze_video(selected_video.name, creator_name)
            
            if result:
                print(f"\n🎉 Success! Check results in: {result}")
        else:
            print("❌ Invalid selection")
            
    except (ValueError, KeyboardInterrupt):
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")


if __name__ == "__main__":
    main()