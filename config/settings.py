from pathlib import Path

class Config:
    # Workspace paths
    WORKSPACE = Path("C:/Attune-Voice-Platform")
    TOOLS_DIR = WORKSPACE / "tools"
    DATA_DIR = WORKSPACE / "data"
    
    # Tool paths
    OPENSMILE_PATH = TOOLS_DIR / "opensmile"
    FFMPEG_PATH = TOOLS_DIR / "ffmpeg" / "ffmpeg.exe"
    
    # Analysis settings
    AUDIO_FORMAT = "wav"
    AUDIO_SAMPLE_RATE = 16000
    AUDIO_CHANNELS = 1
    
    # Output settings
    SAVE_INTERMEDIATE_FILES = True
    GENERATE_REPORTS = True
    
    @classmethod
    def validate(cls):
        """Check all required paths exist"""
        required_paths = [
            cls.OPENSMILE_PATH / "bin" / "SMILExtract.exe",
            cls.FFMPEG_PATH
        ]
        
        for path in required_paths:
            if not path.exists():
                raise FileNotFoundError(f"Required tool not found: {path}")
        
        return True