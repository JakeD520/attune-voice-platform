import json
import sys
from pathlib import Path
import subprocess
from typing import List, Dict, Any, Tuple
from datetime import datetime

# --- Start: Fix for relative import when running directly ---
current_file_path = Path(__file__).resolve()
project_root = current_file_path.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
# --- End: Fix for relative import when running directly ---

from attune_analyzer import AttuneVoiceAnalyzer
from config.settings import Config


class MultiSpeakerVoiceAnalyzer(AttuneVoiceAnalyzer):
    """
    Enhanced Voice Analyzer for Multi-Speaker Content.
    Integrates Demucs for voice separation and (optionally) pyannote.audio for speaker identification.
    """

    def __init__(self, platform_root: str = None):
        super().__init__(platform_root)
        print("\n🚀 Initializing Multi-Speaker Voice Analyzer...")
        self.demucs_available = self._check_demucs()
        print(f"🎵 Demucs Available: {'✅ Yes' if self.demucs_available else '❌ No (Install with: pip install demucs)'}")
        # pyannote.audio will be checked/loaded in the diarization method

    def _check_demucs(self) -> bool:
        """Checks if Demucs is installed and callable."""
        try:
            result = subprocess.run([sys.executable, '-m', 'demucs', '-h'], capture_output=True, text=True, check=True, timeout=10)
            return result.returncode == 0
        except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
            return False
        except Exception as e:
            print(f"Demucs check failed with unexpected error: {e}")
            return False

    def separate_voices_with_demucs(self, audio_file_path: Path, output_base_dir: Path) -> Path:
        """
        Stage 1: Separates vocals from background audio using Demucs.
        Assumes audio_file_path is a valid audio file (e.g., WAV, MP3).
        The output will be a 'vocals.wav' and 'no_vocals.wav' within a demucs-created subfolder.
        Returns the path to the isolated vocals track.
        """
        if not self.demucs_available:
            print("❌ Demucs is not installed or not found. Cannot perform voice separation.")
            return None

        demucs_output_dir = output_base_dir / "demucs_separated"
        demucs_output_dir.mkdir(parents=True, exist_ok=True)

        print(f"🎼 Stage 1: Separating voices with Demucs from '{audio_file_path.name}'...")
        cmd = [
            sys.executable, '-m', 'demucs',
            '--two-stems', 'vocals',
            '-o', str(demucs_output_dir),
            '-n', 'htdemucs_ft',
            str(audio_file_path)
        ]

        try:
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            for line in process.stdout:
                print(f"Demucs STDOUT: {line.strip()}")
            process.wait()

            if process.returncode == 0:
                print("✅ Demucs separation complete.")
                stderr_output = process.stderr.read()
                if stderr_output:
                    print("Demucs STDERR (after completion):\n", stderr_output)
            else:
                stderr_output = process.stderr.read()
                print(f"❌ Demucs separation failed with exit code {process.returncode}.")
                print(f"Demucs STDERR:\n", stderr_output)
                raise subprocess.CalledProcessError(process.returncode, cmd, output=process.stdout.read(), stderr=stderr_output)

            separated_vocals_path = demucs_output_dir / 'htdemucs_ft' / audio_file_path.stem / 'vocals.wav'

            if separated_vocals_path.exists():
                print(f"🎵 Isolated vocals track: {separated_vocals_path.name}")
                return separated_vocals_path
            else:
                print(f"❌ Demucs output 'vocals.wav' not found at {separated_vocals_path}.")
                return None
        except FileNotFoundError:
            print("❌ 'demucs' command (or python -m demucs) not found. Ensure Demucs is installed and in your PATH.")
            return None
        except Exception as e:
            print(f"❌ An unexpected error occurred during Demucs separation: {e}")
            return None

    def identify_speakers_with_pyannote(self, audio_file_path: Path, output_base_dir: Path) -> Dict[str, Path]:
        """
        Stage 2: Identifies and labels individual speakers using pyannote.audio's diarization.
        Takes an isolated vocals track and outputs individual speaker audio files.
        Returns a dictionary mapping speaker IDs (e.g., "SPEAKER_00") to their audio file paths.
        """
        try:
            from pyannote.audio import Pipeline
        except ImportError:
            print("❌ pyannote.audio is not installed. Please install with: pip install pyannote.audio")
            return None
        if not audio_file_path.exists():
            print(f"❌ Audio file for diarization not found: {audio_file_path}")
            return None

        print(f"🧠 Stage 2: Identifying speakers with pyannote.audio from '{audio_file_path.name}'...")
        diar_output_dir = output_base_dir / "pyannote_diarization"
        diar_output_dir.mkdir(parents=True, exist_ok=True)

        # Load pretrained pipeline
        pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization")
        diarization = pipeline(str(audio_file_path))

        # Parse segments and export audio
        speaker_segments_map = {}
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            if speaker not in speaker_segments_map:
                speaker_segments_map[speaker] = []
            speaker_segments_map[speaker].append((turn.start, turn.end))

        # Export individual speaker audio files
        return self._export_speaker_audios(audio_file_path, speaker_segments_map, diar_output_dir / "individual_speakers")

    def _export_speaker_audios(self, original_audio_path: Path, speaker_segments_map: Dict[str, List[Tuple[float, float]]], output_dir: Path) -> Dict[str, Path]:
        """Exports individual speaker audio files from the original audio using FFmpeg."""
        output_dir.mkdir(parents=True, exist_ok=True)
        exported_audios = {}
        
        print(f"Exporting individual speaker audio files to: {output_dir}")

        for speaker_id, segments in speaker_segments_map.items():
            speaker_audio_parts = []
            for i, (start, end) in enumerate(segments):
                output_part_path = output_dir / f"{original_audio_path.stem}_{speaker_id}_part_{i}.wav"
                
                # FFmpeg command to extract segment
                cmd = [
                    str(self.ffmpeg_path), # Use FFmpeg path from base analyzer
                    "-i", str(original_audio_path),
                    "-ss", str(start),
                    "-to", str(end),
                    "-ar", "16000", # Resample to 16kHz for consistency
                    "-ac", "1",    # Mono audio
                    "-y", # Overwrite if exists
                    str(output_part_path)
                ]
                
                try:
                    subprocess.run(cmd, capture_output=True, text=True, check=True)
                    speaker_audio_parts.append(str(output_part_path))
                except subprocess.CalledProcessError as e:
                    print(f"❌ Error extracting audio segment for {speaker_id} part {i}: {e.stderr}")
                    continue
            
            if speaker_audio_parts:
                # Concatenate all parts for this speaker if there are multiple segments
                if len(speaker_audio_parts) > 1:
                    list_file_path = output_dir / f"{speaker_id}_list.txt"
                    with open(list_file_path, "w") as f:
                        for part in speaker_audio_parts:
                            f.write(f"file '{part}'\n")
                    
                    final_speaker_audio_path = output_dir / f"{original_audio_path.stem}_{speaker_id}.wav"
                    concat_cmd = [
                        str(self.ffmpeg_path),
                        "-f", "concat",
                        "-safe", "0", # Required for concat with absolute paths
                        "-i", str(list_file_path),
                        "-c", "copy",
                        "-y",
                        str(final_speaker_audio_path)
                    ]
                    try:
                        subprocess.run(concat_cmd, capture_output=True, text=True, check=True)
                        exported_audios[speaker_id] = final_speaker_audio_path
                        print(f"✅ Exported {speaker_id} audio: {final_speaker_audio_path.name}")
                        # Clean up temporary part files
                        for part in speaker_audio_parts:
                            Path(part).unlink()
                        list_file_path.unlink()
                    except subprocess.CalledProcessError as e:
                        print(f"❌ Error concatenating audio for {speaker_id}: {e.stderr}")
                else:
                    # If only one part, just use that
                    exported_audios[speaker_id] = Path(speaker_audio_parts[0])
                    print(f"✅ Exported {speaker_id} audio: {Path(speaker_audio_parts[0]).name}")

        return exported_audios


    def analyze_multi_speaker_content(self, video_path: str, project_name: str) -> Dict:
        """
        Orchestrates the multi-speaker analysis pipeline.
        Implements Stage 1 (Demucs Voice Separation) and Stage 2 (pyannote.audio Speaker Identification).
        """
        print(f"\n✨ Starting Multi-Speaker Analysis for: {project_name} ({video_path})")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = self.platform_root / "output" / f"{project_name}_MultiSpeaker_{Path(video_path).stem}_{timestamp}"
        output_dir.mkdir(parents=True, exist_ok=True)

        results = {
            'project_name': project_name,
            'status': 'started',
            'pipeline_stages': {}
        }

        try:
            # Stage 0: Extract Audio
            print("🎬 Extracting audio from video...")
            audio_path = self.extract_audio(Path(video_path), output_dir)
            if not audio_path:
                raise Exception("Audio extraction failed.")
            results['pipeline_stages']['audio_extraction'] = str(audio_path)
            print("✅ Audio extraction complete.")

            # Stage 1: Voice Separation (Demucs)
            isolated_vocals_path = self.separate_voices_with_demucs(audio_path, output_dir)
            if not isolated_vocals_path:
                raise Exception("Voice separation with Demucs failed.")
            results['pipeline_stages']['voice_separation'] = str(isolated_vocals_path)
            print("✅ Voice separation complete.")

            # Stage 2: Speaker Identification (pyannote.audio)
            individual_speaker_audios = self.identify_speakers_with_pyannote(isolated_vocals_path, output_dir)
            if not individual_speaker_audios:
                raise Exception("Speaker identification with pyannote.audio failed or found no speakers.")
            results['pipeline_stages']['speaker_identification'] = {
                speaker_id: str(path) for speaker_id, path in individual_speaker_audios.items()
            }
            print(f"✅ Speaker identification complete. Found {len(individual_speaker_audios)} speakers.")

            # --- Future Stages (Placeholders) ---
            # Stage 3: Individual Voice Analysis (Enhanced openSMILE)
            # individual_profiles = {}
            # for speaker_id, audio_segment_path in individual_speaker_audios.items():
            #     profile = self.analyze_creator(audio_segment_path, f"{project_name}_{speaker_id}") # Assuming analyze_creator takes audio path
            #     individual_profiles[speaker_id] = profile
            # results['pipeline_stages']['individual_analysis'] = individual_profiles

            # Stage 4: Combined Multi-Speaker Report
            # comparative_report = self.generate_speaker_comparison(individual_profiles, output_dir)
            # results['pipeline_stages']['comparative_report'] = comparative_report

            results['status'] = 'completed_stage2'
            print(f"\n✨ Multi-Speaker Pipeline (Stage 2) complete for {project_name}.")
            return results

        except Exception as e:
            results['status'] = 'failed'
            results['error'] = str(e)
            print(f"❌ Multi-Speaker Pipeline failed: {e}")
            return results

# Example Usage (for testing this file directly)
if __name__ == "__main__":
    print("--- Testing MultiSpeakerVoiceAnalyzer (Stage 1 & 2) ---")
    test_video_file = "C:/Attune-Voice-Platform/input/test_multi_speaker.mp4" # <--- IMPORTANT: Ensure this path is correct and file exists
    test_project_name = "MultiSpeakerTest"

    if not Path(test_video_file).exists():
        print(f"❌ Test video not found: {test_video_file}. Please create this file or update the path.")
        print("💡 You can place a sample multi-speaker video in C:/Attune-Voice-Platform/input/")
    else:
        analyzer = MultiSpeakerVoiceAnalyzer(platform_root="C:/Attune-Voice-Platform")
        analysis_result = analyzer.analyze_multi_speaker_content(test_video_file, test_project_name)
        print("\nMulti-Speaker Analysis Result:")
        print(json.dumps(analysis_result, indent=2))