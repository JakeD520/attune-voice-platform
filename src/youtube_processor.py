import yt_dlp
from pathlib import Path
from typing import List, Dict, Any
import re
import sys
import json

# --- Start: Fix for relative import when running directly ---
current_file_path = Path(__file__).resolve()
project_root = current_file_path.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
# --- End: Fix for relative import when running directly ---

from attune_analyzer import AttuneVoiceAnalyzer
from config.settings import Config

class YouTubeVoiceProcessor:
    """
    Professional YouTube video processing for voice analysis
    """

    def __init__(self, workspace_path: str = None):
        self.analyzer = AttuneVoiceAnalyzer(platform_root=Config.WORKSPACE)
        self.download_dir = self.analyzer.platform_root / "data" / "raw" / "youtube"
        self.download_dir.mkdir(parents=True, exist_ok=True)
        print(f"✅ YouTube download directory: {self.download_dir}")

    def extract_creator_name(self, url: str) -> str:
        """
        Extracts the creator/uploader name from a YouTube URL using yt-dlp's info extraction.
        Falls back to a generic name if extraction fails.
        """
        try:
            ydl_opts = {
                'quiet': True,
                'no_warnings': True,
                'force_generic_extractor': True,
            }
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=False)
                creator_name = info.get('uploader') or info.get('channel') or info.get('uploader_id')

                if creator_name:
                    creator_name = re.sub(r'[^\w\s-]', '', creator_name).strip().replace(' ', '_')
                    return creator_name if creator_name else "unknown_creator"
                else:
                    return "unknown_creator"
        except Exception as e:
            print(f"⚠️ Could not extract creator name from URL: {e}. Using 'unknown_creator'.")
            return "unknown_creator"

    def download_video(self, url: str, creator_name: str) -> Path:
        """
        Downloads the best quality video (up to 720p) and extracts audio if necessary.
        Returns the path to the downloaded video file.
        """
        output_template = str(self.download_dir / f'{creator_name}_%(id)s.%(ext)s')

        ydl_opts = {
            'format': 'bestvideo[height<=720]+bestaudio/best[height<=720]',
            'outtmpl': output_template,
            'merge_output_format': 'mp4',
            'noplaylist': True,
            'progress_hooks': [self._download_progress_hook],
            'postprocessors': [{
                'key': 'FFmpegVideoRemuxer',
                'preferedformat': 'mp4',
            }],
            'retries': 3,
        }

        downloaded_file_path = None
        try:
            print(f"⬇️ Downloading YouTube video from: {url}")
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                download_result = ydl.extract_info(url, download=True)
                if 'requested_downloads' in download_result and download_result['requested_downloads']:
                    downloaded_file_path = Path(download_result['requested_downloads'][0]['filepath'])
                else:
                    downloaded_file_path = Path(ydl.prepare_filename(download_result, outtmpl=output_template))
                    if ydl_opts['merge_output_format'] and not str(downloaded_file_path).endswith(ydl_opts['merge_output_format']):
                        downloaded_file_path = downloaded_file_path.with_suffix(f".{ydl_opts['merge_output_format']}")

                if downloaded_file_path.exists():
                    print(f"✅ Downloaded: {downloaded_file_path.name}")
                    return downloaded_file_path
                else:
                    raise FileNotFoundError(f"Downloaded file not found at expected path: {downloaded_file_path}")

        except Exception as e:
            print(f"❌ Error downloading YouTube video: {e}")
            raise

    def _download_progress_hook(self, d):
        """Internal hook for yt-dlp download progress."""
        if d['status'] == 'downloading':
            print(f"   Downloading: {d['_percent_str']} of {d['_total_bytes_str']} at {d['_speed_str']} ETA {d['_eta_str']}")
        elif d['status'] == 'finished':
            print(f"   Finished download: {d['filename']}")

    def analyze_youtube_video(self, url: str, creator_name: str = None) -> Dict:
        """Download and analyze a YouTube video."""
        print(f"\n🚀 Starting YouTube video analysis for URL: {url}")

        if not creator_name:
            creator_name = self.extract_creator_name(url)
            print(f"   Inferred creator name: {creator_name}")

        try:
            video_path = self.download_video(url, creator_name)

            print(f"   Passing downloaded video to AttuneVoiceAnalyzer: {video_path}")
            analysis_results_dir = self.analyzer.analyze_video(str(video_path), creator_name)

            if analysis_results_dir:
                report_path = analysis_results_dir / "report.json"
                if report_path.exists():
                    with open(report_path, 'r') as f:
                        report_data = json.load(f)
                    report_data['youtube_url'] = url
                    report_data['downloaded_video_path'] = str(video_path)
                    report_data['status'] = 'complete' # <--- ADD THIS LINE
                    with open(report_path, 'w') as f:
                        json.dump(report_data, f, indent=2)
                    print(f"✅ Updated report.json with YouTube info for {creator_name}")
                    return report_data
                else:
                    print("⚠️ Analysis completed but report.json not found.")
                    # Ensure a status is returned even if report.json is missing but analysis finished
                    return {"status": "complete", "creator": creator_name, "video_path": str(video_path), "report_missing": True}
            else:
                print("❌ AttuneVoiceAnalyzer did not return a successful analysis result.")
                return {"status": "failed", "creator": creator_name, "video_path": str(video_path), "error": "Analysis failed by core analyzer"}

        except Exception as e:
            print(f"❌ YouTube analysis pipeline failed: {e}")
            return {"status": "failed", "creator": creator_name, "error": str(e)}

    def analyze_creator_channel(self, channel_url: str, max_videos: int = 10) -> List[Dict]:
        """
        Analyze multiple videos from a creator's channel.
        This is a more advanced feature and would require iterating through channel videos.
        For now, it's a placeholder.
        """
        print(f"🚧 Channel analysis feature not yet implemented.")
        return []

# Simple usage for testing (if run directly)
if __name__ == "__main__":
    test_url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"

    yt_processor = YouTubeVoiceProcessor()

    try:
        print(f"\n--- Testing YouTubeVoiceProcessor directly with {test_url} ---")
        results = yt_processor.analyze_youtube_video(test_url, "TestCreatorDirectRun")
        print("\nYouTube Analysis Results:")
        print(json.dumps(results, indent=2))
    except Exception as e:
        print(f"\nFailed to analyze YouTube video during direct run test: {e}")