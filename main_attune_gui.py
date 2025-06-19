import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from pathlib import Path
import sys
import subprocess
import json
import pandas as pd
from datetime import datetime

# Import components from your existing files
# Ensure these are correctly placed relative to main_attune_gui.py
from attune_analyzer import AttuneVoiceAnalyzer
from config.settings import Config
from src.youtube_processor import YouTubeVoiceProcessor # <--- NEW IMPORT

class MainAttuneGUI:
    def __init__(self, master):
        self.master = master
        master.title("Attune Voice Platform - Unified Analyzer")
        master.geometry("1000x700")

        self.analyzer = AttuneVoiceAnalyzer(platform_root=Config.WORKSPACE)
        self.yt_processor = YouTubeVoiceProcessor(workspace_path=Config.WORKSPACE) # <--- NEW: Initialize YouTube Processor

        self.setup_gui()
        self.log("Welcome to Attune Voice Platform!")
        self.log(f"Platform Root: {self.analyzer.platform_root}")
        self.refresh_video_list()

    def setup_gui(self):
        # Create a main notebook (tabbed interface)
        self.notebook = ttk.Notebook(self.master)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Analysis Tab
        self.analysis_frame = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(self.analysis_frame, text="Local Video Analysis") # Renamed tab
        self.setup_analysis_tab()

        # YouTube Analysis Tab <--- NEW TAB
        self.youtube_frame = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(self.youtube_frame, text="YouTube Analysis")
        self.setup_youtube_tab()

        # Tools Tab (for launching other standalone functionalities)
        self.tools_frame = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(self.tools_frame, text="Standalone Tools") # Renamed tab
        self.setup_tools_tab()

        # Output Log
        self.log_frame = ttk.LabelFrame(self.master, text="Application Log", padding="10")
        self.log_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))

        self.log_text = tk.Text(self.log_frame, wrap=tk.WORD, state=tk.DISABLED, font=('Consolas', 10))
        log_scrollbar = ttk.Scrollbar(self.log_frame, orient=tk.VERTICAL, command=self.log_text.yview)
        self.log_text.config(yscrollcommand=log_scrollbar.set)
        self.log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        log_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    def setup_analysis_tab(self):
        # Input Section
        input_frame = ttk.LabelFrame(self.analysis_frame, text="Local Video Input", padding="10")
        input_frame.pack(fill=tk.X, pady=10)

        ttk.Label(input_frame, text="Select Video:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        self.video_path_var = tk.StringVar()
        ttk.Entry(input_frame, textvariable=self.video_path_var, width=60).grid(row=0, column=1, padx=5, pady=5)
        ttk.Button(input_frame, text="Browse", command=self.browse_local_video).grid(row=0, column=2, padx=5, pady=5)

        ttk.Label(input_frame, text="Creator Name:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        self.creator_name_var = tk.StringVar()
        ttk.Entry(input_frame, textvariable=self.creator_name_var, width=30).grid(row=1, column=1, sticky=tk.W, padx=5, pady=5)

        ttk.Button(input_frame, text="Analyze Local Video", command=self.run_local_analysis).grid(row=2, column=0, columnspan=3, pady=10)

        # Available Videos Section
        available_videos_frame = ttk.LabelFrame(self.analysis_frame, text="Available Videos in Input Folder", padding="10")
        available_videos_frame.pack(fill=tk.BOTH, expand=True, pady=10)

        self.video_listbox = tk.Listbox(available_videos_frame, height=10)
        self.video_listbox.pack(fill=tk.BOTH, expand=True)
        self.video_listbox.bind("<<ListboxSelect>>", self.on_local_video_select)

    def setup_youtube_tab(self): # <--- NEW METHOD
        yt_input_frame = ttk.LabelFrame(self.youtube_frame, text="YouTube Video Input", padding="10")
        yt_input_frame.pack(fill=tk.X, pady=10)

        ttk.Label(yt_input_frame, text="YouTube URL:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        self.yt_url_var = tk.StringVar()
        ttk.Entry(yt_input_frame, textvariable=self.yt_url_var, width=70).grid(row=0, column=1, padx=5, pady=5)

        ttk.Label(yt_input_frame, text="Creator Name (Optional):").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        self.yt_creator_name_var = tk.StringVar()
        ttk.Entry(yt_input_frame, textvariable=self.yt_creator_name_var, width=30).grid(row=1, column=1, sticky=tk.W, padx=5, pady=5)

        ttk.Button(yt_input_frame, text="Download & Analyze YouTube Video", command=self.run_youtube_analysis).grid(row=2, column=0, columnspan=2, pady=10)

        # Optional: Add a button for channel analysis later if desired
        # ttk.Button(yt_input_frame, text="Analyze YouTube Channel (Coming Soon)", state=tk.DISABLED).grid(row=3, column=0, columnspan=2, pady=5)


    def setup_tools_tab(self):
        ttk.Label(self.tools_frame, text="These tools will launch in separate windows.", font=('Arial', 10, 'italic')).pack(pady=10)
        ttk.Button(self.tools_frame, text="Open ARFF Fixer", command=self.open_arff_fixer).pack(fill=tk.X, pady=5, padx=10)
        ttk.Button(self.tools_frame, text="Open Data Inspector", command=self.open_data_inspector).pack(fill=tk.X, pady=5, padx=10)
        ttk.Button(self.tools_frame, text="Open JSON Reader", command=self.open_json_reader).pack(fill=tk.X, pady=5, padx=10)
        ttk.Button(self.tools_frame, text="Open Voice Pattern Analyzer", command=self.open_voice_pattern_analyzer).pack(fill=tk.X, pady=5, padx=10)


    def log(self, message):
        self.log_text.config(state=tk.NORMAL)
        self.log_text.insert(tk.END, message + "\n")
        self.log_text.see(tk.END)
        self.log_text.config(state=tk.DISABLED)
        self.master.update_idletasks() # Ensure log updates immediately

    def browse_local_video(self): # Renamed method
        video_file = filedialog.askopenfilename(
            title="Select Local Video File",
            filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv"), ("All files", "*.*")],
            initialdir=self.analyzer.platform_root / "input"
        )
        if video_file:
            self.video_path_var.set(video_file)
            self.creator_name_var.set(Path(video_file).stem)
            self.log(f"Selected local video: {Path(video_file).name}")

    def refresh_video_list(self):
        self.video_listbox.delete(0, tk.END)
        videos = self.analyzer.list_videos()
        if not videos:
            self.log(f"No videos found in {self.analyzer.platform_root / 'input'}")
            return

        for i, video in enumerate(videos):
            self.video_listbox.insert(tk.END, video.name)
        self.log(f"Found {len(videos)} videos in input folder.")

    def on_local_video_select(self, event): # Renamed method
        selected_index = self.video_listbox.curselection()
        if selected_index:
            video_name = self.video_listbox.get(selected_index[0])
            full_path = self.analyzer.platform_root / "input" / video_name
            self.video_path_var.set(str(full_path))
            self.creator_name_var.set(Path(video_name).stem)
            self.log(f"Selected from local list: {video_name}")

    def run_local_analysis(self): # Renamed method
        video_path_str = self.video_path_var.get()
        creator_name = self.creator_name_var.get()

        if not video_path_str:
            messagebox.showwarning("Input Error", "Please select a local video file.")
            return
        if not creator_name:
            messagebox.showwarning("Input Error", "Please enter a creator name.")
            return

        video_file = Path(video_path_str)
        if not video_file.exists():
            messagebox.showerror("File Error", f"Video file not found: {video_file}")
            return

        self.log(f"Starting local video analysis for '{video_file.name}' by '{creator_name}'...")

        # Redirect stdout to capture analyze_video prints
        original_stdout = sys.stdout
        sys.stdout = self
        try:
            # The analyze_video method of AttuneVoiceAnalyzer expects a relative path if platform_root is set
            # Or an absolute path if platform_root is not governing the input dir.
            # Passing absolute path is generally safer for files outside 'input' but downloaded ones.
            relative_video_path = video_file.relative_to(self.analyzer.platform_root / "input") \
                                  if video_file.is_relative_to(self.analyzer.platform_root / "input") \
                                  else video_file.name

            result_dir = self.analyzer.analyze_video(relative_video_path, creator_name)

            if result_dir:
                messagebox.showinfo("Analysis Complete", f"Analysis successful! Results in: {result_dir}")
                self.log(f"Analysis complete for {creator_name}. Results in: {result_dir}")
            else:
                messagebox.showerror("Analysis Failed", "Local video analysis failed. Check log for details.")
                self.log(f"Local video analysis failed for {creator_name}.")
        except Exception as e:
            messagebox.showerror("Analysis Error", f"An unexpected error occurred during local analysis: {e}")
            self.log(f"ERROR during local analysis: {e}")
        finally:
            sys.stdout = original_stdout # Restore stdout

    def run_youtube_analysis(self): # <--- NEW METHOD
        youtube_url = self.yt_url_var.get()
        creator_name = self.yt_creator_name_var.get() # Optional creator name

        if not youtube_url:
            messagebox.showwarning("Input Error", "Please enter a YouTube URL.")
            return

        self.log(f"Starting YouTube analysis for URL: {youtube_url}")
        if creator_name:
            self.log(f"Using creator name: {creator_name}")

        # Redirect stdout to capture YouTube processor prints
        original_stdout = sys.stdout
        sys.stdout = self
        try:
            # Call the analyze_youtube_video method from the YouTubeVoiceProcessor instance
            results = self.yt_processor.analyze_youtube_video(youtube_url, creator_name)

            if results and results.get('status') == 'complete':
                messagebox.showinfo("YouTube Analysis Complete", f"YouTube video analysis successful! Results in: {results.get('report_path', 'N/A')}")
                self.log(f"YouTube analysis complete for {results.get('creator')}. Report: {results.get('report_path', 'N/A')}")
            else:
                messagebox.showerror("YouTube Analysis Failed", f"YouTube video analysis failed. Error: {results.get('error', 'Unknown error')}. Check log for details.")
                self.log(f"YouTube analysis failed for {results.get('creator', 'N/A')}. Error: {results.get('error', 'Unknown error')}")
        except Exception as e:
            messagebox.showerror("YouTube Analysis Error", f"An unexpected error occurred during YouTube analysis: {e}")
            self.log(f"ERROR during YouTube analysis: {e}")
        finally:
            sys.stdout = original_stdout # Restore stdout


    def write(self, text):
        """Method to redirect stdout to the log_text widget."""
        self.log_text.config(state=tk.NORMAL)
        self.log_text.insert(tk.END, text)
        self.log_text.see(tk.END)
        self.log_text.config(state=tk.DISABLED)
        self.master.update_idletasks() # Ensure log updates immediately

    def flush(self):
        """Required for file-like object."""
        pass

    def open_arff_fixer(self):
        self.log("Launching ARFF Fixer...")
        try:
            # Use subprocess to run the other GUI scripts as separate processes
            subprocess.Popen([sys.executable, str(self.analyzer.platform_root / "arff_fixer.py")])
            self.log("ARFF Fixer launched successfully.")
        except Exception as e:
            self.log(f"Failed to launch ARFF Fixer: {e}")
            messagebox.showerror("Launch Error", f"Failed to launch ARFF Fixer: {e}")

    def open_data_inspector(self):
        self.log("Launching Data Inspector...")
        try:
            subprocess.Popen([sys.executable, str(self.analyzer.platform_root / "data_inspector.py")])
            self.log("Data Inspector launched successfully.")
        except Exception as e:
            self.log(f"Failed to launch Data Inspector: {e}")
            messagebox.showerror("Launch Error", f"Failed to launch Data Inspector: {e}")

    def open_json_reader(self):
        self.log("Launching JSON Reader...")
        try:
            subprocess.Popen([sys.executable, str(self.analyzer.platform_root / "json_reader.py")])
            self.log("JSON Reader launched successfully.")
        except Exception as e:
            self.log(f"Failed to launch JSON Reader: {e}")
            messagebox.showerror("Launch Error", f"Failed to launch JSON Reader: {e}")

    def open_voice_pattern_analyzer(self):
        self.log("Launching Voice Pattern Analyzer...")
        try:
            subprocess.Popen([sys.executable, str(self.analyzer.platform_root / "voice_pattern_analyzer.py")])
            self.log("Voice Pattern Analyzer launched successfully.")
        except Exception as e:
            self.log(f"Failed to launch Voice Pattern Analyzer: {e}")
            messagebox.showerror("Launch Error", f"Failed to launch Voice Pattern Analyzer: {e}")

def main():
    root = tk.Tk()
    app = MainAttuneGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()