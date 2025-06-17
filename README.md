# 🎤 Attune Voice Platform

**Mathematical Voice Analysis & Archetype Discovery for Creator Development**

[![Python](https://img.shields.io/badge/Python-3.13+-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Voice Features](https://img.shields.io/badge/Voice%20Features-88-brightgreen.svg)](docs/features-guide.md)

> Democratizing professional voice coaching through scientific acoustic analysis

---

## 🎯 Overview

Attune transforms expensive voice coaching ($150-500/hour) into accessible, data-driven insights for content creators. Using academic-grade acoustic analysis, it extracts **88 mathematical features** from voice recordings to identify distinct voice archetypes and provide personalized coaching recommendations.

### ✨ Key Features

- 🎼 **88-Feature Voice Analysis** using openSMILE eGeMAPS
- 🎭 **Voice Archetype Discovery** with machine learning clustering  
- 📊 **Interactive Pattern Visualization** with real-time filtering
- 🎯 **Creator Coaching Insights** based on successful voice patterns
- 📈 **Multi-format Data Export** (JSON, CSV, Excel)
- 🔍 **Comparative Voice Analysis** for archetype classification

---

## 🚀 Quick Start

### Prerequisites

```bash
# Python 3.13+ required
python --version

# Install dependencies
pip install pandas numpy matplotlib seaborn scikit-learn openpyxl
```

### Setup External Tools

1. **Download openSMILE** from [GitHub releases](https://github.com/audeering/opensmile/releases)
2. **Extract to:** `tools/opensmile/`
3. **Download FFmpeg** from [ffmpeg.org](https://ffmpeg.org/download.html)
4. **Place ffmpeg.exe in:** `tools/ffmpeg/`

### Run Analysis

```bash
# 1. Analyze a video
python attune_analyzer.py

# 2. Explore voice patterns  
python voice_pattern_analyzer.py

# 3. Export and view data
python json_reader.py
```

---

## 🎼 What Gets Analyzed

### Voice Feature Categories (88 Total)

| Category | Features | What It Reveals |
|----------|----------|-----------------|
| **🎵 F0/Pitch** | 10 features | Voice character & expressiveness |
| **🔊 Loudness** | 10 features | Energy & audience engagement |
| **🎨 Spectral** | 18 features | Voice quality & uniqueness |
| **💎 Voice Quality** | 6 features | Professional delivery indicators |
| **⏱️ Temporal** | 14+ features | Rhythm & communication style |
| **🔬 Advanced** | 30+ features | Detailed acoustic characteristics |

### Example Features Extracted

```python
# Pitch & Character
F0semitoneFrom27.5Hz_sma3nz_amean: 32.35  # Average pitch
F0semitoneFrom27.5Hz_sma3nz_stddevNorm: 0.12  # Pitch variation

# Energy & Presence  
loudness_sma3_amean: 0.51  # Average loudness
loudness_sma3_stddevNorm: 0.70  # Energy dynamics

# Voice Quality
jitterLocal_sma3nz_amean: 0.016  # Voice steadiness
HNRdBACF_sma3nz_amean: 21.47  # Voice clarity
```

---

## 🎭 Voice Archetypes Discovered

### Supported Creator Voice Types

| Archetype | Characteristics | Use Cases |
|-----------|-----------------|-----------|
| **🎯 Authority Figure** | Low pitch, controlled energy, professional quality | News anchors, business leaders |
| **⚡ Enthusiastic Host** | High variation, dynamic energy, engaging patterns | YouTubers, podcast hosts |
| **📖 Intimate Storyteller** | Warm tone, gentle dynamics, flowing rhythm | Audiobook narrators, meditation guides |
| **🎓 Professional Educator** | Structured delivery, clear articulation | Online instructors, course creators |
| **👥 Conversational Friend** | Natural variation, authentic patterns | Casual podcasters, lifestyle creators |

---

## 📊 Analysis Pipeline

```mermaid
graph LR
    A[Video Input] --> B[Audio Extraction]
    B --> C[openSMILE Analysis]
    C --> D[88 Features]
    D --> E[Pattern Recognition]
    E --> F[Archetype Classification]
    F --> G[Coaching Insights]
```

### Technical Flow

1. **Video → Audio**: FFmpeg extracts 16kHz mono audio
2. **Acoustic Analysis**: openSMILE eGeMAPS feature extraction
3. **Pattern Discovery**: ML clustering identifies voice relationships
4. **Visualization**: Interactive charts reveal voice characteristics
5. **Export**: JSON/CSV/Excel formats for further analysis

---

## 🛠️ Platform Architecture

```
Attune-Voice-Platform/
├── 🎬 input/                    # Video files for analysis
├── 📊 output/                   # Analysis results & reports
├── 🔧 tools/                    # External tools (openSMILE, FFmpeg)
├── ⚙️ config/                   # Configuration files
├── 📝 docs/                     # Documentation
├── 🎤 attune_analyzer.py        # Main analysis engine
├── 📈 voice_pattern_analyzer.py # Visual pattern interface
├── 📋 json_reader.py           # Data export & conversion
└── 🔧 arff_fixer.py            # ARFF format processor
```

---

## 🎯 Use Cases & Applications

### 🎬 Creator Development
- **Voice coaching** for YouTubers and podcasters
- **Delivery improvement** through data-driven insights
- **Archetype matching** to successful creators in same niche
- **Voice evolution tracking** over creator career

### 🏢 Brand & Business
- **Brand voice consistency** across multiple platforms
- **Team voice training** for consistent delivery
- **Executive presence** development for leadership
- **Customer service** voice optimization

### 🎮 Entertainment & Media
- **Character voice design** for gaming and film
- **Voice acting** consistency across projects
- **Dialogue optimization** for narrative content
- **Localization** voice matching across languages

### 🎓 Education & Research
- **Communication skills** assessment and training
- **Academic voice development** for presentations
- **Voice pattern research** and linguistic studies
- **Automated assessment** tools for education

---

## 📈 Sample Analysis Output

### Voice Profile Summary
```json
{
  "creator": "Sample Creator",
  "voice_archetype": "Enthusiastic Host",
  "confidence_score": 0.87,
  "key_characteristics": {
    "pitch_range": "High variation (engaging)",
    "energy_level": "Dynamic and expressive", 
    "voice_quality": "Professional clarity",
    "rhythm_pattern": "Natural conversational flow"
  },
  "coaching_recommendations": [
    "Maintain current energy levels - highly engaging",
    "Consider slight pitch range expansion for emphasis",
    "Voice quality excellent - no technical issues"
  ]
}
```

---

## 🚀 Getting Started Examples

### Analyze Your First Video

```python
from attune_analyzer import AttuneVoiceAnalyzer

# Initialize analyzer
analyzer = AttuneVoiceAnalyzer()

# Analyze a creator video
result = analyzer.analyze_video("my_video.mp4", "Creator Name")

# View results
print(f"Analysis complete: {result}")
```

### Explore Voice Patterns

```python
from voice_pattern_analyzer import VoicePatternAnalyzer

# Start visual analysis
analyzer = VoicePatternAnalyzer()
analyzer.run()  # Opens interactive GUI
```

### Export Data for Research

```python
from json_reader import JSONVoiceDataReader

# Convert analysis to Excel/CSV
reader = JSONVoiceDataReader()
reader.run()  # Opens conversion interface
```

---

## 🔬 Scientific Foundation

### Academic Validation
- **openSMILE eGeMAPS**: Scientifically validated feature set
- **88 Features**: Proven effective for emotion and voice analysis
- **Statistical Methods**: Machine learning clustering and correlation analysis
- **Reproducible Results**: Mathematical precision vs. subjective assessment

### Research Applications
- Voice pattern discovery in creator content
- Correlation analysis between voice characteristics and engagement
- Longitudinal studies of creator voice evolution
- Cross-cultural voice archetype research

---

## 🤝 Contributing

We welcome contributions to expand voice analysis capabilities:

### Areas for Enhancement
- **Additional Archetypes**: Help define new creator voice types
- **Feature Engineering**: Develop new acoustic measurements
- **Visualization**: Enhance pattern discovery interfaces  
- **Integration**: Connect with creator platforms and tools
- **Validation**: Test archetype accuracy across niches

### Development Setup

```bash
# Clone repository
git clone https://github.com/YOUR_USERNAME/attune-voice-platform.git
cd attune-voice-platform

# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
python -m pytest tests/
```

---

## 📄 Documentation

- **[Feature Guide](docs/features-guide.md)**: Complete explanation of 88 voice features
- **[API Reference](docs/api-reference.md)**: Function documentation
- **[Archetype Guide](docs/archetypes.md)**: Voice type classification system
- **[Setup Guide](docs/setup.md)**: Detailed installation instructions
- **[Research Notes](docs/research.md)**: Scientific background and validation

---

## 🎤 Real-World Impact

### Target Market
- **10M+ content creators** seeking voice improvement
- **Creator economy** valued at $104B+ annually
- **Voice coaching market** traditionally limited to high-budget productions
- **Democratic access** to professional-grade analysis tools

### Success Metrics
- Voice coaching accessible at **$29/month vs $500/hour**
- **Mathematical precision** vs subjective feedback
- **Scalable delivery** for millions of creators simultaneously  
- **Data-driven insights** for measurable improvement

---

## 📞 Support & Community

- **Issues**: [GitHub Issues](https://github.com/JakeD520/attune-voice-platform/issues)
- **Discussions**: [GitHub Discussions](https://github.com/JakeD520/attune-voice-platform/discussions)
- **Email**: [jd01234.com](mailto:jd01234@gmail.com)

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🏆 Acknowledgments

- **openSMILE** team at audEERING for acoustic analysis framework
- **eGeMAPS** researchers for validated feature set
- **Creator community** for inspiration and validation
- **Open source contributors** making voice analysis accessible

---

<div align="center">

**Built for the creator economy - Democratizing professional voice analysis**

[⭐ Star this repo](https://github.com/YOUR_USERNAME/attune-voice-platform) if you find it useful!

</div>
