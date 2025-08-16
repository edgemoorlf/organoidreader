# OrganoidReader

An AI-powered organoids image analysis system for automated identification and quantitative analysis of drug-treated organoid images. The system provides comprehensive assessment of growth status, cell viability, apoptosis detection, and drug efficacy evaluation.

## AI & Technical Capabilities

### Core AI Architecture
- **Deep Learning Framework**: PyTorch/TensorFlow implementation
- **Segmentation Models**: 
  - Multi-scale U-Net (>85% accuracy demonstrated)
  - Mask R-CNN variants for instance segmentation
  - RDAU-Net with dynamic convolution and attention mechanisms
  - YOLOX-based detection for lightweight processing
- **Classification Models**: VGG19, ResNet50v2, DenseNet121, Xception
- **Foundation Model Integration**: SAM/ViT-based approaches for enhanced accuracy

### Advanced Analysis Features
- **Organoid Segmentation**: Deep learning-based boundary detection with >90% target accuracy
- **Viability Analysis**: Live/dead cell classification and membrane integrity assessment
- **Apoptosis Detection**: Morphological feature recognition and nuclear fragmentation detection
- **Time Series Analysis**: Temporal tracking and growth trend modeling
- **Multi-Modal Support**: Bright-field, fluorescence, and confocal microscopy

### Technical Performance
- **Target Accuracy**: ≥90% organoid detection (industry-leading vs current 80-85%)
- **Processing Speed**: <30 seconds per image
- **Batch Processing**: >100 images capability
- **Real-time Analysis**: GPU-accelerated with CUDA optimization

### Computer Vision Pipeline
- **Image Processing**: OpenCV, scikit-image integration
- **Multi-format Support**: TIFF, JPEG, PNG compatibility
- **Preprocessing**: Noise reduction, contrast enhancement, standardization
- **Feature Extraction**: Quantitative morphological parameter calculation

### Research & Development Edge
- **Foundation Model Innovation**: First comprehensive SAM-based organoid solution
- **3D Structure Analysis**: Multi-focal plane processing and temporal tracking
- **Cross-Platform Compatibility**: Hardware-agnostic preprocessing and analysis
- **Standardized Pipeline**: Addressing current field fragmentation

## Installation & Setup

### System Requirements
- **OS**: Windows 10+, macOS 10.15+, or Linux (Ubuntu 18.04+)
- **Python**: 3.8 or higher
- **GPU**: NVIDIA GPU with CUDA support (recommended for optimal performance)
- **RAM**: 16GB minimum, 32GB recommended
- **Storage**: 10GB free space for models and datasets

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-org/organoidreader.git
   cd organoidreader
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download pre-trained models**
   ```bash
   python scripts/download_models.py
   ```

### Quick Start

1. **Launch the application**
   ```bash
   python main.py
   ```

2. **Process single image**
   ```bash
   python -m organoidreader.cli analyze --input image.tiff --output results/
   ```

3. **Batch processing**
   ```bash
   python -m organoidreader.cli batch --input-dir images/ --output-dir results/
   ```

### Configuration

Create a `config.yaml` file to customize analysis parameters:
```yaml
segmentation:
  model: "unet_v2"
  confidence_threshold: 0.8
  
analysis:
  viability_analysis: true
  apoptosis_detection: true
  time_series: false

output:
  save_masks: true
  generate_reports: true
  export_format: "csv"
```

### Development Setup

1. **Install development dependencies**
   ```bash
   pip install -r requirements-dev.txt
   ```

2. **Run tests**
   ```bash
   python -m pytest tests/
   ```

3. **Code formatting**
   ```bash
   black organoidreader/
   ruff organoidreader/
   ```

## Usage Examples

### Python API
```python
from organoidreader import OrganoidAnalyzer

analyzer = OrganoidAnalyzer()
results = analyzer.analyze_image("sample.tiff")
print(f"Detected {results.organoid_count} organoids")
```

### Command Line Interface
```bash
# Basic analysis
organoidreader analyze --input sample.tiff

# Advanced analysis with custom parameters
organoidreader analyze --input sample.tiff --model unet_v2 --viability --apoptosis

# Generate detailed report
organoidreader report --input results/ --format pdf
```

## Documentation

- [API Documentation](docs/api.md)
- [User Guide](docs/user_guide.md)
- [Model Architecture](docs/models.md)
- [Contributing Guidelines](CONTRIBUTING.md)
