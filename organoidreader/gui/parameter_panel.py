"""
Parameter Panel for OrganoidReader

This module provides the parameter configuration panel for adjusting
analysis settings and viewing configuration options.
"""

import logging
from typing import Dict, Any, Optional

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QFormLayout,
    QSpinBox, QDoubleSpinBox, QCheckBox, QComboBox, QLineEdit,
    QPushButton, QFileDialog, QLabel, QSlider, QTabWidget,
    QScrollArea
)
from PyQt5.QtCore import Qt, pyqtSignal

from organoidreader.config.config_manager import Config

logger = logging.getLogger(__name__)


class ParameterPanel(QWidget):
    """
    Parameter configuration panel for analysis settings.
    
    Provides controls for adjusting preprocessing, segmentation,
    and analysis parameters.
    """
    
    # Signals
    parameters_changed = pyqtSignal()
    
    def __init__(self, config: Config, parent=None):
        super().__init__(parent)
        
        self.config = config
        self.parameter_widgets = {}
        
        self.init_ui()
        self.load_default_parameters()
    
    def init_ui(self):
        """Initialize the user interface."""
        layout = QVBoxLayout(self)
        
        # Create scroll area for parameters
        scroll_area = QScrollArea()
        scroll_widget = QWidget()
        scroll_layout = QVBoxLayout(scroll_widget)
        
        # Create parameter tabs
        self.tab_widget = QTabWidget()
        
        # Preprocessing parameters
        preprocessing_tab = self.create_preprocessing_tab()
        self.tab_widget.addTab(preprocessing_tab, "Preprocessing")
        
        # Segmentation parameters
        segmentation_tab = self.create_segmentation_tab()
        self.tab_widget.addTab(segmentation_tab, "Segmentation")
        
        # Analysis parameters
        analysis_tab = self.create_analysis_tab()
        self.tab_widget.addTab(analysis_tab, "Analysis")
        
        # Advanced parameters
        advanced_tab = self.create_advanced_tab()
        self.tab_widget.addTab(advanced_tab, "Advanced")
        
        scroll_layout.addWidget(self.tab_widget)
        scroll_area.setWidget(scroll_widget)
        scroll_area.setWidgetResizable(True)
        layout.addWidget(scroll_area)
        
        # Control buttons
        button_layout = QHBoxLayout()
        
        reset_btn = QPushButton("Reset to Default")
        reset_btn.clicked.connect(self.reset_parameters)
        button_layout.addWidget(reset_btn)
        
        load_btn = QPushButton("Load Config...")
        load_btn.clicked.connect(self.load_configuration)
        button_layout.addWidget(load_btn)
        
        save_btn = QPushButton("Save Config...")
        save_btn.clicked.connect(self.save_configuration)
        button_layout.addWidget(save_btn)
        
        button_layout.addStretch()
        layout.addLayout(button_layout)
    
    def create_preprocessing_tab(self) -> QWidget:
        """Create preprocessing parameters tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Image enhancement group
        enhancement_group = QGroupBox("Image Enhancement")
        enhancement_layout = QFormLayout(enhancement_group)
        
        # Contrast enhancement
        self.contrast_checkbox = QCheckBox()
        self.contrast_checkbox.setChecked(True)
        enhancement_layout.addRow("Enable Contrast Enhancement:", self.contrast_checkbox)
        
        self.contrast_limit = QDoubleSpinBox()
        self.contrast_limit.setRange(0.01, 0.1)
        self.contrast_limit.setValue(0.03)
        self.contrast_limit.setSingleStep(0.01)
        self.contrast_limit.setDecimals(3)
        enhancement_layout.addRow("Contrast Limit:", self.contrast_limit)
        
        # Noise reduction
        self.denoise_checkbox = QCheckBox()
        self.denoise_checkbox.setChecked(True)
        enhancement_layout.addRow("Enable Noise Reduction:", self.denoise_checkbox)
        
        self.denoise_strength = QDoubleSpinBox()
        self.denoise_strength.setRange(0.05, 0.3)
        self.denoise_strength.setValue(0.1)
        self.denoise_strength.setSingleStep(0.05)
        self.denoise_strength.setDecimals(2)
        enhancement_layout.addRow("Denoise Strength:", self.denoise_strength)
        
        layout.addWidget(enhancement_group)
        
        # Normalization group
        norm_group = QGroupBox("Normalization")
        norm_layout = QFormLayout(norm_group)
        
        self.normalize_checkbox = QCheckBox()
        self.normalize_checkbox.setChecked(True)
        norm_layout.addRow("Enable Normalization:", self.normalize_checkbox)
        
        self.norm_method = QComboBox()
        self.norm_method.addItems(["minmax", "zscore", "percentile"])
        self.norm_method.setCurrentText("percentile")
        norm_layout.addRow("Normalization Method:", self.norm_method)
        
        layout.addWidget(norm_group)
        
        layout.addStretch()
        
        # Register widgets
        self.parameter_widgets.update({\n            'contrast_enhancement': self.contrast_checkbox,
            'contrast_limit': self.contrast_limit,
            'denoise': self.denoise_checkbox,
            'denoise_strength': self.denoise_strength,
            'normalize': self.normalize_checkbox,
            'norm_method': self.norm_method\n        })
        
        return tab
    
    def create_segmentation_tab(self) -> QWidget:
        """Create segmentation parameters tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Model settings group
        model_group = QGroupBox("Model Settings")
        model_layout = QFormLayout(model_group)
        
        self.model_path = QLineEdit()
        self.model_path.setPlaceholderText("Leave empty to use default model")
        model_layout.addRow("Segmentation Model:", self.model_path)
        
        browse_btn = QPushButton("Browse...")
        browse_btn.clicked.connect(self.browse_segmentation_model)
        model_layout.addRow("", browse_btn)
        
        layout.addWidget(model_group)
        
        # Post-processing group
        postproc_group = QGroupBox("Post-processing")
        postproc_layout = QFormLayout(postproc_group)
        
        self.min_organoid_size = QSpinBox()
        self.min_organoid_size.setRange(10, 1000)
        self.min_organoid_size.setValue(100)
        postproc_layout.addRow("Min Organoid Size (pixels):", self.min_organoid_size)
        
        self.max_organoid_size = QSpinBox()
        self.max_organoid_size.setRange(1000, 50000)
        self.max_organoid_size.setValue(10000)
        postproc_layout.addRow("Max Organoid Size (pixels):", self.max_organoid_size)
        
        self.fill_holes = QCheckBox()
        self.fill_holes.setChecked(True)
        postproc_layout.addRow("Fill Holes:", self.fill_holes)
        
        layout.addWidget(postproc_group)
        
        layout.addStretch()
        
        # Register widgets
        self.parameter_widgets.update({
            'segmentation_model_path': self.model_path,
            'min_organoid_size': self.min_organoid_size,
            'max_organoid_size': self.max_organoid_size,
            'fill_holes': self.fill_holes
        })
        
        return tab
    
    def create_analysis_tab(self) -> QWidget:
        """Create analysis parameters tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Viability analysis group
        viability_group = QGroupBox("Viability Analysis")
        viability_layout = QFormLayout(viability_group)
        
        self.enable_viability = QCheckBox()
        self.enable_viability.setChecked(True)
        viability_layout.addRow("Enable Viability Analysis:", self.enable_viability)
        
        self.viability_model_path = QLineEdit()
        self.viability_model_path.setPlaceholderText("Leave empty to use default model")
        viability_layout.addRow("Viability Model:", self.viability_model_path)
        
        browse_viability_btn = QPushButton("Browse...")
        browse_viability_btn.clicked.connect(self.browse_viability_model)
        viability_layout.addRow("", browse_viability_btn)
        
        layout.addWidget(viability_group)
        
        # Apoptosis analysis group
        apoptosis_group = QGroupBox("Apoptosis Analysis")
        apoptosis_layout = QFormLayout(apoptosis_group)
        
        self.enable_apoptosis = QCheckBox()
        self.enable_apoptosis.setChecked(True)
        apoptosis_layout.addRow("Enable Apoptosis Detection:", self.enable_apoptosis)
        
        self.apoptosis_mode = QComboBox()
        self.apoptosis_mode.addItems(["morphological", "tunel", "combined"])
        self.apoptosis_mode.setCurrentText("morphological")
        apoptosis_layout.addRow("Detection Mode:", self.apoptosis_mode)
        
        self.nuclear_channel = QLineEdit()
        self.nuclear_channel.setPlaceholderText("Path to nuclear channel image (optional)")
        apoptosis_layout.addRow("Nuclear Channel:", self.nuclear_channel)
        
        browse_nuclear_btn = QPushButton("Browse...")
        browse_nuclear_btn.clicked.connect(self.browse_nuclear_channel)
        apoptosis_layout.addRow("", browse_nuclear_btn)
        
        layout.addWidget(apoptosis_group)
        
        # Parameter extraction group
        extraction_group = QGroupBox("Parameter Extraction")
        extraction_layout = QFormLayout(extraction_group)
        
        self.extract_morphological = QCheckBox()
        self.extract_morphological.setChecked(True)
        extraction_layout.addRow("Morphological Features:", self.extract_morphological)
        
        self.extract_intensity = QCheckBox()
        self.extract_intensity.setChecked(True)
        extraction_layout.addRow("Intensity Features:", self.extract_intensity)
        
        self.extract_texture = QCheckBox()
        self.extract_texture.setChecked(True)
        extraction_layout.addRow("Texture Features:", self.extract_texture)
        
        self.extract_spatial = QCheckBox()
        self.extract_spatial.setChecked(True)
        extraction_layout.addRow("Spatial Features:", self.extract_spatial)
        
        layout.addWidget(extraction_group)
        
        layout.addStretch()
        
        # Register widgets
        self.parameter_widgets.update({
            'enable_viability': self.enable_viability,
            'viability_model_path': self.viability_model_path,
            'enable_apoptosis': self.enable_apoptosis,
            'apoptosis_mode': self.apoptosis_mode,
            'nuclear_channel_path': self.nuclear_channel,
            'extract_morphological': self.extract_morphological,
            'extract_intensity': self.extract_intensity,
            'extract_texture': self.extract_texture,
            'extract_spatial': self.extract_spatial
        })
        
        return tab
    
    def create_advanced_tab(self) -> QWidget:
        """Create advanced parameters tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Performance group
        performance_group = QGroupBox("Performance")
        performance_layout = QFormLayout(performance_group)
        
        self.use_gpu = QCheckBox()
        self.use_gpu.setChecked(True)
        performance_layout.addRow("Use GPU (if available):", self.use_gpu)
        
        self.num_threads = QSpinBox()
        self.num_threads.setRange(1, 16)
        self.num_threads.setValue(4)
        performance_layout.addRow("Number of Threads:", self.num_threads)
        
        layout.addWidget(performance_group)
        
        # Quality control group
        quality_group = QGroupBox("Quality Control")
        quality_layout = QFormLayout(quality_group)
        
        self.min_quality_score = QDoubleSpinBox()
        self.min_quality_score.setRange(0.0, 1.0)
        self.min_quality_score.setValue(0.3)
        self.min_quality_score.setSingleStep(0.1)
        self.min_quality_score.setDecimals(2)
        quality_layout.addRow("Min Quality Score:", self.min_quality_score)
        
        self.exclude_border = QCheckBox()
        self.exclude_border.setChecked(True)
        quality_layout.addRow("Exclude Border Organoids:", self.exclude_border)
        
        layout.addWidget(quality_group)
        
        # Time series group
        timeseries_group = QGroupBox("Time Series")
        timeseries_layout = QFormLayout(timeseries_group)
        
        self.timestamp = QDoubleSpinBox()
        self.timestamp.setRange(0.0, 1000.0)
        self.timestamp.setValue(0.0)
        self.timestamp.setSingleStep(1.0)
        self.timestamp.setSuffix(" hours")
        timeseries_layout.addRow("Timestamp:", self.timestamp)
        
        self.treatment_info = QLineEdit()
        self.treatment_info.setPlaceholderText("Treatment information (JSON format)")
        timeseries_layout.addRow("Treatment Info:", self.treatment_info)
        
        layout.addWidget(timeseries_group)
        
        layout.addStretch()
        
        # Register widgets
        self.parameter_widgets.update({
            'use_gpu': self.use_gpu,
            'num_threads': self.num_threads,
            'min_quality_score': self.min_quality_score,
            'exclude_border': self.exclude_border,
            'timestamp': self.timestamp,
            'treatment_info': self.treatment_info
        })
        
        return tab
    
    def browse_segmentation_model(self):
        """Browse for segmentation model file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Segmentation Model", "",
            "Model Files (*.pth *.pt *.pkl);;All Files (*)"
        )
        if file_path:
            self.model_path.setText(file_path)
    
    def browse_viability_model(self):
        """Browse for viability model file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Viability Model", "",
            "Model Files (*.pth *.pt *.pkl);;All Files (*)"
        )
        if file_path:
            self.viability_model_path.setText(file_path)
    
    def browse_nuclear_channel(self):
        """Browse for nuclear channel image."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Nuclear Channel Image", "",
            "Image Files (*.png *.jpg *.jpeg *.tiff *.tif *.bmp);;All Files (*)"
        )
        if file_path:
            self.nuclear_channel.setText(file_path)
    
    def load_default_parameters(self):
        """Load default parameter values."""
        # Set default values based on config
        try:
            # Preprocessing defaults
            self.contrast_checkbox.setChecked(True)
            self.contrast_limit.setValue(0.03)
            self.denoise_checkbox.setChecked(True)
            self.denoise_strength.setValue(0.1)
            self.normalize_checkbox.setChecked(True)
            self.norm_method.setCurrentText("percentile")
            
            # Segmentation defaults
            self.min_organoid_size.setValue(100)
            self.max_organoid_size.setValue(10000)
            self.fill_holes.setChecked(True)
            
            # Analysis defaults
            self.enable_viability.setChecked(True)
            self.enable_apoptosis.setChecked(True)
            self.apoptosis_mode.setCurrentText("morphological")
            self.extract_morphological.setChecked(True)
            self.extract_intensity.setChecked(True)
            self.extract_texture.setChecked(True)
            self.extract_spatial.setChecked(True)
            
            # Advanced defaults
            self.use_gpu.setChecked(True)
            self.num_threads.setValue(4)
            self.min_quality_score.setValue(0.3)
            self.exclude_border.setChecked(True)
            self.timestamp.setValue(0.0)
            
            logger.info("Loaded default parameters")
            
        except Exception as e:
            logger.error(f"Failed to load default parameters: {e}")
    
    def reset_parameters(self):
        """Reset all parameters to default values."""
        self.load_default_parameters()
        self.parameters_changed.emit()
    
    def load_configuration(self):
        """Load parameter configuration from file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Load Configuration", "",
            "JSON Files (*.json);;All Files (*)"
        )
        
        if file_path:
            try:
                # TODO: Implement configuration loading
                logger.info(f"Would load configuration from: {file_path}")
                # For now, just show a message
                from PyQt5.QtWidgets import QMessageBox
                QMessageBox.information(self, "Load Configuration", 
                                      "Configuration loading will be implemented in a future version.")
            except Exception as e:
                logger.error(f"Failed to load configuration: {e}")
    
    def save_configuration(self):
        """Save current parameter configuration to file."""
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Configuration", "organoid_config.json",
            "JSON Files (*.json);;All Files (*)"
        )
        
        if file_path:
            try:
                config = self.get_analysis_config()
                
                import json
                with open(file_path, 'w') as f:
                    json.dump(config, f, indent=2)
                
                logger.info(f"Saved configuration to: {file_path}")
                
                from PyQt5.QtWidgets import QMessageBox
                QMessageBox.information(self, "Save Configuration", 
                                      f"Configuration saved to:\n{file_path}")
                
            except Exception as e:
                logger.error(f"Failed to save configuration: {e}")
    
    def get_analysis_config(self) -> Dict[str, Any]:
        """Get current analysis configuration as dictionary."""
        config = {}
        
        try:
            # Preprocessing parameters
            config['preprocessing'] = {
                'contrast_enhancement': self.contrast_checkbox.isChecked(),
                'contrast_limit': self.contrast_limit.value(),
                'denoise': self.denoise_checkbox.isChecked(),
                'denoise_strength': self.denoise_strength.value(),
                'normalize': self.normalize_checkbox.isChecked(),
                'normalization_method': self.norm_method.currentText()
            }
            
            # Segmentation parameters
            config['segmentation'] = {
                'model_path': self.model_path.text().strip() or None,
                'min_organoid_size': self.min_organoid_size.value(),
                'max_organoid_size': self.max_organoid_size.value(),
                'fill_holes': self.fill_holes.isChecked()
            }
            
            # Analysis parameters
            config['analysis'] = {
                'enable_viability': self.enable_viability.isChecked(),
                'viability_model_path': self.viability_model_path.text().strip() or None,
                'enable_apoptosis': self.enable_apoptosis.isChecked(),
                'apoptosis_mode': self.apoptosis_mode.currentText(),
                'nuclear_channel_path': self.nuclear_channel.text().strip() or None,
                'extract_morphological': self.extract_morphological.isChecked(),
                'extract_intensity': self.extract_intensity.isChecked(),
                'extract_texture': self.extract_texture.isChecked(),
                'extract_spatial': self.extract_spatial.isChecked()
            }
            
            # Advanced parameters
            config['advanced'] = {
                'use_gpu': self.use_gpu.isChecked(),
                'num_threads': self.num_threads.value(),
                'min_quality_score': self.min_quality_score.value(),
                'exclude_border': self.exclude_border.isChecked()
            }
            
            # Time series parameters
            config['timestamp'] = self.timestamp.value() if self.timestamp.value() > 0 else None
            
            treatment_text = self.treatment_info.text().strip()
            if treatment_text:
                try:
                    import json
                    config['treatment_info'] = json.loads(treatment_text)
                except:
                    config['treatment_info'] = {'description': treatment_text}
            else:
                config['treatment_info'] = None
            
        except Exception as e:
            logger.error(f"Failed to get analysis config: {e}")
            config = {}
        
        return config