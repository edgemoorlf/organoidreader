"""
Main Application Window for OrganoidReader

This module provides the main PyQt5-based GUI application for OrganoidReader,
including the central window, menu system, and main control interface.
"""

import sys
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QMenuBar, QStatusBar, QToolBar, QAction, QSplitter, QTabWidget,
    QFileDialog, QMessageBox, QProgressBar, QLabel
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt5.QtGui import QIcon, QFont, QPixmap

from organoidreader.config.config_manager import Config
from organoidreader.analysis.pipeline import AdvancedAnalysisPipeline
from organoidreader.gui.image_viewer import ImageViewer
from organoidreader.gui.parameter_panel import ParameterPanel
from organoidreader.gui.analysis_panel import AnalysisPanel
from organoidreader.gui.progress_dialog import ProgressDialog

logger = logging.getLogger(__name__)


class AnalysisWorker(QThread):
    """Worker thread for running analysis operations."""
    
    # Signals
    progress_updated = pyqtSignal(int, str)  # progress, status
    analysis_completed = pyqtSignal(object)  # results
    analysis_failed = pyqtSignal(str)  # error message
    
    def __init__(self, pipeline: AdvancedAnalysisPipeline, image_path: str, config: Dict[str, Any]):
        super().__init__()
        self.pipeline = pipeline
        self.image_path = image_path
        self.config = config
        self._is_cancelled = False
    
    def run(self):
        """Run analysis in background thread."""
        try:
            self.progress_updated.emit(0, "Starting analysis...")
            
            if self._is_cancelled:
                return
            
            self.progress_updated.emit(25, "Analyzing image...")
            
            # Run single image analysis
            result = self.pipeline.analyze_single_image(
                self.image_path,
                timestamp=self.config.get('timestamp'),
                nuclear_channel_path=self.config.get('nuclear_channel_path'),
                treatment_info=self.config.get('treatment_info')
            )
            
            if self._is_cancelled:
                return
            
            self.progress_updated.emit(100, "Analysis complete")
            self.analysis_completed.emit(result)
            
        except Exception as e:
            logger.error(f"Analysis failed: {e}")
            self.analysis_failed.emit(str(e))
    
    def cancel(self):
        """Cancel the analysis."""
        self._is_cancelled = True


class OrganoidReaderApp(QMainWindow):
    """
    Main application window for OrganoidReader.
    
    Provides the central interface for image loading, analysis configuration,
    and result visualization.
    """
    
    def __init__(self):
        super().__init__()
        
        # Initialize application state
        self.config = Config()
        self.pipeline = None
        self.current_image_path = None
        self.current_results = None
        self.analysis_worker = None
        
        # Setup UI
        self.init_ui()
        self.setup_menu_bar()
        self.setup_toolbar()
        self.setup_status_bar()
        
        # Initialize analysis pipeline
        self.init_analysis_pipeline()
        
        logger.info("OrganoidReader application initialized")
    
    def init_ui(self):
        """Initialize the main user interface."""
        self.setWindowTitle("OrganoidReader - AI-Powered Organoid Analysis")
        self.setGeometry(100, 100, 1400, 900)
        
        # Set application icon if available
        try:
            self.setWindowIcon(QIcon("assets/icon.png"))
        except:
            pass
        
        # Create central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Create main layout
        main_layout = QHBoxLayout(central_widget)
        
        # Create main splitter
        main_splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(main_splitter)
        
        # Left panel - Image viewer and controls
        left_panel = self.create_left_panel()
        main_splitter.addWidget(left_panel)
        
        # Right panel - Parameters and analysis results
        right_panel = self.create_right_panel()
        main_splitter.addWidget(right_panel)
        
        # Set splitter proportions
        main_splitter.setSizes([800, 600])
        
        # Apply styling
        self.apply_styling()
    
    def create_left_panel(self) -> QWidget:
        """Create the left panel with image viewer."""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # Image viewer
        self.image_viewer = ImageViewer()
        layout.addWidget(self.image_viewer)
        
        # Image controls
        controls_widget = self.create_image_controls()
        layout.addWidget(controls_widget)
        
        return panel
    
    def create_right_panel(self) -> QWidget:
        """Create the right panel with parameter controls and results."""
        # Create tab widget for different panels
        tab_widget = QTabWidget()
        
        # Analysis parameters tab
        self.parameter_panel = ParameterPanel(self.config)
        tab_widget.addTab(self.parameter_panel, "Parameters")
        
        # Analysis results tab
        self.analysis_panel = AnalysisPanel()
        tab_widget.addTab(self.analysis_panel, "Results")
        
        return tab_widget
    
    def create_image_controls(self) -> QWidget:
        """Create image control widgets."""
        controls = QWidget()
        layout = QHBoxLayout(controls)
        
        # Load image button
        load_btn = self.create_button("Load Image", self.load_image)
        layout.addWidget(load_btn)
        
        # Analyze button
        self.analyze_btn = self.create_button("Analyze", self.start_analysis)
        self.analyze_btn.setEnabled(False)
        layout.addWidget(self.analyze_btn)
        
        # Export results button
        self.export_btn = self.create_button("Export Results", self.export_results)
        self.export_btn.setEnabled(False)
        layout.addWidget(self.export_btn)
        
        layout.addStretch()
        
        return controls
    
    def create_button(self, text: str, callback) -> QWidget:
        """Create a standardized button."""
        from PyQt5.QtWidgets import QPushButton
        btn = QPushButton(text)
        btn.clicked.connect(callback)
        btn.setMinimumHeight(35)
        return btn
    
    def setup_menu_bar(self):
        """Setup the application menu bar."""
        menubar = self.menuBar()
        
        # File menu
        file_menu = menubar.addMenu('File')
        
        # Open image action
        open_action = QAction('Open Image...', self)
        open_action.setShortcut('Ctrl+O')
        open_action.triggered.connect(self.load_image)
        file_menu.addAction(open_action)
        
        # Open project action
        open_project_action = QAction('Open Project...', self)
        open_project_action.triggered.connect(self.open_project)
        file_menu.addAction(open_project_action)
        
        file_menu.addSeparator()
        
        # Save results action
        save_action = QAction('Save Results...', self)
        save_action.setShortcut('Ctrl+S')
        save_action.triggered.connect(self.export_results)
        save_action.setEnabled(False)
        file_menu.addAction(save_action)
        self.save_action = save_action
        
        file_menu.addSeparator()
        
        # Exit action
        exit_action = QAction('Exit', self)
        exit_action.setShortcut('Ctrl+Q')
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # Analysis menu
        analysis_menu = menubar.addMenu('Analysis')
        
        # Single image analysis
        single_analysis_action = QAction('Analyze Current Image', self)
        single_analysis_action.setShortcut('F5')
        single_analysis_action.triggered.connect(self.start_analysis)
        single_analysis_action.setEnabled(False)
        analysis_menu.addAction(single_analysis_action)
        self.single_analysis_action = single_analysis_action
        
        # Batch analysis
        batch_analysis_action = QAction('Batch Analysis...', self)
        batch_analysis_action.triggered.connect(self.start_batch_analysis)
        analysis_menu.addAction(batch_analysis_action)
        
        # Time series analysis
        timeseries_action = QAction('Time Series Analysis...', self)
        timeseries_action.triggered.connect(self.start_timeseries_analysis)
        analysis_menu.addAction(timeseries_action)
        
        # View menu
        view_menu = menubar.addMenu('View')
        
        # Show/hide panels
        view_parameters_action = QAction('Parameters Panel', self)
        view_parameters_action.setCheckable(True)
        view_parameters_action.setChecked(True)
        view_menu.addAction(view_parameters_action)
        
        view_results_action = QAction('Results Panel', self)
        view_results_action.setCheckable(True)
        view_results_action.setChecked(True)
        view_menu.addAction(view_results_action)
        
        # Help menu
        help_menu = menubar.addMenu('Help')
        
        # About action
        about_action = QAction('About OrganoidReader', self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)
        
        # Documentation action
        docs_action = QAction('Documentation', self)
        docs_action.triggered.connect(self.open_documentation)
        help_menu.addAction(docs_action)
    
    def setup_toolbar(self):
        """Setup the main toolbar."""
        toolbar = QToolBar()
        self.addToolBar(toolbar)
        
        # Add common actions to toolbar
        toolbar.addAction("Open", self.load_image)
        toolbar.addSeparator()
        
        self.toolbar_analyze_action = toolbar.addAction("Analyze", self.start_analysis)
        self.toolbar_analyze_action.setEnabled(False)
        
        toolbar.addAction("Export", self.export_results)
        toolbar.addSeparator()
        
        # Add zoom controls
        toolbar.addAction("Zoom In", self.image_viewer.zoom_in)
        toolbar.addAction("Zoom Out", self.image_viewer.zoom_out)
        toolbar.addAction("Fit to Window", self.image_viewer.fit_to_window)
    
    def setup_status_bar(self):
        """Setup the status bar."""
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        
        # Add permanent widgets
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setMaximumWidth(200)
        self.status_bar.addPermanentWidget(self.progress_bar)
        
        # Status labels
        self.image_info_label = QLabel("No image loaded")
        self.status_bar.addWidget(self.image_info_label)
        
        self.status_bar.showMessage("Ready")
    
    def apply_styling(self):
        """Apply application styling."""
        # Set a modern font
        font = QFont("Arial", 9)
        self.setFont(font)
        
        # Apply stylesheet for modern look
        stylesheet = """
            QMainWindow {
                background-color: #f0f0f0;
            }
            QTabWidget::pane {
                border: 1px solid #c0c0c0;
                background-color: white;
            }
            QTabBar::tab {
                background-color: #e0e0e0;
                padding: 8px 16px;
                margin-right: 2px;
            }
            QTabBar::tab:selected {
                background-color: white;
                border-bottom: 2px solid #0078d4;
            }
            QPushButton {
                background-color: #0078d4;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #106ebe;
            }
            QPushButton:disabled {
                background-color: #cccccc;
            }
        """
        self.setStyleSheet(stylesheet)
    
    def init_analysis_pipeline(self):
        """Initialize the analysis pipeline."""
        try:
            self.pipeline = AdvancedAnalysisPipeline(
                config=self.config,
                viability_model_path=None,  # Will load default/untrained model
                segmentation_model_path=None
            )
            logger.info("Analysis pipeline initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize analysis pipeline: {e}")
            QMessageBox.critical(self, "Error", 
                               f"Failed to initialize analysis pipeline:\n{str(e)}")
    
    def load_image(self):
        """Load an image file."""
        file_dialog = QFileDialog()
        file_path, _ = file_dialog.getOpenFileName(
            self,
            "Select Image File",
            "",
            "Image Files (*.png *.jpg *.jpeg *.tiff *.tif *.bmp *.nd2 *.czi);;All Files (*)"
        )
        
        if file_path:
            try:
                self.current_image_path = file_path
                
                # Load and display image
                self.image_viewer.load_image(file_path)
                
                # Update UI state
                self.analyze_btn.setEnabled(True)
                self.single_analysis_action.setEnabled(True)
                self.toolbar_analyze_action.setEnabled(True)
                
                # Update status
                file_name = Path(file_path).name
                self.image_info_label.setText(f"Image: {file_name}")
                self.status_bar.showMessage(f"Loaded {file_name}")
                
                # Clear previous results
                self.current_results = None
                self.analysis_panel.clear_results()
                self.export_btn.setEnabled(False)
                self.save_action.setEnabled(False)
                
                logger.info(f"Loaded image: {file_path}")
                
            except Exception as e:
                logger.error(f"Failed to load image: {e}")
                QMessageBox.critical(self, "Error", f"Failed to load image:\n{str(e)}")
    
    def start_analysis(self):
        """Start analysis of the current image."""
        if not self.current_image_path or not self.pipeline:
            return
        
        try:
            # Get analysis configuration from parameter panel
            analysis_config = self.parameter_panel.get_analysis_config()
            
            # Create and start analysis worker
            self.analysis_worker = AnalysisWorker(
                self.pipeline, 
                self.current_image_path, 
                analysis_config
            )
            
            # Connect signals
            self.analysis_worker.progress_updated.connect(self.update_progress)
            self.analysis_worker.analysis_completed.connect(self.analysis_completed)
            self.analysis_worker.analysis_failed.connect(self.analysis_failed)
            
            # Update UI state
            self.analyze_btn.setEnabled(False)
            self.single_analysis_action.setEnabled(False)
            self.toolbar_analyze_action.setEnabled(False)
            self.progress_bar.setVisible(True)
            
            # Start analysis
            self.analysis_worker.start()
            
            self.status_bar.showMessage("Running analysis...")
            logger.info("Started image analysis")
            
        except Exception as e:
            logger.error(f"Failed to start analysis: {e}")
            QMessageBox.critical(self, "Error", f"Failed to start analysis:\n{str(e)}")
    
    def update_progress(self, progress: int, status: str):
        """Update analysis progress."""
        self.progress_bar.setValue(progress)
        self.status_bar.showMessage(status)
    
    def analysis_completed(self, results):
        """Handle completed analysis."""
        self.current_results = results
        
        # Update results panel
        self.analysis_panel.display_results(results)
        
        # Update image viewer with overlay
        self.image_viewer.display_analysis_results(results)
        
        # Update UI state
        self.analyze_btn.setEnabled(True)
        self.single_analysis_action.setEnabled(True)
        self.toolbar_analyze_action.setEnabled(True)
        self.export_btn.setEnabled(True)
        self.save_action.setEnabled(True)
        self.progress_bar.setVisible(False)
        
        # Update status
        organoid_count = results.organoid_count
        quality_score = results.quality_score
        self.status_bar.showMessage(
            f"Analysis complete: {organoid_count} organoids found, quality: {quality_score:.2f}"
        )
        
        logger.info(f"Analysis completed: {organoid_count} organoids, quality: {quality_score:.2f}")
    
    def analysis_failed(self, error_message: str):
        """Handle failed analysis."""
        # Update UI state
        self.analyze_btn.setEnabled(True)
        self.single_analysis_action.setEnabled(True)
        self.toolbar_analyze_action.setEnabled(True)
        self.progress_bar.setVisible(False)
        
        self.status_bar.showMessage("Analysis failed")
        
        # Show error message
        QMessageBox.critical(self, "Analysis Failed", f"Analysis failed:\n{error_message}")
        
        logger.error(f"Analysis failed: {error_message}")
    
    def export_results(self):
        """Export analysis results."""
        if not self.current_results:
            QMessageBox.information(self, "No Results", "No analysis results to export.")
            return
        
        file_dialog = QFileDialog()
        file_path, file_type = file_dialog.getSaveFileName(
            self,
            "Export Results",
            "organoid_analysis_results",
            "JSON Files (*.json);;CSV Files (*.csv);;All Files (*)"
        )
        
        if file_path:
            try:
                # Determine format from file extension or selected filter
                if file_path.endswith('.csv') or 'CSV' in file_type:
                    export_format = 'csv'
                else:
                    export_format = 'json'
                
                # Export results
                self.pipeline.export_results(self.current_results, file_path, export_format)
                
                self.status_bar.showMessage(f"Results exported to {Path(file_path).name}")
                QMessageBox.information(self, "Export Complete", 
                                      f"Results exported successfully to:\n{file_path}")
                
                logger.info(f"Exported results to: {file_path}")
                
            except Exception as e:
                logger.error(f"Failed to export results: {e}")
                QMessageBox.critical(self, "Export Failed", 
                                   f"Failed to export results:\n{str(e)}")
    
    def start_batch_analysis(self):
        """Start batch analysis of multiple images."""
        # TODO: Implement batch analysis dialog
        QMessageBox.information(self, "Feature Coming Soon", 
                              "Batch analysis will be implemented in a future version.")
    
    def start_timeseries_analysis(self):
        """Start time series analysis."""
        # TODO: Implement time series analysis dialog
        QMessageBox.information(self, "Feature Coming Soon", 
                              "Time series analysis will be implemented in a future version.")
    
    def open_project(self):
        """Open a saved project."""
        # TODO: Implement project loading
        QMessageBox.information(self, "Feature Coming Soon", 
                              "Project management will be implemented in a future version.")
    
    def show_about(self):
        """Show about dialog."""
        about_text = """
        <h2>OrganoidReader</h2>
        <p><b>AI-Powered Organoid Analysis Platform</b></p>
        <p>Version 1.0.0</p>
        
        <p>OrganoidReader provides comprehensive analysis of organoid images including:</p>
        <ul>
        <li>Automated segmentation and parameter extraction</li>
        <li>Viability analysis with machine learning</li>
        <li>Apoptosis detection and quantification</li>
        <li>Time series tracking and growth analysis</li>
        </ul>
        
        <p>Built with Python, PyQt5, PyTorch, and scikit-image.</p>
        """
        
        QMessageBox.about(self, "About OrganoidReader", about_text)
    
    def open_documentation(self):
        """Open documentation."""
        # TODO: Implement documentation opening
        QMessageBox.information(self, "Documentation", 
                              "Documentation will be available online and in help files.")
    
    def closeEvent(self, event):
        """Handle application close event."""
        # Cancel any running analysis
        if self.analysis_worker and self.analysis_worker.isRunning():
            self.analysis_worker.cancel()
            self.analysis_worker.wait(3000)  # Wait up to 3 seconds
        
        # Save application settings
        try:
            # TODO: Save window geometry and settings
            pass
        except Exception as e:
            logger.error(f"Failed to save settings: {e}")
        
        event.accept()


def run_gui():
    """Run the OrganoidReader GUI application."""
    app = QApplication(sys.argv)
    
    # Set application properties
    app.setApplicationName("OrganoidReader")
    app.setApplicationVersion("1.0.0")
    app.setOrganizationName("OrganoidReader Team")
    
    # Create and show main window
    window = OrganoidReaderApp()
    window.show()
    
    # Start event loop
    return app.exec_()


if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Run application
    sys.exit(run_gui())