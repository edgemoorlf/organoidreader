"""
Image Viewer Widget for OrganoidReader

This module provides a custom image viewer widget with pan, zoom, and overlay
capabilities for displaying organoid images and analysis results.
"""

import logging
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple

import numpy as np
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QScrollArea,
    QSizePolicy, QFrame, QSlider, QCheckBox, QPushButton
)
from PyQt5.QtCore import Qt, QRect, QPoint, pyqtSignal
from PyQt5.QtGui import (
    QPixmap, QPainter, QPen, QBrush, QColor, QFont, 
    QWheelEvent, QMouseEvent, QPaintEvent
)

from organoidreader.core.image_loader import ImageLoader
from organoidreader.analysis.pipeline import AnalysisResults

logger = logging.getLogger(__name__)


class ImageDisplay(QLabel):
    """Custom QLabel for displaying images with pan and zoom capabilities."""
    
    # Signals
    mouse_position_changed = pyqtSignal(int, int)  # x, y coordinates
    zoom_changed = pyqtSignal(float)  # zoom factor
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # Image data
        self.original_pixmap = None
        self.current_pixmap = None
        self.zoom_factor = 1.0
        self.min_zoom = 0.1
        self.max_zoom = 10.0
        
        # Pan/drag state
        self.pan_start_point = QPoint()
        self.pan_active = False
        
        # Overlay data
        self.show_organoids = True
        self.show_labels = True
        self.organoid_contours = []
        self.organoid_labels = []
        self.analysis_results = None
        
        # Setup widget
        self.setMinimumSize(400, 300)
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet("border: 1px solid gray; background-color: #2b2b2b;")
        self.setScaledContents(False)
        self.setMouseTracking(True)
        
        # Enable drag and drop
        self.setAcceptDrops(True)
    
    def load_image(self, image_path: str):
        """Load and display an image."""
        try:
            loader = ImageLoader()
            image_array, metadata = loader.load_image(image_path)
            
            # Convert numpy array to QPixmap
            self.original_pixmap = self.array_to_pixmap(image_array)
            
            # Reset view
            self.zoom_factor = 1.0
            self.update_display()
            
            # Emit zoom changed signal
            self.zoom_changed.emit(self.zoom_factor)
            
            logger.info(f"Loaded image: {Path(image_path).name}")
            
        except Exception as e:
            logger.error(f"Failed to load image: {e}")
            raise
    
    def array_to_pixmap(self, array: np.ndarray) -> QPixmap:
        """Convert numpy array to QPixmap."""
        if array.ndim == 2:
            # Grayscale image
            # Normalize to 0-255
            if array.dtype != np.uint8:
                array = ((array - array.min()) / (array.max() - array.min()) * 255).astype(np.uint8)
            
            height, width = array.shape
            bytes_per_line = width
            
            # Convert to QPixmap
            from PyQt5.QtGui import QImage
            q_image = QImage(array.data, width, height, bytes_per_line, QImage.Format_Grayscale8)
            return QPixmap.fromImage(q_image)
            
        elif array.ndim == 3 and array.shape[2] == 3:
            # RGB image
            if array.dtype != np.uint8:
                array = ((array - array.min()) / (array.max() - array.min()) * 255).astype(np.uint8)
            
            height, width, channels = array.shape
            bytes_per_line = channels * width
            
            from PyQt5.QtGui import QImage
            q_image = QImage(array.data, width, height, bytes_per_line, QImage.Format_RGB888)
            return QPixmap.fromImage(q_image)
        
        else:
            raise ValueError(f"Unsupported image array shape: {array.shape}")
    
    def update_display(self):
        """Update the displayed image with current zoom and overlays."""
        if self.original_pixmap is None:
            return
        
        # Scale the pixmap
        scaled_size = self.original_pixmap.size() * self.zoom_factor
        self.current_pixmap = self.original_pixmap.scaled(
            scaled_size, Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        
        # Create overlay if analysis results are available
        if self.analysis_results and (self.show_organoids or self.show_labels):
            self.current_pixmap = self.add_overlays(self.current_pixmap)
        
        # Set the pixmap
        self.setPixmap(self.current_pixmap)
        self.update()
    
    def add_overlays(self, pixmap: QPixmap) -> QPixmap:
        """Add analysis result overlays to the pixmap."""
        if not self.analysis_results:
            return pixmap
        
        # Create a copy to draw on
        overlay_pixmap = QPixmap(pixmap)
        painter = QPainter(overlay_pixmap)
        painter.setRenderHint(QPainter.Antialiasing)
        
        try:
            # Draw organoid contours
            if self.show_organoids:
                self.draw_organoid_contours(painter)
            
            # Draw labels
            if self.show_labels:
                self.draw_organoid_labels(painter)
            
        finally:
            painter.end()
        
        return overlay_pixmap
    
    def draw_organoid_contours(self, painter: QPainter):
        """Draw organoid contours on the image."""
        pen = QPen(QColor(0, 255, 0), 2)  # Green contours
        painter.setPen(pen)
        
        for params in self.analysis_results.organoid_parameters:
            # Get bounding box (scaled for zoom)
            bbox = params.morphological.get('bbox', None)
            if bbox:
                min_row, min_col, max_row, max_col = bbox
                
                # Scale coordinates
                x = int(min_col * self.zoom_factor)
                y = int(min_row * self.zoom_factor)
                width = int((max_col - min_col) * self.zoom_factor)
                height = int((max_row - min_row) * self.zoom_factor)
                
                # Draw bounding rectangle
                painter.drawRect(x, y, width, height)
                
                # Color code based on viability if available
                if hasattr(self.analysis_results, 'viability_results') and self.analysis_results.viability_results:
                    try:
                        viability_idx = params.label - 1
                        if 0 <= viability_idx < len(self.analysis_results.viability_results):
                            viability = self.analysis_results.viability_results[viability_idx]
                            
                            # Set color based on viability
                            if viability.viability_classification == 'viable':
                                pen.setColor(QColor(0, 255, 0))  # Green
                            elif viability.viability_classification == 'compromised':
                                pen.setColor(QColor(255, 255, 0))  # Yellow
                            else:
                                pen.setColor(QColor(255, 0, 0))  # Red
                            
                            painter.setPen(pen)
                            painter.drawRect(x, y, width, height)
                    except:
                        pass  # Fall back to default green
    
    def draw_organoid_labels(self, painter: QPainter):
        """Draw organoid labels on the image."""
        font = QFont("Arial", max(8, int(10 * self.zoom_factor)))
        painter.setFont(font)
        painter.setPen(QPen(QColor(255, 255, 255), 1))  # White text
        
        for params in self.analysis_results.organoid_parameters:
            # Get centroid position
            centroid = params.spatial.get('centroid_row', 0), params.spatial.get('centroid_col', 0)
            
            # Scale coordinates
            x = int(centroid[1] * self.zoom_factor)
            y = int(centroid[0] * self.zoom_factor)
            
            # Draw label
            label_text = f"#{params.label}"
            painter.drawText(x, y, label_text)
            
            # Add viability info if available
            if hasattr(self.analysis_results, 'viability_results') and self.analysis_results.viability_results:
                try:
                    viability_idx = params.label - 1
                    if 0 <= viability_idx < len(self.analysis_results.viability_results):
                        viability = self.analysis_results.viability_results[viability_idx]
                        viability_text = f"V: {viability.viability_score:.2f}"
                        painter.drawText(x, y + 15, viability_text)
                except:
                    pass
    
    def zoom_in(self):
        """Zoom in on the image."""
        if self.zoom_factor < self.max_zoom:
            self.zoom_factor = min(self.zoom_factor * 1.25, self.max_zoom)
            self.update_display()
            self.zoom_changed.emit(self.zoom_factor)
    
    def zoom_out(self):
        """Zoom out on the image."""
        if self.zoom_factor > self.min_zoom:
            self.zoom_factor = max(self.zoom_factor / 1.25, self.min_zoom)
            self.update_display()
            self.zoom_changed.emit(self.zoom_factor)
    
    def fit_to_window(self):
        """Fit image to window size."""
        if self.original_pixmap is None:
            return
        
        # Calculate zoom to fit
        widget_size = self.size()
        pixmap_size = self.original_pixmap.size()
        
        zoom_x = widget_size.width() / pixmap_size.width()
        zoom_y = widget_size.height() / pixmap_size.height()
        
        self.zoom_factor = min(zoom_x, zoom_y)
        self.update_display()
        self.zoom_changed.emit(self.zoom_factor)
    
    def reset_zoom(self):
        """Reset zoom to 100%."""
        self.zoom_factor = 1.0
        self.update_display()
        self.zoom_changed.emit(self.zoom_factor)
    
    def set_overlay_visibility(self, show_organoids: bool, show_labels: bool):
        """Set overlay visibility."""
        self.show_organoids = show_organoids
        self.show_labels = show_labels
        self.update_display()
    
    def display_analysis_results(self, results):
        """Display analysis results as overlays."""
        self.analysis_results = results
        self.update_display()
    
    def wheelEvent(self, event: QWheelEvent):
        """Handle mouse wheel for zooming."""
        if event.angleDelta().y() > 0:
            self.zoom_in()
        else:
            self.zoom_out()
    
    def mousePressEvent(self, event: QMouseEvent):
        """Handle mouse press for panning."""
        if event.button() == Qt.LeftButton:
            self.pan_start_point = event.pos()
            self.pan_active = True
    
    def mouseMoveEvent(self, event: QMouseEvent):
        """Handle mouse move for panning and position tracking."""
        # Emit mouse position (in image coordinates)
        if self.original_pixmap:
            image_x = int(event.x() / self.zoom_factor)
            image_y = int(event.y() / self.zoom_factor)
            self.mouse_position_changed.emit(image_x, image_y)
        
        # Handle panning
        if self.pan_active and event.buttons() & Qt.LeftButton:
            # TODO: Implement panning for large images
            pass
    
    def mouseReleaseEvent(self, event: QMouseEvent):
        """Handle mouse release."""
        if event.button() == Qt.LeftButton:
            self.pan_active = False


class ImageViewer(QWidget):
    """
    Complete image viewer widget with controls.
    
    Provides image display, zoom controls, and overlay options.
    """
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # Initialize UI
        self.init_ui()
        
        # Connect signals
        self.image_display.zoom_changed.connect(self.update_zoom_display)
        self.image_display.mouse_position_changed.connect(self.update_position_display)
    
    def init_ui(self):
        """Initialize the user interface."""
        layout = QVBoxLayout(self)
        
        # Image display area
        self.scroll_area = QScrollArea()
        self.image_display = ImageDisplay()
        self.scroll_area.setWidget(self.image_display)
        self.scroll_area.setWidgetResizable(True)
        layout.addWidget(self.scroll_area)
        
        # Control panel
        controls = self.create_controls()
        layout.addWidget(controls)
        
        # Info panel
        info_panel = self.create_info_panel()
        layout.addWidget(info_panel)
    
    def create_controls(self) -> QWidget:
        """Create zoom and overlay controls."""
        controls = QFrame()
        controls.setFrameStyle(QFrame.StyledPanel)
        layout = QHBoxLayout(controls)
        
        # Zoom controls
        layout.addWidget(QLabel("Zoom:"))
        
        zoom_in_btn = QPushButton("+")
        zoom_in_btn.setMaximumWidth(30)
        zoom_in_btn.clicked.connect(self.zoom_in)
        layout.addWidget(zoom_in_btn)
        
        zoom_out_btn = QPushButton("-")
        zoom_out_btn.setMaximumWidth(30)
        zoom_out_btn.clicked.connect(self.zoom_out)
        layout.addWidget(zoom_out_btn)
        
        fit_btn = QPushButton("Fit")
        fit_btn.clicked.connect(self.fit_to_window)
        layout.addWidget(fit_btn)
        
        reset_btn = QPushButton("100%")
        reset_btn.clicked.connect(self.reset_zoom)
        layout.addWidget(reset_btn)
        
        layout.addWidget(QLabel("|"))  # Separator
        
        # Overlay controls
        layout.addWidget(QLabel("Show:"))
        
        self.organoids_checkbox = QCheckBox("Organoids")
        self.organoids_checkbox.setChecked(True)
        self.organoids_checkbox.toggled.connect(self.update_overlays)
        layout.addWidget(self.organoids_checkbox)
        
        self.labels_checkbox = QCheckBox("Labels")
        self.labels_checkbox.setChecked(True)
        self.labels_checkbox.toggled.connect(self.update_overlays)
        layout.addWidget(self.labels_checkbox)
        
        layout.addStretch()
        
        return controls
    
    def create_info_panel(self) -> QWidget:
        """Create information display panel."""
        info_panel = QFrame()
        info_panel.setFrameStyle(QFrame.StyledPanel)
        info_panel.setMaximumHeight(40)
        layout = QHBoxLayout(info_panel)
        
        self.zoom_label = QLabel("Zoom: 100%")
        layout.addWidget(self.zoom_label)
        
        layout.addWidget(QLabel("|"))
        
        self.position_label = QLabel("Position: (0, 0)")
        layout.addWidget(self.position_label)
        
        layout.addStretch()
        
        return info_panel
    
    def load_image(self, image_path: str):
        """Load an image for viewing."""
        self.image_display.load_image(image_path)
    
    def display_analysis_results(self, results):
        """Display analysis results as overlays."""
        self.image_display.display_analysis_results(results)
    
    def zoom_in(self):
        """Zoom in on the image."""
        self.image_display.zoom_in()
    
    def zoom_out(self):
        """Zoom out on the image."""
        self.image_display.zoom_out()
    
    def fit_to_window(self):
        """Fit image to window size."""
        self.image_display.fit_to_window()
    
    def reset_zoom(self):
        """Reset zoom to 100%."""
        self.image_display.reset_zoom()
    
    def update_overlays(self):
        """Update overlay visibility."""
        show_organoids = self.organoids_checkbox.isChecked()
        show_labels = self.labels_checkbox.isChecked()
        self.image_display.set_overlay_visibility(show_organoids, show_labels)
    
    def update_zoom_display(self, zoom_factor: float):
        """Update zoom display."""
        self.zoom_label.setText(f"Zoom: {zoom_factor*100:.0f}%")
    
    def update_position_display(self, x: int, y: int):
        """Update position display."""
        self.position_label.setText(f"Position: ({x}, {y})")