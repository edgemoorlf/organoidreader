"""
Analysis Panel for OrganoidReader

This module provides the analysis results display panel for viewing
organoid analysis results, statistics, and detailed information.
"""

import logging
from typing import Dict, Any, Optional, List

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QFormLayout,
    QLabel, QTableWidget, QTableWidgetItem, QTabWidget,
    QTextEdit, QScrollArea, QFrame, QPushButton, QHeaderView
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont, QPixmap, QPainter, QColor

from organoidreader.analysis.pipeline import AnalysisResults

logger = logging.getLogger(__name__)


class SummaryWidget(QWidget):
    """Widget for displaying analysis summary statistics."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()
    
    def init_ui(self):
        """Initialize the user interface."""
        layout = QVBoxLayout(self)
        
        # Overall statistics group
        overall_group = QGroupBox("Overall Statistics")
        overall_layout = QFormLayout(overall_group)
        
        self.organoid_count_label = QLabel("0")
        overall_layout.addRow("Total Organoids:", self.organoid_count_label)
        
        self.quality_score_label = QLabel("0.00")
        overall_layout.addRow("Quality Score:", self.quality_score_label)
        
        self.viable_count_label = QLabel("0 (0%)")
        overall_layout.addRow("Viable Organoids:", self.viable_count_label)
        
        self.apoptotic_count_label = QLabel("0 (0%)")
        overall_layout.addRow("Apoptotic Organoids:", self.apoptotic_count_label)
        
        layout.addWidget(overall_group)
        
        # Morphological statistics group
        morphology_group = QGroupBox("Morphological Statistics")
        morphology_layout = QFormLayout(morphology_group)
        
        self.mean_area_label = QLabel("0.0 ± 0.0 pixels")
        morphology_layout.addRow("Mean Area:", self.mean_area_label)
        
        self.mean_circularity_label = QLabel("0.00")
        morphology_layout.addRow("Mean Circularity:", self.mean_circularity_label)
        
        self.total_area_label = QLabel("0.0 pixels")
        morphology_layout.addRow("Total Area:", self.total_area_label)
        
        layout.addWidget(morphology_group)
        
        # Viability statistics group
        viability_group = QGroupBox("Viability Statistics")
        viability_layout = QFormLayout(viability_group)
        
        self.mean_viability_label = QLabel("0.00 ± 0.00")
        viability_layout.addRow("Mean Viability Score:", self.mean_viability_label)
        
        self.viable_percentage_label = QLabel("0.0%")
        viability_layout.addRow("Viable Percentage:", self.viable_percentage_label)
        
        layout.addWidget(viability_group)
        
        # Apoptosis statistics group
        apoptosis_group = QGroupBox("Apoptosis Statistics")
        apoptosis_layout = QFormLayout(apoptosis_group)
        
        self.mean_apoptosis_label = QLabel("0.00 ± 0.00")
        apoptosis_layout.addRow("Mean Apoptosis Score:", self.mean_apoptosis_label)
        
        self.apoptotic_percentage_label = QLabel("0.0%")
        apoptosis_layout.addRow("Apoptotic Percentage:", self.apoptotic_percentage_label)
        
        layout.addWidget(apoptosis_group)
        
        layout.addStretch()
    
    def update_summary(self, results: AnalysisResults):
        """Update summary with analysis results."""
        try:
            # Overall statistics
            self.organoid_count_label.setText(str(results.organoid_count))
            self.quality_score_label.setText(f"{results.quality_score:.3f}")
            
            # Summary statistics
            if results.summary_statistics:
                stats = results.summary_statistics
                
                # Morphological statistics
                if 'morphological' in stats:
                    morph = stats['morphological']
                    mean_area = morph.get('mean_area', 0)
                    std_area = morph.get('std_area', 0)
                    self.mean_area_label.setText(f"{mean_area:.1f} ± {std_area:.1f} pixels")
                    self.mean_circularity_label.setText(f"{morph.get('mean_circularity', 0):.3f}")
                    self.total_area_label.setText(f"{morph.get('total_area', 0):.1f} pixels")
                
                # Viability statistics
                if 'viability' in stats:
                    viab = stats['viability']
                    mean_score = viab.get('mean_score', 0)
                    std_score = viab.get('std_score', 0)
                    self.mean_viability_label.setText(f"{mean_score:.3f} ± {std_score:.3f}")
                    self.viable_percentage_label.setText(f"{viab.get('viable_percentage', 0):.1f}%")
                    
                    viable_count = viab.get('viable_count', 0)
                    self.viable_count_label.setText(f"{viable_count} ({viab.get('viable_percentage', 0):.1f}%)")
                
                # Apoptosis statistics
                if 'apoptosis' in stats:
                    apop = stats['apoptosis']
                    mean_score = apop.get('mean_score', 0)
                    std_score = apop.get('std_score', 0)
                    self.mean_apoptosis_label.setText(f"{mean_score:.3f} ± {std_score:.3f}")
                    self.apoptotic_percentage_label.setText(f"{apop.get('apoptotic_percentage', 0):.1f}%")
                    
                    apoptotic_count = apop.get('apoptotic_count', 0)
                    self.apoptotic_count_label.setText(f"{apoptotic_count} ({apop.get('apoptotic_percentage', 0):.1f}%)")
            
        except Exception as e:
            logger.error(f"Failed to update summary: {e}")
    
    def clear_summary(self):
        """Clear all summary displays."""
        labels = [
            self.organoid_count_label, self.quality_score_label,
            self.viable_count_label, self.apoptotic_count_label,
            self.mean_area_label, self.mean_circularity_label,
            self.total_area_label, self.mean_viability_label,
            self.viable_percentage_label, self.mean_apoptosis_label,
            self.apoptotic_percentage_label
        ]
        
        for label in labels:
            if "±" in label.text() or "%" in label.text():
                label.setText("0.00 ± 0.00" if "±" in label.text() else "0.0%")
            else:
                label.setText("0" if label == self.organoid_count_label else "0.00")


class OrganoidTableWidget(QTableWidget):
    """Table widget for displaying individual organoid data."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_table()
    
    def init_table(self):
        """Initialize the table structure."""
        # Set column headers
        headers = [
            "ID", "Area", "Circularity", "Solidity",
            "Viability Score", "Viability Class",
            "Apoptosis Score", "Apoptosis Stage",
            "Centroid X", "Centroid Y"
        ]
        
        self.setColumnCount(len(headers))
        self.setHorizontalHeaderLabels(headers)
        
        # Configure table appearance
        self.setAlternatingRowColors(True)
        self.setSelectionBehavior(QTableWidget.SelectRows)
        self.setSortingEnabled(True)
        
        # Auto-resize columns
        header = self.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeToContents)
        header.setStretchLastSection(True)
    
    def update_data(self, results: AnalysisResults):
        """Update table with organoid data."""
        try:
            organoid_count = len(results.organoid_parameters)
            self.setRowCount(organoid_count)
            
            for i, params in enumerate(results.organoid_parameters):
                # Basic information
                self.setItem(i, 0, QTableWidgetItem(str(params.label)))
                
                # Morphological features
                area = params.morphological.get('area_pixels', 0)
                circularity = params.morphological.get('circularity', 0)
                solidity = params.morphological.get('solidity', 0)
                
                self.setItem(i, 1, QTableWidgetItem(f"{area:.1f}"))
                self.setItem(i, 2, QTableWidgetItem(f"{circularity:.3f}"))
                self.setItem(i, 3, QTableWidgetItem(f"{solidity:.3f}"))
                
                # Viability data
                if results.viability_results and i < len(results.viability_results):
                    viability = results.viability_results[i]
                    self.setItem(i, 4, QTableWidgetItem(f"{viability.viability_score:.3f}"))
                    self.setItem(i, 5, QTableWidgetItem(viability.viability_classification))
                else:
                    self.setItem(i, 4, QTableWidgetItem("N/A"))
                    self.setItem(i, 5, QTableWidgetItem("N/A"))
                
                # Apoptosis data
                if results.apoptosis_results and i < len(results.apoptosis_results):
                    apoptosis = results.apoptosis_results[i]
                    self.setItem(i, 6, QTableWidgetItem(f"{apoptosis.apoptosis_score:.3f}"))
                    self.setItem(i, 7, QTableWidgetItem(apoptosis.apoptosis_stage))
                else:
                    self.setItem(i, 6, QTableWidgetItem("N/A"))
                    self.setItem(i, 7, QTableWidgetItem("N/A"))
                
                # Spatial information
                centroid_row = params.spatial.get('centroid_row', 0)
                centroid_col = params.spatial.get('centroid_col', 0)
                self.setItem(i, 8, QTableWidgetItem(f"{centroid_col:.1f}"))
                self.setItem(i, 9, QTableWidgetItem(f"{centroid_row:.1f}"))
            
            # Color-code rows based on viability
            if results.viability_results:
                self.color_code_rows(results.viability_results)
            
        except Exception as e:
            logger.error(f"Failed to update table data: {e}")
    
    def color_code_rows(self, viability_results):
        """Color-code table rows based on viability classification."""
        for i, viability in enumerate(viability_results):
            color = None
            
            if viability.viability_classification == "viable":
                color = QColor(220, 255, 220)  # Light green
            elif viability.viability_classification == "compromised":
                color = QColor(255, 255, 200)  # Light yellow
            elif viability.viability_classification == "non-viable":
                color = QColor(255, 220, 220)  # Light red
            
            if color:
                for col in range(self.columnCount()):
                    item = self.item(i, col)
                    if item:
                        item.setBackground(color)
    
    def clear_data(self):
        """Clear all table data."""
        self.setRowCount(0)


class MetadataWidget(QTextEdit):
    """Widget for displaying analysis metadata."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setReadOnly(True)
        self.setMaximumHeight(200)
        
        # Set monospace font for better formatting
        font = QFont("Consolas", 9)
        font.setStyleHint(QFont.TypeWriter)
        self.setFont(font)
    
    def update_metadata(self, results: AnalysisResults):
        """Update metadata display."""
        try:
            metadata_text = []
            
            # Image information
            metadata_text.append("=== IMAGE INFORMATION ===")
            metadata_text.append(f"Image Path: {results.image_path}")
            if results.timestamp is not None:
                metadata_text.append(f"Timestamp: {results.timestamp}")
            
            # Processing metadata
            if results.processing_metadata:
                metadata = results.processing_metadata
                
                metadata_text.append("\\n=== PROCESSING METADATA ===")
                
                if 'preprocessing_steps' in metadata:
                    steps = metadata['preprocessing_steps']
                    metadata_text.append(f"Preprocessing Steps: {len(steps)}")
                    for step in steps:
                        metadata_text.append(f"  - {step}")
                
                if 'segmentation_metrics' in metadata:
                    seg_metrics = metadata['segmentation_metrics']
                    metadata_text.append("\\nSegmentation Metrics:")
                    for key, value in seg_metrics.items():
                        if isinstance(value, float):
                            metadata_text.append(f"  {key}: {value:.3f}")
                        else:
                            metadata_text.append(f"  {key}: {value}")
                
                if 'image_metadata' in metadata:
                    img_metadata = metadata['image_metadata']
                    metadata_text.append("\\nImage Metadata:")
                    for key, value in img_metadata.items():
                        metadata_text.append(f"  {key}: {value}")
                
                if 'treatment_info' in metadata and metadata['treatment_info']:
                    treatment = metadata['treatment_info']
                    metadata_text.append("\\nTreatment Information:")
                    if isinstance(treatment, dict):
                        for key, value in treatment.items():
                            metadata_text.append(f"  {key}: {value}")
                    else:
                        metadata_text.append(f"  {treatment}")
            
            self.setPlainText("\\n".join(metadata_text))
            
        except Exception as e:
            logger.error(f"Failed to update metadata: {e}")
            self.setPlainText(f"Error displaying metadata: {str(e)}")
    
    def clear_metadata(self):
        """Clear metadata display."""
        self.clear()


class AnalysisPanel(QWidget):
    """
    Main analysis results panel.
    
    Provides tabbed interface for viewing analysis results,
    including summary statistics, individual organoid data, and metadata.
    """
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.current_results = None
        self.init_ui()
    
    def init_ui(self):
        """Initialize the user interface."""
        layout = QVBoxLayout(self)
        
        # Create tab widget
        self.tab_widget = QTabWidget()
        
        # Summary tab
        self.summary_widget = SummaryWidget()
        self.tab_widget.addTab(self.summary_widget, "Summary")
        
        # Detailed data tab
        self.table_widget = OrganoidTableWidget()
        self.tab_widget.addTab(self.table_widget, "Organoid Data")
        
        # Metadata tab
        self.metadata_widget = MetadataWidget()
        self.tab_widget.addTab(self.metadata_widget, "Metadata")
        
        layout.addWidget(self.tab_widget)
        
        # Control buttons
        button_layout = QHBoxLayout()
        
        self.export_table_btn = QPushButton("Export Table...")
        self.export_table_btn.clicked.connect(self.export_table_data)
        self.export_table_btn.setEnabled(False)
        button_layout.addWidget(self.export_table_btn)
        
        self.generate_report_btn = QPushButton("Generate Report...")
        self.generate_report_btn.clicked.connect(self.generate_report)
        self.generate_report_btn.setEnabled(False)
        button_layout.addWidget(self.generate_report_btn)
        
        button_layout.addStretch()
        layout.addLayout(button_layout)
    
    def display_results(self, results: AnalysisResults):
        """Display analysis results in all tabs."""
        try:
            self.current_results = results
            
            # Update all tabs
            self.summary_widget.update_summary(results)
            self.table_widget.update_data(results)
            self.metadata_widget.update_metadata(results)
            
            # Enable action buttons
            self.export_table_btn.setEnabled(True)
            self.generate_report_btn.setEnabled(True)
            
            logger.info("Analysis results displayed successfully")
            
        except Exception as e:
            logger.error(f"Failed to display results: {e}")
    
    def clear_results(self):
        """Clear all result displays."""
        try:
            self.current_results = None
            
            # Clear all tabs
            self.summary_widget.clear_summary()
            self.table_widget.clear_data()
            self.metadata_widget.clear_metadata()
            
            # Disable action buttons
            self.export_table_btn.setEnabled(False)
            self.generate_report_btn.setEnabled(False)
            
            logger.info("Analysis results cleared")
            
        except Exception as e:
            logger.error(f"Failed to clear results: {e}")
    
    def export_table_data(self):
        """Export table data to CSV file."""
        if not self.current_results:
            return
        
        try:
            from PyQt5.QtWidgets import QFileDialog, QMessageBox
            
            file_path, _ = QFileDialog.getSaveFileName(
                self, "Export Table Data", "organoid_table_data.csv",
                "CSV Files (*.csv);;All Files (*)"
            )
            
            if file_path:
                # Export table data
                self.table_widget.export_to_csv(file_path)
                
                QMessageBox.information(self, "Export Complete", 
                                      f"Table data exported to:\\n{file_path}")
                
                logger.info(f"Exported table data to: {file_path}")
        
        except Exception as e:
            logger.error(f"Failed to export table data: {e}")
            from PyQt5.QtWidgets import QMessageBox
            QMessageBox.critical(self, "Export Failed", f"Failed to export table data:\\n{str(e)}")
    
    def generate_report(self):
        """Generate and save analysis report."""
        if not self.current_results:
            return
        
        try:
            from PyQt5.QtWidgets import QFileDialog, QMessageBox
            from organoidreader.analysis.pipeline import create_analysis_report
            
            file_path, _ = QFileDialog.getSaveFileName(
                self, "Save Analysis Report", "organoid_analysis_report.txt",
                "Text Files (*.txt);;All Files (*)"
            )
            
            if file_path:
                # Generate report
                report = create_analysis_report(self.current_results)
                
                # Save to file
                with open(file_path, 'w') as f:
                    f.write(report)
                
                QMessageBox.information(self, "Report Generated", 
                                      f"Analysis report saved to:\\n{file_path}")
                
                logger.info(f"Generated analysis report: {file_path}")
        
        except Exception as e:
            logger.error(f"Failed to generate report: {e}")
            from PyQt5.QtWidgets import QMessageBox
            QMessageBox.critical(self, "Report Failed", f"Failed to generate report:\\n{str(e)}")


# Add CSV export capability to OrganoidTableWidget
def export_to_csv(self, file_path: str):
    """Export table data to CSV file."""
    try:
        import csv
        
        with open(file_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            
            # Write headers
            headers = []
            for col in range(self.columnCount()):
                headers.append(self.horizontalHeaderItem(col).text())
            writer.writerow(headers)
            
            # Write data rows
            for row in range(self.rowCount()):
                row_data = []
                for col in range(self.columnCount()):
                    item = self.item(row, col)
                    row_data.append(item.text() if item else "")
                writer.writerow(row_data)
        
        logger.info(f"Exported {self.rowCount()} rows to CSV: {file_path}")
        
    except Exception as e:
        logger.error(f"Failed to export to CSV: {e}")
        raise

# Add the export method to OrganoidTableWidget
OrganoidTableWidget.export_to_csv = export_to_csv