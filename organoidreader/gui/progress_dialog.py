"""
Progress Dialog for OrganoidReader

This module provides a progress dialog for displaying the progress of
long-running analysis operations.
"""

import logging
from typing import Optional

from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel,
    QProgressBar, QPushButton, QTextEdit
)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QFont

logger = logging.getLogger(__name__)


class ProgressDialog(QDialog):
    """
    Progress dialog for long-running operations.
    
    Provides progress bar, status text, and optional cancellation.
    """
    
    # Signals
    cancelled = pyqtSignal()
    
    def __init__(self, title: str = "Processing", parent=None):
        super().__init__(parent)
        
        self.setWindowTitle(title)
        self.setModal(True)
        self.setFixedSize(400, 200)
        
        # State
        self.is_cancelled = False
        
        self.init_ui()
    
    def init_ui(self):
        """Initialize the user interface."""
        layout = QVBoxLayout(self)
        
        # Main status label
        self.status_label = QLabel("Starting...")
        self.status_label.setAlignment(Qt.AlignCenter)
        font = QFont()
        font.setPointSize(10)
        self.status_label.setFont(font)
        layout.addWidget(self.status_label)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        layout.addWidget(self.progress_bar)
        
        # Detailed status (optional)
        self.detail_text = QTextEdit()
        self.detail_text.setMaximumHeight(60)
        self.detail_text.setReadOnly(True)
        self.detail_text.setVisible(False)
        layout.addWidget(self.detail_text)
        
        # Button layout
        button_layout = QHBoxLayout()
        
        # Toggle details button
        self.details_btn = QPushButton("Show Details")
        self.details_btn.clicked.connect(self.toggle_details)
        button_layout.addWidget(self.details_btn)
        
        button_layout.addStretch()
        
        # Cancel button
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.clicked.connect(self.cancel_operation)
        button_layout.addWidget(self.cancel_btn)
        
        layout.addLayout(button_layout)
        
        # Timer for auto-close when completed
        self.auto_close_timer = QTimer()
        self.auto_close_timer.setSingleShot(True)
        self.auto_close_timer.timeout.connect(self.accept)
    
    def update_progress(self, progress: int, status: str, details: Optional[str] = None):
        """
        Update progress display.
        
        Args:
            progress: Progress percentage (0-100)
            status: Main status message
            details: Optional detailed status message
        """
        self.progress_bar.setValue(progress)
        self.status_label.setText(status)
        
        if details:
            self.detail_text.append(details)
            # Auto-scroll to bottom
            self.detail_text.moveCursor(self.detail_text.textCursor().End)
        
        # Auto-close on completion
        if progress >= 100:
            self.cancel_btn.setText("Close")
            # Auto-close after 3 seconds
            self.auto_close_timer.start(3000)
    
    def set_indeterminate(self, indeterminate: bool = True):
        """Set progress bar to indeterminate mode."""
        if indeterminate:
            self.progress_bar.setRange(0, 0)
        else:
            self.progress_bar.setRange(0, 100)
    
    def toggle_details(self):
        """Toggle detail view visibility."""
        if self.detail_text.isVisible():
            self.detail_text.setVisible(False)
            self.details_btn.setText("Show Details")
            self.setFixedSize(400, 140)
        else:
            self.detail_text.setVisible(True)
            self.details_btn.setText("Hide Details")
            self.setFixedSize(400, 240)
    
    def cancel_operation(self):
        """Handle cancel button click."""
        if self.progress_bar.value() >= 100:
            # Operation completed, just close
            self.accept()
        else:
            # Cancel operation
            self.is_cancelled = True
            self.cancelled.emit()
            
            # Update UI
            self.status_label.setText("Cancelling...")
            self.cancel_btn.setEnabled(False)
            self.progress_bar.setStyleSheet("QProgressBar::chunk { background-color: #ff6b6b; }")
    
    def closeEvent(self, event):
        """Handle dialog close event."""
        if self.progress_bar.value() < 100 and not self.is_cancelled:
            # Ask for confirmation if operation is in progress
            from PyQt5.QtWidgets import QMessageBox
            reply = QMessageBox.question(
                self, "Cancel Operation",
                "Operation is still in progress. Do you want to cancel?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )
            
            if reply == QMessageBox.Yes:
                self.cancel_operation()
                event.accept()
            else:
                event.ignore()
        else:
            event.accept()


class BatchProgressDialog(ProgressDialog):
    """
    Extended progress dialog for batch operations.
    
    Shows overall progress and current item being processed.
    """
    
    def __init__(self, title: str = "Batch Processing", parent=None):
        super().__init__(title, parent)
        
        # Add current item label
        self.current_item_label = QLabel("Ready to start...")
        self.current_item_label.setAlignment(Qt.AlignLeft)
        
        # Insert before progress bar
        layout = self.layout()
        layout.insertWidget(1, self.current_item_label)
        
        # Adjust size
        self.setFixedSize(450, 220)
        
        # Batch state
        self.current_item = 0
        self.total_items = 0
    
    def set_batch_info(self, total_items: int):
        """Set total number of items to process."""
        self.total_items = total_items
        self.current_item = 0
        self.update_item_display()
    
    def update_batch_progress(self, 
                            item_index: int, 
                            item_name: str,
                            item_progress: int,
                            status: str,
                            details: Optional[str] = None):
        """
        Update batch progress.
        
        Args:
            item_index: Current item index (0-based)
            item_name: Name of current item being processed
            item_progress: Progress of current item (0-100)
            status: Status message
            details: Optional detailed message
        """
        self.current_item = item_index + 1
        
        # Update current item display
        self.current_item_label.setText(f"Processing: {item_name}")
        
        # Calculate overall progress
        if self.total_items > 0:
            # Overall progress based on completed items + current item progress
            completed_items = max(0, self.current_item - 1)
            overall_progress = (completed_items * 100 + item_progress) / self.total_items
            overall_progress = min(100, max(0, overall_progress))
        else:
            overall_progress = item_progress
        
        # Update status to include batch info
        batch_status = f"({self.current_item}/{self.total_items}) {status}"
        
        # Update progress
        self.update_progress(int(overall_progress), batch_status, details)
    
    def update_item_display(self):
        """Update the current item display."""
        if self.total_items > 0:
            self.current_item_label.setText(f"Item {self.current_item} of {self.total_items}")
        else:
            self.current_item_label.setText("Ready to start...")


class SimpleProgressDialog(QDialog):
    """
    Simplified progress dialog for quick operations.
    
    Shows only a progress bar and status message.
    """
    
    def __init__(self, title: str = "Processing", message: str = "Please wait...", parent=None):
        super().__init__(parent)
        
        self.setWindowTitle(title)
        self.setModal(True)
        self.setFixedSize(300, 100)
        self.setWindowFlags(Qt.Dialog | Qt.CustomizeWindowHint | Qt.WindowTitleHint)
        
        layout = QVBoxLayout(self)
        
        # Status message
        status_label = QLabel(message)
        status_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(status_label)
        
        # Progress bar (indeterminate)
        progress_bar = QProgressBar()
        progress_bar.setRange(0, 0)  # Indeterminate
        layout.addWidget(progress_bar)


# Factory functions for common progress dialogs
def create_analysis_progress_dialog(parent=None) -> ProgressDialog:
    """Create progress dialog for analysis operations."""
    dialog = ProgressDialog("Analyzing Image", parent)
    dialog.update_progress(0, "Initializing analysis pipeline...")
    return dialog


def create_batch_analysis_progress_dialog(parent=None) -> BatchProgressDialog:
    """Create progress dialog for batch analysis operations."""
    dialog = BatchProgressDialog("Batch Analysis", parent)
    return dialog


def create_export_progress_dialog(parent=None) -> ProgressDialog:
    """Create progress dialog for export operations."""
    dialog = ProgressDialog("Exporting Results", parent)
    dialog.update_progress(0, "Preparing export...")
    return dialog