import os
import sys
import shutil
import numpy as np
import cv2
import imageio
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QLabel, QPushButton, QScrollArea, QMessageBox, QShortcut, QFileDialog
)
from PyQt5.QtGui import QPixmap, QImage, QKeySequence, QPainter
from PyQt5.QtCore import Qt


# Enhanced depth visualization function
def enhance_depth_visualization(img):
    """
    Improved depth image visualization processing, optimized contrast and brightness
    Parameters:
        img: input depth image (uint16, float32, etc.)
    Returns:
        uint8 8-bit grayscale image with optimized visual effect
    """
    # Handle 16-bit unsigned integer format
    if img.dtype == np.uint16:
        # Convert to float for processing
        img = img.astype(np.float32)

        # Calculate valid depth range (excluding outliers)
        valid_pixels = img[img > 0]  # Exclude zero-value pixels
        if len(valid_pixels) == 0:
            return np.zeros_like(img, dtype=np.uint8)

        # Use percentiles to avoid extreme value impact
        min_val = np.percentile(valid_pixels, 2)  # 2nd percentile as min
        max_val = np.percentile(valid_pixels, 98)  # 98th percentile as max

        # Ensure range is not zero
        depth_range = max(max_val - min_val, 1.0)

        # Normalize to [0,1] range
        normalized = (img - min_val) / depth_range
        normalized[normalized < 0] = 0
        normalized[normalized > 1] = 1

        # Apply gamma correction for visual enhancement
        gamma = 0.8  # <1 enhances dark area details
        normalized = np.power(normalized, gamma)

        # Convert to 8-bit
        return (normalized * 255).astype(np.uint8)

    # Handle float formats
    elif img.dtype == np.float32 or img.dtype == np.float64:
        # Calculate valid depth range
        valid_pixels = img[img > 0]  # Exclude zero-value pixels
        if len(valid_pixels) == 0:
            return np.zeros_like(img, dtype=np.uint8)

        min_val = np.percentile(valid_pixels, 2)
        max_val = np.percentile(valid_pixels, 98)
        depth_range = max(max_val - min_val, 1e-5)  # Avoid division by zero

        # Normalize
        normalized = (img - min_val) / depth_range
        normalized[normalized < 0] = 0
        normalized[normalized > 1] = 1

        # Apply gamma correction
        gamma = 0.8
        normalized = np.power(normalized, gamma)

        return (normalized * 255).astype(np.uint8)

    # For other formats return directly
    else:
        return img


def extract_key(filename):
    """
    Extract first 5 groups of numbers as matching key from filename
    Example: "03-01-01-99-240523093208-00-99-01-02.png" -> "03-01-01-99-240523093208"
    """
    # Remove extension
    base_name = os.path.splitext(filename)[0]

    # Split by hyphen
    parts = base_name.split('-')

    # Take first 5 parts (or all if less than 5)
    key_parts = parts[:min(5, len(parts))]

    # Recombine into matching key
    return '-'.join(key_parts)


def load_image_with_fallback(path):
    """Try multiple methods to load image, solving OpenCV read issues"""
    try:
        # First try OpenCV
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is not None:
            return img

        # If OpenCV fails, try imageio
        img = imageio.imread(path)
        return img
    except Exception as e:
        print(f"Image loading failed: {path}, error: {str(e)}")
        return None


class DepthImageReviewer(QMainWindow):
    """Depth and color image viewer main window"""

    def __init__(self, depth_folder, color_folder, output_folder):
        super().__init__()
        self.depth_folder = depth_folder  # Depth image folder
        self.color_folder = color_folder  # Color image folder
        self.output_folder = output_folder  # Output save folder
        self.current_index = 0  # Current image index

        # Create subfolders in output folder
        self.depth_output = os.path.join(output_folder, "depth_images")
        self.color_output = os.path.join(output_folder, "color_images")
        os.makedirs(self.depth_output, exist_ok=True)
        os.makedirs(self.color_output, exist_ok=True)

        # Get image pairs
        self.image_pairs = self.get_image_pairs()

        # Exit if no matching pairs found
        if not self.image_pairs:
            QMessageBox.critical(self, "Error", "No matching image pairs found in specified folders.")
            sys.exit(1)

        # Initialize UI
        self.init_ui()
        self.load_current_images()
        self.setWindowTitle("Depth and Color Image Viewer")  # Set window title

    def get_image_pairs(self):
        """Get list of paired depth and color images"""
        # Get all image files in depth folder
        depth_files = [f for f in os.listdir(self.depth_folder)
                       if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.tif', '.dcm', '.bmp'))]

        # Get all image files in color folder
        color_files = [f for f in os.listdir(self.color_folder)
                       if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.tif', '.bmp'))]

        # Create image matching dictionary (key = first 5 groups, value = file paths)
        depth_dict = {}
        for depth_file in depth_files:
            key = extract_key(depth_file)
            if key not in depth_dict:
                depth_dict[key] = []  # Create list for each key
            depth_dict[key].append(depth_file)

        color_dict = {}
        for color_file in color_files:
            key = extract_key(color_file)
            if key not in color_dict:
                color_dict[key] = []
            color_dict[key].append(color_file)

        # Create image pairs (only keys present in both folders)
        pairs = []
        matched_keys = set()  # Track matched keys

        # Iterate through all keys
        all_keys = set(depth_dict.keys()) | set(color_dict.keys())
        for key in sorted(all_keys):  # Sort keys
            if key in depth_dict and key in color_dict:
                # Get all depth and color images for this key
                depth_list = depth_dict[key]
                color_list = color_dict[key]

                # Find matching color image for each depth image
                for depth_file in depth_list:
                    # Find best matching color image (closest filename)
                    best_color = None
                    min_diff = float('inf')

                    for color_file in color_list:
                        # Calculate filename similarity (based on length difference)
                        diff = abs(len(depth_file) - len(color_file))

                        if diff < min_diff:
                            min_diff = diff
                            best_color = color_file

                    if best_color:
                        depth_path = os.path.join(self.depth_folder, depth_file)
                        color_path = os.path.join(self.color_folder, best_color)

                        # Check file existence
                        if os.path.exists(depth_path) and os.path.exists(color_path):
                            pairs.append((depth_path, color_path, key))
                            matched_keys.add(key)

        # Output matching statistics
        print(f"Found {len(pairs)} matching image pairs")
        print(f"Depth folder contains {len(depth_files)} images")
        print(f"Color folder contains {len(color_files)} images")
        print(f"Matched {len(matched_keys)} unique keys")

        return pairs

    def init_ui(self):
        """Initialize user interface"""
        # Create central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Main layout
        main_layout = QVBoxLayout(central_widget)

        # Status label
        self.status_label = QLabel()
        self.status_label.setStyleSheet("font-size: 14px; font-weight: bold;")
        main_layout.addWidget(self.status_label)

        # Image display area
        image_layout = QHBoxLayout()

        # Depth image display area
        depth_widget = QWidget()
        depth_layout = QVBoxLayout(depth_widget)
        depth_layout.addWidget(QLabel("<b>Depth Image</b>"))
        self.depth_scroll = QScrollArea()
        self.depth_scroll.setWidgetResizable(True)  # Allow widget resizing
        self.depth_label = QLabel("Loading depth image...")
        self.depth_label.setAlignment(Qt.AlignCenter)  # Center alignment
        self.depth_label.setMinimumSize(400, 300)  # Minimum size
        self.depth_scroll.setWidget(self.depth_label)
        depth_layout.addWidget(self.depth_scroll)

        # Color image display area
        color_widget = QWidget()
        color_layout = QVBoxLayout(color_widget)
        color_layout.addWidget(QLabel("<b>Color Image</b>"))
        self.color_scroll = QScrollArea()
        self.color_scroll.setWidgetResizable(True)
        self.color_label = QLabel("Loading color image...")
        self.color_label.setAlignment(Qt.AlignCenter)
        self.color_label.setMinimumSize(400, 300)
        self.color_scroll.setWidget(self.color_label)
        color_layout.addWidget(self.color_scroll)

        # Add left-right image display areas
        image_layout.addWidget(depth_widget)
        image_layout.addWidget(QLabel("<h2>⇄</h2>"))  # Add double arrow indicator
        image_layout.addWidget(color_widget)

        main_layout.addLayout(image_layout, 1)  # Add to main layout

        # Button area
        button_layout = QHBoxLayout()

        # Previous button
        self.prev_button = QPushButton("Previous (←)")
        self.prev_button.setStyleSheet("font-size: 14px; padding: 8px;")
        self.prev_button.clicked.connect(self.prev_image)
        button_layout.addWidget(self.prev_button)

        # Save button
        self.save_button = QPushButton("Save (Ctrl+S)")
        self.save_button.setStyleSheet("font-size: 14px; padding: 8px; background-color: #4CAF50; color: white;")
        self.save_button.clicked.connect(self.save_image)
        button_layout.addWidget(self.save_button)

        # Next button
        self.next_button = QPushButton("Next (→)")
        self.next_button.setStyleSheet("font-size: 14px; padding: 8px;")
        self.next_button.clicked.connect(self.next_image)
        button_layout.addWidget(self.next_button)

        main_layout.addLayout(button_layout)

        # Set shortcuts
        QShortcut(QKeySequence("Left"), self).activated.connect(self.prev_image)  # Left arrow
        QShortcut(QKeySequence("Right"), self).activated.connect(self.next_image)  # Right arrow
        QShortcut(QKeySequence("Ctrl+S"), self).activated.connect(self.save_image)  # Ctrl+S

        # Set initial window size
        self.resize(1200, 700)

    def load_depth_image(self, path):
        """Load depth image and convert to QPixmap"""
        try:
            # Load image using multiple methods
            img = load_image_with_fallback(path)
            if img is None:  # Return empty if read fails
                print(f"Unable to load depth image: {path}")
                return QPixmap()

            # Save original depth image for saving (unchanged)
            self.original_depth = img.copy()

            # Apply enhanced depth visualization
            if img.dtype != np.uint8:
                img = enhance_depth_visualization(img)

            # Convert to QImage
            if len(img.shape) == 2:  # Grayscale image
                height, width = img.shape
                bytes_per_line = width  # Bytes per line
                # Create grayscale QImage
                qimg = QImage(img.data, width, height, bytes_per_line, QImage.Format_Grayscale8)
            elif len(img.shape) == 3:  # Color image
                # Convert to RGB format
                if img.shape[2] == 4:  # RGBA format
                    img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
                elif img.shape[2] == 1:  # Single channel but 3D
                    img = img[:, :, 0]
                    height, width = img.shape
                    bytes_per_line = width
                    qimg = QImage(img.data, width, height, bytes_per_line, QImage.Format_Grayscale8)
                    return QPixmap.fromImage(qimg)

                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                height, width, channel = img.shape
                bytes_per_line = 3 * width
                qimg = QImage(img.data, width, height, bytes_per_line, QImage.Format_RGB888)
            else:
                print(f"Unsupported image format: {img.shape}")
                return QPixmap()

            return QPixmap.fromImage(qimg)
        except Exception as e:
            print(f"Error loading depth image {path}: {str(e)}")
            return QPixmap()

    def load_color_image(self, path):
        """Load color image and convert to QPixmap"""
        try:
            # Load image using multiple methods
            img = load_image_with_fallback(path)
            if img is None:  # Return empty if read fails
                print(f"Unable to load color image: {path}")
                return QPixmap()

            # Save original color image for saving (unchanged)
            self.original_color = img.copy()

            # Convert to RGB format
            if len(img.shape) == 2:  # If grayscale, convert to color
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            elif len(img.shape) == 3:  # Color image
                if img.shape[2] == 4:  # RGBA format
                    img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
                else:  # BGR format
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Convert to QImage
            height, width, channel = img.shape
            bytes_per_line = 3 * width
            qimg = QImage(img.data, width, height, bytes_per_line, QImage.Format_RGB888)

            return QPixmap.fromImage(qimg)
        except Exception as e:
            print(f"Error loading color image {path}: {str(e)}")
            return QPixmap()

    def load_current_images(self):
        """Load and display current image pair"""
        if not self.image_pairs or self.current_index >= len(self.image_pairs):
            return

        # Get current image pair paths and matching key
        depth_path, color_path, key = self.image_pairs[self.current_index]
        depth_filename = os.path.basename(depth_path)
        color_filename = os.path.basename(color_path)

        # Load depth image
        depth_pixmap = self.load_depth_image(depth_path)
        if depth_pixmap.isNull():
            self.depth_label.setText(f"Failed to load depth image:\n{depth_filename}\nPath: {depth_path}")
            self.depth_label.setStyleSheet("color: red; font-weight: bold;")
        else:
            # Scale image to fit display area (maintain aspect ratio)
            scaled_pixmap = depth_pixmap.scaled(
                self.depth_scroll.width(),
                self.depth_scroll.height(),
                Qt.KeepAspectRatio,  # Maintain aspect ratio
                Qt.SmoothTransformation  # Smooth transformation
            )
            self.depth_label.setPixmap(scaled_pixmap)
            self.depth_label.setToolTip(f"Depth image: {depth_filename}\nPath: {depth_path}")  # Set tooltip
            self.depth_label.setStyleSheet("")  # Clear error style

        # Load color image
        color_pixmap = self.load_color_image(color_path)
        if color_pixmap.isNull():
            self.color_label.setText(f"Failed to load color image:\n{color_filename}\nPath: {color_path}")
            self.color_label.setStyleSheet("color: red; font-weight: bold;")
        else:
            # Scale image to fit display area
            scaled_pixmap = color_pixmap.scaled(
                self.color_scroll.width(),
                self.color_scroll.height(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
            self.color_label.setPixmap(scaled_pixmap)
            self.color_label.setToolTip(f"Color image: {color_filename}\nPath: {color_path}")
            self.color_label.setStyleSheet("")  # Clear error style

        # Update status label
        self.status_label.setText(
            f"Showing: {self.current_index + 1}/{len(self.image_pairs)} | Key: {key}\n"
            f"Depth: {depth_filename}\n"
            f"Color: {color_filename}"
        )

    def resizeEvent(self, event):
        """Window resize event handling"""
        super().resizeEvent(event)
        self.load_current_images()  # Reload images to fit new size

    def prev_image(self):
        """Show previous image pair"""
        if self.current_index > 0:
            self.current_index -= 1
            self.load_current_images()

    def next_image(self):
        """Show next image pair"""
        if self.current_index < len(self.image_pairs) - 1:
            self.current_index += 1
            self.load_current_images()

    def save_image(self):
        """Save current depth and color images to output folder"""
        if not self.image_pairs:
            return

        # Get current image info
        depth_path, color_path, key = self.image_pairs[self.current_index]
        depth_filename = os.path.basename(depth_path)
        color_filename = os.path.basename(color_path)

        # Create output paths
        depth_output = os.path.join(self.depth_output, depth_filename)
        color_output = os.path.join(self.color_output, color_filename)

        # Show confirmation dialog
        reply = QMessageBox.question(
            self, "Confirm Save",
            f"Save image pair '{key}' to output folder?\n\n"
            f"Depth image will be saved to:\n{depth_output}\n\n"
            f"Color image will be saved to:\n{color_output}",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.Yes
        )

        if reply == QMessageBox.Yes:
            try:
                # Save original depth data (without visualization enhancement)
                if hasattr(self, 'original_depth'):
                    # Save based on original data type
                    if self.original_depth.dtype == np.uint16:
                        # Save 16-bit depth as PNG
                        cv2.imwrite(depth_output, self.original_depth)
                    else:
                        # Use imageio for other formats
                        imageio.imwrite(depth_output, self.original_depth)
                else:
                    # Copy directly if no original depth available
                    shutil.copy2(depth_path, depth_output)

                # Save color image
                if hasattr(self, 'original_color'):
                    # Save as JPG or PNG
                    if color_output.lower().endswith('.jpg') or color_output.lower().endswith('.jpeg'):
                        cv2.imwrite(color_output, cv2.cvtColor(self.original_color, cv2.COLOR_RGB2BGR))
                    else:
                        imageio.imwrite(color_output, self.original_color)
                else:
                    shutil.copy2(color_path, color_output)

                # Remove saved pair from list
                self.image_pairs.pop(self.current_index)

                # Adjust index position
                if self.current_index >= len(self.image_pairs):
                    self.current_index = max(0, len(self.image_pairs) - 1)

                # Load next image
                self.load_current_images()

                # Show success message
                QMessageBox.information(
                    self, "Save Successful",
                    f"Image pair saved to output folder:\n"
                    f"Depth: {depth_output}\n"
                    f"Color: {color_output}"
                )

            except Exception as e:
                # Show error message
                QMessageBox.critical(
                    self, "Save Failed",
                    f"Error saving images:\n{str(e)}"
                )


def select_directory(title):
    """Open directory selection dialog"""
    dialog = QFileDialog()
    dialog.setFileMode(QFileDialog.Directory)  # Set to directory mode
    dialog.setOption(QFileDialog.ShowDirsOnly, True)  # Show only directories
    dialog.setWindowTitle(title)  # Set dialog title

    if dialog.exec_() == QFileDialog.Accepted:
        return dialog.selectedFiles()[0]  # Return first selected path
    return None


if __name__ == "__main__":
    app = QApplication(sys.argv)

    # Step 1: Select depth image folder
    depth_folder = select_directory("Select Depth Image Folder")
    if not depth_folder:
        sys.exit()  # Exit if not selected

    # Step 2: Select color image folder
    color_folder = select_directory("Select Color Image Folder")
    if not color_folder:
        sys.exit()

    # Step 3: Select output folder
    output_folder = select_directory("Select Output Folder")
    if not output_folder:
        sys.exit()

    # Create and show main window
    window = DepthImageReviewer(depth_folder, color_folder, output_folder)
    window.show()

    sys.exit(app.exec_())
