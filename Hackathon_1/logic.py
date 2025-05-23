from PyQt5 import QtWidgets, QtCore, QtGui
import cv2
import numpy as np

from welcome_page_ui import Ui_uploadDialog
from second_page import Ui_secondDialog


class WelcomePage(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.ui = Ui_uploadDialog()
        self.ui.setupUi(self)


class SecondPage(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.ui = Ui_secondDialog()
        self.ui.setupUi(self)


class MainApp(QtWidgets.QDialog):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Image Transformer App")
        self.resize(600, 400)

        self.stack = QtWidgets.QStackedWidget(self)
        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(self.stack)

        self.welcome_page = WelcomePage()
        self.second_page = SecondPage()

        self.stack.addWidget(self.welcome_page)
        self.stack.addWidget(self.second_page)

        self.welcome_page.ui.uploadButton.clicked.connect(self.open_image)

        self.second_page.ui.resetButton.clicked.connect(self.reset_image)
        self.second_page.ui.saveButton.clicked.connect(self.save_image)

        self.second_page.ui.blurSlider.valueChanged.connect(self.apply_transformations)
        self.second_page.ui.brightnessSlider.valueChanged.connect(self.apply_transformations)
        self.second_page.ui.angleSlider.valueChanged.connect(self.apply_transformations)
        self.second_page.ui.transitionCombobox.currentIndexChanged.connect(self.apply_transformations)

        self.original_image = None
        self.processed_image = None

        self.setup_transitions()

    def setup_transitions(self):
        transitions = [
            "Original",
            "Grayscale",
            "Gaussian Blur",
            "Median Blur",
            "Sobel Edge Detection",
            "Canny Edge Detection",
            "Thresholding",
            "Rotate",
            "Resize (50%)",
            "Erosion",
            "Dilation",
            "Brightness Adjust",
            "Contrast Adjust",
            "Flip Horizontal",
            "Flip Vertical"
        ]
        self.second_page.ui.transitionCombobox.addItems(transitions)

        # Set sliders default ranges and values
        self.second_page.ui.blurSlider.setMinimum(1)
        self.second_page.ui.blurSlider.setMaximum(21) # Increased max blur for more noticeable effect
        self.second_page.ui.blurSlider.setValue(1)  # Start with no blur (kernel 1x1)
        self.second_page.ui.blurSlider.setSingleStep(2) # Ensures odd kernel sizes

        self.second_page.ui.brightnessSlider.setMinimum(-100)
        self.second_page.ui.brightnessSlider.setMaximum(100)
        self.second_page.ui.brightnessSlider.setValue(0) # Default to no change

        self.second_page.ui.angleSlider.setMinimum(-180)
        self.second_page.ui.angleSlider.setMaximum(180)
        self.second_page.ui.angleSlider.setValue(0) # Default to no rotation

    def open_image(self):
        fname, _ = QtWidgets.QFileDialog.getOpenFileName(self, 'Open Image', '',
                                                         "Image files (*.jpg *.jpeg *.png *.bmp *.tif)")
        if not fname:
            QtWidgets.QMessageBox.warning(self, "Invalid Input", "No image file selected.")
            return

        img = cv2.imread(fname)
        if img is None:
            QtWidgets.QMessageBox.warning(self, "Invalid Input", "Failed to load image. Please select a valid image file.")
            return

        self.original_image = img
        self.processed_image = img.copy()

        self.show_image(self.processed_image)
        self.stack.setCurrentWidget(self.second_page)

        # Reset sliders and combo box to defaults on new image load
        self.second_page.ui.blurSlider.setValue(1)
        self.second_page.ui.brightnessSlider.setValue(0)
        self.second_page.ui.angleSlider.setValue(0)
        self.second_page.ui.transitionCombobox.setCurrentIndex(0) # Select "Original"

    def show_image(self, img):
        if img is None:
            self.second_page.ui.imageLable.clear()
            self.second_page.ui.imageLable.setText("Image will be displayed here")
            return

        # Ensure image is 3-channel for display if it's grayscale
        if len(img.shape) == 2:
            rgb_image = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            h, w = img.shape
            ch = 3
        else:
            rgb_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape

        bytes_per_line = ch * w
        qt_image = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        pixmap = QtGui.QPixmap.fromImage(qt_image)
        label = self.second_page.ui.imageLable

        label.setPixmap(pixmap)
        label.update()

    def reset_image(self):
        if self.original_image is not None:
            self.processed_image = self.original_image.copy()
            self.show_image(self.processed_image)
            self.second_page.ui.blurSlider.setValue(1)
            self.second_page.ui.brightnessSlider.setValue(0)
            self.second_page.ui.angleSlider.setValue(0)
            self.second_page.ui.transitionCombobox.setCurrentIndex(0)

    def save_image(self):
        if self.processed_image is None:
            QtWidgets.QMessageBox.warning(self, "No Image", "There is no image to save.")
            return

        fname, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save Image", "", "PNG Files (*.png);;JPEG Files (*.jpg)")
        if fname:
            if len(self.processed_image.shape) == 2: # If grayscale, convert to BGR for saving
                img_to_save = cv2.cvtColor(self.processed_image, cv2.COLOR_GRAY2BGR)
            else:
                img_to_save = self.processed_image
            cv2.imwrite(fname, img_to_save)
            QtWidgets.QMessageBox.information(self, "Image Saved", f"Image saved successfully to {fname}")

    def apply_transformations(self):
        if self.original_image is None:
            return

        # Always start with a fresh copy of the original image for the base transformation
        current_img = self.original_image.copy()

        # Get current values from UI elements
        choice = self.second_page.ui.transitionCombobox.currentText()
        blur_val = self.second_page.ui.blurSlider.value()
        brightness_val = self.second_page.ui.brightnessSlider.value()
        angle_val = self.second_page.ui.angleSlider.value()

        # --- Phase 1: Apply the primary transformation based on ComboBox choice ---
        # This handles the specific selections like "Gaussian Blur" or "Resize (50%)"
        # When these are selected, their corresponding slider values (if any) are applied.
        # Other slider values will be applied in Phase 2 unless specifically excluded.
        if choice == "Original":
            pass # current_img remains original

        elif choice == "Grayscale":
            current_img = cv2.cvtColor(current_img, cv2.COLOR_BGR2GRAY)

        elif choice == "Gaussian Blur":
            k = blur_val # Use the current blur slider value
            if k % 2 == 0: k += 1 # Ensure odd kernel size
            if k < 1: k = 1 # Ensure k is at least 1
            # If the image is currently grayscale (e.g., from a previous filter), convert to BGR for blur
            if len(current_img.shape) == 2:
                current_img = cv2.cvtColor(current_img, cv2.COLOR_GRAY2BGR)
            current_img = cv2.GaussianBlur(current_img, (k, k), 0)

        elif choice == "Median Blur":
            k = blur_val # Use the current blur slider value
            if k % 2 == 0: k += 1 # Ensure odd kernel size
            if k < 1: k = 1 # Ensure k is at least 1
            # If the image is currently grayscale, convert to BGR for blur
            if len(current_img.shape) == 2:
                current_img = cv2.cvtColor(current_img, cv2.COLOR_GRAY2BGR)
            current_img = cv2.medianBlur(current_img, k)

        elif choice == "Sobel Edge Detection":
            gray = cv2.cvtColor(current_img, cv2.COLOR_BGR2GRAY)
            sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            sobel = cv2.magnitude(sobelx, sobely)
            current_img = np.uint8(np.clip(sobel, 0, 255)) # Output is single channel

        elif choice == "Canny Edge Detection":
            gray = cv2.cvtColor(current_img, cv2.COLOR_BGR2GRAY)
            current_img = cv2.Canny(gray, 100, 200) # Output is single channel

        elif choice == "Thresholding":
            gray = cv2.cvtColor(current_img, cv2.COLOR_BGR2GRAY)
            _, current_img = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY) # Output is single channel

        elif choice == "Rotate":
            h, w = current_img.shape[:2]
            M = cv2.getRotationMatrix2D((w / 2, h / 2), angle_val, 1)
            cos = np.abs(M[0, 0])
            sin = np.abs(M[0, 1])
            nW = int((h * sin) + (w * cos))
            nH = int((h * cos) + (w * sin))
            M[0, 2] += (nW / 2) - (w / 2)
            M[1, 2] += (nH / 2) - (h / 2)
            current_img = cv2.warpAffine(current_img, M, (nW, nH))

        elif choice == "Resize (50%)":
            # This is a fixed resize. The slider doesn't control this specific value.
            current_img = cv2.resize(current_img, (0, 0), fx=0.5, fy=0.5)

        elif choice == "Erosion":
            kernel = np.ones((5, 5), np.uint8)
            current_img = cv2.erode(current_img, kernel, iterations=1)

        elif choice == "Dilation":
            kernel = np.ones((5, 5), np.uint8)
            current_img = cv2.dilate(current_img, kernel, iterations=1)

        elif choice == "Brightness Adjust":
            current_img = cv2.convertScaleAbs(current_img, alpha=1, beta=brightness_val)

        elif choice == "Contrast Adjust":
            alpha = 1.0 + (brightness_val / 100.0) # Using brightness_val for contrast slider
            alpha = max(0.1, min(alpha, 3.0)) # Clamp alpha to prevent extreme values
            current_img = cv2.convertScaleAbs(current_img, alpha=alpha, beta=0)

        elif choice == "Flip Horizontal":
            current_img = cv2.flip(current_img, 1)

        elif choice == "Flip Vertical":
            current_img = cv2.flip(current_img, 0)

        # --- Phase 2: Apply cumulative effects from sliders if they are NOT the primary choice ---
        # This allows the sliders to act as "global" modifiers unless their specific function
        # is already being handled by the combobox selection.

        # Apply Blur if the combobox choice was NOT specifically a blur filter, and slider value > 1
        if choice not in ["Gaussian Blur", "Median Blur"] and blur_val > 1:
            k = blur_val
            if k % 2 == 0: k += 1 # Ensure odd kernel size
            if k < 1: k = 1

            # Important: Ensure the image is 3-channel for GaussianBlur if it became grayscale from a previous filter
            if len(current_img.shape) == 2:
                temp_img = cv2.cvtColor(current_img, cv2.COLOR_GRAY2BGR)
            else:
                temp_img = current_img
            current_img = cv2.GaussianBlur(temp_img, (k, k), 0)


        # Apply Brightness if the combobox choice was NOT specifically brightness/contrast adjust, and value not 0
        if choice not in ["Brightness Adjust", "Contrast Adjust"] and brightness_val != 0:
            current_img = cv2.convertScaleAbs(current_img, alpha=1, beta=brightness_val)

        # Apply Rotation if the combobox choice was NOT specifically "Rotate", and angle not 0
        if choice != "Rotate" and angle_val != 0:
            h, w = current_img.shape[:2]
            M = cv2.getRotationMatrix2D((w / 2, h / 2), angle_val, 1)
            cos = np.abs(M[0, 0])
            sin = np.abs(M[0, 1])
            nW = int((h * sin) + (w * cos))
            nH = int((h * cos) + (w * sin))
            M[0, 2] += (nW / 2) - (w / 2)
            M[1, 2] += (nH / 2) - (h / 2)
            current_img = cv2.warpAffine(current_img, M, (nW, nH))


        self.processed_image = current_img
        self.show_image(self.processed_image)