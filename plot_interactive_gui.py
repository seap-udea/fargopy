import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QComboBox, QCheckBox, QLineEdit,
    QGridLayout, QSpinBox, QDoubleSpinBox, QGroupBox, QFrame, QSizePolicy, QFileDialog,
    QDialog, QTextEdit, QFormLayout, QDialogButtonBox
)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QFont, QIcon, QPixmap
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import sys
import fargopy as fp
import matplotlib as plt
plt.style.use('seaborn-v0_8-whitegrid')

class SimInfoDialog(QDialog):
    def __init__(self, sim, parent=None):
        super().__init__(parent)
        self.sim = sim
        self.setWindowTitle("Simulation Info & Units")
        self.setMinimumWidth(500)
        layout = QFormLayout(self)

        # Store initial units for reset
        self.initial_unitsystem = getattr(self.sim, "unitsystem", "cgs")
        self.initial_UL = getattr(self.sim, "UL", 1.0)
        self.initial_UM = getattr(self.sim, "UM", 1.0)

        # --- Units selector ---
        self.units_combo = QComboBox()
        self.units_combo.addItems(["CGS", "MKS"])
        try:
            current_units = self.sim.unitsystem.upper()
        except Exception:
            current_units = "CGS"
        self.units_combo.setCurrentText(current_units)
        layout.addRow("Units system:", self.units_combo)

        # --- UL controls ---
        self.ul_spin = QDoubleSpinBox()
        self.ul_spin.setDecimals(3)
        self.ul_spin.setMaximum(1e20)
        self.ul_spin.setValue(getattr(self.sim, "UL", 1.0))
        layout.addRow("UL (length unit):", self.ul_spin)

        self.ul_unit_combo = QComboBox()
        self.ul_unit_combo.addItems([
            "cm (CGS)", "m (MKS)", "Earth radii", "Jupiter radii", "Solar radii", "AU"
        ])
        layout.addRow("UL as:", self.ul_unit_combo)

        # --- UM controls ---
        self.um_spin = QDoubleSpinBox()
        self.um_spin.setDecimals(3)
        self.um_spin.setMaximum(1e30)
        self.um_spin.setValue(getattr(self.sim, "UM", 1.0))
        layout.addRow("UM (mass unit):", self.um_spin)

        self.um_unit_combo = QComboBox()
        self.um_unit_combo.addItems([
            "g (CGS)", "kg (MKS)", "Earth masses", "Jupiter masses", "Solar masses"
        ])
        layout.addRow("UM as:", self.um_unit_combo)

        # Info area
        self.info_text = QTextEdit()
        self.info_text.setReadOnly(True)
        self.update_info_text()
        layout.addRow("Simulation properties:", self.info_text)

        # Buttons
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Apply)
        layout.addRow(buttons)
        buttons.accepted.connect(self.accept)
        buttons.button(QDialogButtonBox.Apply).clicked.connect(self.apply_changes)

        # Reset units button
        self.reset_button = QPushButton("Reset Units")
        layout.addRow(self.reset_button)
        self.reset_button.clicked.connect(self.reset_units)

        # Connect signals
        self.units_combo.currentTextChanged.connect(self.apply_unitsystem)
        self.ul_unit_combo.currentTextChanged.connect(self.update_ul_display)
        self.um_unit_combo.currentTextChanged.connect(self.update_um_display)

    def update_ul_display(self):
        ul = getattr(self.sim, "UL", 1.0)
        unitsys = self.units_combo.currentText().upper()
        AU = 1.495978707e13 if unitsys == "CGS" else 1.495978707e11
        RE = 6.371e8 if unitsys == "CGS" else 6.371e6
        RJ = 7.1492e9 if unitsys == "CGS" else 7.1492e7
        RS = 6.957e10 if unitsys == "CGS" else 6.957e8
        if self.ul_unit_combo.currentText().startswith("Earth radii"):
            value = ul / RE
        elif self.ul_unit_combo.currentText().startswith("Jupiter radii"):
            value = ul / RJ
        elif self.ul_unit_combo.currentText().startswith("Solar radii"):
            value = ul / RS
        elif self.ul_unit_combo.currentText().endswith("AU"):
            value = ul / AU
        elif self.ul_unit_combo.currentText().endswith("cm"):
            value = ul if unitsys == "CGS" else ul * 100
        elif self.ul_unit_combo.currentText().endswith("m"):
            value = ul if unitsys == "MKS" else ul / 100
        else:
            value = ul
        self.ul_spin.setValue(value)

    def update_um_display(self):
        um = getattr(self.sim, "UM", 1.0)
        unitsys = self.units_combo.currentText().upper()
        ME = 5.9722e27 if unitsys == "CGS" else 5.9722e24
        MJ = 1.89813e30 if unitsys == "CGS" else 1.89813e27
        MS = 1.98847e33 if unitsys == "CGS" else 1.98847e30
        if self.um_unit_combo.currentText().startswith("Earth masses"):
            value = um / ME
        elif self.um_unit_combo.currentText().startswith("Jupiter masses"):
            value = um / MJ
        elif self.um_unit_combo.currentText().startswith("Solar masses"):
            value = um / MS
        elif self.um_unit_combo.currentText().endswith("g"):
            value = um if unitsys == "CGS" else um * 1000
        elif self.um_unit_combo.currentText().endswith("kg"):
            value = um if unitsys == "MKS" else um / 1000
        else:
            value = um
        self.um_spin.setValue(value)

    def update_info_text(self):
        try:
            props = self.sim.load_properties()
            domain_info = ""
            try:
                rmin = self.sim.domains.r.min()
                rmax = self.sim.domains.r.max()
                domain_info += f"r domain: [{rmin:.3g}, {rmax:.3g}]\n"
            except Exception:
                domain_info += "r domain: not available\n"
            try:
                thetamin = self.sim.domains.theta.min()
                thetamax = self.sim.domains.theta.max()
                domain_info += f"theta domain: [{thetamin:.3g}, {thetamax:.3g}]\n"
            except Exception:
                domain_info += "theta domain: not available\n"
            try:
                phimin = self.sim.domains.phi.min()
                phimax = self.sim.domains.phi.max()
                domain_info += f"phi domain: [{phimin:.3g}, {phimax:.3g}]\n"
            except Exception:
                domain_info += "phi domain: not available\n"
            full_info = f"{props}\n\n{domain_info}"
            self.info_text.setText(full_info)
        except Exception as e:
            self.info_text.setText(f"Error loading properties:\n{e}")

    def apply_unitsystem(self, text):
        self.sim.units(text.lower())
        self.ul_spin.setValue(getattr(self.sim, "UL", 1.0))
        self.um_spin.setValue(getattr(self.sim, "UM", 1.0))
        self.update_ul_display()
        self.update_um_display()
        self.update_info_text()
        if self.parent() and hasattr(self.parent(), "plot_density"):
            self.parent().plot_density(self.units_combo.currentText())

    def apply_changes(self):
        unitsys = self.units_combo.currentText().upper()
        AU = 1.495978707e13 if unitsys == "CGS" else 1.495978707e11
        RE = 6.371e8 if unitsys == "CGS" else 6.371e6
        RJ = 7.1492e9 if unitsys == "CGS" else 7.1492e7
        RS = 6.957e10 if unitsys == "CGS" else 6.957e8
        ME = 5.9722e27 if unitsys == "CGS" else 5.9722e24
        MJ = 1.89813e30 if unitsys == "CGS" else 1.89813e27
        MS = 1.98847e33 if unitsys == "CGS" else 1.98847e30

        ul_val = self.ul_spin.value()
        ul_unit = self.ul_unit_combo.currentText()
        if ul_unit.startswith("Earth radii"):
            ul = ul_val * RE
        elif ul_unit.startswith("Jupiter radii"):
            ul = ul_val * RJ
        elif ul_unit.startswith("Solar radii"):
            ul = ul_val * RS
        elif ul_unit.endswith("AU"):
            ul = ul_val * AU
        elif ul_unit.endswith("cm"):
            ul = ul_val if unitsys == "CGS" else ul_val * 100
        elif ul_unit.endswith("m"):
            ul = ul_val if unitsys == "MKS" else ul_val / 100
        else:
            ul = ul_val

        um_val = self.um_spin.value()
        um_unit = self.um_unit_combo.currentText()
        if um_unit.startswith("Earth masses"):
            um = um_val * ME
        elif um_unit.startswith("Jupiter masses"):
            um = um_val * MJ
        elif um_unit.startswith("Solar masses"):
            um = um_val * MS
        elif um_unit.endswith("g"):
            um = um_val if unitsys == "CGS" else um_val * 1000
        elif um_unit.endswith("kg"):
            um = um_val if unitsys == "MKS" else um_val / 1000
        else:
            um = um_val

        self.sim.set_units(UL=ul, UM=um)
        self.update_ul_display()
        self.update_um_display()
        self.update_info_text()
        if self.parent() and hasattr(self.parent(), "plot_density"):
            self.parent().plot_density(self.units_combo.currentText())

    def reset_units(self):
        self.sim.units(str(self.initial_unitsystem).lower())
        self.sim.set_units(UL=self.initial_UL, UM=self.initial_UM)
        self.units_combo.setCurrentText(str(self.initial_unitsystem).upper())
        self.ul_spin.setValue(self.initial_UL)
        self.um_spin.setValue(self.initial_UM)
        self.update_ul_display()
        self.update_um_display()
        self.update_info_text()
        if self.parent() and hasattr(self.parent(), "plot_density"):
            self.parent().plot_density(self.units_combo.currentText())

class PlotInteractiveWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.sim = None

        # --- Length scale factors for axes ---
        self.length_scale_factors = {
            "Simulation UL": 1.0,
            "cm (CGS)": 1.0,
            "m (MKS)": 1.0,
            "AU": 1.495978707e13,         # CGS
            "Earth radii": 6.371e8,       # CGS
            "Jupiter radii": 7.1492e9,    # CGS
            "Solar radii": 6.957e10       # CGS
        }
        self.length_scale_factors_mks = {
            "Simulation UL": 1.0,
            "cm (CGS)": 0.01,
            "m (MKS)": 1.0,
            "AU": 1.495978707e11,         # MKS
            "Earth radii": 6.371e6,       # MKS
            "Jupiter radii": 7.1492e7,    # MKS
            "Solar radii": 6.957e8        # MKS
        }

        self.init_ui()

        self.slice_type = "theta"  # default
        self.last_slice_str = ""

    def init_ui(self):
        self.setFont(QFont("Segoe UI", 13))

        logo_label = QLabel()
        # Reduce logo size to occupy less vertical space
        logo_pixmap = QPixmap("fargopy_logo.png")
        logo_pixmap = logo_pixmap.scaledToWidth(300, Qt.SmoothTransformation)  # <-- previously 340
        logo_label.setPixmap(logo_pixmap)
        logo_label.setAlignment(Qt.AlignCenter)
        logo_label.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)

        controls_group = QGroupBox("Visualization controls")
        controls_group.setFont(QFont("Segoe UI", 15, QFont.Bold))
        controls_layout = QGridLayout()
        controls_layout.setHorizontalSpacing(14)
        controls_layout.setVerticalSpacing(10)

        # --- Path selection for simulation (FIRST) ---
        self.path_line = QLineEdit()
        self.path_line.setText("")
        self.path_line.setPlaceholderText("Select simulation path...")
        self.path_line.setReadOnly(True)
        self.browse_button = QPushButton("Browse...")
        self.browse_button.setStyleSheet("""
            QPushButton {
                background-color: #ff9800;
                color: white;
                font-size: 13px;
                font-weight: bold;
                border-radius: 8px;
                padding: 6px 12px;
            }
            QPushButton:hover {
                background-color: #e65100;
            }
        """)
        controls_layout.addWidget(QLabel("Simulation path:"), 0, 0)
        controls_layout.addWidget(self.path_line, 0, 1)
        controls_layout.addWidget(self.browse_button, 1, 1)

        # --- Simulation Info/Units button ---
        self.info_button = QPushButton("Simulation Info / Units")
        self.info_button.setStyleSheet("""
            QPushButton {
                background-color: #ff9800;
                color: white;
                font-size: 14px;
                font-weight: bold;
                border-radius: 8px;
                padding: 8px 16px;
            }
            QPushButton:hover {
                background-color: #e65100;
            }
        """)
        self.info_button.setEnabled(False)
        controls_layout.addWidget(self.info_button, 1, 0)

        # --- Snapshot (disabled until sim loaded) ---
        self.time_slider = QSpinBox()
        self.time_slider.setEnabled(False)
        controls_layout.addWidget(QLabel("Snapshot:"), 2, 0)
        controls_layout.addWidget(self.time_slider, 2, 1)

        # --- Slices: compact and aligned (disabled until sim loaded) ---
        slice_grid = QGridLayout()
        slice_grid.setHorizontalSpacing(6)
        slice_grid.setVerticalSpacing(4)
        slice_grid.addWidget(QLabel(""), 0, 0, alignment=Qt.AlignCenter)
        min_label = QLabel("min")
        min_label.setAlignment(Qt.AlignCenter)
        slice_grid.addWidget(min_label, 0, 1)
        max_label = QLabel("max")
        max_label.setAlignment(Qt.AlignCenter)
        slice_grid.addWidget(max_label, 0, 2)

        self.r_min = QLineEdit()
        self.r_max = QLineEdit()
        self.theta_min = QLineEdit()
        self.theta_max = QLineEdit()
        self.phi_min = QLineEdit()
        self.phi_max = QLineEdit()
        for edit in [self.r_min, self.r_max, self.theta_min, self.theta_max, self.phi_min, self.phi_max]:
            edit.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            edit.setMaximumHeight(32)
            edit.setEnabled(False)

        slice_grid.addWidget(QLabel("r:"), 1, 0, alignment=Qt.AlignRight | Qt.AlignVCenter)
        slice_grid.addWidget(self.r_min, 1, 1)
        slice_grid.addWidget(self.r_max, 1, 2)
        slice_grid.addWidget(QLabel("θ:"), 2, 0, alignment=Qt.AlignRight | Qt.AlignVCenter)
        slice_grid.addWidget(self.theta_min, 2, 1)
        slice_grid.addWidget(self.theta_max, 2, 2)
        slice_grid.addWidget(QLabel("φ:"), 3, 0, alignment=Qt.AlignRight | Qt.AlignVCenter)
        slice_grid.addWidget(self.phi_min, 3, 1)
        slice_grid.addWidget(self.phi_max, 3, 2)

        # --- Slice type selector inside the slice box ---
        self.slice_type_combo = QComboBox()
        self.slice_type_combo.addItems(['theta', 'phi'])
        self.slice_type_combo.setCurrentText('theta')
        self.slice_type_combo.currentTextChanged.connect(self.on_slice_type_change)
        slice_grid.addWidget(QLabel("Slice type:"), 4, 0)
        slice_grid.addWidget(self.slice_type_combo, 4, 1, 1, 2)

        slice_box = QGroupBox("Slices")
        slice_box.setLayout(slice_grid)
        slice_box.setStyleSheet("""
            QGroupBox {
                font-weight: normal;
                border: 1px solid #bdbdbd;
                border-radius: 6px;
                margin-top: 4px;
                padding: 2px 2px 2px 2px;
                color: #f5f5f5;
                background: #23272b;
            }
        """)
        slice_box.setSizePolicy(slice_box.sizePolicy().horizontalPolicy(), QSizePolicy.Fixed)
        controls_layout.addWidget(slice_box, 3, 0, 1, 2, alignment=Qt.AlignLeft | Qt.AlignTop)

        # --- Other controls (disabled until sim loaded) ---
        self.res_slider = QSpinBox()
        self.res_slider.setMinimum(50)
        self.res_slider.setMaximum(1000)
        self.res_slider.setSingleStep(10)
        self.res_slider.setValue(500)
        self.res_slider.setEnabled(False)
        controls_layout.addWidget(QLabel("Resolution:"), 4, 0)
        controls_layout.addWidget(self.res_slider, 4, 1)

        self.interp_toggle = QCheckBox("Interpolate")
        self.interp_toggle.setEnabled(False)
        controls_layout.addWidget(self.interp_toggle, 5, 0)

        self.streamlines_toggle = QCheckBox("Streamlines")
        self.streamlines_toggle.setEnabled(False)
        controls_layout.addWidget(self.streamlines_toggle, 5, 1)

        self.density_slider = QDoubleSpinBox()
        self.density_slider.setMinimum(1)
        self.density_slider.setMaximum(10)
        self.density_slider.setSingleStep(0.5)
        self.density_slider.setValue(3)
        self.density_slider.setEnabled(False)
        controls_layout.addWidget(QLabel("Streamline density:"), 6, 0)
        controls_layout.addWidget(self.density_slider, 6, 1)

        self.hill_frac_slider = QDoubleSpinBox()
        self.hill_frac_slider.setMinimum(0.1)
        self.hill_frac_slider.setMaximum(2.0)
        self.hill_frac_slider.setSingleStep(0.05)
        self.hill_frac_slider.setValue(1.0)
        self.hill_frac_slider.setEnabled(False)
        controls_layout.addWidget(QLabel("Hill fraction:"), 7, 0)
        controls_layout.addWidget(self.hill_frac_slider, 7, 1)

        self.show_circle_toggle = QCheckBox("Show Hill")
        self.show_circle_toggle.setEnabled(False)
        controls_layout.addWidget(self.show_circle_toggle, 8, 0)

        self.cmap_dropdown = QComboBox()
        self.cmap_dropdown.addItems(['Spectral_r', 'viridis', 'plasma', 'inferno', 'magma', 'cividis', 'YlGnBu', 'cubehelix', 'twilight', 'turbo'])
        self.cmap_dropdown.setEnabled(False)
        controls_layout.addWidget(QLabel("Colormap:"), 9, 0)
        controls_layout.addWidget(self.cmap_dropdown, 9, 1)

        # --- New dropdown for streamlines colormap ---
        self.stream_cmap_dropdown = QComboBox()
        self.stream_cmap_dropdown.addItems(['Spectral_r', 'viridis', 'plasma', 'inferno', 'magma', 'cividis', 'YlGnBu', 'cubehelix', 'twilight', 'turbo'])
        self.stream_cmap_dropdown.setEnabled(False)
        controls_layout.addWidget(QLabel("Streamlines cmap:"), 10, 0)
        controls_layout.addWidget(self.stream_cmap_dropdown, 10, 1)
        # Adjust indices of following controls +1
        self.map_dropdown = QComboBox()
        self.map_dropdown.addItems(['Density', 'Energy', 'Velocity'])
        self.map_dropdown.setEnabled(False)
        controls_layout.addWidget(QLabel("Map type:"), 11, 0)
        controls_layout.addWidget(self.map_dropdown, 11, 1)

        self.vel_dropdown = QComboBox()
        self.vel_dropdown.addItems(['vx', 'vy', 'vz'])
        self.vel_dropdown.setEnabled(False)
        controls_layout.addWidget(QLabel("Velocity component:"), 12, 0)
        controls_layout.addWidget(self.vel_dropdown, 12, 1)

        self.update_button = QPushButton("Update plot")
        self.update_button.setIcon(QIcon.fromTheme("view-refresh"))
        self.update_button.setEnabled(False)
        controls_layout.addWidget(self.update_button, 13, 0, 1, 2)

        self.status_label = QLabel("")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setFont(QFont("Segoe UI", 12, QFont.Bold))
        controls_layout.addWidget(self.status_label, 14, 0, 1, 2)

        # --- Length scale selector for axes (always allow all units) ---
        self.length_scale_combo = QComboBox()
        self.length_scale_combo.addItems([
            "Simulation UL", "cm (CGS)", "m (MKS)", "AU", "Earth radii", "Jupiter radii", "Solar radii"
        ])
        self.length_scale_combo.setCurrentText("Simulation UL")
        controls_layout.addWidget(QLabel("Length axis in:"), 15, 0)
        controls_layout.addWidget(self.length_scale_combo, 15, 1)
        self.length_scale_combo.currentTextChanged.connect(lambda _: self.plot_density())

        # Remove slice_type_combo from controls_layout if present (now inside slice_box)
        # ---

        controls_group.setLayout(controls_layout)
        controls_group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                border: 1.5px solid #1976D2;
                border-radius: 8px;
                margin-top: 8px;
                padding: 16px;
                color: #f5f5f5;
            }
            QGroupBox:title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 3px 0 3px;
            }
        """)

        self.update_button.setStyleSheet("""
            QPushButton {
                background-color: #ff9800;
                color: white;
                font-size: 15px;
                font-weight: bold;
                border-radius: 8px;
                padding: 10px 20px;
            }
            QPushButton:hover {
                background-color: #e65100;
            }
        """)
        for widget in [self.time_slider, self.res_slider, self.density_slider, self.hill_frac_slider]:
            widget.setStyleSheet("""
                QSpinBox, QDoubleSpinBox {
                    background: #2c3136;
                    border: 1px solid #444;
                    border-radius: 4px;
                    padding: 2px 4px;
                    color: #f5f5f5;
                    font-size: 14px;
                }
            """)

        left_panel_widget = QWidget()
        left_panel_layout = QVBoxLayout(left_panel_widget)
        left_panel_layout.setSpacing(8)  # <-- reduce spacing between widgets
        left_panel_layout.setContentsMargins(24, 12, 24, 12)  # <-- reduce top/bottom margin
        left_panel_layout.addWidget(logo_label, alignment=Qt.AlignHCenter)
        left_panel_layout.addWidget(controls_group, stretch=1)  # <-- prioritize space for controls
        # Remove addStretch(1) so controls are not pushed down
        # left_panel_layout.addStretch(1)  # <-- remove this line

        left_panel_widget.setMinimumWidth(420)
        left_panel_widget.setMaximumWidth(520)
        left_panel_widget.setStyleSheet("""
            QWidget {
                background-color: #23272b;
            }
            QLabel, QCheckBox {
                color: #f5f5f5;
                background: #23272b;
                font-size: 15px;
            }
            QGroupBox {
                color: #f5f5f5;
            }
            QLineEdit, QSpinBox, QDoubleSpinBox {
                background: #2c3136;
                border: 1px solid #444;
                color: #f5f5f5;
                font-size: 15px;
            }
            QComboBox {
                background: #2c3136;
                border: 1px solid #444;
                color: #f5f5f5;
                selection-background-color: #1976D2;
                selection-color: #fff;
                font-size: 15px;
            }
            QComboBox QAbstractItemView {
                background: #23272b;
                color: #f5f5f5;
                selection-background-color: #1976D2;
                selection-color: #fff;
                font-size: 15px;
            }
            QPushButton {
                background-color: #ff9800;
                color: white;
                font-size: 15px;
            }
            QPushButton:hover {
                background-color: #e65100;
            }
        """)

        self.figure = Figure(figsize=(7, 5))
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)

        self.show_logo_on_canvas()

        right_panel = QVBoxLayout()
        right_panel.addWidget(self.toolbar)
        right_panel.addWidget(self.canvas)

        h_layout = QHBoxLayout()
        h_layout.addWidget(left_panel_widget, 0)
        line = QFrame()
        line.setFrameShape(QFrame.VLine)
        line.setFrameShadow(QFrame.Sunken)
        h_layout.addWidget(line)
        h_layout.addLayout(right_panel, 1)

        self.setLayout(h_layout)

        # Connect signals
        self.update_button.clicked.connect(self.update_plot)
        self.map_dropdown.currentTextChanged.connect(self.on_map_change)
        self.browse_button.clicked.connect(self.select_simulation_path)
        self.info_button.clicked.connect(self.show_sim_info)
        for edit in [self.r_min, self.r_max, self.theta_min, self.theta_max, self.phi_min, self.phi_max]:
            edit.editingFinished.connect(lambda e=edit: self.normalize_decimal(e))
            edit.editingFinished.connect(self.on_slice_change)

        self.canvas.mpl_connect('button_release_event', self.on_zoom_release)

    def normalize_decimal(self, lineedit):
        text = lineedit.text()
        if ',' in text:
            lineedit.setText(text.replace(',', '.'))

    def show_logo_on_canvas(self):
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.axis('off')
        ax.set_facecolor('#23272b')
        self.figure.set_facecolor('#23272b')
        try:
            import matplotlib.image as mpimg
            img = mpimg.imread("fargopy_logo.png")
            ax.imshow(img, aspect='auto')
        except Exception:
            ax.text(0.5, 0.5, "FARGOpy", fontsize=40, ha='center', va='center', color='white')
        self.canvas.draw()

    def select_simulation_path(self):
        path = QFileDialog.getExistingDirectory(self, "Select simulation output directory")
        if path:
            self.path_line.setText(path)
            self.load_simulation(path)

    def load_simulation(self, path):
        self.sim = fp.Simulation(output_dir=path)
        # self.sim.units('CGS')  # <-- Remove this line, respeta las unidades originales
        self.time_slider.setEnabled(True)
        self.time_slider.setMinimum(0)
        self.time_slider.setMaximum(self.sim._get_nsnaps()-1)
        self.time_slider.setValue(1)
        # --- Reset all slice fields and state ---
        self.r_min.setText("")
        self.r_max.setText("")
        self.theta_min.setText(str(self.sim.domains.theta.max()))
        self.theta_max.setText('')
        self.phi_min.setText("")
        self.phi_max.setText("")
        self.last_slice_str = ""  # Reset slice string
        for edit in [self.r_min, self.r_max, self.theta_min, self.theta_max, self.phi_min, self.phi_max]:
            edit.setEnabled(True)
        self.res_slider.setEnabled(True)
        self.interp_toggle.setEnabled(True)
        self.streamlines_toggle.setEnabled(True)
        self.density_slider.setEnabled(True)
        self.hill_frac_slider.setEnabled(True)
        self.show_circle_toggle.setEnabled(True)
        self.cmap_dropdown.setEnabled(True)
        self.stream_cmap_dropdown.setEnabled(True)  # <-- habilita el nuevo dropdown
        self.map_dropdown.setEnabled(True)
        self.vel_dropdown.setEnabled(True)
        self.update_button.setEnabled(True)
        self.info_button.setEnabled(True)
        self.on_map_change(self.map_dropdown.currentText())
        self.plot_density()

    def show_sim_info(self):
        if self.sim is not None:
            dlg = SimInfoDialog(self.sim, self)
            dlg.exec_()

    def on_slice_type_change(self, text):
        self.slice_type = text
        # When changing slice type, reset fields according to convention
        if self.slice_type == "theta":
            self.theta_min.setText(str(self.sim.domains.theta.max()))
            self.theta_max.setText('')
            self.phi_min.setText("")
            self.phi_max.setText("")
        else:
            self.phi_min.setText("0")
            self.phi_max.setText("0")
            self.theta_min.setText("")
            self.theta_max.setText("")
        self.r_min.setText("")
        self.r_max.setText("")
        self.last_slice_str = ""  # <-- Ensure previous slice is cleared
        self.plot_density()

    def on_zoom_release(self, event):
        # Only if user used zoom (right button or wheel)
        if event.button not in [1, 3]:
            return
        ax = self.figure.gca()
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        scale_label = self.length_scale_combo.currentText()

        x_min_plot, x_max_plot = xlim
        y_min_plot, y_max_plot = ylim
        x_min_ul = x_min_plot
        x_max_ul = x_max_plot
        y_min_ul = y_min_plot
        y_max_ul = y_max_plot

        # Now, x_ul, y_ul are in UL (simulation units)
        corners = [
            (x_min_ul, y_min_ul),
            (x_min_ul, y_max_ul),
            (x_max_ul, y_min_ul),
            (x_max_ul, y_max_ul)
        ]

        r_list = []
        theta_list = []
        phi_list = []
        for x, y in corners:
            if self.slice_type == "theta":
                z = 0.0
                r = np.sqrt(x**2 + y**2 + z**2)
                theta = np.arccos(z / r) if r != 0 else 0.0
                phi = np.arctan2(y, x)
            else:
                y_ = 0.0
                z = y
                r = np.sqrt(x**2 + y_**2 + z**2)
                theta = np.arccos(z / r) if r != 0 else 0.0
                phi = np.arctan2(y_, x)
            r_list.append(r)
            theta_list.append(theta)
            phi_list.append(phi)


            r_min = np.min(r_list)
            r_max = np.max(r_list)

        theta_min = np.min(theta_list)
        theta_max = np.max(theta_list)
        phi_min = np.min(phi_list)
        phi_max = np.max(phi_list)

        if self.slice_type == "theta":
            theta_val = str(self.sim.domains.theta.max())
            self.theta_min.setText(theta_val)
            self.theta_max.setText(theta_val)
            self.r_min.setText(f"{r_min:.5f}")
            self.r_max.setText(f"{r_max:.5f}")
            self.phi_min.setText(f"{phi_min:.5f}")
            self.phi_max.setText(f"{phi_max:.5f}")
            slice_str = (
                f"theta={theta_val},"
                f"r=[{r_min:.5f},{r_max:.5f}],"
                f"phi=[{phi_min:.5f},{phi_max:.5f}]"
            )
        else:
            phi_val = self.phi_min.text() if self.phi_min.text() != "" else "0"
            self.phi_min.setText(phi_val)
            self.phi_max.setText(phi_val)
            self.r_min.setText(f"{r_min:.5f}")
            self.r_max.setText(f"{r_max:.5f}")
            self.theta_min.setText(f"{theta_min:.5f}")
            self.theta_max.setText(f"{theta_max:.5f}")
            slice_str = (
                f"phi={phi_val},"
                f"r=[{r_min:.5f},{r_max:.5f}],"
                f"theta=[{theta_min:.5f},{theta_max:.5f}]"
            )
        self.last_slice_str = slice_str
        self._fields_edited = False
        # DO NOT call self.plot_density() here

    def build_slice_str(self):
        # Si el usuario ha editado algún campo manualmente DESDE el último update, ignora last_slice_str
        if getattr(self, "_fields_edited", False):
            return self._manual_slice_str()
        # Si no, usa el slice del zoom si existe
        if self.last_slice_str:
            return self.last_slice_str
        return self._manual_slice_str()

    def _manual_slice_str(self):
        def norm(txt):
            return txt.replace(',', '.').strip()
        r_min_val = norm(self.r_min.text())
        r_max_val = norm(self.r_max.text())
        theta_min_val = norm(self.theta_min.text())
        theta_max_val = norm(self.theta_max.text())
        phi_min_val = norm(self.phi_min.text())
        phi_max_val = norm(self.phi_max.text())

        slice_parts = []
        # Si ambos r_min y r_max están vacíos, y ambos phi_min y phi_max están vacíos, y theta está definido, es un slice en theta
        if not r_min_val and not r_max_val and not phi_min_val and not phi_max_val and theta_min_val:
            slice_parts.append(f"theta={theta_min_val}")
        else:
            if r_min_val and r_max_val:
                if r_min_val == r_max_val:
                    slice_parts.append(f"r={r_min_val}")
                else:
                    slice_parts.append(f"r=[{r_min_val},{r_max_val}]")
            elif r_min_val:
                slice_parts.append(f"r={r_min_val}")
            elif r_max_val:
                slice_parts.append(f"r={r_max_val}")
            if theta_min_val and theta_max_val:
                if theta_min_val == theta_max_val:
                    slice_parts.append(f"theta={theta_min_val}")
                else:
                    slice_parts.append(f"theta=[{theta_min_val},{theta_max_val}]")
            elif theta_min_val:
                slice_parts.append(f"theta={theta_min_val}")
            elif theta_max_val:
                slice_parts.append(f"theta={theta_max_val}")
            if phi_min_val and phi_max_val:
                if phi_min_val == phi_max_val:
                    slice_parts.append(f"phi={phi_min_val}")
                else:
                    slice_parts.append(f"phi=[{phi_min_val},{phi_max_val}]")
            elif phi_min_val:
                slice_parts.append(f"phi={phi_min_val}")
            elif phi_max_val:
                slice_parts.append(f"phi={phi_max_val}")
        return ",".join(slice_parts)

    def on_slice_change(self):
        # Marca que hubo edición manual y limpia el slice del zoom
        self._fields_edited = True
        self.last_slice_str = ""

    def update_plot(self):
        # Llamar a esto solo desde el botón Update plot
        self._fields_edited = False  # Reset flag, ahora los campos son la fuente de verdad
        self.last_slice_str = ""     # Siempre usar los campos manuales al actualizar
        self.plot_density()

    def on_map_change(self, text):
        if text == 'Velocity':
            self.vel_dropdown.setEnabled(True)
        else:
            self.vel_dropdown.setEnabled(False)

    def plot_density(self, unitsys_override=None):
        import re

        if not self.sim:
            return

        self.status_label.setText("🐍 The Python snake is exploring the disk...")
        QApplication.processEvents()

        slice_str = self.build_slice_str()
        res = self.res_slider.value()
        interpolate = self.interp_toggle.isChecked()
        show_streamlines = self.streamlines_toggle.isChecked()
        stream_density = self.density_slider.value()
        hill_frac = self.hill_frac_slider.value()
        show_circle = self.show_circle_toggle.isChecked()
        cmap = self.cmap_dropdown.currentText()
        stream_cmap = self.stream_cmap_dropdown.currentText()  # <-- use selected streamlines cmap
        map_type = self.map_dropdown.currentText()
        vel_comp = self.vel_dropdown.currentText()
        n = self.time_slider.value()

        # --- UNITS ---
        sim_unitsys = getattr(self.sim, "unitsystem", "cgs").lower()
        length_unit = self.length_scale_combo.currentText()

        unit_factors_cgs = {
            "Simulation UL": self.sim.UL,
            "cm (CGS)": 1.0,
            "m (MKS)": 100.0,
            "AU": 1.495978707e13,
            "Earth radii": 6.371e8,
            "Jupiter radii": 7.1492e9,
            "Solar radii": 6.957e10
        }
        unit_factors_mks = {
            "Simulation UL": self.sim.UL,
            "cm (CGS)": 0.01,
            "m (MKS)": 1.0,
            "AU": 1.495978707e11,
            "Earth radii": 6.371e6,
            "Jupiter radii": 7.1492e7,
            "Solar radii": 6.957e8
        }

        if sim_unitsys == "mks":
            unit_factors = unit_factors_mks
            axis_unit_label_map = {
                "Simulation UL": f"UL (m)",
                "cm (CGS)": "cm",
                "m (MKS)": "m",
                "AU": "AU",
                "Earth radii": "R$_\\oplus$",
                "Jupiter radii": "R$_J$",
                "Solar radii": "R$_\\odot$"
            }
        elif sim_unitsys == "cgs":
            unit_factors = unit_factors_cgs
            axis_unit_label_map = {
                "Simulation UL": f"UL",
                "cm (CGS)": "cm",
                "m (MKS)": "m",
                "AU": "AU",
                "Earth radii": "R$_\\oplus$",
                "Jupiter radii": "R$_J$",
                "Solar radii": "R$_\\odot$"
            }
        else:
            unit_factors = unit_factors_cgs  # Default to CGS if unknown
            axis_unit_label_map = {
                "Simulation UL": f"UL",
                "cm (CGS)": "cm",
                "m (MKS)": "m",
                "AU": "AU",
                "Earth radii": "R$_\\oplus$",
                "Jupiter radii": "R$_J$",
                "Solar radii": "R$_\\odot$"
            }

        scale_factor = unit_factors.get(length_unit, self.sim.UL)

        if sim_unitsys == "cgs":
            dens_unit = "g/cm³"
            vel_unit = "cm/s"
            v_factor = self.sim.UL / self.sim.UT  # cm/s
        elif sim_unitsys == "mks":
            dens_unit = "kg/m³"
            vel_unit = "m/s"
            v_factor = self.sim.UL / self.sim.UT  # m/s
        else:
            dens_unit = r"UM/UL$^3$"
            vel_unit = "UL/UT"
            v_factor = self.sim.UL / self.sim.UT  # cm/s
        # Determine axes and mesh names based on fixed coordinates
        def is_fixed(var, slice_str):
            match = re.search(rf'{var}=([^\[\],]+)', slice_str.replace(' ', ''))
            return match is not None

        if is_fixed('theta', slice_str):
            mesh_x_name = 'var1_mesh'
            mesh_y_name = 'var2_mesh'
            self.vel_dropdown.clear()
            self.vel_dropdown.addItems(['vx', 'vy'])
        elif is_fixed('phi', slice_str):
            mesh_x_name = 'var1_mesh'
            mesh_y_name = 'var3_mesh'
            self.vel_dropdown.clear()
            self.vel_dropdown.addItems(['vx', 'vz'])
        else:
            mesh_x_name = 'var1_mesh'
            mesh_y_name = 'var2_mesh'
            self.vel_dropdown.clear()
            self.vel_dropdown.addItems(['vx', 'vy'])

        # Load data according to selection
        if map_type == 'Density':
            gasdens, gasv = self.sim.load_field(
                fields=['gasdens', 'gasv'],
                slice=slice_str,
                snapshot=n,
                interpolate=True
            )
        elif map_type == 'Energy':
            gasenergy = self.sim.load_field(
                fields='gasenergy',
                slice=slice_str,
                snapshot=n,
                interpolate=True
            )
            gasv = self.sim.load_field(
                fields='gasv',
                slice=slice_str,
                snapshot=n,
                interpolate=True
            )
        elif map_type == 'Velocity':
            gasv = self.sim.load_field(
                fields='gasv',
                slice=slice_str,
                snapshot=n,
                interpolate=True
            )

        # --- Interpolation ---
        if interpolate:
            if mesh_y_name == 'var2_mesh':
                xmin, xmax = getattr(gasv, mesh_x_name)[0].min(), getattr(gasv, mesh_x_name)[0].max()
                ymin, ymax = getattr(gasv, mesh_y_name)[0].min(), getattr(gasv, mesh_y_name)[0].max()
                xs = np.linspace(xmin, xmax, res)
                ys = np.linspace(ymin, ymax, res)
                X, Y = np.meshgrid(xs, ys)
                if map_type == 'Density':
                    data_map = gasdens.evaluate(time=n, var1=X, var2=Y)
                    data_map = np.log10(data_map * self.sim.URHO)
                    vel = gasv.evaluate(time=n, var1=X, var2=Y)
                    vx = vel[0]
                    vy = vel[1]
                    vmag = np.sqrt(vx**2 + vy**2)
                elif map_type == 'Energy':
                    data_map = gasenergy.evaluate(time=n, var1=X, var2=Y)
                    vel = gasv.evaluate(time=n, var1=X, var2=Y)
                    vx = vel[0]
                    vy = vel[1]
                    vmag = np.sqrt(vx**2 + vy**2)
                elif map_type == 'Velocity':
                    vel = gasv.evaluate(time=n, var1=X, var2=Y)
                    idx = {'vx': 0, 'vy': 1, 'vz': 2}[vel_comp]
                    data_map = vel[idx]
                    vx = vel[0]
                    vy = vel[1]
                    vmag = np.sqrt(vx**2 + vy**2)
            else:
                xmin, xmax = getattr(gasv, mesh_x_name)[0].min(), getattr(gasv, mesh_x_name)[0].max()
                zmin, zmax = getattr(gasv, mesh_y_name)[0].min(), getattr(gasv, mesh_y_name)[0].max()
                xs = np.linspace(xmin, xmax, res)
                zs = np.linspace(zmin, zmax, res)
                X, Y = np.meshgrid(xs, zs)
                if map_type == 'Density':
                    data_map = gasdens.evaluate(time=n, var1=X, var3=Y)
                    data_map = np.log10(data_map * self.sim.URHO)
                    vel = gasv.evaluate(time=n, var1=X, var3=Y)
                    vx = vel[0]
                    vy = vel[2]
                    vmag = np.sqrt(vx**2 + vy**2)
                elif map_type == 'Energy':
                    data_map = gasenergy.evaluate(time=n, var1=X, var3=Y)
                    vel = gasv.evaluate(time=n, var1=X, var3=Y)
                    vx = vel[0]
                    vy = vel[2]
                    vmag = np.sqrt(vx**2 + vy**2)
                elif map_type == 'Velocity':
                    vel = gasv.evaluate(time=n, var1=X, var3=Y)
                    idx = {'vx': 0, 'vy': 1, 'vz': 2}[vel_comp]
                    data_map = vel[idx]
                    vx = vel[0]
                    vy = vel[2]
                    vmag = np.sqrt(vx**2 + vy**2)
        else:
            if mesh_y_name == 'var2_mesh':
                X = getattr(gasv, mesh_x_name)[0]
                Y = getattr(gasv, mesh_y_name)[0]
                if map_type == 'Density':
                    data_map = np.log10(gasdens.gasdens_mesh[0] * self.sim.URHO)
                    vel = gasv.gasv_mesh[0]
                    vx = vel[0]
                    vy = vel[1]
                    vmag = np.sqrt(vx**2 + vy**2)
                elif map_type == 'Energy':
                    data_map = gasenergy.gasenergy_mesh[0]
                    vel = gasv.gasv_mesh[0]
                    vx = vel[0]
                    vy = vel[1]
                    vmag = np.sqrt(vx**2 + vy**2)
                elif map_type == 'Velocity':
                    vel = gasv.gasv_mesh[0]
                    idx = {'vx': 0, 'vy': 1, 'vz': 2}[vel_comp]
                    data_map = vel[idx]
                    vx = vel[0]
                    vy = vel[1]
                    vmag = np.sqrt(vx**2 + vy**2)
            else:
                X = getattr(gasv, mesh_x_name)[0]
                Y = getattr(gasv, mesh_y_name)[0]
                if map_type == 'Density':
                    data_map = np.log10(gasdens.gasdens_mesh[0] * self.sim.URHO)
                    vel = gasv.gasv_mesh[0]
                    vx = vel[0]
                    vy = vel[2]
                    vmag = np.sqrt(vx**2 + vy**2)
                elif map_type == 'Energy':
                    data_map = gasenergy.gasenergy_mesh[0]
                    vel = gasv.gasv_mesh[0]
                    vx = vel[0]
                    vy = vel[2]
                    vmag = np.sqrt(vx**2 + vy**2)
                elif map_type == 'Velocity':
                    vel = gasv.gasv_mesh[0]
                    idx = {'vx': 0, 'vy': 1, 'vz': 2}[vel_comp]
                    data_map = vel[idx]
                    vx = vel[0]
                    vy = vel[2]
                    vmag = np.sqrt(vx**2 + vy**2)

        # --- Apply units and scaling to axes and velocity ---
        # X, Y are in simulation UL units. Convert to cm or m, then to target unit.
        X_plot = X * self.sim.UL / scale_factor
        Y_plot = Y * self.sim.UL / scale_factor

        if map_type == 'Velocity':
            vx = vx * v_factor
            vy = vy * v_factor
            vmag = vmag * v_factor
        # --- Always convert vmag to km/s for streamlines colorbar ---
        vmag_kms = None
        if show_streamlines and vmag is not None:
            # vmag está en cm/s o m/s, convertir a km/s
            if sim_unitsys == "cgs":
                vmag_kms = vmag
            elif sim_unitsys == "mks":
                vmag_kms = vmag
            else:
                vmag_kms = vmag

        # --- Masking and plotting ---
        # r in plot units (same as X_plot/Y_plot)
        r = np.sqrt(X_plot**2 + Y_plot**2)
        # Extract r_min/r_max from slice_str (these are in the original simulation units, no rescaling)
        r_match = re.search(r"r=\[([0-9\.]+),([0-9\.]+)\]", slice_str.replace(" ", ""))
        if r_match:
            # r_min_sim and r_max_sim are in simulation units (no rescaling)
            r_min_sim = float(r_match.group(1))
            r_max_sim = float(r_match.group(2))
            # Convert to plot units for masking
            r_min = r_min_sim * self.sim.UL / scale_factor
            r_max = r_max_sim * self.sim.UL / scale_factor
        else:
            r_min = None
            r_max = None

        # Mask using r in plot units
        if r_min is not None and r_max is not None:
            mask = (r >= r_min) & (r <= r_max)
            data_map = np.where(mask, data_map, np.nan)
            if show_streamlines and vx is not None and vy is not None and vmag is not None:
                vx = np.where(mask, vx, np.nan)
                vy = np.where(mask, vy, np.nan)
                vmag = np.where(mask, vmag, np.nan)

        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.set_facecolor('white')
        self.figure.set_facecolor('white')
        pcm = ax.pcolormesh(X_plot, Y_plot, data_map, shading='auto', cmap=cmap)
      # --- Axis label according to scaling ---
        axis_unit_label = axis_unit_label_map.get(length_unit, length_unit)
        xlabel, ylabel = f'X [{axis_unit_label}]', f'Y [{axis_unit_label}]'

        stream_obj = None
        if show_streamlines and vx is not None and vy is not None:
            stream_obj = ax.streamplot(
                X_plot, Y_plot, vx, vy,
                color=vmag_kms if vmag_kms is not None else None,
                linewidth=0.5,
                density=stream_density,
                cmap=stream_cmap,  # <-- use streamlines cmap
                arrowsize=1
            )

        planets = self.sim.load_planets(snapshot=n)
        if planets:
            center_x = planets[0].pos.x * self.sim.UL / scale_factor
            center_y = planets[0].pos.y * self.sim.UL / scale_factor
            radius = hill_frac * planets[0].hill_radius * self.sim.UL / scale_factor
        else:
            center_x = 0
            center_y = 0
            radius = 0

        if show_circle:
            if is_fixed('theta', slice_str):
                from matplotlib.patches import Circle
                circle = Circle((center_x, center_y), radius, color='red', fill=False, linestyle='--', linewidth=3,label=fr'${hill_frac:.1f}\,R_H$')
                ax.add_patch(circle)
            elif is_fixed('phi', slice_str):
                theta = np.linspace(0, np.pi, 100)
                x = center_x + radius * np.cos(theta)
                y = center_y + radius * np.sin(theta)
                ax.plot(x, y, color='red', linewidth=3,label=f'{hill_frac:.1f}'+r'$R_H$', linestyle='--')

        # Change font and size of axis labels
        font_properties = {'fontsize': 15, 'fontname': 'Serif'}
        ax.set_xlabel(xlabel, **font_properties)
        ax.set_ylabel(ylabel, **font_properties)
        fp.Plot.fargopy_mark(ax)
        ax.legend(fontsize=15, prop={'family': 'Serif'}, loc='upper left')
        ax.grid(0.1)

        # --- Colorbar label ---
        if show_streamlines and stream_obj is not None and vmag_kms is not None:
            cbar = self.figure.colorbar(stream_obj.lines, ax=ax)
            cbar.set_label('$|v|$ [km/s]', fontsize=15, fontname='Serif')
        else:
            if map_type == 'Density':
                cbar_label = r'$\log_{10}(\rho)$' + f' [{dens_unit}]'
            elif map_type == 'Energy':
                cbar_label = r'$\log_{10}(\mathrm{{energy}})$'
            else:
                cbar_label = f'{vel_comp} [{vel_unit}]'
            cbar = self.figure.colorbar(pcm, ax=ax)
            cbar.set_label(cbar_label, fontsize=18, fontname='Serif')

        self.canvas.draw()
        QTimer.singleShot(800, lambda: self.status_label.setText(""))

if __name__ == "__main__":
    print("Starting GUI...")
    app = QApplication(sys.argv)
    window = PlotInteractiveWindow()
    window.setWindowTitle("FARGOpy Interactive Plot")
    window.resize(1350, 800)
    window.show()
    print("Window shown. Running app...")
    sys.exit(app.exec_())