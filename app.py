import tkinter as tk
from tkinter import ttk, filedialog, messagebox, colorchooser
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from scipy.fft import fft, fftfreq
from scipy.signal.windows import blackman, hann, hamming
import json
import os
from datetime import datetime

class FFTAnalyzerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("FFT Analyzer - Flight Stand Data Analysis")
        self.root.geometry("1400x900")
        self.root.minsize(1600, 960)
        
        # Data storage
        self.df = None
        self.fft_results = {}  # Store multiple FFT results for combining
        self.current_colors = ['#000000', "#8B8B8B", "#0095ff", '#ff7f0e', "#22d322", "#ff0000", "#a94cff", '#8c564b']
        self.color_index = 0
        
        # Settings
        self.settings = {
            'peak_labels_count': 5,
            'default_colors': self.current_colors.copy(),
            'window_function': 'none',
            'peak_threshold_mode': 'relative',  # 'relative', 'absolute', or 'statistical'
            'peak_relative_threshold': 0.1,  # 10% of max amplitude
            'peak_absolute_threshold': 0.001,  # Absolute amplitude value
            'peak_statistical_factor': 1.0,  # Factor for statistical threshold (mean + factor * std)
            'peak_min_distance': 10,  # Minimum distance between peaks (in frequency bins)
            'skip_dc_component': True,  # Skip DC (0 Hz) component
            'peak_window_size': 3  # Window size for local maximum detection
        }
        
        self.setup_ui()
        self.load_settings()
    
    def setup_ui(self):
        # Create main notebook for tabs
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Main Analysis Tab
        self.main_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.main_frame, text="FFT Analysis")
        
        # Settings Tab
        self.settings_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.settings_frame, text="Settings")
        
        # Results Tab
        self.results_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.results_frame, text="Combined Results")
        
        self.setup_main_tab()
        self.setup_settings_tab()
        self.setup_results_tab()
    
    def setup_main_tab(self):
        # Create paned window for resizable sections
        paned = ttk.PanedWindow(self.main_frame, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True)
        
        # Left panel for controls
        left_frame = ttk.Frame(paned)
        paned.add(left_frame, weight=1)
        
        # Right panel for plot
        right_frame = ttk.Frame(paned)
        paned.add(right_frame, weight=3)
        
        self.setup_control_panel(left_frame)
        self.setup_plot_panel(right_frame)
    
    def setup_control_panel(self, parent):
        # File selection section
        file_frame = ttk.LabelFrame(parent, text="Data File Selection", padding="10")
        file_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.file_path = tk.StringVar()
        ttk.Label(file_frame, text="Selected File:").pack(anchor=tk.W)
        ttk.Entry(file_frame, textvariable=self.file_path, state="readonly").pack(fill=tk.X, pady=(5, 10))
        ttk.Button(file_frame, text="Select CSV File", command=self.select_file).pack(fill=tk.X)
        
        # Data configuration section
        data_frame = ttk.LabelFrame(parent, text="Data Configuration", padding="10")
        data_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Column selection
        ttk.Label(data_frame, text="Select Column:").pack(anchor=tk.W)
        self.column_var = tk.StringVar()
        self.column_combo = ttk.Combobox(data_frame, textvariable=self.column_var, state="readonly")
        self.column_combo.pack(fill=tk.X, pady=(5, 10))
        self.column_combo.bind('<<ComboboxSelected>>', self.on_column_selected)
        
        # Column rename
        ttk.Label(data_frame, text="Rename Column (optional):").pack(anchor=tk.W)
        self.column_name = tk.StringVar()
        ttk.Entry(data_frame, textvariable=self.column_name).pack(fill=tk.X, pady=(5, 10))
        
        # Data range selection
        range_section = ttk.LabelFrame(data_frame, text="Data Range Selection", padding="5")
        range_section.pack(fill=tk.X, pady=(0, 10))
        
        # Start line
        ttk.Label(range_section, text="Start Line (row number):").pack(anchor=tk.W)
        self.start_line_var = tk.IntVar(value=1)
        start_frame = ttk.Frame(range_section)
        start_frame.pack(fill=tk.X, pady=(5, 10))
        
        self.start_scale = ttk.Scale(start_frame, from_=1, to=10000, 
                                   variable=self.start_line_var, orient=tk.HORIZONTAL)
        self.start_scale.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        self.start_label = ttk.Label(start_frame, text="1")
        self.start_label.pack(side=tk.RIGHT, padx=(10, 0))
        self.start_scale.configure(command=self.update_start_label)
        
        # Number of lines
        ttk.Label(range_section, text="Number of Lines to Use:").pack(anchor=tk.W)
        self.lines_var = tk.IntVar(value=1000)
        lines_frame = ttk.Frame(range_section)
        lines_frame.pack(fill=tk.X, pady=(5, 10))
        
        self.lines_scale = ttk.Scale(lines_frame, from_=1, to=100000, 
                                   variable=self.lines_var, orient=tk.HORIZONTAL)
        self.lines_scale.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        self.lines_label = ttk.Label(lines_frame, text="1000")
        self.lines_label.pack(side=tk.RIGHT, padx=(10, 0))
        self.lines_scale.configure(command=self.update_lines_label)
        
        # Range info
        self.range_info = ttk.Label(range_section, text="Range: Row 1 to 1001 (1000 points)", 
                                   foreground="blue", font=("TkDefaultFont", 8))
        self.range_info.pack(anchor=tk.W, pady=(5, 0))        # Acquisition frequency
        ttk.Label(data_frame, text="Acquisition Frequency (Hz):").pack(anchor=tk.W)
        self.freq_var = tk.DoubleVar(value=1000.0)
        freq_frame = ttk.Frame(data_frame)
        freq_frame.pack(fill=tk.X, pady=(5, 10))
        
        ttk.Entry(freq_frame, textvariable=self.freq_var, width=10).pack(side=tk.LEFT)
        ttk.Label(freq_frame, text="Hz").pack(side=tk.LEFT, padx=(5, 0))
        
        # Study information
        study_frame = ttk.LabelFrame(parent, text="Study Information", padding="10")
        study_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(study_frame, text="Study Name:").pack(anchor=tk.W)
        self.study_name = tk.StringVar(value="FFT Analysis")
        ttk.Entry(study_frame, textvariable=self.study_name).pack(fill=tk.X, pady=(5, 10))
        
        # Window function
        ttk.Label(study_frame, text="Window Function:").pack(anchor=tk.W)
        self.window_var = tk.StringVar(value="none")
        window_combo = ttk.Combobox(study_frame, textvariable=self.window_var, 
                                values=["none", "blackman", "hann", "hamming"], 
                                state="readonly")
        window_combo.pack(fill=tk.X, pady=(5, 10))
        
        # Analysis button
        ttk.Button(study_frame, text="Run FFT Analysis", 
                command=self.run_fft_analysis, style="Accent.TButton").pack(fill=tk.X, pady=(10, 0))
        
        # Export section
        export_frame = ttk.LabelFrame(parent, text="Export", padding="10")
        export_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Button(export_frame, text="Export Data (CSV)", 
                command=self.export_data).pack(fill=tk.X, pady=(0, 5))
        ttk.Button(export_frame, text="Export Plot (PNG)", 
                command=self.export_plot).pack(fill=tk.X, pady=(0, 5))
        ttk.Button(export_frame, text="Save to Combined Results", 
                command=self.save_to_results).pack(fill=tk.X)
    
    def setup_plot_panel(self, parent):
        # Create matplotlib figure
        self.fig = Figure(figsize=(10, 6), dpi=100)
        self.ax = self.fig.add_subplot(111)
        
        # Create canvas
        self.canvas = FigureCanvasTkAgg(self.fig, parent)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        
        # Create toolbar
        toolbar = NavigationToolbar2Tk(self.canvas, parent)
        toolbar.update()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        
        # Initial empty plot
        self.ax.set_xlabel('Frequency (Hz)')
        self.ax.set_ylabel('Amplitude')
        self.ax.set_title('FFT Analysis Results')
        self.ax.grid(True, alpha=0.3)
        self.fig.tight_layout()
    
    def setup_settings_tab(self):
        settings_main = ttk.Frame(self.settings_frame)
        settings_main.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Create scrollable frame for settings
        canvas = tk.Canvas(settings_main)
        scrollbar = ttk.Scrollbar(settings_main, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Peak Detection Settings
        peak_frame = ttk.LabelFrame(scrollable_frame, text="Peak Detection Settings", padding="15")
        peak_frame.pack(fill=tk.X, pady=(0, 15))
        
        # Number of peaks to label
        peaks_count_frame = ttk.Frame(peak_frame)
        peaks_count_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(peaks_count_frame, text="Number of peaks to label:").pack(side=tk.LEFT)
        self.peak_count_var = tk.IntVar(value=self.settings['peak_labels_count'])
        peak_count_spinbox = ttk.Spinbox(peaks_count_frame, from_=0, to=50, width=5, 
                                       textvariable=self.peak_count_var)
        peak_count_spinbox.pack(side=tk.LEFT, padx=(10, 0))
        ttk.Label(peaks_count_frame, text="(0 = no labels)").pack(side=tk.LEFT, padx=(10, 0))
        
        # Peak detection threshold mode
        threshold_frame = ttk.Frame(peak_frame)
        threshold_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(threshold_frame, text="Threshold Mode:").pack(anchor=tk.W)
        self.threshold_mode_var = tk.StringVar(value=self.settings['peak_threshold_mode'])
        threshold_combo = ttk.Combobox(threshold_frame, textvariable=self.threshold_mode_var,
                                     values=["relative", "absolute", "statistical"], 
                                     state="readonly", width=15)
        threshold_combo.pack(anchor=tk.W, pady=(5, 0))
        threshold_combo.bind('<<ComboboxSelected>>', self.on_threshold_mode_changed)
        
        # Threshold value frames (will be shown/hidden based on mode)
        self.relative_frame = ttk.Frame(peak_frame)
        rel_label_frame = ttk.Frame(self.relative_frame)
        rel_label_frame.pack(fill=tk.X)
        ttk.Label(rel_label_frame, text="Relative Threshold (% of max amplitude):").pack(side=tk.LEFT)
        self.relative_threshold_var = tk.DoubleVar(value=self.settings['peak_relative_threshold'])
        rel_scale = ttk.Scale(self.relative_frame, from_=0.01, to=0.5, 
                            variable=self.relative_threshold_var, orient=tk.HORIZONTAL)
        rel_scale.pack(fill=tk.X, pady=(5, 0))
        self.rel_value_label = ttk.Label(self.relative_frame, text=f"{self.settings['peak_relative_threshold']:.2f}")
        self.rel_value_label.pack(anchor=tk.W)
        rel_scale.configure(command=self.update_relative_label)
        
        self.absolute_frame = ttk.Frame(peak_frame)
        abs_label_frame = ttk.Frame(self.absolute_frame)
        abs_label_frame.pack(fill=tk.X)
        ttk.Label(abs_label_frame, text="Absolute Threshold (amplitude value):").pack(side=tk.LEFT)
        self.absolute_threshold_var = tk.DoubleVar(value=self.settings['peak_absolute_threshold'])
        abs_entry = ttk.Entry(self.absolute_frame, textvariable=self.absolute_threshold_var, width=15)
        abs_entry.pack(anchor=tk.W, pady=(5, 0))
        
        self.statistical_frame = ttk.Frame(peak_frame)
        stat_label_frame = ttk.Frame(self.statistical_frame)
        stat_label_frame.pack(fill=tk.X)
        ttk.Label(stat_label_frame, text="Statistical Factor (mean + factor × std):").pack(side=tk.LEFT)
        self.statistical_factor_var = tk.DoubleVar(value=self.settings['peak_statistical_factor'])
        stat_scale = ttk.Scale(self.statistical_frame, from_=0.1, to=3.0, 
                             variable=self.statistical_factor_var, orient=tk.HORIZONTAL)
        stat_scale.pack(fill=tk.X, pady=(5, 0))
        self.stat_value_label = ttk.Label(self.statistical_frame, text=f"{self.settings['peak_statistical_factor']:.1f}")
        self.stat_value_label.pack(anchor=tk.W)
        stat_scale.configure(command=self.update_statistical_label)
        
        # Minimum distance between peaks
        distance_frame = ttk.Frame(peak_frame)
        distance_frame.pack(fill=tk.X, pady=(10, 10))
        
        ttk.Label(distance_frame, text="Minimum distance between peaks (frequency bins):").pack(anchor=tk.W)
        self.min_distance_var = tk.IntVar(value=self.settings['peak_min_distance'])
        distance_scale = ttk.Scale(distance_frame, from_=1, to=100, 
                                 variable=self.min_distance_var, orient=tk.HORIZONTAL)
        distance_scale.pack(fill=tk.X, pady=(5, 0))
        self.distance_label = ttk.Label(distance_frame, text=str(self.settings['peak_min_distance']))
        self.distance_label.pack(anchor=tk.W)
        distance_scale.configure(command=self.update_distance_label)
        
        # Window size for local maximum detection
        window_frame = ttk.Frame(peak_frame)
        window_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(window_frame, text="Peak detection window size:").pack(side=tk.LEFT)
        self.window_size_var = tk.IntVar(value=self.settings['peak_window_size'])
        window_spinbox = ttk.Spinbox(window_frame, from_=1, to=10, width=5, 
                                   textvariable=self.window_size_var)
        window_spinbox.pack(side=tk.LEFT, padx=(10, 0))
        ttk.Label(window_frame, text="(larger = fewer peaks)").pack(side=tk.LEFT, padx=(10, 0))
        
        # Skip DC component
        self.skip_dc_var = tk.BooleanVar(value=self.settings['skip_dc_component'])
        ttk.Checkbutton(peak_frame, text="Skip DC component (0 Hz)", 
                       variable=self.skip_dc_var).pack(anchor=tk.W, pady=(10, 0))
        
        # Show appropriate threshold frame
        self.on_threshold_mode_changed()
        
        # Color settings
        color_frame = ttk.LabelFrame(scrollable_frame, text="Color Settings", padding="15")
        color_frame.pack(fill=tk.X, pady=(0, 15))
        
        ttk.Button(color_frame, text="Reset Colors to Default", 
                command=self.reset_colors).pack(anchor=tk.W, pady=(0, 10))
        
        # Current colors display
        colors_display_frame = ttk.Frame(color_frame)
        colors_display_frame.pack(fill=tk.X)
        
        ttk.Label(colors_display_frame, text="Current Colors:").pack(anchor=tk.W)
        self.colors_frame = ttk.Frame(colors_display_frame)
        self.colors_frame.pack(fill=tk.X, pady=(5, 0))
        self.update_color_display()
        
        # Save settings button
        ttk.Button(scrollable_frame, text="Save Settings", 
                command=self.save_settings, style="Accent.TButton").pack(pady=20)
        
        # Pack scrollable components
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
    
    def setup_results_tab(self):
        results_main = ttk.Frame(self.results_frame)
        results_main.pack(fill=tk.BOTH, expand=True)
        
        # Create paned window
        results_paned = ttk.PanedWindow(results_main, orient=tk.HORIZONTAL)
        results_paned.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Left panel for results list and controls
        results_left = ttk.Frame(results_paned)
        results_paned.add(results_left, weight=1)
        
        # Right panel for combined plot
        results_right = ttk.Frame(results_paned)
        results_paned.add(results_right, weight=3)
        
        # Results list
        list_frame = ttk.LabelFrame(results_left, text="Saved Results", padding="10")
        list_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        # Treeview for results
        self.results_tree = ttk.Treeview(list_frame, columns=('Name', 'Date'), show='tree headings')
        self.results_tree.heading('#0', text='ID')
        self.results_tree.heading('Name', text='Study Name')
        self.results_tree.heading('Date', text='Date')
        
        self.results_tree.column('#0', width=50)
        self.results_tree.column('Name', width=150)
        self.results_tree.column('Date', width=100)
        
        results_scrollbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=self.results_tree.yview)
        self.results_tree.configure(yscrollcommand=results_scrollbar.set)
        
        self.results_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        results_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Control buttons
        controls_frame = ttk.Frame(results_left)
        controls_frame.pack(fill=tk.X)
        
        ttk.Button(controls_frame, text="Plot Selected", 
                command=self.plot_combined_results).pack(fill=tk.X, pady=(0, 5))
        ttk.Button(controls_frame, text="Remove Selected", 
                command=self.remove_result).pack(fill=tk.X, pady=(0, 5))
        ttk.Button(controls_frame, text="Clear All", 
                command=self.clear_all_results).pack(fill=tk.X, pady=(0, 5))
        ttk.Button(controls_frame, text="Export Combined Plot", 
                command=self.export_combined_plot).pack(fill=tk.X)
        
        # Combined plot
        self.combined_fig = Figure(figsize=(10, 6), dpi=100)
        self.combined_ax = self.combined_fig.add_subplot(111)
        
        self.combined_canvas = FigureCanvasTkAgg(self.combined_fig, results_right)
        self.combined_canvas.draw()
        self.combined_canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        
        combined_toolbar = NavigationToolbar2Tk(self.combined_canvas, results_right)
        combined_toolbar.update()
        
        # Initial empty plot
        self.combined_ax.set_xlabel('Frequency (Hz)')
        self.combined_ax.set_ylabel('Amplitude')
        self.combined_ax.set_title('Combined FFT Results')
        self.combined_ax.grid(True, alpha=0.3)
        self.combined_fig.tight_layout()
    
    def select_file(self):
        file_path = filedialog.askopenfilename(
            title="Select CSV file",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                self.df = pd.read_csv(file_path)
                self.file_path.set(file_path)
                
                # Update column combo
                columns = list(self.df.columns)
                self.column_combo['values'] = columns
                if columns:
                    self.column_combo.set(columns[0])
                    self.column_name.set(columns[0])
                
                # Update slider ranges based on data size
                max_lines = len(self.df)
                
                # Update start line scale maximum
                self.start_scale.configure(to=max_lines)
                
                # Update lines scale maximum
                self.lines_scale.configure(to=max_lines)
                
                # Set reasonable defaults
                self.start_line_var.set(1)
                self.lines_var.set(min(1000, max_lines))
                
                # Update range info
                self.update_range_info()
                
                messagebox.showinfo("Success", f"File loaded successfully!\nRows: {len(self.df)}\nColumns: {len(self.df.columns)}")
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load file:\n{str(e)}")
    
    def on_column_selected(self, event=None):
        selected_column = self.column_var.get()
        if selected_column and not self.column_name.get():
            self.column_name.set(selected_column)
    
    def update_start_label(self, value):
        start_val = int(float(value))
        self.start_label.configure(text=str(start_val))
        self.update_range_info()
    
    def update_lines_label(self, value):
        lines_val = int(float(value))
        self.lines_label.configure(text=str(lines_val))
        self.update_range_info()
    
    def update_range_info(self):
        """Update the range information display"""
        start = self.start_line_var.get()
        lines = self.lines_var.get()
        end = start + lines - 1
        
        # Check if we have data loaded to validate range
        if hasattr(self, 'df') and self.df is not None:
            max_rows = len(self.df)
            if end > max_rows:
                end = max_rows
                actual_lines = end - start + 1
                self.range_info.configure(
                    text=f"Range: Row {start} to {end} ({actual_lines} points) - Limited by data size",
                    foreground="orange"
                )
            else:
                self.range_info.configure(
                    text=f"Range: Row {start} to {end} ({lines} points)",
                    foreground="blue"
                )
        else:
            self.range_info.configure(
                text=f"Range: Row {start} to {end} ({lines} points)",
                foreground="blue"
            )
    
    def run_fft_analysis(self):
        if self.df is None:
            messagebox.showerror("Error", "Please select a CSV file first.")
            return
        
        if not self.column_var.get():
            messagebox.showerror("Error", "Please select a column to analyze.")
            return
        
        try:
            # Get parameters
            column = self.column_var.get()
            start_line = self.start_line_var.get()
            n_lines = self.lines_var.get()
            freq_hz = self.freq_var.get()
            window_func = self.window_var.get()
            
            # Calculate actual indices (convert from 1-based to 0-based indexing)
            start_idx = start_line - 1  # Convert to 0-based index
            end_idx = start_idx + n_lines
            
            # Validate range
            max_rows = len(self.df)
            if start_idx >= max_rows:
                messagebox.showerror("Error", f"Start line ({start_line}) exceeds data size ({max_rows} rows).")
                return
            
            # Adjust end index if it exceeds data size
            if end_idx > max_rows:
                end_idx = max_rows
                actual_lines = end_idx - start_idx
                messagebox.showwarning("Warning", f"Requested range exceeds data size. Using {actual_lines} lines instead of {n_lines}.")
            
            # Extract data from the specified range
            data = self.df[column].iloc[start_idx:end_idx].values
            data = data[~np.isnan(data)]  # Remove NaN values
            
            if len(data) == 0:
                messagebox.showerror("Error", "No valid data found in selected range.")
                return
            
            # Apply window function
            if window_func == "blackman":
                window = blackman(len(data))
                data = data * window
            elif window_func == "hann":
                window = hann(len(data))
                data = data * window
            elif window_func == "hamming":
                window = hamming(len(data))
                data = data * window
            
            # Perform FFT
            yf = fft(data)
            T = 1.0 / freq_hz
            xf = fftfreq(len(data), T)[:len(data)//2]
            amplitude = 2.0/len(data) * np.abs(yf[:len(data)//2])
            
            # Plot results
            self.ax.clear()
            
            color = self.current_colors[self.color_index % len(self.current_colors)]
            self.ax.semilogy(xf[1:], amplitude[1:], color=color, linewidth=1.5)
            
            # Customize plot
            display_name = self.column_name.get() or column
            range_text = f"Rows {start_line}-{start_idx + len(data)}"
            self.ax.set_xlabel('Frequency (Hz)')
            self.ax.set_ylabel('Amplitude')
            self.ax.set_title(f'FFT Analysis: {display_name} ({range_text})')
            self.ax.grid(True, alpha=0.3)
            
            # Add frequency labels if enabled
            peak_count = self.peak_count_var.get()
            if peak_count > 0:
                peaks_idx = self.detect_peaks_advanced(amplitude, xf)
                
                # Sort peaks by amplitude (highest first) and take the requested number
                if peaks_idx:
                    peaks_with_amplitude = [(idx, amplitude[idx]) for idx in peaks_idx if idx < len(xf)]
                    peaks_with_amplitude.sort(key=lambda x: x[1], reverse=True)
                    
                    # Add annotations with better positioning to avoid overlap
                    for i, (idx, amp) in enumerate(peaks_with_amplitude[:peak_count]):
                        # Vary the annotation position to reduce overlap
                        x_offset = 10 + (i % 3) * 15  # Stagger x position
                        y_offset = 10 + (i % 2) * 20  # Alternate y position
                        
                        self.ax.annotate(f'{xf[idx]:.1f} Hz\n{amp:.2e}', 
                                       xy=(xf[idx], amplitude[idx]), 
                                       xytext=(x_offset, y_offset), textcoords='offset points',
                                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8),
                                       arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.1'),
                                       fontsize=8)
            
            self.fig.tight_layout()
            self.canvas.draw()
            
            # Store current analysis data
            self.current_fft_data = {
                'frequencies': xf,
                'amplitudes': amplitude,
                'column': column,
                'display_name': display_name,
                'study_name': self.study_name.get(),
                'freq_hz': freq_hz,
                'start_line': start_line,
                'n_lines': len(data),  # Actual number of lines used
                'range_text': range_text,
                'window_func': window_func,
                'color': color,
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
            messagebox.showinfo("Success", f"FFT analysis completed successfully!\nAnalyzed {len(data)} data points from {range_text}")
            
        except Exception as e:
            messagebox.showerror("Error", f"FFT analysis failed:\n{str(e)}")
    
    def export_data(self):
        if not hasattr(self, 'current_fft_data'):
            messagebox.showerror("Error", "No FFT results to export. Run analysis first.")
            return
        
        file_path = filedialog.asksaveasfilename(
            title="Export FFT Data",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                data = self.current_fft_data
                df_export = pd.DataFrame({
                    'Frequency_Hz': data['frequencies'],
                    'Amplitude': data['amplitudes']
                })
                
                df_export.to_csv(file_path, index=False)
                messagebox.showinfo("Success", f"Data exported to {file_path}")
                
            except Exception as e:
                messagebox.showerror("Error", f"Export failed:\n{str(e)}")
    
    def export_plot(self):
        if not hasattr(self, 'current_fft_data'):
            messagebox.showerror("Error", "No plot to export. Run analysis first.")
            return
        
        file_path = filedialog.asksaveasfilename(
            title="Export Plot",
            defaultextension=".png",
            filetypes=[("PNG files", "*.png"), ("PDF files", "*.pdf"), ("SVG files", "*.svg"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                self.fig.savefig(file_path, dpi=300, bbox_inches='tight')
                messagebox.showinfo("Success", f"Plot exported to {file_path}")
                
            except Exception as e:
                messagebox.showerror("Error", f"Export failed:\n{str(e)}")
    
    def save_to_results(self):
        if not hasattr(self, 'current_fft_data'):
            messagebox.showerror("Error", "No results to save. Run analysis first.")
            return
        
        try:
            # Generate unique ID
            result_id = len(self.fft_results) + 1
            self.fft_results[result_id] = self.current_fft_data.copy()
            
            # Add to treeview
            self.results_tree.insert('', 'end', 
                                text=str(result_id),
                                values=(self.current_fft_data['study_name'], 
                                        self.current_fft_data['timestamp']))
            
            self.color_index += 1  # Move to next color for next analysis
            messagebox.showinfo("Success", "Results saved to Combined Results tab!")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save results:\n{str(e)}")
    
    def plot_combined_results(self):
        selected_items = self.results_tree.selection()
        if not selected_items:
            messagebox.showwarning("Warning", "Please select results to plot.")
            return
        
        try:
            self.combined_ax.clear()
            
            for item in selected_items:
                result_id = int(self.results_tree.item(item, 'text'))
                data = self.fft_results[result_id]
                
                self.combined_ax.semilogy(data['frequencies'][1:], data['amplitudes'][1:], 
                                        color=data['color'], linewidth=1.5, 
                                        label=data['display_name'], alpha=0.8)
            
            self.combined_ax.set_xlabel('Frequency (Hz)')
            self.combined_ax.set_ylabel('Amplitude')
            self.combined_ax.set_title('Combined FFT Results')
            self.combined_ax.grid(True, alpha=0.3)
            self.combined_ax.legend()
            
            self.combined_fig.tight_layout()
            self.combined_canvas.draw()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to plot results:\n{str(e)}")
    
    def remove_result(self):
        selected_items = self.results_tree.selection()
        if not selected_items:
            messagebox.showwarning("Warning", "Please select results to remove.")
            return
        
        if messagebox.askyesno("Confirm", "Remove selected results?"):
            for item in selected_items:
                result_id = int(self.results_tree.item(item, 'text'))
                if result_id in self.fft_results:
                    del self.fft_results[result_id]
                self.results_tree.delete(item)
    
    def clear_all_results(self):
        if messagebox.askyesno("Confirm", "Clear all saved results?"):
            self.fft_results.clear()
            for item in self.results_tree.get_children():
                self.results_tree.delete(item)
            self.combined_ax.clear()
            self.combined_ax.set_xlabel('Frequency (Hz)')
            self.combined_ax.set_ylabel('Amplitude')
            self.combined_ax.set_title('Combined FFT Results')
            self.combined_ax.grid(True, alpha=0.3)
            self.combined_canvas.draw()
    
    def export_combined_plot(self):
        if not self.fft_results:
            messagebox.showwarning("Warning", "No results to export.")
            return
        
        file_path = filedialog.asksaveasfilename(
            title="Export Combined Plot",
            defaultextension=".png",
            filetypes=[("PNG files", "*.png"), ("PDF files", "*.pdf"), ("SVG files", "*.svg"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                self.combined_fig.savefig(file_path, dpi=300, bbox_inches='tight')
                messagebox.showinfo("Success", f"Combined plot exported to {file_path}")
                
            except Exception as e:
                messagebox.showerror("Error", f"Export failed:\n{str(e)}")
    
    def detect_peaks_advanced(self, amplitude, frequencies):
        """Advanced peak detection using configurable settings"""
        peaks_idx = []
        
        # Get settings
        threshold_mode = self.threshold_mode_var.get()
        min_distance = self.min_distance_var.get()
        window_size = self.window_size_var.get()
        skip_dc = self.skip_dc_var.get()
        
        # Determine start index (skip DC if requested)
        start_idx = 1 if skip_dc else 0
        
        # Calculate threshold based on mode
        if threshold_mode == "relative":
            threshold = np.max(amplitude) * self.relative_threshold_var.get()
        elif threshold_mode == "absolute":
            threshold = self.absolute_threshold_var.get()
        elif threshold_mode == "statistical":
            mean_amp = np.mean(amplitude[start_idx:])
            std_amp = np.std(amplitude[start_idx:])
            threshold = mean_amp + self.statistical_factor_var.get() * std_amp
        else:
            threshold = 0  # Fallback
        
        # Find local maxima
        for i in range(start_idx + window_size, len(amplitude) - window_size):
            # Check if current point is a local maximum within the window
            is_maximum = True
            for j in range(-window_size, window_size + 1):
                if j != 0 and amplitude[i] <= amplitude[i + j]:
                    is_maximum = False
                    break
            
            # Check if above threshold
            if is_maximum and amplitude[i] > threshold:
                # Check minimum distance from existing peaks
                too_close = False
                for existing_peak in peaks_idx:
                    if abs(i - existing_peak) < min_distance:
                        # Keep the higher peak
                        if amplitude[i] > amplitude[existing_peak]:
                            peaks_idx.remove(existing_peak)
                        else:
                            too_close = True
                        break
                
                if not too_close:
                    peaks_idx.append(i)
        
        return peaks_idx
    
    def on_threshold_mode_changed(self, event=None):
        """Show/hide appropriate threshold setting frame based on mode"""
        mode = self.threshold_mode_var.get()
        
        # Hide all frames first
        self.relative_frame.pack_forget()
        self.absolute_frame.pack_forget()
        self.statistical_frame.pack_forget()
        
        # Show the appropriate frame
        if mode == "relative":
            self.relative_frame.pack(fill=tk.X, pady=(5, 10))
        elif mode == "absolute":
            self.absolute_frame.pack(fill=tk.X, pady=(5, 10))
        elif mode == "statistical":
            self.statistical_frame.pack(fill=tk.X, pady=(5, 10))
    
    def update_relative_label(self, value):
        """Update relative threshold label"""
        val = float(value)
        self.rel_value_label.configure(text=f"{val:.2f}")
    
    def update_statistical_label(self, value):
        """Update statistical factor label"""
        val = float(value)
        self.stat_value_label.configure(text=f"{val:.1f}")
    
    def update_distance_label(self, value):
        """Update minimum distance label"""
        val = int(float(value))
        self.distance_label.configure(text=str(val))
    
    def update_color_display(self):
        # Clear existing color widgets
        for widget in self.colors_frame.winfo_children():
            widget.destroy()
        
        for i, color in enumerate(self.current_colors):
            color_frame = ttk.Frame(self.colors_frame)
            color_frame.pack(side=tk.LEFT, padx=2)
            
            color_label = tk.Label(color_frame, text="  ", bg=color, width=3, height=1, relief=tk.RAISED)
            color_label.pack()
            
            # Bind click to change color
            color_label.bind("<Button-1>", lambda e, idx=i: self.change_color(idx))
    
    def change_color(self, index):
        color = colorchooser.askcolor(color=self.current_colors[index])
        if color[1]:  # If color was selected
            self.current_colors[index] = color[1]
            self.update_color_display()
    
    def reset_colors(self):
        self.current_colors = self.settings['default_colors'].copy()
        self.update_color_display()
    
    def save_settings(self):
        # Save all peak detection settings
        self.settings['peak_labels_count'] = self.peak_count_var.get()
        self.settings['peak_threshold_mode'] = self.threshold_mode_var.get()
        self.settings['peak_relative_threshold'] = self.relative_threshold_var.get()
        self.settings['peak_absolute_threshold'] = self.absolute_threshold_var.get()
        self.settings['peak_statistical_factor'] = self.statistical_factor_var.get()
        self.settings['peak_min_distance'] = self.min_distance_var.get()
        self.settings['peak_window_size'] = self.window_size_var.get()
        self.settings['skip_dc_component'] = self.skip_dc_var.get()
        self.settings['default_colors'] = self.current_colors.copy()
        
        try:
            with open('fft_analyzer_settings.json', 'w') as f:
                json.dump(self.settings, f, indent=2)
            messagebox.showinfo("Success", "Settings saved successfully!")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save settings:\n{str(e)}")
    
    def load_settings(self):
        try:
            if os.path.exists('fft_analyzer_settings.json'):
                with open('fft_analyzer_settings.json', 'r') as f:
                    loaded_settings = json.load(f)
                    self.settings.update(loaded_settings)
                    self.current_colors = self.settings.get('default_colors', self.current_colors)
                    
                    # Update UI variables if they exist
                    if hasattr(self, 'peak_count_var'):
                        self.peak_count_var.set(self.settings.get('peak_labels_count', 5))
                    if hasattr(self, 'threshold_mode_var'):
                        self.threshold_mode_var.set(self.settings.get('peak_threshold_mode', 'relative'))
                    if hasattr(self, 'relative_threshold_var'):
                        self.relative_threshold_var.set(self.settings.get('peak_relative_threshold', 0.1))
                    if hasattr(self, 'absolute_threshold_var'):
                        self.absolute_threshold_var.set(self.settings.get('peak_absolute_threshold', 0.001))
                    if hasattr(self, 'statistical_factor_var'):
                        self.statistical_factor_var.set(self.settings.get('peak_statistical_factor', 1.0))
                    if hasattr(self, 'min_distance_var'):
                        self.min_distance_var.set(self.settings.get('peak_min_distance', 10))
                    if hasattr(self, 'window_size_var'):
                        self.window_size_var.set(self.settings.get('peak_window_size', 3))
                    if hasattr(self, 'skip_dc_var'):
                        self.skip_dc_var.set(self.settings.get('skip_dc_component', True))
        except Exception as e:
            print(f"Failed to load settings: {e}")

def main():
    root = tk.Tk()
    
    # Configure ttk style
    style = ttk.Style()
    
    # Create accent button style
    style.configure("Accent.TButton", 
                foreground="black", 
                background="#0078d4",
                borderwidth=0,
                focuscolor="none",
                padding=(20, 10))
    
    style.map("Accent.TButton",
            background=[('active', '#106ebe'), ('pressed', '#005a9e')],
            foreground=[('active', 'black'), ('pressed', 'black')])
    
    app = FFTAnalyzerApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()