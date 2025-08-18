# FFT Analyzer - Flight Stand Data Analysis Tool

A comprehensive graphical interface for performing FFT analysis on flight stand data with professional visualization and export capabilities.

## Features

### ✅ **Complete Requirements Implementation**

1. **Graphical Interface** - No code editing required
2. **File Selection** - Easy CSV file loading with preview
3. **Data Configuration**:
   - Column selection from CSV
   - Column renaming capability
   - Adjustable number of lines (with slider)
   - Flexible acquisition frequency setting
4. **Study Management** - Name your analysis studies
5. **Settings Panel**:
   - Toggle frequency labels on/off
   - Customizable graph colors
   - Multiple window functions (Blackman, Hann, Hamming)
6. **Combined Results** - Multiple FFT results in one graph
7. **Export Options**:
   - Export frequency and amplitude data as CSV
   - Export graphs as high-resolution images (PNG, PDF, SVG)
   - Save and manage multiple analysis results

## Installation & Setup

### Option 1: Quick Start (Recommended)
1. Double-click `run_fft_analyzer.bat`
2. The script will automatically check and install dependencies
3. The application will launch automatically

### Option 2: Manual Setup
1. Ensure Python 3.7+ is installed
2. Install dependencies: `pip install -r requirements.txt`
3. Run the application: `python app.py`

## Usage Guide

### 1. **Main Analysis Tab**
- **Load Data**: Click "Select CSV File" to load your flight stand data
- **Configure Analysis**:
  - Select the column to analyze from dropdown
  - Optionally rename the column for display
  - Adjust number of lines with the slider (100-10,000)
  - Set acquisition frequency in Hz
  - Name your study
  - Choose window function (optional)
- **Run Analysis**: Click "Run FFT Analysis"
- **Export**: Save data as CSV or plot as image
- **Save Results**: Add to combined results for comparison

### 2. **Settings Tab**
- **Display Options**: Toggle frequency labels on peaks
- **Colors**: Customize plot colors by clicking color squares
- **Reset**: Restore default color scheme
- **Save**: Persist your settings

### 3. **Combined Results Tab**
- **View Saved Studies**: All your saved FFT analyses
- **Plot Multiple**: Select multiple results and plot together
- **Manage**: Remove individual results or clear all
- **Export**: Save combined plots as images

## File Formats

### Input CSV Format
- First row should contain column headers
- Numeric data in columns (Time, Load Cell readings, etc.)
- Example columns: `Time`, `Load_Cell_1`, `Load_Cell_2`, `Thrust`, `RPM`, `Voltage`, `Current`

### Export Formats
- **Data Export**: CSV with Frequency_Hz and Amplitude columns
- **Image Export**: PNG (default), PDF, or SVG formats at 300 DPI

## Window Functions

Choose from different window functions to reduce spectral leakage:
- **None**: Raw data (rectangular window)
- **Blackman**: Good for general purposes, low spectral leakage
- **Hann**: Good frequency resolution, moderate spectral leakage  
- **Hamming**: Similar to Hann with slightly different characteristics

## Tips for Best Results

1. **Sampling Rate**: Ensure your acquisition frequency is accurate for proper frequency scaling
2. **Data Length**: More data points provide better frequency resolution
3. **Window Functions**: Use windowing for non-periodic signals to reduce artifacts
4. **Peak Detection**: Enable frequency labels to identify dominant frequencies
5. **Color Coding**: Use different colors when comparing multiple datasets

## Sample Data

A sample CSV file (`sample_data.csv`) is included for testing the application.

## Technical Details

- **FFT Implementation**: Using SciPy's optimized FFT algorithms
- **GUI Framework**: Tkinter with modern styling
- **Plotting**: Matplotlib with interactive navigation
- **Data Handling**: Pandas for robust CSV processing
- **Windowing**: Multiple window functions for signal conditioning

## Troubleshooting

### Common Issues:
1. **"No module named..." error**: Run `pip install -r requirements.txt`
2. **Empty plot**: Check that your CSV has numeric data in selected column
3. **Performance issues**: Reduce number of lines for very large datasets
4. **Memory issues**: Process data in smaller chunks for very large files

### System Requirements:
- Windows 7+ (tested on Windows 10/11)
- Python 3.7 or higher
- 4GB RAM recommended for large datasets
- 100MB free disk space

## Support

For issues or feature requests, check your CSV data format and ensure all dependencies are properly installed.
