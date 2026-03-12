
This project is a graduate-level web application built for STATGR5243. It provides a complete pipeline for data science tasks, allowing users to upload datasets (CSV, Excel, JSON, Parquet), clean and preprocess them, engineer new features, and perform exploratory data analysis (EDA) through an intuitive and responsive interface.

## Features

- **Data Loading**: Support for multiple file formats with robust error handling.
- **Data Cleaning**: Interactive missing value imputation, duplicate removal, outlier filtering, scaling, and categorical encoding.
- **Feature Engineering**: Arithmetic operations, mathematical transformations, datetime extraction, and column management.
- **Exploratory Data Analysis**: Dynamic visualizations (Histogram, Box Plot, Bar Chart, Scatter Plot, Heatmap) and summary statistics using Plotly.
- **Export**: Download the processed dataset.
- **User Guide**: Built-in documentation for easy onboarding.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/qz2573-sketch/STATGR5243-Project2.git
   cd STATGR5243-Project2
   ```

2. Create a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

Run the application locally:

```bash
shiny run app.py
```

Navigate to `http://localhost:8000` in your browser.

## Project Structure

```
STATGR5243-Project2/
├── app.py                  # Main application entry point
├── requirements.txt        # Python dependencies
├── README.md               # Project documentation
├── data/                   # Sample datasets
├── src/                    # Source code modules
│   ├── data_loader.py      # Data loading logic
│   ├── preprocessing.py    # Cleaning and preprocessing functions
│   ├── feature_engineering.py # Feature engineering logic
│   ├── eda.py              # Visualization and EDA functions
│   ├── ui_helpers.py       # UI components
│   └── utils.py            # Utility functions
└── report/                 # Project report
    └── project_report.md   # Draft report
```

## Note for Course Submission

This project fulfills the requirements for Project 2 by demonstrating:
- **Modular Code**: Logic is separated into `src/` modules.
- **Interactivity**: Real-time updates using Shiny's reactive graph.
- **Robustness**: Error handling for file uploads and operations.
- **Deployment**: Configuration for shinyapps.io.
