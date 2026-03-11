# Project Report: Data Explorer Pro

## 1. Introduction

Data Explorer Pro is an interactive web application designed to democratize the initial stages of the data science lifecycle. Built using Python for Shiny, it enables users to upload raw datasets and guide them through cleaning, preprocessing, feature engineering, and exploratory data analysis (EDA) without writing a single line of code.

## 2. Architecture

The application follows a modular architecture, separating the user interface (UI) definition, server logic, and core data processing functions.

- **Frontend**: Built with Shiny UI components, leveraging `nav_panel` for a tabbed layout and `layout_sidebar` for control panels.
- **Backend**: Python functions in the `src/` directory handle data manipulation using `pandas` and `numpy`.
- **Reactivity**: Shiny's reactive graph manages the state of the data (`current_df`), ensuring that changes in one stage (e.g., cleaning) automatically propagate to downstream stages (e.g., EDA).

## 3. Key Features

### 3.1 Data Loading
The app supports CSV, Excel, JSON, and Parquet formats. It automatically infers file types and provides immediate feedback on the dataset's dimensions and structure.

### 3.2 Data Cleaning
Users can interactively:
- Handle missing values (drop, impute with mean/median/mode/constant).
- Remove duplicates.
- Filter outliers using IQR or Z-score methods.
- Scale numeric features (Standardization, Min-Max).
- Encode categorical variables (One-Hot, Label).

### 3.3 Feature Engineering
The app allows for the creation of new features through:
- Arithmetic operations between columns.
- Mathematical transformations (Log, Square, Sqrt).
- Extraction of datetime components (Year, Month, Day, etc.).

### 3.4 Exploratory Data Analysis
Interactive visualizations are powered by Plotly, offering:
- Histograms for distribution analysis.
- Box plots for outlier detection and distribution comparison.
- Bar charts for categorical frequency and aggregation.
- Scatter plots for relationship analysis.
- Heatmaps for correlation analysis.

## 4. Challenges and Solutions

- **State Management**: Managing the state of the dataframe across multiple transformation steps was a key challenge. We solved this by using a central reactive value `current_df` that is updated by "Apply" actions, ensuring a linear but reversible workflow.
- **Performance**: To ensure responsiveness with larger datasets, we optimized the plotting functions to use Plotly's efficient rendering and limited the preview to the first few rows.

## 5. Conclusion

Data Explorer Pro successfully meets the project requirements, providing a robust and user-friendly tool for data analysis. Its modular design allows for easy extensibility, such as adding machine learning models or more advanced visualization types in the future.
