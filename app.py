from shiny import App, ui, reactive, render, req
import pandas as pd
import numpy as np
from pathlib import Path
from src.data_loader import load_dataset, get_dataset_info
from src.preprocessing import (
    handle_missing_values, remove_duplicates, filter_outliers, 
    scale_features, encode_categorical, convert_dtypes
)
from src.feature_engineering import (
    create_arithmetic_feature, transform_feature, 
    extract_datetime_features, drop_columns
)
from src.eda import (
    get_summary_statistics, get_categorical_summary, 
    plot_histogram, plot_box, plot_bar, plot_scatter, plot_heatmap
)
from src.ui_helpers import card_header, info_box
from shinywidgets import output_widget, render_widget
import faicons as fa

# --- UI Definitions ---

# User Guide Tab
user_guide_tab = ui.nav_panel(
    "User Guide",
    ui.div(
        ui.h2("Welcome to Data Explorer Pro"),
        ui.p("This application allows you to upload datasets, clean them, engineer features, and perform exploratory data analysis."),
        ui.hr(),
        ui.h4("How to use:"),
        ui.accordion(
            ui.accordion_panel(
                "1. Data Upload",
                "Upload a CSV, Excel, JSON, or Parquet file. You can also load sample datasets to test the app."
            ),
            ui.accordion_panel(
                "2. Cleaning & Preprocessing",
                "Handle missing values, remove duplicates, filter outliers, scale features, and encode categorical variables. Changes are applied immediately."
            ),
            ui.accordion_panel(
                "3. Feature Engineering",
                "Create new features using arithmetic operations, transformations, or datetime extraction. You can also drop unwanted columns."
            ),
            ui.accordion_panel(
                "4. Exploratory Data Analysis (EDA)",
                "Visualize your data using interactive plots (Histogram, Box Plot, Bar Chart, Scatter Plot, Heatmap) and view summary statistics."
            ),
            ui.accordion_panel(
                "5. Export",
                "Download your processed dataset as a CSV file."
            )
        ),
        class_="p-4"
    ),
    icon=fa.icon_svg("book")
)

# Data Upload Tab
data_upload_tab = ui.nav_panel(
    "Data Upload",
    ui.layout_sidebar(
        ui.sidebar(
            card_header("Upload Dataset", "upload"),
            ui.input_file("file_upload", "Choose a file", accept=[".csv", ".xlsx", ".json", ".parquet"], multiple=False),
            ui.hr(),
            ui.h5("Or load sample data:"),
            ui.input_action_button("load_sample_1", "Load Employee Data", class_="btn-outline-primary w-100 mb-2"),
            ui.input_action_button("load_sample_2", "Load Product Data", class_="btn-outline-primary w-100"),
            width=300
        ),
        ui.div(
            ui.output_ui("data_info_boxes"),
            ui.card(
                card_header("Data Preview", "table"),
                ui.output_data_frame("data_preview"),
                full_screen=True
            ),
            class_="p-3"
        )
    ),
    icon=fa.icon_svg("upload")
)

# Cleaning Tab
cleaning_tab = ui.nav_panel(
    "Cleaning",
    ui.layout_sidebar(
        ui.sidebar(
            ui.accordion(
                ui.accordion_panel(
                    "Missing Values",
                    ui.input_select("mv_method", "Method", 
                                    {"drop_rows": "Drop Rows", "drop_cols": "Drop Columns", 
                                     "mean": "Mean Imputation", "median": "Median Imputation", 
                                     "mode": "Mode Imputation", "constant": "Constant Value"}),
                    ui.panel_conditional(
                        "input.mv_method == 'constant'",
                        ui.input_text("mv_fill_value", "Fill Value", "0")
                    ),
                    ui.input_select("mv_cols", "Columns (Optional)", choices=[], multiple=True),
                    ui.input_action_button("apply_mv", "Apply", class_="btn-primary w-100")
                ),
                ui.accordion_panel(
                    "Duplicates",
                    ui.p("Remove duplicate rows."),
                    ui.input_action_button("apply_dedup", "Remove Duplicates", class_="btn-primary w-100")
                ),
                ui.accordion_panel(
                    "Outliers",
                    ui.input_select("outlier_col", "Column", choices=[]),
                    ui.input_select("outlier_method", "Method", {"iqr": "IQR", "zscore": "Z-Score"}),
                    ui.input_numeric("outlier_threshold", "Threshold", 1.5, step=0.1),
                    ui.input_action_button("apply_outlier", "Filter Outliers", class_="btn-primary w-100")
                ),
                ui.accordion_panel(
                    "Scaling",
                    ui.input_select("scale_cols", "Columns", choices=[], multiple=True),
                    ui.input_select("scale_method", "Method", {"standard": "Standardization", "minmax": "Min-Max Scaling"}),
                    ui.input_action_button("apply_scale", "Apply Scaling", class_="btn-primary w-100")
                ),
                ui.accordion_panel(
                    "Encoding",
                    ui.input_select("encode_cols", "Columns", choices=[], multiple=True),
                    ui.input_select("encode_method", "Method", {"onehot": "One-Hot Encoding", "label": "Label Encoding"}),
                    ui.input_action_button("apply_encode", "Apply Encoding", class_="btn-primary w-100")
                ),
                ui.accordion_panel(
                    "Data Types",
                    ui.input_select("dtype_col", "Column", choices=[]),
                    ui.input_select("dtype_target", "Target Type", 
                                    {"numeric": "Numeric", "string": "String", 
                                     "datetime": "Datetime", "category": "Category"}),
                    ui.input_action_button("apply_dtype", "Convert", class_="btn-primary w-100")
                ),
                id="cleaning_accordion"
            ),
            ui.hr(),
            ui.input_action_button("reset_data", "Reset Data", class_="btn-danger w-100"),
            width=350
        ),
        ui.card(
            card_header("Current Data State", "database"),
            ui.output_text_verbatim("cleaning_status"),
            ui.output_data_frame("cleaning_preview"),
            full_screen=True
        )
    ),
    icon=fa.icon_svg("broom")
)

# Feature Engineering Tab
feature_eng_tab = ui.nav_panel(
    "Feature Engineering",
    ui.layout_sidebar(
        ui.sidebar(
            ui.accordion(
                ui.accordion_panel(
                    "Arithmetic",
                    ui.input_select("fe_arith_col1", "Column 1", choices=[]),
                    ui.input_select("fe_arith_op", "Operation", 
                                    {"add": "+", "subtract": "-", "multiply": "*", "divide": "/"}),
                    ui.input_select("fe_arith_col2", "Column 2", choices=[]),
                    ui.input_text("fe_arith_name", "New Column Name", "new_feature"),
                    ui.input_action_button("apply_arith", "Create Feature", class_="btn-primary w-100")
                ),
                ui.accordion_panel(
                    "Transformations",
                    ui.input_select("fe_trans_col", "Column", choices=[]),
                    ui.input_select("fe_trans_method", "Method", 
                                    {"log": "Log", "square": "Square", 
                                     "sqrt": "Square Root", "abs": "Absolute Value", "binning": "Binning"}),
                    ui.input_action_button("apply_trans", "Apply Transform", class_="btn-primary w-100")
                ),
                ui.accordion_panel(
                    "Datetime Extraction",
                    ui.input_select("fe_dt_col", "Datetime Column", choices=[]),
                    ui.input_checkbox_group("fe_dt_features", "Extract", 
                                            {"year": "Year", "month": "Month", "day": "Day", 
                                             "weekday": "Weekday", "quarter": "Quarter"}),
                    ui.input_action_button("apply_dt", "Extract", class_="btn-primary w-100")
                ),
                ui.accordion_panel(
                    "Drop Columns",
                    ui.input_select("fe_drop_cols", "Columns to Drop", choices=[], multiple=True),
                    ui.input_action_button("apply_drop", "Drop Columns", class_="btn-warning w-100")
                )
            ),
            width=350
        ),
        ui.card(
            card_header("Engineered Data", "gears"),
            ui.output_data_frame("fe_preview"),
            full_screen=True
        )
    ),
    icon=fa.icon_svg("flask")
)

# EDA Tab
eda_tab = ui.nav_panel(
    "EDA",
    ui.layout_sidebar(
        ui.sidebar(
            ui.input_select("eda_plot_type", "Plot Type", 
                            {"histogram": "Histogram", "box": "Box Plot", 
                             "bar": "Bar Chart", "scatter": "Scatter Plot", 
                             "heatmap": "Correlation Heatmap"}),
            ui.panel_conditional(
                "input.eda_plot_type != 'heatmap'",
                ui.input_select("eda_x", "X Axis", choices=[]),
                ui.input_select("eda_color", "Color (Optional)", choices=["None"])
            ),
            ui.panel_conditional(
                "input.eda_plot_type == 'box' || input.eda_plot_type == 'bar' || input.eda_plot_type == 'scatter'",
                ui.input_select("eda_y", "Y Axis", choices=[])
            ),
            ui.panel_conditional(
                "input.eda_plot_type == 'bar'",
                ui.input_select("eda_agg", "Aggregation", {"count": "Count", "mean": "Mean", "sum": "Sum"})
            ),
            ui.panel_conditional(
                "input.eda_plot_type == 'heatmap'",
                ui.input_select("eda_heat_cols", "Columns", choices=[], multiple=True)
            ),
            width=300
        ),
        ui.div(
            ui.card(
                card_header("Visualization", "chart-simple"),
                output_widget("eda_plot"),
                full_screen=True
            ),
            ui.card(
                card_header("Summary Statistics", "table"),
                ui.output_data_frame("eda_summary")
            )
        )
    ),
    icon=fa.icon_svg("chart-line")
)

# Export Tab
export_tab = ui.nav_panel(
    "Export",
    ui.div(
        ui.card(
            card_header("Download Processed Data", "download"),
            ui.p("Download the final dataset with all cleaning and feature engineering steps applied."),
            ui.download_button("download_data", "Download CSV", class_="btn-success"),
            class_="text-center p-5"
        ),
        class_="d-flex justify-content-center align-items-center h-100"
    ),
    icon=fa.icon_svg("file-export")
)

app_ui = ui.page_navbar(
    user_guide_tab,
    data_upload_tab,
    cleaning_tab,
    feature_eng_tab,
    eda_tab,
    export_tab,
    title="Data Explorer Pro",
    id="navbar_id",
    fillable=True
)

# --- Server Logic ---

def server(input, output, session):
    # Reactive value to store the current dataframe
    # We use a single mutable reactive value to represent the state of the data
    # as it flows through the pipeline.
    current_df = reactive.Value(None)
    
    # Status message for cleaning operations
    status_message = reactive.Value("Ready")

    # --- Data Loading ---
    
    @reactive.Effect
    @reactive.event(input.file_upload)
    def _():
        file_infos = input.file_upload()
        if not file_infos:
            return
        
        file_info = file_infos[0]
        df, error = load_dataset(file_info["datapath"])
        
        if df is not None:
            current_df.set(df)
            status_message.set(f"Loaded {file_info['name']} with {df.shape[0]} rows and {df.shape[1]} columns.")
            ui.notification_show("File loaded successfully!", type="message")
        else:
            ui.notification_show(f"Error loading file: {error}", type="error")

    @reactive.Effect
    @reactive.event(input.load_sample_1)
    def _():
        df, _ = load_dataset("data/sample_dataset_1.csv")
        current_df.set(df)
        status_message.set("Loaded sample dataset 1.")
        ui.notification_show("Sample data loaded!", type="message")

    @reactive.Effect
    @reactive.event(input.load_sample_2)
    def _():
        df, _ = load_dataset("data/sample_dataset_2.csv")
        current_df.set(df)
        status_message.set("Loaded sample dataset 2.")
        ui.notification_show("Sample data loaded!", type="message")

    @reactive.Effect
    @reactive.event(input.reset_data)
    def _():
        # Ideally we would reload the original file, but for simplicity we just clear or warn.
        # A better implementation would store 'original_df' separately.
        # Here we just notify the user to reload.
        current_df.set(None)
        status_message.set("Data reset.")
        ui.notification_show("Data cleared. Please upload again.", type="warning")

    # --- Data Info & Preview ---

    @output
    @render.ui
    def data_info_boxes():
        df = current_df()
        if df is None:
            return ui.div()
        
        info = get_dataset_info(df)
        return ui.layout_column_wrap(
            info_box("Rows", str(info["rows"]), "table-list"),
            info_box("Columns", str(info["cols"]), "columns"),
            info_box("Missing Values", str(sum(info["missing_values"].values())), "triangle-exclamation", "bg-warning-subtle" if sum(info["missing_values"].values()) > 0 else "bg-light"),
            info_box("Duplicates", str(info["duplicates"]), "copy", "bg-danger-subtle" if info["duplicates"] > 0 else "bg-light"),
            width=1/4
        )

    @output
    @render.data_frame
    def data_preview():
        return render.DataGrid(current_df())

    @output
    @render.data_frame
    def cleaning_preview():
        return render.DataGrid(current_df())
    
    @output
    @render.data_frame
    def fe_preview():
        return render.DataGrid(current_df())

    @output
    @render.text
    def cleaning_status():
        return status_message()

    # --- Dynamic UI Updates ---
    
    @reactive.Effect
    def update_column_choices():
        df = current_df()
        if df is None:
            choices = []
        else:
            choices = list(df.columns)
            
        # Update all column selectors
        ui.update_select("mv_cols", choices=choices)
        ui.update_select("outlier_col", choices=choices)
        ui.update_select("scale_cols", choices=choices)
        ui.update_select("encode_cols", choices=choices)
        ui.update_select("dtype_col", choices=choices)
        
        ui.update_select("fe_arith_col1", choices=choices)
        ui.update_select("fe_arith_col2", choices=choices)
        ui.update_select("fe_trans_col", choices=choices)
        ui.update_select("fe_dt_col", choices=choices)
        ui.update_select("fe_drop_cols", choices=choices)
        
        ui.update_select("eda_x", choices=choices)
        ui.update_select("eda_y", choices=choices)
        ui.update_select("eda_color", choices=["None"] + choices)
        ui.update_select("eda_heat_cols", choices=choices)

    # --- Cleaning Operations ---

    @reactive.Effect
    @reactive.event(input.apply_mv)
    def _():
        df = current_df()
        if df is None: return
        
        cols = input.mv_cols()
        cols = list(cols) if cols else None
        
        # Handle fill value for constant
        fill_val = None
        if input.mv_method() == 'constant':
            try:
                fill_val = float(input.mv_fill_value())
            except:
                fill_val = input.mv_fill_value()

        new_df = handle_missing_values(df, input.mv_method(), cols, fill_val)
        current_df.set(new_df)
        status_message.set(f"Applied missing value handling: {input.mv_method()}")
        ui.notification_show("Missing values handled.", type="message")

    @reactive.Effect
    @reactive.event(input.apply_dedup)
    def _():
        df = current_df()
        if df is None: return
        new_df = remove_duplicates(df)
        diff = len(df) - len(new_df)
        current_df.set(new_df)
        status_message.set(f"Removed {diff} duplicate rows.")
        ui.notification_show(f"Removed {diff} duplicates.", type="message")

    @reactive.Effect
    @reactive.event(input.apply_outlier)
    def _():
        df = current_df()
        col = input.outlier_col()
        if df is None or not col: return
        
        new_df = filter_outliers(df, col, input.outlier_method(), input.outlier_threshold())
        diff = len(df) - len(new_df)
        current_df.set(new_df)
        status_message.set(f"Filtered {diff} outliers from {col}.")
        ui.notification_show(f"Filtered {diff} outliers.", type="message")

    @reactive.Effect
    @reactive.event(input.apply_scale)
    def _():
        df = current_df()
        cols = list(input.scale_cols())
        if df is None or not cols: return
        
        new_df = scale_features(df, cols, input.scale_method())
        current_df.set(new_df)
        status_message.set(f"Scaled columns: {', '.join(cols)}")
        ui.notification_show("Scaling applied.", type="message")

    @reactive.Effect
    @reactive.event(input.apply_encode)
    def _():
        df = current_df()
        cols = list(input.encode_cols())
        if df is None or not cols: return
        
        new_df = encode_categorical(df, cols, input.encode_method())
        current_df.set(new_df)
        status_message.set(f"Encoded columns: {', '.join(cols)}")
        ui.notification_show("Encoding applied.", type="message")

    @reactive.Effect
    @reactive.event(input.apply_dtype)
    def _():
        df = current_df()
        col = input.dtype_col()
        if df is None or not col: return
        
        new_df = convert_dtypes(df, col, input.dtype_target())
        current_df.set(new_df)
        status_message.set(f"Converted {col} to {input.dtype_target()}")
        ui.notification_show("Data type converted.", type="message")

    # --- Feature Engineering Operations ---

    @reactive.Effect
    @reactive.event(input.apply_arith)
    def _():
        df = current_df()
        if df is None: return
        
        new_df = create_arithmetic_feature(
            df, input.fe_arith_col1(), input.fe_arith_col2(), 
            input.fe_arith_op(), input.fe_arith_name()
        )
        current_df.set(new_df)
        ui.notification_show("Arithmetic feature created.", type="message")

    @reactive.Effect
    @reactive.event(input.apply_trans)
    def _():
        df = current_df()
        if df is None: return
        
        new_df = transform_feature(df, input.fe_trans_col(), input.fe_trans_method())
        current_df.set(new_df)
        ui.notification_show("Transformation applied.", type="message")

    @reactive.Effect
    @reactive.event(input.apply_dt)
    def _():
        df = current_df()
        if df is None: return
        
        new_df = extract_datetime_features(df, input.fe_dt_col(), list(input.fe_dt_features()))
        current_df.set(new_df)
        ui.notification_show("Datetime features extracted.", type="message")

    @reactive.Effect
    @reactive.event(input.apply_drop)
    def _():
        df = current_df()
        cols = list(input.fe_drop_cols())
        if df is None or not cols: return
        
        new_df = drop_columns(df, cols)
        current_df.set(new_df)
        ui.notification_show("Columns dropped.", type="message")

    # --- EDA ---

    @output
    @render_widget
    def eda_plot():
        df = current_df()
        if df is None: return None
        
        plot_type = input.eda_plot_type()
        color = input.eda_color()
        if color == "None": color = None
        
        if plot_type == "histogram":
            return plot_histogram(df, input.eda_x(), color=color)
        elif plot_type == "box":
            return plot_box(df, input.eda_y(), input.eda_x(), color)
        elif plot_type == "bar":
            return plot_bar(df, input.eda_x(), input.eda_y(), color, input.eda_agg())
        elif plot_type == "scatter":
            return plot_scatter(df, input.eda_x(), input.eda_y(), color)
        elif plot_type == "heatmap":
            cols = list(input.eda_heat_cols())
            if not cols: return None
            return plot_heatmap(df, cols)
        return None

    @output
    @render.data_frame
    def eda_summary():
        df = current_df()
        if df is None: return None
        
        # If numeric column selected in X, show stats. If categorical, show counts.
        # But simpler to just show stats for whole DF or specific selection logic.
        # Let's show generic describe for now.
        return render.DataGrid(get_summary_statistics(df))

    # --- Export ---

    @render.download(filename="processed_data.csv")
    def download_data():
        df = current_df()
        if df is not None:
            yield df.to_csv(index=False)

app = App(app_ui, server)
