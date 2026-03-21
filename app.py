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

custom_css = """
/* ── Base & Brand ── */
:root {
    --brand-primary:   #4f46e5;
    --brand-secondary: #7c3aed;
    --brand-accent:    #06b6d4;
    --brand-success:   #10b981;
    --brand-warning:   #f59e0b;
    --brand-danger:    #ef4444;
    --surface:         #f8fafc;
    --card-bg:         #ffffff;
    --text-primary:    #1e293b;
    --text-muted:      #64748b;
    --border:          #e2e8f0;
    --shadow-sm:       0 1px 3px rgba(0,0,0,.08), 0 1px 2px rgba(0,0,0,.05);
    --shadow-md:       0 4px 12px rgba(0,0,0,.10);
    --radius:          10px;
}

body {
    background: var(--surface);
    color: var(--text-primary);
    font-family: 'Inter', system-ui, -apple-system, sans-serif;
}

/* ── Navbar ── */
.navbar {
    background: linear-gradient(135deg, var(--brand-primary) 0%, var(--brand-secondary) 100%) !important;
    box-shadow: 0 2px 8px rgba(79,70,229,.35);
    padding: 0 1.5rem;
}
.navbar-brand {
    font-size: 1.25rem !important;
    font-weight: 700 !important;
    color: #fff !important;
    letter-spacing: -.3px;
}
.navbar .nav-link {
    color: rgba(255,255,255,.85) !important;
    font-weight: 500;
    padding: .75rem 1rem !important;
    border-radius: 6px;
    transition: background .2s, color .2s;
}
.navbar .nav-link:hover,
.navbar .nav-link.active {
    color: #fff !important;
    background: rgba(255,255,255,.15) !important;
}

/* ── Cards ── */
.card {
    background: var(--card-bg);
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
    box-shadow: var(--shadow-sm);
    transition: box-shadow .2s;
}
.card:hover { box-shadow: var(--shadow-md); }
.card-title {
    font-size: .95rem;
    font-weight: 600;
    color: var(--text-primary);
    display: flex;
    align-items: center;
    gap: .5rem;
}

/* ── Sidebar ── */
.bslib-sidebar-layout > .sidebar {
    background: var(--card-bg) !important;
    border-right: 1px solid var(--border) !important;
}

/* ── Buttons ── */
.btn-primary {
    background: var(--brand-primary) !important;
    border-color: var(--brand-primary) !important;
    font-weight: 500;
    border-radius: 7px !important;
    transition: opacity .15s, transform .1s;
}
.btn-primary:hover  { opacity: .88; transform: translateY(-1px); }
.btn-outline-primary {
    color: var(--brand-primary) !important;
    border-color: var(--brand-primary) !important;
    font-weight: 500;
    border-radius: 7px !important;
    transition: background .15s, color .15s;
}
.btn-outline-primary:hover {
    background: var(--brand-primary) !important;
    color: #fff !important;
}
.btn-warning  { border-radius: 7px !important; font-weight: 500; }
.btn-danger   { border-radius: 7px !important; font-weight: 500; }
.btn-success  {
    background: var(--brand-success) !important;
    border-color: var(--brand-success) !important;
    font-weight: 500;
    border-radius: 7px !important;
    font-size: 1.05rem;
    padding: .6rem 1.6rem !important;
    transition: opacity .15s, transform .1s;
}
.btn-success:hover { opacity: .88; transform: translateY(-1px); }

/* ── Info Boxes ── */
.info-box {
    background: var(--card-bg);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    box-shadow: var(--shadow-sm);
    padding: 1rem 1.25rem;
    display: flex;
    align-items: center;
    gap: 1rem;
    transition: box-shadow .2s;
}
.info-box:hover { box-shadow: var(--shadow-md); }
.info-box-icon {
    width: 48px; height: 48px;
    border-radius: 10px;
    display: flex; align-items: center; justify-content: center;
    flex-shrink: 0;
}
.info-box-icon.primary { background: rgba(79,70,229,.12); color: var(--brand-primary); }
.info-box-icon.success { background: rgba(16,185,129,.12); color: var(--brand-success); }
.info-box-icon.warning { background: rgba(245,158,11,.15); color: var(--brand-warning); }
.info-box-icon.danger  { background: rgba(239,68,68,.12);  color: var(--brand-danger);  }
.info-box-label { font-size: .78rem; color: var(--text-muted); font-weight: 500; text-transform: uppercase; letter-spacing: .05em; }
.info-box-value { font-size: 1.6rem; font-weight: 700; color: var(--text-primary); line-height: 1.1; }

/* ── Accordion ── */
.accordion-button {
    font-weight: 600 !important;
    font-size: .88rem !important;
    color: var(--text-primary) !important;
    background: transparent !important;
}
.accordion-button:not(.collapsed) {
    color: var(--brand-primary) !important;
    box-shadow: none !important;
}
.accordion-item { border-color: var(--border) !important; }

/* ── Form Controls ── */
.form-control, .form-select {
    border-radius: 7px !important;
    border-color: var(--border) !important;
    font-size: .88rem;
    transition: border-color .2s, box-shadow .2s;
}
.form-control:focus, .form-select:focus {
    border-color: var(--brand-primary) !important;
    box-shadow: 0 0 0 3px rgba(79,70,229,.15) !important;
}
label { font-size: .83rem; font-weight: 500; color: var(--text-muted); }

/* ── Status / verbatim text ── */
pre.shiny-text-output {
    background: #f1f5f9;
    border: 1px solid var(--border);
    border-radius: 7px;
    font-size: .82rem;
    color: var(--text-muted);
    padding: .6rem 1rem;
}

/* ── User Guide ── */
.guide-hero {
    background: linear-gradient(135deg, var(--brand-primary) 0%, var(--brand-secondary) 100%);
    border-radius: var(--radius);
    padding: 2.5rem 2rem;
    color: #fff;
    margin-bottom: 1.5rem;
}
.guide-hero h2 { font-size: 1.8rem; font-weight: 700; margin-bottom: .4rem; }
.guide-hero p  { opacity: .85; margin: 0; font-size: 1rem; }
.guide-step-card {
    background: var(--card-bg);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1.2rem 1.4rem;
    display: flex; gap: 1rem; align-items: flex-start;
    margin-bottom: .75rem;
    transition: box-shadow .2s;
}
.guide-step-card:hover { box-shadow: var(--shadow-md); }
.guide-step-num {
    width: 36px; height: 36px; border-radius: 50%;
    background: linear-gradient(135deg, var(--brand-primary), var(--brand-secondary));
    color: #fff; font-weight: 700; font-size: .95rem;
    display: flex; align-items: center; justify-content: center; flex-shrink: 0;
}
.guide-step-title { font-weight: 600; font-size: .95rem; margin-bottom: .2rem; }
.guide-step-desc  { font-size: .85rem; color: var(--text-muted); margin: 0; }

/* ── Export page ── */
.export-card {
    max-width: 480px;
    margin: 3rem auto;
    text-align: center;
    padding: 3rem 2rem;
    border-radius: var(--radius);
    background: var(--card-bg);
    border: 1px solid var(--border);
    box-shadow: var(--shadow-md);
}
.export-icon {
    width: 72px; height: 72px; border-radius: 50%;
    background: linear-gradient(135deg, var(--brand-primary), var(--brand-secondary));
    display: flex; align-items: center; justify-content: center;
    margin: 0 auto 1.25rem;
    color: #fff;
}

/* ── File upload drop zone ── */
.shiny-input-container input[type=file] { border-radius: 7px; }

/* ── Notification ── */
.shiny-notification {
    border-radius: var(--radius) !important;
    box-shadow: var(--shadow-md) !important;
    font-weight: 500;
}

/* ── Fade-out upload progress ── */
.shiny-file-input-progress {
    animation: fadeOut 3s forwards;
    animation-delay: 2s;
}
@keyframes fadeOut {
    0%   { opacity: 1; }
    100% { opacity: 0; visibility: hidden; }
}
"""

# User Guide Tab
user_guide_tab = ui.nav_panel(
    "User Guide",
    ui.div(
        ui.div(
            ui.h2(fa.icon_svg("chart-line"), " Data Explorer Pro"),
            ui.p("A complete, no-code pipeline for uploading, cleaning, engineering, and visualizing your data."),
            class_="guide-hero"
        ),
        ui.div(
            ui.div(
                ui.div("1", class_="guide-step-num"),
                ui.div(
                    ui.p("Data Upload", class_="guide-step-title"),
                    ui.p("Upload CSV, Excel, JSON, or Parquet files — or load one of the built-in sample datasets to explore the app instantly.", class_="guide-step-desc"),
                ),
            class_="guide-step-card"),
            ui.div(
                ui.div("2", class_="guide-step-num"),
                ui.div(
                    ui.p("Cleaning & Preprocessing", class_="guide-step-title"),
                    ui.p("Handle missing values, remove duplicates, filter outliers, scale numeric features, and encode categorical variables interactively.", class_="guide-step-desc"),
                ),
            class_="guide-step-card"),
            ui.div(
                ui.div("3", class_="guide-step-num"),
                ui.div(
                    ui.p("Feature Engineering", class_="guide-step-title"),
                    ui.p("Create new columns via arithmetic operations, math transforms (log, sqrt, square), or extract components from datetime columns.", class_="guide-step-desc"),
                ),
            class_="guide-step-card"),
            ui.div(
                ui.div("4", class_="guide-step-num"),
                ui.div(
                    ui.p("Exploratory Data Analysis", class_="guide-step-title"),
                    ui.p("Explore your data with interactive Plotly charts: Histogram, Box Plot, Bar Chart, Scatter Plot, and Correlation Heatmap.", class_="guide-step-desc"),
                ),
            class_="guide-step-card"),
            ui.div(
                ui.div("5", class_="guide-step-num"),
                ui.div(
                    ui.p("Export", class_="guide-step-title"),
                    ui.p("Download your fully processed dataset as a CSV file at any point in the pipeline.", class_="guide-step-desc"),
                ),
            class_="guide-step-card"),
        ),
        ui.div(
            ui.tags.b(fa.icon_svg("lightbulb"), " Tip"),
            ui.tags.p("Changes are applied sequentially and persist across tabs. Use Reset Data in the Cleaning tab to start over.", style="margin:0; font-size:.85rem; color:var(--text-muted)"),
            class_="alert alert-info mt-3", style="border-radius:var(--radius); border:none; background:rgba(79,70,229,.08);"
        ),
        class_="p-4", style="max-width:780px; margin:0 auto;"
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
        ui.div(
            ui.div(
                fa.icon_svg("file-arrow-down", width="32px"),
                class_="export-icon"
            ),
            ui.h4("Download Processed Dataset", style="font-weight:700; margin-bottom:.5rem;"),
            ui.p(
                "All cleaning, preprocessing, and feature engineering steps are included.",
                style="color:var(--text-muted); margin-bottom:1.75rem; font-size:.95rem;"
            ),
            ui.download_button("download_data", "Download as CSV", class_="btn-success"),
            ui.hr(style="margin:1.5rem 0; border-color:var(--border);"),
            ui.p(
                fa.icon_svg("circle-info"), " The exported file reflects the current state of the data.",
                style="font-size:.82rem; color:var(--text-muted); margin:0;"
            ),
            class_="export-card"
        ),
        style="display:flex; justify-content:center; align-items:flex-start; padding:2rem;"
    ),
    icon=fa.icon_svg("file-export")
)

app_ui = ui.page_navbar(
    ui.head_content(ui.tags.style(custom_css)),
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
            info_box("Columns", str(info["cols"]), "table-columns"),
            info_box("Missing Values", str(sum(info["missing_values"].values())), "triangle-exclamation", "bg-warning-subtle" if sum(info["missing_values"].values()) > 0 else "bg-light"),
            info_box("Duplicates", str(info["duplicates"]), "copy", "bg-danger-subtle" if info["duplicates"] > 0 else "bg-light"),
            width=1/4
        )

    @output
    @render.data_frame
    def data_preview():
        df = current_df()
        if df is None:
            return render.DataGrid(pd.DataFrame())
        return render.DataGrid(df)

    @output
    @render.data_frame
    def cleaning_preview():
        df = current_df()
        if df is None:
            return render.DataGrid(pd.DataFrame())
        return render.DataGrid(df)
    
    @output
    @render.data_frame
    def fe_preview():
        df = current_df()
        if df is None:
            return render.DataGrid(pd.DataFrame())
        return render.DataGrid(df)

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
        if df is None: return render.DataGrid(pd.DataFrame())
        
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