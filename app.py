from shiny import App, ui, reactive, render, req
import pandas as pd
import numpy as np
from pathlib import Path
from src.data_loader import load_dataset, get_dataset_info
import plotly.graph_objects as go
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
import pandas.api.types as ptypes

# --- UI Definitions ---

custom_css = """
.shiny-file-input-progress {
    animation: fadeOut 3s forwards;
    animation-delay: 2s;
}
@keyframes fadeOut {
    0% { opacity: 1; }
    100% { opacity: 0; visibility: hidden; }
}
"""

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
                    ui.input_select("mv_cols", "Columns (Optional)", choices=[], multiple=True, selectize=True),
                    ui.input_action_button("apply_mv", "Apply", class_="btn-primary w-100")
                ),
                ui.accordion_panel(
                    "Duplicates",
                    ui.input_action_button("apply_dedup", "Remove Duplicates", class_="btn-primary w-100")
                ),
                ui.accordion_panel(
                    "Outliers",
                    ui.input_select("outlier_col", "Column", choices=[], selectize=True),
                    ui.input_select("outlier_method", "Method", {"iqr": "IQR", "zscore": "Z-Score"}),
                    ui.input_numeric("outlier_threshold", "Threshold", 1.5, step=0.1),
                    ui.input_action_button("apply_outlier", "Filter Outliers", class_="btn-primary w-100")
                ),
                ui.accordion_panel(
                    "Scaling",
                    ui.input_select("scale_cols", "Columns", choices=[], multiple=True, selectize=True),
                    ui.input_select("scale_method", "Method", {"standard": "Standardization", "minmax": "Min-Max Scaling"}),
                    ui.input_action_button("apply_scale", "Apply Scaling", class_="btn-primary w-100")
                ),
                ui.accordion_panel(
                    "Encoding",
                    ui.input_select("encode_cols", "Columns", choices=[], multiple=True, selectize=True),
                    ui.input_select("encode_method", "Method", {"onehot": "One-Hot Encoding", "label": "Label Encoding"}),
                    ui.input_action_button("apply_encode", "Apply Encoding", class_="btn-primary w-100")
                ),
                ui.accordion_panel(
                    "Data Types",
                    ui.input_select("dtype_col", "Column", choices=[], selectize=True),
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
        ui.layout_sidebar(
            ui.sidebar(
                card_header("Operation History", "clock-rotate-left"),
                ui.output_ui("action_history_ui"),
                position="right",
                open="closed",
                id="history_sidebar",
                width=350
            ),
            ui.card(
                ui.card_header(
                    ui.row(
                        ui.column(6, card_header("Current Data State", "database"),
                                  ui.tags.small("Tip: Click the right-side arrow for History", class_="text-muted ms-2"),
                                  class_="d-flex align-items-center"
                                ),
                        ui.column(6, 
                            ui.div(
                                ui.input_action_button("undo_btn", "Undo", icon=fa.icon_svg("rotate-left"), class_="btn-sm btn-outline-secondary me-2"),
                                ui.input_action_button("redo_btn", "Redo", icon=fa.icon_svg("rotate-right"), class_="btn-sm btn-outline-secondary"),
                                class_="d-flex justify-content-end align-items-center"
                            )
                        )
                    )
                ),
                ui.output_text_verbatim("cleaning_status"),
                ui.output_data_frame("cleaning_preview"),
                full_screen=True
            ),
            fillable=True
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
                    "Custom Expression for new feature",
                    ui.p("Enter a math expression using column names. Common operators:"),
                    ui.tags.ul(
                        ui.tags.li("Add / Subtract: ", ui.tags.code("+"), " / ", ui.tags.code("-")),
                        ui.tags.li("Multiply / Divide: ", ui.tags.code("*"), " / ", ui.tags.code("/")),
                        ui.tags.li("Power: use ", ui.tags.code("**"), " (not ^), e.g., ", ui.tags.code("score ** 2"))
                    ),
                    ui.p(
                        "For more info, please refer to the ", 
                        ui.tags.a("Pandas eval() documentation", href="https://pandas.pydata.org/docs/reference/api/pandas.eval.html", target="_blank"),
                        "."
                    ),
                    ui.input_text("fe_arith_expr", "Expression", ""),
                    ui.input_text("fe_arith_name", "New Column Name", "new_feature"),
                    ui.input_action_button("apply_arith", "Create Feature", class_="btn-primary w-100")
                ),
                ui.accordion_panel(
                    "One-Click Transforms",
                    ui.input_select("fe_trans_col", "Column", choices=[], selectize=True),
                    ui.input_select("fe_trans_method", "Method", 
                                    {"log": "Log", "square": "Square", 
                                     "sqrt": "Square Root", "abs": "Absolute Value", "binning": "Binning"}),
                    ui.input_action_button("apply_trans", "Apply Transform", class_="btn-primary w-100")
                ),
                ui.accordion_panel(
                    "Datetime Extraction",
                    ui.input_select("fe_dt_col", "Datetime Column", choices=[], selectize=True),
                    ui.input_checkbox_group("fe_dt_features", "Extract", 
                                            {"year": "Year", "month": "Month", "day": "Day", 
                                             "weekday": "Weekday", "quarter": "Quarter"}),
                    ui.input_action_button("apply_dt", "Extract", class_="btn-primary w-100")
                ),
                ui.accordion_panel(
                    "Drop Columns",
                    ui.input_select("fe_drop_cols", "Columns to Drop", choices=[], multiple=True, selectize=True),
                    ui.input_action_button("apply_drop", "Drop Columns", class_="btn-warning w-100")
                )
            ),
            width=350
        ),
        ui.layout_sidebar(
            ui.sidebar(
                card_header("Operation History", "clock-rotate-left"),
                # Unique ID to avoid "Duplicate Output ID" error
                ui.output_ui("action_history_ui_fe"), 
                position="right",
                open="closed",
                id="history_sidebar_fe",
                width=350
            ),
            ui.card(
                ui.card_header(
                    ui.row(
                        ui.column(6, card_header("Engineered Data", "gears"),
                                  ui.tags.small("Tip: Click the right-side arrow for History", class_="text-muted ms-2"),
                                  class_="d-flex align-items-center"
                                  ),
                        ui.column(6, 
                            ui.div(
                                ui.div(
                                    ui.input_action_button("undo_btn_fe", "Undo", icon=fa.icon_svg("rotate-left"), class_="btn-sm btn-outline-secondary me-2"),
                                    ui.input_action_button("redo_btn_fe", "Redo", icon=fa.icon_svg("rotate-right"), class_="btn-sm btn-outline-secondary"),
                                    class_="d-flex justify-content-end align-items-center"
                                ),
                                class_="text-end"
                            )
                        )
                    )
                ),
                ui.output_data_frame("fe_preview"),
                full_screen=True
            ),
            fillable=True
        )
    ),
    icon=fa.icon_svg("flask")
)

# EDA Tab
eda_tab = ui.nav_panel(
    "EDA",
    ui.layout_sidebar(
        ui.sidebar(
            # --- 1. graph settings ---
            ui.input_select("eda_plot_type", "Plot Type", 
                            {"histogram": "Histogram", "box": "Box Plot", 
                             "bar": "Bar Chart", "scatter": "Scatter Plot", 
                             "heatmap": "Correlation Heatmap"}),
            ui.panel_conditional(
                "input.eda_plot_type != 'heatmap'",
                ui.input_select("eda_x", "X Axis", choices=[], selectize=True),
                ui.input_select("eda_color", "Color (Optional)", choices=["None"], selectize=True)
            ),
            ui.panel_conditional(
                "input.eda_plot_type == 'box' || input.eda_plot_type == 'bar' || input.eda_plot_type == 'scatter'",
                ui.input_select("eda_y", "Y Axis", choices=[], selectize=True)
            ),
            ui.panel_conditional(
                "input.eda_plot_type == 'bar'",
                ui.input_select("eda_agg", "Aggregation", {"count": "Count", "mean": "Mean", "sum": "Sum"})
            ),
            ui.panel_conditional(
                "input.eda_plot_type == 'heatmap'",
                ui.input_select(
                    "eda_heat_cols", 
                    "Columns", 
                    choices=[], 
                    multiple=True, 
                    selectize=True 
                )
            ),            
            ui.hr(), 

            # --- 2. Global Filters ---
            ui.h6("📊 Global Filters"),
            
            ui.input_select("filter_num_col", "Numeric Filter", choices=["None"]),
            ui.output_ui("dynamic_num_slider"), 
            
            ui.tags.br(),
            
            ui.input_select("filter_cat_col", "Category Filter", choices=["None"]),
            ui.output_ui("dynamic_cat_selector"), 
            
            ui.hr(), 
            
            ui.input_action_button(
                "run_eda", 
                "Generate Plot", 
                class_="btn-primary w-100", 
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

    # --- Store History ---
    # ==========================================
    # --- (Undo/Redo) ---
    # ==========================================
    
    current_df = reactive.Value(None)
    status_message = reactive.Value("Ready")
    
    history_log = reactive.Value([])
    undo_stack_df = reactive.Value([])
    redo_stack_df = reactive.Value([])
    redo_stack_log = reactive.Value([])
    
    MAX_UNDO = 10 

    def save_action_state(df, action_msg):
        undo_dfs = undo_stack_df.get().copy()
        undo_dfs.append(df.copy())
        
        if len(undo_dfs) > MAX_UNDO:
            undo_dfs.pop(0)
            
        undo_stack_df.set(undo_dfs)
        
        redo_stack_df.set([])
        redo_stack_log.set([])
        
        logs = history_log.get().copy()
        logs.append(action_msg)
        history_log.set(logs)

    def perform_undo():
        undo_dfs = undo_stack_df.get().copy()
        
        if not undo_dfs:
            ui.notification_show("Cannot undo: You have reached the oldest available state (Max 10 steps).", type="warning")
            return
        
        prev_df = undo_dfs.pop()
        
        redo_dfs = redo_stack_df.get().copy()
        redo_dfs.append(current_df.get().copy())
        redo_stack_df.set(redo_dfs)
        
        logs = history_log.get().copy()
        if logs:
            undone_msg = logs.pop()
            redo_logs = redo_stack_log.get().copy()
            redo_logs.append(undone_msg)
            redo_stack_log.set(redo_logs)
            history_log.set(logs)
            
            msg = f"Undid: {undone_msg}"
            status_message.set(msg)
            ui.notification_show(msg, type="message")

        undo_stack_df.set(undo_dfs)
        current_df.set(prev_df)

    def perform_redo():
        redo_dfs = redo_stack_df.get().copy()
        
        if not redo_dfs:
            ui.notification_show("Cannot redo: You are already at the newest state.", type="warning")
            return
            
        next_df = redo_dfs.pop()
        
        undo_dfs = undo_stack_df.get().copy()
        undo_dfs.append(current_df.get().copy())
        if len(undo_dfs) > MAX_UNDO:
            undo_dfs.pop(0)
        undo_stack_df.set(undo_dfs)
        
        redo_logs = redo_stack_log.get().copy()
        if redo_logs:
            next_log = redo_logs.pop()
            logs = history_log.get().copy()
            logs.append(next_log)
            history_log.set(logs)
            redo_stack_log.set(redo_logs)
            
            msg = f"Redid: {next_log}"
            status_message.set(msg)
            ui.notification_show(msg, type="message")
        
        redo_stack_df.set(redo_dfs)
        current_df.set(next_df)

    @reactive.Effect
    @reactive.event(input.undo_btn, getattr(input, "undo_btn_fe", None))
    def _():
        perform_undo()

    @reactive.Effect
    @reactive.event(input.redo_btn, getattr(input, "redo_btn_fe", None))
    def _():
        perform_redo()
#---History Bar---
    @render.ui
    def action_history_ui():
        logs = history_log.get()
        if not logs: return ui.p("No actions performed yet.", class_="text-muted")
        list_items = [ui.tags.li(log, class_="list-group-item") for log in reversed(logs)]
        return ui.tags.ul(*list_items, class_="list-group list-group-flush")

    @render.ui
    def action_history_ui_fe():
        logs = history_log.get()
        if not logs: return ui.p("No actions performed yet.", class_="text-muted")
        list_items = [ui.tags.li(log, class_="list-group-item") for log in reversed(logs)]
        return ui.tags.ul(*list_items, class_="list-group list-group-flush")

#---EDA Trigger---

    plot_trigger = reactive.Value(0)
    @reactive.Effect
    @reactive.event(input.run_eda)
    def check_before_plotting():
        df = filtered_df()
        if df is None: return

        is_large_data = len(df) > 50000
        
        is_huge_heatmap = (input.eda_plot_type() == 'heatmap') and (len(list(input.eda_heat_cols())) > 15)
        
        color_col = input.eda_color()
        is_high_cardinality = False
        unique_categories = 0
        
        if color_col != "None" and color_col in df.columns:
            unique_categories = df[color_col].nunique()
            if unique_categories > 20: 
                is_high_cardinality = True

        if is_large_data or is_huge_heatmap or is_high_cardinality:
            
            warning_msg = []
            if is_large_data:
                warning_msg.append(f"• Large dataset ({len(df)} rows).")
            if is_huge_heatmap:
                warning_msg.append("• Too many columns for a heatmap.")
            if is_high_cardinality:
                warning_msg.append(f"• The color column '{color_col}' has {unique_categories} unique values. This will generate too many categories/legends and may crash the browser.")

            m = ui.modal(
                ui.p("⚠️ Warning: Your current plot settings might cause performance issues:"),
                ui.tags.ul([ui.tags.li(msg) for msg in warning_msg]), 
                ui.p("Are you sure you want to proceed?"),
                title="Performance & Logic Warning",
                footer=ui.tags.div(
                    ui.modal_button("Cancel", class_="btn-secondary"),
                    ui.input_action_button("confirm_plot", "Force Plot", class_="btn-danger"),
                    class_="d-flex gap-2"
                ),
                easy_close=True,
            )
            ui.modal_show(m)
        else:
            plot_trigger.set(plot_trigger.get() + 1)
    @reactive.Effect
    @reactive.event(input.confirm_plot)
    def force_plot():
        ui.modal_remove()
        plot_trigger.set(plot_trigger.get() + 1)

#---EDA filtering---

    @reactive.Effect
    def update_filter_cols():
        df = current_df()
        if df is not None:
            num_cols = df.select_dtypes(include=np.number).columns.tolist()
            ui.update_select("filter_num_col", choices=["None"] + num_cols)
            
            cat_cols = df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
            ui.update_select("filter_cat_col", choices=["None"] + cat_cols)
    @output
    @render.ui
    def dynamic_num_slider():
        df = current_df()
        col = input.filter_num_col()
        
        if df is None or col == "None" or col not in df.columns:
            return None
        
        min_val = float(df[col].min())
        max_val = float(df[col].max())
        
        is_int = pd.api.types.is_integer_dtype(df[col])
        step_val = 1 if is_int else (max_val - min_val) / 100
        
        return ui.TagList(
            ui.input_slider(
                "num_slider_range", 
                f"Range for {col}", 
                min=min_val, max=max_val, value=[min_val, max_val], step=step_val
            ),
            ui.layout_columns(
                ui.input_numeric("num_filter_min", "Exact Min", value=min_val),
                ui.input_numeric("num_filter_max", "Exact Max", value=max_val)
            )
        )
    @reactive.Effect
    @reactive.event(input.num_slider_range)
    def sync_slider_to_box():
        vals = input.num_slider_range()
        if vals:
            ui.update_numeric("num_filter_min", value=vals[0])
            ui.update_numeric("num_filter_max", value=vals[1])

    @reactive.Effect
    @reactive.event(input.num_filter_min, input.num_filter_max)
    def sync_box_to_slider():
        min_v = input.num_filter_min()
        max_v = input.num_filter_max()
        if min_v is not None and max_v is not None:
            ui.update_slider("num_slider_range", value=[min_v, max_v])
    
    @output
    @render.ui
    def dynamic_cat_selector():
        df = current_df()
        col = input.filter_cat_col()
        
        if df is None or col == "None" or col not in df.columns:
            return None
        
        unique_values = sorted(df[col].dropna().unique().tolist())
        
        return ui.input_selectize(
            "cat_filter_values", 
            f"Select values for {col}", 
            choices=unique_values,
            selected=None, 
            multiple=True,
            options={"plugins": ["remove_button"]} 
        )
    @reactive.calc
    def filtered_df():
        df = current_df()
        if df is None: return None
        
        num_col = input.filter_num_col()
        if num_col != "None" and num_col in df.columns:
            min_v = input.num_filter_min()
            max_v = input.num_filter_max()
            if min_v is not None and max_v is not None:
                df = df[(df[num_col] >= min_v) & (df[num_col] <= max_v)]
        
        cat_col = input.filter_cat_col()
        if cat_col != "None" and cat_col in df.columns:
            selected_cats = input.cat_filter_values()
            if selected_cats:
                df = df[df[cat_col].isin(selected_cats)]
            else:
                return df.iloc[0:0] 
                
        return df

   # --- Data Loading ---
    @reactive.Effect
    @reactive.event(input.file_upload)
    def _():
        file_infos = input.file_upload()
        if not file_infos: return
    
        file_info = file_infos[0]
        df, error = load_dataset(file_info["datapath"])
    
        if df is not None:
            current_df.set(df)
            
            # --- Clear all history and Undo/Redo stacks for new dataset ---
            undo_stack_df.set([])
            redo_stack_df.set([])
            redo_stack_log.set([])
            
            msg = f"Loaded {file_info['name']} with {df.shape[0]} rows and {df.shape[1]} columns."
            status_message.set(msg)
            
            # Reset history log to start fresh
            history_log.set([msg])
            ui.notification_show("File loaded successfully!", type="message")
        else:
            ui.notification_show(f"Error loading file: {error}", type="error")

    @reactive.Effect
    @reactive.event(input.load_sample_1)
    def _():
        df, _ = load_dataset("data/sample_dataset_1.csv")
        current_df.set(df)
        
        # --- Clear all history and Undo/Redo stacks for new dataset ---
        undo_stack_df.set([])
        redo_stack_df.set([])
        redo_stack_log.set([])
        
        msg = "Loaded sample dataset 1."
        status_message.set(msg)
        
        # Reset history log to start fresh
        history_log.set([msg])
        ui.notification_show("Sample data loaded!", type="message")

    @reactive.Effect
    @reactive.event(input.load_sample_2)
    def _():
        df, _ = load_dataset("data/sample_dataset_2.csv")
        current_df.set(df)
        
        # --- Clear all history and Undo/Redo stacks for new dataset ---
        undo_stack_df.set([])
        redo_stack_df.set([])
        redo_stack_log.set([])
        
        msg = "Loaded sample dataset 2."
        status_message.set(msg)
        
        # Reset history log to start fresh
        history_log.set([msg])
        ui.notification_show("Sample data loaded!", type="message")

    @reactive.Effect
    @reactive.event(input.reset_data)
    def _():
        current_df.set(None)
        
        # --- Clear EVERYTHING on reset ---
        undo_stack_df.set([])
        redo_stack_df.set([])
        redo_stack_log.set([])
        history_log.set([])
        
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
    def build_history_ui(logs):
        if not logs:
            return ui.p("No operations performed yet. Your data processing steps will appear here.")
        items = [ui.tags.li(log, class_="list-group-item") for log in logs]
        return ui.tags.ol(items, class_="list-group list-group-numbered")

    @output
    @render.ui
    def action_history_ui():
        return build_history_ui(history_log.get())
    @output
    @render.ui
    def action_history_ui_fe():
        return build_history_ui(history_log.get())
    
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
            all_cols = []
            str_cols = []
            dt_cols = []
            num_cols = []
            cat_cols = []
        else:
            all_cols = list(df.columns)
            num_cols = list(df.select_dtypes(include=['number']).columns)
            str_cols = list(df.select_dtypes(include=['object', 'string']).columns)
            dt_cols = list(df.select_dtypes(include=['datetime', 'datetimetz']).columns)
            cat_cols = list(df.select_dtypes(include=['category']).columns)
        # Update all column selectors
        ui.update_select("mv_cols", choices=all_cols)
        ui.update_select("outlier_col", choices=num_cols)
        ui.update_select("scale_cols", choices=num_cols)
        ui.update_select("encode_cols", choices=str_cols+cat_cols)
        ui.update_select("dtype_col", choices=all_cols)
        
        ui.update_select("fe_trans_col", choices=num_cols)
        ui.update_select("fe_dt_col", choices=dt_cols)
        ui.update_select("fe_drop_cols", choices=all_cols)
        
        ui.update_select("eda_x", choices=all_cols)
        ui.update_select("eda_y", choices=all_cols)#?
        ui.update_select("eda_color", choices=["None"] + all_cols)
        ui.update_select("eda_heat_cols", choices=all_cols)#?

    # --- Cleaning Operations ---

#--missing values---
    @reactive.Effect
    @reactive.event(input.apply_mv)
    def _():
        df = current_df()
        if df is None: return
        
        raw_cols = input.mv_cols()
        cols = list(raw_cols) if raw_cols else list(df.columns)
        
        fill_val = None
        method = input.mv_method()
        if method == 'constant':
            try:
                fill_val = float(input.mv_fill_value())
            except:
                fill_val = input.mv_fill_value()

        missing_before = df[cols].isna().sum().sum()
        new_df = handle_missing_values(df, method, list(raw_cols) if raw_cols else None, fill_val)
        missing_after = new_df[cols].isna().sum().sum()
        missing_diff = missing_before - missing_after
        
        cols_formatted = ", ".join([f"'{c}'" for c in cols])
        
        if method in ["drop_rows", "drop_cols"]:
            msg = f"Dropped {missing_diff} missing values in column {cols_formatted}."
        else:
            method_display = f"constant '{fill_val}'" if method == 'constant' else method.upper()
            msg = f"Filled {missing_diff} missing values in column {cols_formatted} using {method_display}."
            
        # --- Save state BEFORE modifying data ---
        save_action_state(df, msg)
        
        current_df.set(new_df)
        status_message.set(msg)
        ui.notification_show(msg, type="message")

#---Delete Duplicates---
    @reactive.Effect
    @reactive.event(input.apply_dedup)
    def _():
        df = current_df()
        if df is None: return
        
        rows_before = len(df)
        new_df = remove_duplicates(df)
        rows_after = len(new_df)
        rows_diff = rows_before - rows_after
        
        msg = f"Removed {rows_diff} duplicate rows from the dataset."
        
        # --- NEW: Save state BEFORE modification ---
        save_action_state(df, msg)
        
        current_df.set(new_df)
        status_message.set(msg)
        ui.notification_show(msg, type="message")

#---Filter out outliers---
    @reactive.Effect
    @reactive.event(input.apply_outlier)
    def _():
        df = current_df()
        col = input.outlier_col()
        if df is None or not col: return

        method = input.outlier_method()
        threshold = input.outlier_threshold()
        rows_before = len(df)
        new_df = filter_outliers(df, col, method, threshold)
        rows_after = len(new_df)
        diff = rows_before - rows_after

        method_display = "IQR" if method == "iqr" else "Z-Score"
        msg = f"Filtered {diff} outliers from column '{col}' using {method_display} method (threshold: {threshold})."
        
        # --- NEW: Save state BEFORE modification ---
        save_action_state(df, msg)

        current_df.set(new_df)
        status_message.set(msg)
        ui.notification_show(msg, type="message")     

#---Scaling---

    @reactive.Effect
    @reactive.event(input.apply_scale)
    def _():
        df = current_df()
        cols = list(input.scale_cols())
        if df is None or not cols: return
        
        new_df = scale_features(df, cols, input.scale_method())
        
        method_display = "Standardization" if input.scale_method() == "standard" else "Min-Max Scaling"
        msg = f"Scaled columns: {', '.join(cols)} using {method_display}."

        # --- Save state BEFORE modification (Undo/Redo logic) ---
        save_action_state(df, msg)
        
        current_df.set(new_df)
        status_message.set(msg)
        ui.notification_show(msg, type="message")

#---Encoding---
    @reactive.Effect
    @reactive.event(input.apply_encode)
    def _():
        df = current_df()
        cols = list(input.encode_cols())
        if df is None or not cols: return
        
        new_df = encode_categorical(df, cols, input.encode_method())
        
        method_display = "One-Hot Encoding" if input.encode_method() == "onehot" else "Label Encoding"
        msg = f"Encoded columns: {', '.join(cols)} using {method_display}."

        # --- Save state BEFORE modification (Undo/Redo logic) ---
        save_action_state(df, msg)
        
        current_df.set(new_df)
        status_message.set(msg)
        ui.notification_show(msg, type="message")

#---Dtype---
    @reactive.Effect
    @reactive.event(input.apply_dtype)
    def _():
        df = current_df()
        col = input.dtype_col()
        if df is None or not col: return
        
        target_type = input.dtype_target()
        msg = f"Converted column '{col}' to {target_type}."
        
        # --- Save state BEFORE modification (Undo/Redo logic) ---
        save_action_state(df, msg)
        
        new_df = convert_dtypes(df, col, target_type)
        current_df.set(new_df)
        status_message.set(msg)
        ui.notification_show(msg, type="message")

    # --- Feature Engineering Operations ---

# --- Custom Expression Feature Creation ---
    @reactive.Effect
    @reactive.event(input.apply_arith)
    def _():
        df = current_df()
        if df is None: return
        
        expr = input.fe_arith_expr().strip()
        new_col_name = input.fe_arith_name().strip()
        
        if not expr or not new_col_name:
            ui.notification_show("Please enter both an expression and a new column name.", type="warning")
            return
            
        try:
            # Attempt to calculate the new column based on the expression
            new_col_data = df.eval(expr)
            msg = f"Created feature '{new_col_name}' using expression: {expr}"
            
            # --- Save state BEFORE modification (Undo/Redo logic) ---
            save_action_state(df, msg)
            
            # Add the new column to the dataframe
            new_df = df.copy()
            new_df[new_col_name] = new_col_data
            current_df.set(new_df)
            
            status_message.set(msg)
            ui.notification_show(f"Success: {msg}", type="message")
            
        except Exception as e:
            # Catch evaluation errors (e.g., typos in column names) without crashing the app
            ui.notification_show(f"Evaluation Error: Please check your expression. ({str(e)})", type="error", duration=5)

#---One-Click Transforms---
    @reactive.Effect
    @reactive.event(input.apply_trans)
    def _():
        df = current_df()
        col = input.fe_trans_col()
        method = input.fe_trans_method()
        if df is None or not col: return
        
        # map names for history
        method_names = {
            "log": "Log", 
            "square": "Square", 
            "sqrt": "Square Root", 
            "abs": "Absolute Value", 
            "binning": "Binning"
        }
        display_method = method_names.get(method, method)
        msg = f"Applied {display_method} transformation to column '{col}'."
        
        # --- Save state BEFORE modification (Undo/Redo logic) ---
        save_action_state(df, msg)
        
        new_df = transform_feature(df, col, method)
        current_df.set(new_df)
        status_message.set(msg)
        ui.notification_show(f"Success: {msg}", type="message")

#---Datetime Extraction---
    @reactive.Effect
    @reactive.event(input.apply_dt)
    def _():
        df = current_df()
        col = input.fe_dt_col()
        features = list(input.fe_dt_features())
        
        if df is None or not col or not features: 
            ui.notification_show("Please select a column and at least one feature to extract.", type="warning")
            return
        
        feature_names = ", ".join(features)
        msg = f"Extracted datetime features ({feature_names}) from column '{col}'."
        
        # --- Save state BEFORE modification (Undo/Redo logic) ---
        save_action_state(df, msg)
        
        new_df = extract_datetime_features(df, col, features)
        current_df.set(new_df)
        status_message.set(msg)
        ui.notification_show(f"Success: {msg}", type="message")

#---Drop cols---
    @reactive.Effect
    @reactive.event(input.apply_drop)
    def _():
        df = current_df()
        cols_to_drop = list(input.fe_drop_cols())
        
        if df is None or not cols_to_drop:
            ui.notification_show("Please select at least one column to drop.", type="warning")
            return
        
        try:
            dropped_str = ", ".join(cols_to_drop)
            msg = f"Dropped columns: {dropped_str}"
            
            # --- Save state BEFORE modification (Undo/Redo logic) ---
            save_action_state(df, msg)
            
            new_df = df.drop(columns=cols_to_drop)
            current_df.set(new_df)
            status_message.set(msg)
            ui.notification_show(f"Success: {msg}", type="message")
            
        except Exception as e:
            ui.notification_show(f"Error dropping columns: {str(e)}", type="error")

    # --- EDA ---

    @output
    @render_widget
    @reactive.event(plot_trigger)
    def eda_plot():
        if plot_trigger.get() == 0:
            return go.Figure()

        df = filtered_df()
        if df is None or len(df) == 0:
            ui.notification_show("No data left after filtering!", type="warning")
            return go.Figure()
        
        # --- 1. get input ---
        plot_type = input.eda_plot_type()
        x_col = input.eda_x()
        y_col = input.eda_y()
        color = input.eda_color()
        if color == "None": color = None
        heat_cols = list(input.eda_heat_cols()) 
        # --- 2. Pre-check ---
        
        if plot_type in ["scatter", "box", "histogram"]:
            # check X
            if not pd.api.types.is_numeric_dtype(df[x_col]):
                ui.notification_show(f"Error: X-Axis ({x_col}) must be numeric for {plot_type}!", type="error", duration=5)
                return go.Figure()
            
            # check Y
            if plot_type == "scatter" and not pd.api.types.is_numeric_dtype(df[y_col]):
                ui.notification_show(f"Error: Y-Axis ({y_col}) must be numeric for Scatter Plot!", type="error", duration=5)
                return go.Figure()

        # for Heatmap
        if plot_type == "heatmap":
            if not heat_cols or len(heat_cols) < 2:
                ui.notification_show("Error: Please select at least 2 columns for Heatmap!", type="error")
                return go.Figure()
            
            # Numeric check
            non_num = [c for c in heat_cols if not pd.api.types.is_numeric_dtype(df[c])]
            if non_num:
                ui.notification_show(f"Error: Heatmap only works with numeric columns. Please remove: {', '.join(non_num)}", type="error")
                return go.Figure()

        # --- 3. graph logic---
        try:
            if plot_type == "histogram":
                return plot_histogram(df, x_col, color=color)
            
            elif plot_type == "box":
                return plot_box(df, y_col, x_col, color)
            
            elif plot_type == "bar":
                return plot_bar(df, x_col, y_col, color, input.eda_agg())
            
            elif plot_type == "scatter":
                return plot_scatter(df, x_col, y_col, color)
                
            elif plot_type == "heatmap":
                return plot_heatmap(df, heat_cols)
            
            return go.Figure()
            
        except Exception as e:
            ui.notification_show(f"Plot execution error: {str(e)}", type="error", duration=8)
            return go.Figure()   
         
    @output
    @render.data_frame
    @reactive.event(plot_trigger)
    def eda_summary():
        if plot_trigger.get() == 0:
            return render.DataGrid(pd.DataFrame())

        df = filtered_df() 
        
        if df is None: return render.DataGrid(pd.DataFrame())
        
        try:
            return render.DataGrid(get_summary_statistics(df))
        except Exception as e:
            ui.notification_show(f"Summary table error: {str(e)}", type="error", duration=8)
            return render.DataGrid(df.head())

    # --- Export ---

    @render.download(filename="processed_data.csv")
    def download_data():
        df = current_df()
        if df is not None:
            yield df.to_csv(index=False)

app = App(app_ui, server)
