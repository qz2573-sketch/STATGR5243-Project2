from shiny import ui
import faicons as fa

def card_header(title: str, icon: str = None):
    """Creates a styled card header with an optional icon."""
    if icon:
        return ui.div(
            fa.icon_svg(icon),
            ui.span(title, style="margin-left:.5rem;"),
            class_="card-title"
        )
    return ui.div(title, class_="card-title")

def info_box(title: str, value: str, icon: str, bg_color: str = "bg-light"):
    """Creates a polished info box using the CSS design system."""
    # Map bg_color hint to icon style class
    if "warning" in bg_color:
        icon_class = "warning"
    elif "danger" in bg_color:
        icon_class = "danger"
    elif "success" in bg_color:
        icon_class = "success"
    else:
        icon_class = "primary"

    return ui.div(
        ui.div(
            fa.icon_svg(icon, width="22px"),
            class_=f"info-box-icon {icon_class}"
        ),
        ui.div(
            ui.div(title, class_="info-box-label"),
            ui.div(value, class_="info-box-value"),
        ),
        class_="info-box"
    )

def tooltip_wrapper(element, text: str):
    """Wraps an element in a tooltip."""
    return ui.tooltip(element, text)
