from shiny import ui
import faicons as fa

def card_header(title: str, icon: str = None):
    """
    Creates a card header with an optional icon.
    """
    if icon:
        return ui.h5(fa.icon_svg(icon), " ", title, class_="card-title")
    return ui.h5(title, class_="card-title")

def info_box(title: str, value: str, icon: str, bg_color: str = "bg-light"):
    """
    Creates a small info box.
    """
    return ui.div(
        ui.div(
            ui.span(fa.icon_svg(icon), class_="fa-2x opacity-50"),
            class_="d-flex align-items-center justify-content-center p-3"
        ),
        ui.div(
            ui.h6(title, class_="text-muted mb-0"),
            ui.h4(value, class_="fw-bold mb-0"),
            class_="p-3"
        ),
        class_=f"d-flex flex-row align-items-center bg-white rounded shadow-sm border {bg_color} mb-3"
    )

def tooltip_wrapper(element, text: str):
    """
    Wraps an element in a tooltip.
    """
    return ui.tooltip(element, text)
