##### IMPORTS

import colorsys
from math import ceil, floor, hypot, log10
from pathlib import Path
from statistics import NormalDist
import re
import textwrap
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap, to_hex, to_rgb
from matplotlib.font_manager import FontProperties
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter
from scipy.stats import shapiro, spearmanr


def _save_figure_to_outputs(fig, save_path, transparent="F"):
    """Save a figure to the outputs folder when save_path is provided.

    transparent="T" saves the PNG with no background.
    """
    if save_path is None:
        return None
    if transparent not in {"T", "F"}:
        raise ValueError("transparent must be 'T' or 'F'.")

    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)

    output_name = Path(save_path).name
    output_path = output_dir / output_name

    if output_path.suffix.lower() != ".png":
        output_path = output_path.with_suffix(".png")

    clear = transparent == "T"
    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.25,
        facecolor="none" if clear else "white",
        edgecolor="none",
        transparent=clear,
    )

    return output_path


def _register_roboto_slab_bold():
    """The installed Roboto Slab file is variable, and Matplotlib records it as weight 400.
    A static bold instance lets fontweight='bold' use weight 700.
    """
    from matplotlib import font_manager as fm
    from fontTools.ttLib import TTFont
    from fontTools.varLib.instancer import instantiateVariableFont

    if any(font.name == "Roboto Slab" and font.weight >= 700 for font in fm.fontManager.ttflist):
        return
    source = next(
        (Path(font.fname) for font in fm.fontManager.ttflist if font.name == "Roboto Slab"),
        None,
    )
    if source is None:
        return
    bold_path = Path(__file__).resolve().parent / "fonts" / "RobotoSlab-Bold.ttf"
    if not bold_path.exists():
        bold_path.parent.mkdir(exist_ok=True)
        instantiateVariableFont(TTFont(source), {"wght": 700}, inplace=False).save(bold_path)
    fm.fontManager.addfont(bold_path)


_register_roboto_slab_bold()


def _use_currency_thousands(max_pounds):
    """GrantNav-sized totals (< £10m) plot in thousands; larger ones in millions."""
    return max_pounds < 10_000_000


def _wrap_label_two_lines(label):
    """Split a long label onto two compact lines for chart axes."""
    text = str(label)
    preferred_wraps = {
        "Number of unique total learners in scope": (
            "Unique learners\nin scope"
        ),
        "Total hours of enrichment provision": (
            "Total enrichment\nhours"
        ),
        "Average number of enrichment hours per learner": (
            "Avg hours\nper learner"
        ),
        "Nuneaton & Bedworth Borough Council": (
            "Nuneaton & Bedworth\nBorough Council"
        ),
        "Bath and North East Somerset": (
            "Bath and North\nEast Somerset"
        ),
    }
    if text in preferred_wraps:
        return preferred_wraps[text]

    words = text.split()
    if len(words) <= 1:
        return text

    midpoint = len(text) / 2
    best_index = 1
    best_distance = float("inf")
    for index in range(1, len(words)):
        left = " ".join(words[:index])
        distance = abs(len(left) - midpoint)
        if distance < best_distance:
            best_distance = distance
            best_index = index

    return " ".join(words[:best_index]) + "\n" + " ".join(words[best_index:])


def _format_region_axis_label(label, wrap_at=30):
    """Wrap long region labels onto two lines for bar-chart axes."""
    text = str(label)
    preferred_wraps = {
        "Nuneaton & Bedworth Borough Council",
    }
    if text in preferred_wraps or len(text) >= wrap_at:
        return _wrap_label_two_lines(text)
    return text


def _normalize_filter_values(values):
    """Return None or a list of filter values."""
    if values is None:
        return None
    if isinstance(values, str):
        return [values]
    return list(values)


def _apply_column_value_filter(data, column, values):
    """Keep rows where column is in values. No-op when values is None."""
    values = _normalize_filter_values(values)
    if values is None:
        return data.copy()

    if column not in data.columns:
        raise ValueError(f"Missing required column: {column}")

    filtered = data.loc[data[column].isin(values)].copy()

    if filtered.empty:
        raise ValueError(
            f"No rows found for {column} in: "
            f"{', '.join(sorted(str(value) for value in values))}"
        )

    missing_values = set(values) - set(filtered[column])
    if missing_values:
        raise ValueError(
            f"These {column} values were not found: "
            f"{', '.join(sorted(str(value) for value in missing_values))}"
        )

    return filtered


def _apply_geography_filter(data, regional_breakdown=None, region_name=None):
    """Optionally filter by regional_breakdown and/or region_name columns."""
    filtered = data.copy()
    filtered = _apply_column_value_filter(
        filtered, "regional_breakdown", regional_breakdown
    )
    filtered = _apply_column_value_filter(filtered, "region_name", region_name)
    return filtered


_NOTE_LOC = {
    ("left", "top"): "upper left",
    ("right", "top"): "upper right",
    ("left", "bottom"): "lower left",
    ("right", "bottom"): "lower right",
    ("left", "center"): "center left",
    ("right", "center"): "center right",
    ("center", "top"): "upper center",
    ("center", "bottom"): "lower center",
}


def _place_note_away_from_land(ax, geoms, note_width=0.24, note_height=0.12):
    """Pick an axes-coordinate box that overlaps England the least."""
    from shapely.geometry import box

    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()
    x_span = x_max - x_min
    y_span = y_max - y_min
    if x_span == 0 or y_span == 0 or geoms is None or len(geoms) == 0:
        return 0.03, 0.03, "lower left"

    try:
        land = geoms.union_all()
    except AttributeError:
        land = geoms.unary_union

    inset = 0.03
    candidates = [
        (inset, 1 - inset, "left", "top",
         inset, inset + note_width, 1 - inset - note_height, 1 - inset),
        (1 - inset, 1 - inset, "right", "top",
         1 - inset - note_width, 1 - inset, 1 - inset - note_height, 1 - inset),
        (inset, inset, "left", "bottom",
         inset, inset + note_width, inset, inset + note_height),
        (1 - inset, inset, "right", "bottom",
         1 - inset - note_width, 1 - inset, inset, inset + note_height),
        (inset, 0.50, "left", "center",
         inset, inset + note_width, 0.50 - note_height / 2, 0.50 + note_height / 2),
        (1 - inset, 0.50, "right", "center",
         1 - inset - note_width, 1 - inset, 0.50 - note_height / 2, 0.50 + note_height / 2),
        (0.50, 1 - inset, "center", "top",
         0.50 - note_width / 2, 0.50 + note_width / 2, 1 - inset - note_height, 1 - inset),
        (0.50, inset, "center", "bottom",
         0.50 - note_width / 2, 0.50 + note_width / 2, inset, inset + note_height),
    ]

    best = None
    for axes_x, axes_y, ha, va, left, right, bottom, top in candidates:
        data_box = box(
            x_min + left * x_span,
            y_min + bottom * y_span,
            x_min + right * x_span,
            y_min + top * y_span,
        )
        overlap = data_box.intersection(land).area
        if best is None or overlap < best[0]:
            best = (overlap, axes_x, axes_y, ha, va)
            if overlap == 0:
                break

    _, axes_x, axes_y, ha, va = best
    return axes_x, axes_y, _NOTE_LOC[(ha, va)]


def _add_map_note(ax, handles, geoms, background_color, text_color):
    """Draw a compact legend note in the emptiest sea space."""
    n_items = max(len(handles), 1)
    axes_x, axes_y, loc = _place_note_away_from_land(
        ax,
        geoms,
        note_width=0.22,
        note_height=0.055 * n_items + 0.04,
    )
    ax.legend(
        handles=handles,
        loc=loc,
        bbox_to_anchor=(axes_x, axes_y),
        bbox_transform=ax.transAxes,
        frameon=True,
        fancybox=True,
        framealpha=1,
        fontsize=10,
        labelcolor=text_color,
        borderpad=0.6,
        handlelength=1.2,
        handletextpad=0.6,
        edgecolor="#cccccc",
        facecolor=background_color,
    )


##### FUNCTION TO CALCULATE THE SAMPLE SIZE FOR A SURVEY 

def calculate_survey_sample_size(
    confidence_level=0.95,
    prevalence=0.50,
    population_size=24453,
    margin_of_error=0.05,
    anticipated_response_rate=0.50,
    cluster_sampling=False,
    design_effect=1.0,
):
    """Calculate sample size for a cross-sectional survey estimating a proportion.

    Defaults match the school survey assumptions:
    95% confidence, 50% prevalence, 24,453 schools, 5% margin of error,
    no cluster sampling, and 50% anticipated response rate.
    """
    if not 0 < confidence_level < 1:
        raise ValueError("confidence_level must be between 0 and 1.")
    if not 0 < prevalence < 1:
        raise ValueError("prevalence must be between 0 and 1.")
    if population_size <= 0:
        raise ValueError("population_size must be greater than 0.")
    if not 0 < margin_of_error < 1:
        raise ValueError("margin_of_error must be between 0 and 1.")
    if not 0 < anticipated_response_rate <= 1:
        raise ValueError("anticipated_response_rate must be between 0 and 1.")
    if design_effect < 1:
        raise ValueError("design_effect must be at least 1.")

    z_score = NormalDist().inv_cdf(1 - (1 - confidence_level) / 2)
    initial_sample_size = (
        z_score**2 * prevalence * (1 - prevalence) / margin_of_error**2
    )

    adjusted_sample_size = initial_sample_size
    if cluster_sampling:
        adjusted_sample_size *= design_effect

    finite_population_sample_size = (
        adjusted_sample_size
        / (1 + ((adjusted_sample_size - 1) / population_size))
    )

    completed_responses_required = ceil(finite_population_sample_size)
    schools_to_invite = ceil(
        completed_responses_required / anticipated_response_rate
    )

    return {
        "study_design": "Cross-sectional survey estimating a percentage/prevalence",
        "confidence_level": confidence_level,
        "z_score": z_score,
        "prevalence": prevalence,
        "population_size": population_size,
        "margin_of_error": margin_of_error,
        "cluster_sampling": cluster_sampling,
        "design_effect": design_effect if cluster_sampling else 1.0,
        "anticipated_response_rate": anticipated_response_rate,
        "completed_responses_required": completed_responses_required,
        "schools_to_invite": schools_to_invite,
    }

##### FUNCTION TO PLOT A HORIZONTAL BAR CHART FOR A SELECTED INDICATOR BY REGION


def _foi_question_split(data, scopes):
    """Q1, Q2, and upper-bound expenditure per local authority for one scope."""
    rows = data[
        data["unit"].astype(str).str.strip().eq("expenditure")
        & data["question"].isin(("Q1", "Q2"))
        & data["scope"].astype(str).str.strip().str.lower().isin(scopes)
    ].copy()
    rows["_value"] = pd.to_numeric(
        rows["value"].astype(str).str.strip().replace({"": pd.NA}),
        errors="coerce",
    )
    rows = rows.dropna(subset=["_value"])
    by_question = rows.pivot_table(
        index="LA", columns="question", values="_value", aggfunc="sum", fill_value=0
    )
    for question in ("Q1", "Q2"):
        if question not in by_question.columns:
            by_question[question] = 0.0
    upper_bound = rows["upper_bound"].astype(str).str.strip().str.upper().isin(["Y", "YES"])
    upper = rows.loc[upper_bound].groupby("LA")["_value"].sum()
    frame = pd.DataFrame({
        "q1": by_question["Q1"],
        "q2": by_question["Q2"],
        "upper": upper,
    }).fillna(0)
    frame["total"] = frame["q1"] + frame["q2"]
    return frame.loc[frame["total"] > 0]


def _plot_foi_question_stack(
    data, indicator, top_n, ascending, currency_unit, ax, title, save_path,
    transparent="F", brand_font="ON",
):
    """Stacked horizontal bars of Q1 and Q2 for V14 or V17."""
    scopes, scope_name = ({"in"}, "in scope") if indicator == "V14" else ({"out"}, "out of scope")
    frame = _foi_question_split(data, scopes)
    if frame.empty:
        raise ValueError(f"No local-authority totals found for {indicator}.")
    if top_n is None or top_n <= 0:
        raise ValueError("top_n must be greater than 0.")
    frame = frame.sort_values("total", ascending=ascending).head(top_n).iloc[::-1]
    max_total = float(frame["total"].max())
    if currency_unit is None:
        use_thousands = _use_currency_thousands(max_total)
    elif currency_unit == "thousands":
        use_thousands = True
    elif currency_unit == "millions":
        use_thousands = False
    else:
        raise ValueError("currency_unit must be 'thousands' or 'millions'.")

    def money(amount):
        if use_thousands:
            return f"£{amount / 1_000:,.0f}k"
        shown = amount / 1_000_000
        return f"£{shown:,.1f}M" if shown < 10 else f"£{shown:,.0f}M"

    if brand_font not in {"ON", "OFF"}:
        raise ValueError("brand_font must be ON or OFF.")
    plt.rcParams["font.family"] = (
        ["Roboto Slab", "serif"] if brand_font == "ON" else ["Arial", "sans-serif"]
    )
    q1_color, q2_color = "#FF4D5E", "#B7D6EA"
    ink = "#222222"
    if ax is None:
        _, ax = plt.subplots(figsize=(13.5, max(4.8, len(frame) * 0.55)))
    ax.figure.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#ffffff")
    positions = np.arange(len(frame))
    ax.barh(positions, frame["q1"], color=q1_color, height=0.68)
    ax.barh(positions, frame["q2"], left=frame["q1"], color=q2_color, height=0.68)
    ax.set_yticks(positions)
    ax.set_yticklabels([_format_region_axis_label(name) for name in frame.index])
    ax.set_xlim(0, max_total * 2.05)
    for position, row in enumerate(frame.itertuples(index=False)):
        outside = []
        for key, amount, start, colour in (
            ("Q1", row.q1, 0, "#ffffff"),
            ("Q2", row.q2, row.q1, "#000000"),
        ):
            if amount <= 0:
                continue
            share = 100 * amount / row.total
            text = f"{key}: {money(amount)} ({share:.0f}%)"
            if amount >= max_total * 0.16:
                ax.text(
                    start + amount / 2,
                    position,
                    text,
                    ha="center",
                    va="center",
                    color=colour,
                    fontsize=8,
                    fontweight="bold",
                )
            else:
                outside.append(text)
        end = money(row.total)
        if outside:
            end = " · ".join(outside) + "  " + end
        ax.text(
            row.total + max_total * 0.015,
            position,
            end,
            ha="left",
            va="center",
            color=ink,
            fontsize=9,
            clip_on=False,
        )
    chart_title = title or f"Top {top_n} local authorities by {scope_name} expenditure"
    ax.set_title(chart_title, color=ink, fontsize=14, fontweight="bold", pad=14)
    ax.set_xlabel(
        f"Expenditure (£) in {'thousands' if use_thousands else 'millions'}",
        color=ink,
        fontsize=11,
    )
    ax.set_ylabel("")
    ax.xaxis.set_major_formatter(
        FuncFormatter(
            lambda value, _: money(value) if not use_thousands else f"£{value / 1_000:,.0f}"
        )
    )
    ax.legend(
        handles=[
            Patch(facecolor=q1_color, label="Q1"),
            Patch(facecolor=q2_color, label="Q2"),
        ],
        frameon=False,
        loc="lower right",
    )
    ax.tick_params(axis="both", colors=ink, labelsize=10)
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#000000")
    ax.spines["bottom"].set_color("#000000")
    ax.set_axisbelow(True)
    _save_figure_to_outputs(ax.figure, save_path, transparent)
    return ax


def _lighter_shade(color):
    """One step lighter than color, same hue, mixed toward white."""
    red, green, blue = to_rgb(color)
    hue, light, sat = colorsys.rgb_to_hls(red, green, blue)
    light = light + (1 - light) * 0.55
    return to_hex(colorsys.hls_to_rgb(hue, light, sat))


def _brand_color(layout_branding):
    """'default' keeps the current red. Any other value is a colour code."""
    if layout_branding == "default":
        return "#d93649", "#f6b8c0"
    text = str(layout_branding).strip()
    if len(text) in {6, 8} and all(c in "0123456789abcdefABCDEF" for c in text):
        text = f"#{text}"
    try:
        color = to_hex(to_rgb(text))
    except ValueError:
        raise ValueError(
            "layout_branding must be 'default' or a colour code."
        ) from None
    return color, _lighter_shade(color)


def plot_indicator_horizontal_bar_chart(
    data,
    indicator,
    rank_on=True,
    top_n=None,
    ascending=False,
    region_col="region_name",
    indicator_col="indicator",
    value_col="value",
    regional_breakdown=None,
    region_name=None,
    source=None,
    source_col="source",
    ax=None,
    title=None,
    special_style=False,
    bars_styled=None,
    save_path=None,
    currency_unit=None,
    transparent="F",
    brand_font="ON",
    layout_branding="default",
):
    """Stacked horizontal bars for V14 or V17.

    data is the long FOI extract. Each bar is one local authority.
    V14 is expenditure in scope. V17 is expenditure out of scope.
    The bar is split into Q1 and Q2. A segment wide enough to read is
    labelled with its amount and its share of that authority's total.
    The figure at the end of the bar is the authority total. A Q1 or Q2
    label that does not fit inside its segment is written there too.

    rank_on=True with top_n keeps that many authorities. ascending=False
    puts the highest total at the top. currency_unit is "thousands" or
    "millions". Leave it unset and a largest bar under £10 million is
    drawn in thousands. Pass save_path to save a PNG in outputs.
    transparent="T" saves that PNG with no background.
    brand_font ON uses Roboto Slab. OFF uses Arial. On the GrantNav
    chart, layout_branding "default" keeps the current reds. Any other
    colour code is used for the bars. With special_style, the highlighted
    bars use that colour and the rest use one lighter shade of it.
    """
    indicator_scales = {
        "Total amount awarded": "currency_millions",
        "Total amount awarded per head": "currency",
        "Funding per child aged 5 to 20": "currency",
        "Funding per child aged 5 to 20 (excluding largest grant)": "currency",
        "% of creative and cultural employees": "percentage",
        "% of the population engaged with the arts online in the last 12 months": "percentage",
        "% of the population engaged with the arts in person in the last 12 months": "percentage",
        "Number of unique total learners in scope": "number",
        "Number of total enrichment activities": "number",
        "Average number of enrichment activities per learner": "number_decimal",
        "Total hours of enrichment provision": "number_decimal",
        "Average number of enrichment hours per learner": "number_decimal",
    }

    if indicator in {"V14", "V17"}:
        return _plot_foi_question_stack(
            data, indicator, top_n, ascending, currency_unit, ax, title, save_path,
            transparent, brand_font,
        )
    if indicator in _FOI_GROSS_INDICATORS:
        raise ValueError("This chart plots V14 or V17.")

    required_cols = {region_col, indicator_col, value_col}
    if source is not None:
        required_cols.add(source_col)
    missing_cols = required_cols - set(data.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing_cols))}")

    plot_data = _apply_geography_filter(
        data,
        regional_breakdown=regional_breakdown,
        region_name=region_name,
    )

    plot_data = plot_data.loc[plot_data[indicator_col] == indicator].copy()

    if plot_data.empty:
        raise ValueError(f"No rows found for indicator: {indicator}")

    if source is not None:
        if isinstance(source, str):
            source = [source]

        plot_data = plot_data.loc[plot_data[source_col].isin(source)]

        if plot_data.empty:
            raise ValueError(
                f"No rows found for indicator '{indicator}' with source in: "
                f"{', '.join(sorted(source))}"
            )

        missing_sources = set(source) - set(plot_data[source_col])
        if missing_sources:
            raise ValueError(
                "These sources were not found for this indicator: "
                f"{', '.join(sorted(missing_sources))}"
            )
    elif source_col in plot_data.columns:
        sources_present = sorted(plot_data[source_col].dropna().unique())
        if len(sources_present) > 1:
            raise ValueError(
                f"Indicator '{indicator}' appears in multiple sources: "
                f"{', '.join(sources_present)}. "
                "Pass source=... to choose which source to plot."
            )

    plot_data[value_col] = plot_data[value_col].astype(float)

    if rank_on:
        if top_n is None:
            raise ValueError("top_n must be given when rank_on=True.")
        if top_n <= 0:
            raise ValueError("top_n must be greater than 0.")

        plot_data = (
            plot_data.sort_values(value_col, ascending=ascending)
            .head(top_n)
        )
        ranked_unit = "regions"
        title_prefix = f"{top_n} {'lowest' if ascending else 'highest'} {ranked_unit}"
    else:
        if region_name is None:
            raise ValueError("region_name must be given when rank_on=False.")

        plot_data = plot_data.sort_values(value_col, ascending=ascending)
        title_prefix = "Selected regions"

    scale_type = indicator_scales.get(indicator, "number")
    amount_label = "Total awarded"
    chart_title = title if title is not None else f"{title_prefix}: {indicator}"

    if scale_type == "percentage" and plot_data[value_col].abs().max() <= 1:
        plot_data[value_col] = plot_data[value_col] * 100

    plot_data = plot_data.iloc[::-1]

    if brand_font not in {"ON", "OFF"}:
        raise ValueError("brand_font must be ON or OFF.")
    plt.rcParams["font.family"] = (
        ["Roboto Slab", "serif"] if brand_font == "ON" else ["Arial", "sans-serif"]
    )

    bar_color, light_bar_color = _brand_color(layout_branding)
    background_color = "#ffffff"
    text_color = "#222222"
    axis_color = "#000000"

    if special_style:
        if bars_styled is None:
            bars_styled = [1, 2, 3]
        elif isinstance(bars_styled, int):
            bars_styled = list(range(1, bars_styled + 1))

        invalid_bars = [
            bar_position
            for bar_position in bars_styled
            if bar_position not in {1, 2, 3}
        ]
        if invalid_bars:
            raise ValueError("bars_styled can only include 1, 2, and/or 3.")

        highlighted_rows = {
            len(plot_data) - bar_position
            for bar_position in bars_styled
            if bar_position <= len(plot_data)
        }
        bar_colors = [
            bar_color if row_number in highlighted_rows else light_bar_color
            for row_number in range(len(plot_data))
        ]
    else:
        bar_colors = bar_color

    if ax is None:
        _, ax = plt.subplots(figsize=(8, max(4, len(plot_data) * 0.45)))

    ax.figure.patch.set_facecolor(background_color)
    ax.set_facecolor(background_color)

    region_labels = [
        _format_region_axis_label(region_name)
        for region_name in plot_data[region_col]
    ]

    bars = ax.barh(
        region_labels,
        plot_data[value_col],
        height=0.65,
        color=bar_colors,
        edgecolor=bar_colors,
    )

    max_value = plot_data[value_col].max()
    if currency_unit is None:
        use_thousands = _use_currency_thousands(max_value)
    elif currency_unit == "thousands":
        use_thousands = True
    elif currency_unit == "millions":
        use_thousands = False
    else:
        raise ValueError("currency_unit must be 'thousands' or 'millions'.")
    if max_value > 0:
        # Extra x-axis headroom so bars look shorter in the frame (better for slides).
        ax.set_xlim(right=max_value * 1.45)

    label_offset = max_value * 0.02 if max_value > 0 else 0

    for bar in bars:
        value = bar.get_width()

        if scale_type == "currency_millions":
            if use_thousands:
                value_label = f"£{value / 1_000:,.0f}k"
            elif max_value < 10_000_000:
                value_label = f"£{value / 1_000_000:,.1f}M"
            else:
                value_label = f"£{value / 1_000_000:,.0f}M"
        elif scale_type == "currency":
            value_label = f"£{value:,.2f}"
        elif scale_type == "percentage":
            value_label = f"{value:.0f}%"
        elif scale_type == "number_decimal":
            value_label = f"{value:,.2f}"
        else:
            value_label = f"{value:,.0f}"

        ax.text(
            value + label_offset,
            bar.get_y() + bar.get_height() / 2,
            value_label,
            va="center",
            ha="left",
            color=text_color,
            fontsize=10,
        )

    ax.set_title(
        chart_title,
        color=text_color,
        fontsize=14,
        fontweight="bold",
        pad=14,
    )
    ax.set_ylabel("")

    if scale_type == "percentage":
        ax.set_xlabel("Percentage (%)", color=text_color, fontsize=11)
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.0f}%"))
    elif scale_type == "currency_millions":
        if use_thousands:
            ax.set_xlabel(
                f"{amount_label} (£) in thousands",
                color=text_color,
                fontsize=11,
            )
            ax.xaxis.set_major_formatter(
                FuncFormatter(lambda x, _: f"£{x / 1_000:,.0f}")
            )
        else:
            ax.set_xlabel(
                f"{amount_label} (£) in millions",
                color=text_color,
                fontsize=11,
            )
            ax.xaxis.set_major_formatter(
                FuncFormatter(
                    lambda x, _: (
                        "£0"
                        if abs(x) < 1e-6
                        else f"£{x / 1_000_000:,.0f}"
                        if abs(x / 1_000_000 - round(x / 1_000_000)) < 1e-9
                        else f"£{x / 1_000_000:,.1f}"
                    )
                )
            )
    elif scale_type == "currency":
        ax.set_xlabel("Value (£)", color=text_color, fontsize=11)
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"£{x:,.2f}"))
    elif scale_type == "number_decimal":
        ax.set_xlabel("Value", color=text_color, fontsize=11)
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:,.2f}"))
    else:
        ax.set_xlabel("Value", color=text_color, fontsize=11)
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:,.0f}"))

    ax.tick_params(axis="both", colors=text_color, labelsize=10)
    for label in ax.get_yticklabels():
        label.set_linespacing(1.15)
    ax.grid(False)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(axis_color)
    ax.spines["bottom"].set_color(axis_color)

    ax.set_axisbelow(True)

    _save_figure_to_outputs(ax.figure, save_path, transparent)

    return ax


##### FUNCTION TO PLOT A HISTOGRAM AND PRINT DESCRIPTIVE STATISTICS


def plot_indicator_value_distribution(
    data,
    indicator,
    bins=10,
    indicator_col="indicator",
    value_col="value",
    regional_breakdown=None,
    region_name=None,
    source=None,
    source_col="source",
    ax=None,
    title=None,
    save_path=None,
    currency_unit=None,
):
    """Histogram of one indicator.

    The FOI extract (columns LA, question, unit) plots V9-V17: one
    gross-expenditure total per local authority. Authorities with no figure
    are left out. all_data_final.csv keeps regional_breakdown, region_name,
    and source. Descriptive statistics are printed, not drawn.
    currency_unit is "thousands" or "millions" for currency totals. Leave it
    unset and a largest value under £10 million is drawn in thousands.
    """
    indicator_scales = {
        "Total amount awarded": "currency_millions",
        "Number of grants": "number",
        "Effective number of grants (ENG)": "number_decimal",
        "Total amount awarded per head": "currency",
        "Funding per child aged 5 to 20": "currency",
        "Funding per child aged 5 to 20 (excluding largest grant)": "currency",
        "% of creative and cultural employees": "percentage",
        "% of the population engaged with the arts online in the last 12 months": "percentage",
        "% of the population engaged with the arts in person in the last 12 months": "percentage",
        "Number of unique total learners in scope": "number",
        "Number of total enrichment activities": "number",
        "Average number of enrichment activities per learner": "number_decimal",
        "Total hours of enrichment provision": "number_decimal",
        "Average number of enrichment hours per learner": "number_decimal",
    }
    foi_extract = {"LA", "question", "unit"}.issubset(data.columns)
    if foi_extract:
        values, label = _foi_gross_by_la(data, indicator)
        indicator = f"{indicator}. {label}"
        scale_type = "currency_millions"
    else:
        required_cols = {indicator_col, value_col}
        if source is not None:
            required_cols.add(source_col)
        missing_cols = required_cols - set(data.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns: {', '.join(sorted(missing_cols))}")

        plot_data = _apply_geography_filter(
            data,
            regional_breakdown=regional_breakdown,
            region_name=region_name,
        )
        plot_data = plot_data.loc[plot_data[indicator_col] == indicator].copy()
        if plot_data.empty:
            raise ValueError(f"No rows found for indicator: {indicator}")

        if source is not None:
            if isinstance(source, str):
                source = [source]
            plot_data = plot_data.loc[plot_data[source_col].isin(source)]
            if plot_data.empty:
                raise ValueError(
                    f"No rows found for indicator '{indicator}' with source in: "
                    f"{', '.join(sorted(source))}"
                )
            missing_sources = set(source) - set(plot_data[source_col])
            if missing_sources:
                raise ValueError(
                    "These sources were not found for this indicator: "
                    f"{', '.join(sorted(missing_sources))}"
                )
        elif source_col in plot_data.columns:
            sources_present = sorted(plot_data[source_col].dropna().unique())
            if len(sources_present) > 1:
                raise ValueError(
                    f"Indicator '{indicator}' appears in multiple sources: "
                    f"{', '.join(sources_present)}. "
                    "Pass source=... to choose which source to plot."
                )

        values = plot_data[value_col].astype(float).dropna()
        if values.empty:
            raise ValueError(f"No numeric values found for indicator: {indicator}")
        scale_type = indicator_scales.get(indicator, "number")

    if scale_type == "currency_millions":
        if currency_unit is None:
            use_thousands = _use_currency_thousands(values.max())
        elif currency_unit == "thousands":
            use_thousands = True
        elif currency_unit == "millions":
            use_thousands = False
        else:
            raise ValueError("currency_unit must be 'thousands' or 'millions'.")
        if use_thousands:
            values_to_plot = values / 1_000
            x_label = "Value (£) in thousands"
            unit_label = "£ in thousands"
            x_formatter = FuncFormatter(lambda x, _: f"£{x:,.0f}")
        else:
            values_to_plot = values / 1_000_000
            x_label = "Value (£) in millions"
            unit_label = "£ in millions"
            # Whole millions stay as £1; half-million ticks must stay £0.5.
            # £{x:,.0f} rounded both 0 and 0.5 to £0.
            x_formatter = FuncFormatter(
                lambda x, _: (
                    "£0"
                    if abs(x) < 1e-12
                    else f"£{x:,.0f}"
                    if abs(x - round(x)) < 1e-9
                    else f"£{x:,.1f}"
                )
            )
    elif scale_type == "currency":
        values_to_plot = values
        x_label = "Value (£)"
        unit_label = "£"
        x_formatter = FuncFormatter(lambda x, _: f"£{x:,.2f}")
    elif scale_type == "percentage":
        values_to_plot = values * 100 if values.abs().max() <= 1 else values
        x_label = "Value (%)"
        unit_label = "%"
        x_formatter = FuncFormatter(lambda x, _: f"{x:.0f}%")
    elif scale_type == "number_decimal":
        values_to_plot = values
        x_label = "Value"
        unit_label = "decimal values"
        x_formatter = FuncFormatter(lambda x, _: f"{x:,.2f}")
    else:
        values_to_plot = values
        x_label = "Value"
        unit_label = "count"
        x_formatter = FuncFormatter(lambda x, _: f"{x:,.0f}")

    descriptive_stats = values_to_plot.describe()
    descriptive_stats.loc["median"] = values_to_plot.median()
    skewness = float(values_to_plot.skew())
    # pandas uses Fisher's definition: excess kurtosis (normal distribution = 0)
    kurtosis = float(values_to_plot.kurt())
    descriptive_stats.loc["skewness"] = skewness
    descriptive_stats.loc["kurtosis"] = kurtosis

    n = len(values_to_plot)
    if n < 3:
        shapiro_stat = float("nan")
        shapiro_p = float("nan")
        normality_verdict = (
            "Shapiro-Wilk test not run (fewer than 3 observations); "
            "insufficient data to assess normality"
        )
    elif n > 5000:
        shapiro_stat = float("nan")
        shapiro_p = float("nan")
        normality_verdict = (
            "Shapiro-Wilk test not run (more than 5,000 observations); "
            "the test is not reliable at this sample size"
        )
    else:
        shapiro_stat, shapiro_p = shapiro(values_to_plot)
        shapiro_stat = float(shapiro_stat)
        shapiro_p = float(shapiro_p)
        if shapiro_p >= 0.05:
            normality_verdict = (
                "consistent with a normal distribution at the 5% level; "
                "the Shapiro-Wilk test does not provide evidence against "
                "normality"
            )
        else:
            normality_verdict = (
                "not consistent with a normal distribution at the 5% level; "
                "the Shapiro-Wilk test suggests significant departure from "
                "normality"
            )

    descriptive_stats.loc["shapiro_w"] = shapiro_stat
    descriptive_stats.loc["shapiro_p"] = shapiro_p
    descriptive_stats = descriptive_stats.reindex(
        [
            "count",
            "mean",
            "median",
            "std",
            "min",
            "25%",
            "50%",
            "75%",
            "max",
            "skewness",
            "kurtosis",
            "shapiro_w",
            "shapiro_p",
        ]
    )

    if abs(skewness) < 0.5:
        skew_verdict = (
            "approximately symmetric (little to no skew); "
            "values are fairly evenly spread around the centre, "
            "with no strong pull from unusually high or low outliers"
        )
    elif abs(skewness) < 1:
        if skewness > 0:
            skew_verdict = (
                "moderately right-skewed (positively skewed); "
                "a longer tail on the right suggests some high/large "
                "outliers pulling the mean above the median"
            )
        else:
            skew_verdict = (
                "moderately left-skewed (negatively skewed); "
                "a longer tail on the left suggests some low/small "
                "outliers pulling the mean below the median"
            )
    else:
        if skewness > 0:
            skew_verdict = (
                "highly right-skewed (positively skewed); "
                "a long tail on the right suggests high/large "
                "outliers pulling the mean above the median"
            )
        else:
            skew_verdict = (
                "highly left-skewed (negatively skewed); "
                "a long tail on the left suggests low/small "
                "outliers pulling the mean below the median"
            )

    if abs(kurtosis) < 0.5:
        kurt_verdict = (
            "approximately mesokurtic (similar to a normal distribution); "
            "tails and extreme values are about as heavy as in a normal curve. "
            "Note: kurtosis measures tailedness/extremes, not how tightly "
            "values concentrate around the mean (that is closer to the standard deviation)"
        )
    elif kurtosis >= 0.5:
        intensity = "moderately" if kurtosis < 1 else "highly"
        kurt_verdict = (
            f"{intensity} leptokurtic (positive excess kurtosis); "
            "heavier tails than a normal distribution, so extreme outliers "
            "are more common."
        )
    else:
        intensity = "moderately" if kurtosis > -1 else "highly"
        kurt_verdict = (
            f"{intensity} platykurtic (negative excess kurtosis); "
            "lighter tails than a normal distribution, so extreme outliers "
            "are less common and the distribution is flatter. Note: this is "
            "about tailedness/extremes, not how tightly values concentrate "
            "around the mean"
        )

    print(f"Descriptive statistics for: {indicator}")
    print(f"Unit: {unit_label}")
    print(descriptive_stats.to_string(float_format=lambda x: f"{x:,.2f}"))
    print(
        f"Skewness verdict: {skew_verdict} "
        f"(skewness = {skewness:,.2f})."
    )
    print(
        f"Kurtosis verdict: {kurt_verdict} "
        f"(excess kurtosis = {kurtosis:,.2f})."
    )
    if pd.notna(shapiro_p):
        print(
            f"Normality verdict (Shapiro-Wilk): {normality_verdict} "
            f"(W = {shapiro_stat:,.4f}, p = {shapiro_p:,.4f})."
        )
    else:
        print(f"Normality verdict (Shapiro-Wilk): {normality_verdict}.")

    plt.rcParams["font.family"] = ["Roboto Slab", "serif"]

    bar_color = "#d93649"
    line_color = "#000000"
    background_color = "#ffffff"
    text_color = "#222222"
    axis_color = "#000000"
    chart_title = title if title is not None else f"Distribution of {indicator}"

    if ax is None:
        _, ax = plt.subplots(figsize=(10, 5))

    ax.figure.patch.set_facecolor(background_color)
    ax.set_facecolor(background_color)

    sns.histplot(
        x=values_to_plot,
        bins=bins,
        kde=False,
        stat="density",
        color=bar_color,
        alpha=0.7,
        edgecolor=background_color,
        ax=ax,
    )
    sns.kdeplot(
        x=values_to_plot,
        color=line_color,
        clip=(values_to_plot.min(), values_to_plot.max()),
        cut=0,
        linewidth=2.5,
        ax=ax,
    )

    if values_to_plot.min() >= 0:
        ax.set_xlim(left=0)

    ax.set_title(
        chart_title,
        color=text_color,
        fontsize=14,
        fontweight="bold",
        pad=14,
    )
    ax.set_xlabel(x_label, color=text_color, fontsize=11)

    if x_formatter is not None:
        ax.xaxis.set_major_formatter(x_formatter)

    # Keep true density on the axes, but show plain tick labels by scaling.
    # Tiny densities (common with large x-values) otherwise trigger Matplotlib's
    # floating scientific-notation offset above the chart.
    y_max = ax.get_ylim()[1]
    if y_max > 0 and y_max < 0.01:
        density_scale = 10 ** (-floor(log10(y_max)))
        ax.set_ylabel(
            f"Density (×{density_scale:,.0f})",
            color=text_color,
            fontsize=11,
        )
        ax.yaxis.set_major_formatter(
            FuncFormatter(
                lambda y, _, scale=density_scale: (
                    "0" if abs(y) < 1e-18 else f"{y * scale:g}"
                )
            )
        )
    else:
        ax.set_ylabel("Density", color=text_color, fontsize=11)
        ax.yaxis.set_major_formatter(
            FuncFormatter(
                lambda y, _: "0" if abs(y) < 1e-18 else f"{y:g}"
            )
        )

    ax.yaxis.get_offset_text().set_visible(False)
    ax.xaxis.get_offset_text().set_visible(False)

    ax.tick_params(axis="both", colors=text_color, labelsize=10)
    ax.grid(False)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(axis_color)
    ax.spines["bottom"].set_color(axis_color)

    ax.set_axisbelow(True)

    _save_figure_to_outputs(ax.figure, save_path)

    return ax, descriptive_stats


##### FUNCTION TO PLOT A SCATTERPLOT BETWEEN TWO INDICATORS


def _format_scatter_label_value(value, scale, thousands=False):
    """Compact value text for a scatter point label."""
    if scale == "currency_millions":
        if thousands or abs(value) < 10_000_000:
            return f"£{value / 1_000:,.0f}k"
        return f"£{value / 1_000_000:,.2f}M"
    if scale == "currency":
        return f"£{value:,.2f}"
    if scale == "percentage":
        pct = value * 100 if abs(value) <= 1 else value
        return f"{pct:.2f}%"
    return f"{value:,.2f}"


def _annotate_scatter_labels(
    ax,
    regions,
    x_series,
    y_series,
    text_color,
    plot_data,
    x_indicator,
    y_indicator,
    x_scale,
    y_scale,
    x_thousands,
    y_thousands,
):
    """Place region labels with leader lines, staggered so they do not overlap."""
    names = sorted(
        regions,
        key=lambda name: (float(x_series.loc[name]), float(y_series.loc[name])),
    )
    ax.margins(x=0.22, y=0.08)
    ax.figure.canvas.draw()
    renderer = ax.figure.canvas.get_renderer()
    offsets = [
        (14, 16),
        (-14, 16),
        (14, -18),
        (-14, -18),
        (22, 8),
        (-22, 8),
        (8, 26),
        (-8, 26),
        (30, 18),
        (-30, 18),
    ]
    placed_boxes = []
    for index, name in enumerate(names):
        x_value = float(x_series.loc[name])
        y_value = float(y_series.loc[name])
        label_text = (
            f"{name}\n"
            f"{_format_scatter_label_value(float(plot_data.loc[name, x_indicator]), x_scale, x_thousands)}, "
            f"{_format_scatter_label_value(float(plot_data.loc[name, y_indicator]), y_scale, y_thousands)}"
        )
        # Smaller x → try left first; larger x → try right first.
        ordered = offsets[1:] + offsets[:1] if index % 2 == 0 else offsets
        annotation = None
        for dx, dy in ordered:
            ha = "left" if dx >= 0 else "right"
            va = "bottom" if dy >= 0 else "top"
            annotation = ax.annotate(
                label_text,
                (x_value, y_value),
                textcoords="offset points",
                xytext=(dx, dy),
                ha=ha,
                va=va,
                fontsize=8,
                color=text_color,
                clip_on=False,
                arrowprops={
                    "arrowstyle": "-",
                    "color": text_color,
                    "lw": 0.7,
                    "shrinkA": 0,
                    "shrinkB": 4,
                },
                bbox={
                    "facecolor": "white",
                    "edgecolor": "none",
                    "pad": 0.5,
                    "alpha": 0.9,
                },
            )
            ax.figure.canvas.draw()
            box = annotation.get_window_extent(renderer=renderer).padded(4)
            if not any(box.overlaps(other) for other in placed_boxes):
                placed_boxes.append(box)
                break
            annotation.remove()
            annotation = None
        if annotation is None:
            ax.annotate(
                label_text,
                (x_value, y_value),
                textcoords="offset points",
                xytext=(14, 16),
                ha="left",
                va="bottom",
                fontsize=8,
                color=text_color,
                clip_on=False,
            )


def plot_indicator_scatter(
    data,
    x_indicator,
    y_indicator,
    region_col="region_name",
    indicator_col="indicator",
    value_col="value",
    regional_breakdown=None,
    region_name=None,
    source=None,
    source_col="source",
    label_regions=False,
    label_outliers=None,
    label_high_outliers=False,
    high_outlier_x=1_750_000,
    high_outlier_y=0.5,
    x_label=None,
    y_label=None,
    ax=None,
    title=None,
    save_path=None,
    alpha=0.05,
):
    """Plot a scatterplot for two indicators from long-format data.

    Each point is one region. Data are pivoted so both indicators share the
    same region rows before plotting. Pass source when the same indicator
    appears in more than one source. Pass regional_breakdown and/or
    region_name to restrict geography (same names as the dataset columns).
    Pass save_path to save the chart as a PNG in the outputs folder.

    Axis labels default to readable names for known indicators. Pass x_label
    and/or y_label to override them for a specific chart.

    To label points selectively:
    - label_regions=False: no labels (default)
    - label_regions=True: label every region
    - label_regions=[...]: label only the listed region names
    - label_outliers=3: also label the top and bottom 3 regions on each axis
      (pass True to use 3). Explicit and automatic labels are combined.
    - label_high_outliers=True: label regions with x above high_outlier_x and
      y above high_outlier_y (defaults: £1,750,000 and 0.5, for awards vs IDACI).
      Labels use leader lines and are staggered to avoid overlaps.

    When y_indicator is cyp_avg_score, a dotted reference line is drawn at y=0
    (national average) with an explanatory note about negative scores.

    Returns
    -------
    ax, plot_data, spearman_rho, spearman_pvalue
    """
    indicator_scales = {
        "Total amount awarded": "currency_millions",
        "Total amount awarded per head": "currency",
        "Funding per child aged 5 to 20": "currency",
        "Funding per child aged 5 to 20 (excluding largest grant)": "currency",
        "% of creative and cultural employees": "percentage",
        "% of the population engaged with the arts online in the last 12 months": "percentage",
        "% of the population engaged with the arts in person in the last 12 months": "percentage",
        "Number of unique total learners in scope": "number",
        "Number of total enrichment activities": "number",
        "Average number of enrichment activities per learner": "number_decimal",
        "Total hours of enrichment provision": "number_decimal",
        "Average number of enrichment hours per learner": "number_decimal",
        "hours_per_young_person_5_18": "number_decimal",
        "hours_per_learner": "number_decimal",
        "total_hours": "number_decimal",
        "learners_in_scope": "number",
        "idaci_avg_score": "number_decimal",
        "imd_avg_score": "number_decimal",
        "cyp_avg_score": "number_decimal",
        "Effective number of grants (ENG)": "number_decimal",
        "Number of grants": "number",
    }
    indicator_axis_labels = {
        "hours_per_young_person_5_18": (
            "Enrichment hours per young person (aged 5-18)"
        ),
        "hours_per_learner": "Average enrichment hours per learner",
        "total_hours": "Total enrichment hours",
        "learners_in_scope": "Unique learners in scope",
        "idaci_avg_score": (
            "IDACI average score (children in income-deprived households)"
        ),
        "imd_avg_score": "IMD average score (all-age deprivation)",
        "cyp_avg_score": (
            "CYP average score (children's educational deprivation)"
        ),
        "Effective number of grants (ENG)": "Effective number of grants (ENG)",
    }

    def _resolve_axis_label(indicator, scale, override=None, thousands=False):
        if override is not None:
            if thousands:
                return override.replace("millions", "thousands")
            return override

        label = indicator_axis_labels.get(indicator, indicator)
        if scale == "currency_millions" and "£" not in label:
            unit = "thousands" if thousands else "millions"
            return f"{label} (£ in {unit})"
        if scale == "currency" and "£" not in label:
            return f"{label} (£)"
        return label
    indicators = [x_indicator, y_indicator]

    required_cols = {region_col, indicator_col, value_col}
    if source is not None:
        required_cols.add(source_col)
    missing_cols = required_cols - set(data.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing_cols))}")

    plot_data = _apply_geography_filter(
        data,
        regional_breakdown=regional_breakdown,
        region_name=region_name,
    )

    if source is not None:
        if isinstance(source, str):
            source = [source]
        plot_data = plot_data.loc[plot_data[source_col].isin(source)]
        if plot_data.empty:
            raise ValueError(
                "No rows found for source in: "
                f"{', '.join(sorted(source))}"
            )

    plot_data = plot_data.loc[plot_data[indicator_col].isin(indicators)].copy()
    if plot_data.empty:
        raise ValueError("No rows found for the selected indicators.")

    missing_indicators = [
        indicator
        for indicator in indicators
        if indicator not in set(plot_data[indicator_col])
    ]
    if missing_indicators:
        raise ValueError(
            "These indicators were not found: "
            f"{', '.join(missing_indicators)}"
        )

    if source_col in plot_data.columns:
        ambiguous = (
            plot_data.groupby(indicator_col)[source_col]
            .nunique()
            .loc[lambda counts: counts > 1]
        )
        if not ambiguous.empty:
            raise ValueError(
                "These indicators appear in multiple sources: "
                f"{', '.join(ambiguous.index)}. "
                "Pass source=... to choose which source to use."
            )

    plot_data = (
        plot_data.pivot_table(
            index=region_col,
            columns=indicator_col,
            values=value_col,
            aggfunc="mean",
        )
        .reindex(columns=indicators)
        .apply(pd.to_numeric, errors="coerce")
        .dropna(how="any")
    )

    if plot_data.empty:
        raise ValueError("No complete region rows found for both indicators.")

    if len(plot_data) < 3:
        raise ValueError(
            "Need at least 3 complete region rows to plot a scatter chart."
        )

    x_scale = indicator_scales.get(x_indicator, "number")
    y_scale = indicator_scales.get(y_indicator, "number")

    x_values = plot_data[x_indicator].copy()
    y_values = plot_data[y_indicator].copy()

    x_currency_thousands = (
        x_scale == "currency_millions"
        and _use_currency_thousands(x_values.abs().max())
    )
    y_currency_thousands = (
        y_scale == "currency_millions"
        and _use_currency_thousands(y_values.abs().max())
    )

    x_axis_label = _resolve_axis_label(
        x_indicator, x_scale, x_label, thousands=x_currency_thousands
    )
    y_axis_label = _resolve_axis_label(
        y_indicator, y_scale, y_label, thousands=y_currency_thousands
    )

    if x_scale == "percentage" and x_values.abs().max() <= 1:
        x_values = x_values * 100
    if y_scale == "percentage" and y_values.abs().max() <= 1:
        y_values = y_values * 100
    if x_scale == "currency_millions":
        x_values = x_values / (1_000 if x_currency_thousands else 1_000_000)
    if y_scale == "currency_millions":
        y_values = y_values / (1_000 if y_currency_thousands else 1_000_000)

    spearman_rho, spearman_pvalue = spearmanr(x_values, y_values)

    def _format_p_value(p_value):
        if p_value < 0.001:
            return "p<0.001"
        return f"p={p_value:.3f}"

    if spearman_pvalue < alpha:
        correlation_verdict = (
            f"statistically significant at {alpha:.0%} "
            "(unlikely to be due to chance alone)"
        )
    else:
        correlation_verdict = (
            f"not statistically significant at {alpha:.0%} "
            "(could plausibly be due to chance)"
        )

    print(f"Scatterplot: {x_indicator} vs {y_indicator}")
    print(f"Regions included (complete cases): {len(plot_data)}")
    print(
        f"Spearman ρ = {spearman_rho:.3f}, "
        f"{_format_p_value(spearman_pvalue)} → {correlation_verdict}"
    )

    plt.rcParams["font.family"] = ["Roboto Slab", "serif"]

    point_color = "#d93649"
    background_color = "#ffffff"
    text_color = "#222222"
    axis_color = "#000000"
    chart_title = (
        title if title is not None else f"{y_indicator} vs {x_indicator}"
    )

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 6))

    ax.figure.patch.set_facecolor(background_color)
    ax.set_facecolor(background_color)

    ax.scatter(
        x_values,
        y_values,
        color=point_color,
        alpha=0.85,
        edgecolor=background_color,
        linewidth=0.8,
        s=55,
        zorder=3,
    )

    x_series = pd.Series(x_values.values, index=plot_data.index)
    y_series = pd.Series(y_values.values, index=plot_data.index)

    regions_to_label = set()
    if label_regions is True:
        regions_to_label.update(plot_data.index)
    elif label_regions:
        if isinstance(label_regions, str):
            label_regions = [label_regions]
        missing_regions = set(label_regions) - set(plot_data.index)
        if missing_regions:
            raise ValueError(
                "These region names were not found in the plot data: "
                f"{', '.join(sorted(missing_regions))}"
            )
        regions_to_label.update(label_regions)

    if label_outliers is not None:
        outlier_count = 3 if label_outliers is True else int(label_outliers)
        if outlier_count <= 0:
            raise ValueError("label_outliers must be True or a positive integer.")

        for series in (x_series, y_series):
            regions_to_label.update(series.nlargest(outlier_count).index)
            regions_to_label.update(series.nsmallest(outlier_count).index)

    if label_high_outliers:
        high_hits = plot_data.index[
            (plot_data[x_indicator] > high_outlier_x)
            & (plot_data[y_indicator] > high_outlier_y)
        ]
        regions_to_label.update(high_hits)

    if x_values.min() >= 0:
        ax.set_xlim(left=0)
    if y_values.min() >= 0:
        ax.set_ylim(bottom=0)

    ax.set_title(
        chart_title,
        color=text_color,
        fontsize=14,
        pad=14,
    )

    if x_scale == "percentage":
        ax.set_xlabel(x_axis_label, color=text_color, fontsize=11)
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.0f}%"))
    elif x_scale == "currency_millions":
        ax.set_xlabel(x_axis_label, color=text_color, fontsize=11)
        ax.xaxis.set_major_formatter(
            FuncFormatter(lambda x, _: f"£{x:,.0f}")
            if x_currency_thousands
            else FuncFormatter(
                lambda x, _: (
                    "£0"
                    if abs(x) < 1e-12
                    else f"£{x:,.0f}"
                    if abs(x - round(x)) < 1e-9
                    else f"£{x:,.1f}"
                )
            )
        )
    elif x_scale == "currency":
        ax.set_xlabel(x_axis_label, color=text_color, fontsize=11)
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"£{x:,.2f}"))
    elif x_scale == "number_decimal":
        ax.set_xlabel(x_axis_label, color=text_color, fontsize=11)
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:,.2f}"))
    else:
        ax.set_xlabel(x_axis_label, color=text_color, fontsize=11)
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:,.0f}"))

    if y_scale == "percentage":
        ax.set_ylabel(y_axis_label, color=text_color, fontsize=11)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:.0f}%"))
    elif y_scale == "currency_millions":
        ax.set_ylabel(y_axis_label, color=text_color, fontsize=11)
        ax.yaxis.set_major_formatter(
            FuncFormatter(lambda y, _: f"£{y:,.0f}")
            if y_currency_thousands
            else FuncFormatter(
                lambda y, _: (
                    "£0"
                    if abs(y) < 1e-12
                    else f"£{y:,.0f}"
                    if abs(y - round(y)) < 1e-9
                    else f"£{y:,.1f}"
                )
            )
        )
    elif y_scale == "currency":
        ax.set_ylabel(y_axis_label, color=text_color, fontsize=11)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"£{y:,.2f}"))
    elif y_scale == "number_decimal":
        ax.set_ylabel(y_axis_label, color=text_color, fontsize=11)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:,.2f}"))
    else:
        ax.set_ylabel(y_axis_label, color=text_color, fontsize=11)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:,.0f}"))

    ax.tick_params(axis="both", colors=text_color, labelsize=10)
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(axis_color)
    ax.spines["bottom"].set_color(axis_color)
    ax.set_axisbelow(True)

    if regions_to_label:
        if len(regions_to_label) < len(plot_data):
            print(
                "Regions labelled: "
                f"{', '.join(sorted(regions_to_label, key=str))}"
            )
        _annotate_scatter_labels(
            ax,
            regions_to_label,
            x_series,
            y_series,
            text_color,
            plot_data,
            x_indicator,
            y_indicator,
            x_scale,
            y_scale,
            x_currency_thousands,
            y_currency_thousands,
        )

    if y_indicator == "cyp_avg_score":
        ax.axhline(
            0,
            color=axis_color,
            linestyle=":",
            linewidth=1.2,
            zorder=2,
        )
        x_min, x_max = ax.get_xlim()
        x_label_padding = (x_max - x_min) * 0.02
        national_average_font = FontProperties(
            family=["DejaVu Serif"],
            weight="bold",
            size=8,
        )
        ax.annotate(
            "National Average",
            xy=(x_max - x_label_padding, 0),
            xytext=(0, -6),
            textcoords="offset points",
            va="top",
            ha="right",
            fontproperties=national_average_font,
            color=text_color,
            zorder=4,
        )
        cyp_note = "\n".join(
            textwrap.wrap(
                "Negative scores indicate an area where children have, on average, "
                "better educational outcomes than the national average — higher "
                "attainment, more progression to higher education, and lower absence.",
                width=70,
            )
        )
        ax.text(
            1.1,
            0.99,
            cyp_note,
            transform=ax.transAxes,
            fontsize=7.5,
            color=text_color,
            va="top",
            ha="right",
            linespacing=1.2,
            bbox={
                "boxstyle": "round,pad=0.3",
                "facecolor": background_color,
                "edgecolor": "#cccccc",
                "linewidth": 0.8,
            },
            zorder=4,
        )

    _save_figure_to_outputs(ax.figure, save_path)

    return ax, plot_data, spearman_rho, spearman_pvalue


##### FUNCTION TO PLOT A SPEARMAN RANK CORRELATION MATRIX


def plot_spearman_correlation_matrix(
    data,
    indicators=None,
    region_col="region_name",
    indicator_col="indicator",
    value_col="value",
    regional_breakdown=None,
    region_name=None,
    source=None,
    source_col="source",
    alpha=0.05,
    ax=None,
    title=None,
    save_path=None,
):
    """Plot a Spearman rank correlation matrix for selected indicators.

    Data are expected in long format and are pivoted to one row per region
    before correlations are calculated. Pass source when the same indicator
    appears in more than one source. Pass regional_breakdown and/or
    region_name to restrict geography (same names as the dataset columns).
    Pass save_path to save the chart as a PNG in the outputs folder.

    Cell annotations show Spearman ρ and the two-sided p-value. Significance
    stars use: * p<0.05, ** p<0.01, *** p<0.001.

    Defaults to the Navigate enrichment indicators:
    - Number of unique total learners in scope
    - Total hours of enrichment provision
    - Average number of enrichment hours per learner

    Returns
    -------
    ax, correlation_matrix, pvalue_matrix
    """
    default_indicators = [
        "Number of unique total learners in scope",
        "Total hours of enrichment provision",
        "Average number of enrichment hours per learner",
    ]

    if indicators is None:
        indicators = default_indicators
    elif isinstance(indicators, str):
        indicators = [indicators]

    if len(indicators) < 2:
        raise ValueError("At least two indicators are required for a correlation matrix.")

    required_cols = {region_col, indicator_col, value_col}
    if source is not None:
        required_cols.add(source_col)
    missing_cols = required_cols - set(data.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing_cols))}")

    plot_data = _apply_geography_filter(
        data,
        regional_breakdown=regional_breakdown,
        region_name=region_name,
    )

    if source is not None:
        if isinstance(source, str):
            source = [source]
        plot_data = plot_data.loc[plot_data[source_col].isin(source)]
        if plot_data.empty:
            raise ValueError(
                "No rows found for source in: "
                f"{', '.join(sorted(source))}"
            )

    plot_data = plot_data.loc[plot_data[indicator_col].isin(indicators)].copy()
    if plot_data.empty:
        raise ValueError("No rows found for the selected indicators.")

    missing_indicators = [
        indicator
        for indicator in indicators
        if indicator not in set(plot_data[indicator_col])
    ]
    if missing_indicators:
        raise ValueError(
            "These indicators were not found: "
            f"{', '.join(missing_indicators)}"
        )

    if source_col in plot_data.columns:
        ambiguous = (
            plot_data.groupby(indicator_col)[source_col]
            .nunique()
            .loc[lambda counts: counts > 1]
        )
        if not ambiguous.empty:
            raise ValueError(
                "These indicators appear in multiple sources: "
                f"{', '.join(ambiguous.index)}. "
                "Pass source=... to choose which source to use."
            )

    wide = (
        plot_data.pivot_table(
            index=region_col,
            columns=indicator_col,
            values=value_col,
            aggfunc="mean",
        )
        .reindex(columns=indicators)
        .apply(pd.to_numeric, errors="coerce")
        .dropna(how="any")
    )

    if len(wide) < 3:
        raise ValueError(
            "Need at least 3 complete region rows to compute Spearman correlations."
        )

    rho_values, p_values = spearmanr(wide.to_numpy())
    if wide.shape[1] == 2:
        # spearmanr returns scalars for exactly two variables
        rho_values = [
            [1.0, float(rho_values)],
            [float(rho_values), 1.0],
        ]
        p_values = [
            [0.0, float(p_values)],
            [float(p_values), 0.0],
        ]

    correlation_matrix = pd.DataFrame(
        rho_values,
        index=indicators,
        columns=indicators,
    )
    pvalue_matrix = pd.DataFrame(
        p_values,
        index=indicators,
        columns=indicators,
    )

    def _significance_stars(p_value):
        if p_value < 0.001:
            return "***"
        if p_value < 0.01:
            return "**"
        if p_value < 0.05:
            return "*"
        return ""

    def _format_p_value(p_value):
        if p_value < 0.001:
            return "p<0.001"
        return f"p={p_value:.3f}"

    print("Spearman rank correlation matrix")
    print(f"Regions included (complete cases): {len(wide)}")
    print(correlation_matrix.to_string(float_format=lambda x: f"{x:.3f}"))
    print()
    print("Two-sided p-value matrix")
    print(pvalue_matrix.to_string(float_format=lambda x: f"{x:.4f}"))
    print()
    print(
        "Significance stars: * p<0.05, ** p<0.01, *** p<0.001 "
        f"(threshold used for verdicts: alpha={alpha:.2f})"
    )
    print()
    print("Pairwise verdicts:")
    for i, row_indicator in enumerate(indicators):
        for col_indicator in indicators[i + 1 :]:
            rho = correlation_matrix.loc[row_indicator, col_indicator]
            p_value = pvalue_matrix.loc[row_indicator, col_indicator]
            stars = _significance_stars(p_value)
            if p_value < alpha:
                verdict = (
                    f"statistically significant at {alpha:.0%} "
                    "(unlikely to be due to chance alone)"
                )
            else:
                verdict = (
                    f"not statistically significant at {alpha:.0%} "
                    "(could plausibly be due to chance)"
                )
            print(
                f"- {row_indicator} vs {col_indicator}: "
                f"ρ={rho:.3f}{stars}, {_format_p_value(p_value)} → {verdict}"
            )

    plt.rcParams["font.family"] = ["Roboto Slab", "serif"]

    bar_color = "#d93649"
    background_color = "#ffffff"
    text_color = "#222222"
    chart_title = (
        title if title is not None else "Spearman rank correlation matrix"
    )
    correlation_cmap = LinearSegmentedColormap.from_list(
        "spearman_corr",
        ["#2166ac", "#ffffff", bar_color],
    )

    display_matrix = correlation_matrix.copy()
    wrapped_labels = [
        _wrap_label_two_lines(label) for label in correlation_matrix.columns
    ]
    display_matrix.index = wrapped_labels
    display_matrix.columns = wrapped_labels

    if ax is None:
        fig, ax = plt.subplots(figsize=(7.2, 6.0))
    else:
        fig = ax.figure

    fig.patch.set_facecolor(background_color)
    ax.set_facecolor(background_color)

    heatmap = sns.heatmap(
        display_matrix,
        ax=ax,
        annot=False,
        cmap=correlation_cmap,
        vmin=-1,
        vmax=1,
        center=0,
        square=True,
        linewidths=1.2,
        linecolor=background_color,
        cbar_kws={"shrink": 0.85, "label": "Spearman ρ"},
    )

    # Correlation on top of each cell, p-value directly underneath.
    for row_index, row_indicator in enumerate(indicators):
        for col_index, col_indicator in enumerate(indicators):
            rho = correlation_matrix.loc[row_indicator, col_indicator]
            p_value = pvalue_matrix.loc[row_indicator, col_indicator]
            x_position = col_index + 0.5
            y_position = row_index + 0.5

            if row_indicator == col_indicator:
                ax.text(
                    x_position,
                    y_position,
                    "1.00",
                    ha="center",
                    va="center",
                    color=text_color,
                    fontsize=12,
                    fontweight="bold",
                )
            else:
                stars = _significance_stars(p_value)
                ax.text(
                    x_position,
                    y_position - 0.14,
                    f"{rho:.2f}{stars}",
                    ha="center",
                    va="center",
                    color=text_color,
                    fontsize=12,
                    fontweight="bold",
                )
                ax.text(
                    x_position,
                    y_position + 0.16,
                    _format_p_value(p_value),
                    ha="center",
                    va="center",
                    color=text_color,
                    fontsize=8,
                )

    ax.set_title(
        chart_title,
        color=text_color,
        fontsize=14,
        fontweight="bold",
        pad=14,
    )
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(axis="both", colors=text_color, labelsize=10, length=0, pad=6)

    for label in ax.get_xticklabels():
        label.set_rotation(0)
        label.set_ha("center")
        label.set_va("top")
        label.set_linespacing(1.25)

    for label in ax.get_yticklabels():
        label.set_rotation(0)
        label.set_ha("right")
        label.set_va("center")
        label.set_linespacing(1.25)

    colorbar = heatmap.collections[0].colorbar
    if colorbar is not None:
        colorbar.set_ticks([-1, -0.5, 0, 0.5, 1])
        colorbar.ax.tick_params(labelsize=9, colors=text_color)
        colorbar.set_label("Spearman ρ", color=text_color, fontsize=10)

    fig.subplots_adjust(left=0.20, right=0.88, top=0.88, bottom=0.16)

    if save_path is not None:
        output_dir = Path("outputs")
        output_dir.mkdir(exist_ok=True)
        output_path = output_dir / Path(save_path).name
        if output_path.suffix.lower() != ".png":
            output_path = output_path.with_suffix(".png")
        fig.savefig(
            output_path,
            dpi=300,
            bbox_inches="tight",
            facecolor=background_color,
            edgecolor="none",
            transparent=False,
        )

    return ax, correlation_matrix, pvalue_matrix


##### FUNCTION TO PLOT AN ENGLAND CHOROPLETH FOR ONE INDICATOR


_BOUNDARY_FILES = {
    "utla": {
        "path": Path("data/geo/utla_uk.geojson"),
        "url": (
            "https://services1.arcgis.com/ESMARspQHYMw9BZ9/arcgis/rest/services/"
            "Counties_and_Unitary_Authorities_December_2023_Boundaries_UK_BGC/"
            "FeatureServer/0/query?where=1%3D1&outFields=*&outSR=4326&f=geojson"
        ),
        "name_col": "CTYUA23NM",
        "code_col": "CTYUA23CD",
    },
    "region": {
        "path": Path("data/geo/regions_en.geojson"),
        "url": (
            "https://services1.arcgis.com/ESMARspQHYMw9BZ9/arcgis/rest/services/"
            "Regions_December_2023_Boundaries_EN_BGC/"
            "FeatureServer/0/query?where=1%3D1&outFields=*&outSR=4326&f=geojson"
        ),
        "name_col": "RGN23NM",
        "code_col": "RGN23CD",
    },
}

_BREAKDOWN_TO_LAYER = {
    "upper tier local authority": "utla",
    "best available region": "region",
}


def _normalise_area_name(name):
    """Match GrantNav / ONS names: drop suffixes and ', City of'."""
    text = str(name)
    text = re.sub(r"\s*\((UTLA|CTY|LONB|MD|UA)\)\s*$", "", text)
    text = re.sub(
        r"\s+((City|Borough|County|District|Metropolitan|London)\s+)*"
        r"Council\s*$",
        "",
        text,
        flags=re.I,
    )
    text = re.sub(r",\s*(City|County) of$", "", text, flags=re.I)
    text = re.sub(r"\s+", " ", text.replace("&", "and")).strip().casefold()
    if text == "east":
        return "east of england"
    return text


def _load_england_boundaries(layer_key):
    """Load cached ONS boundaries, downloading them on first use."""
    import geopandas as gpd
    from urllib.request import urlretrieve

    spec = _BOUNDARY_FILES[layer_key]
    path = spec["path"]
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        urlretrieve(spec["url"], path)

    boundaries = gpd.read_file(path)
    england = boundaries.loc[
        boundaries[spec["code_col"]].astype(str).str.startswith("E")
    ].copy()
    england["name_key"] = england[spec["name_col"]].map(_normalise_area_name)
    return england.to_crs(27700)


def _format_map_value(value, scale_type):
    """Format a map value for colour-bar ticks or rank labels."""
    if scale_type == "percentage":
        return f"{value:.0f}%"
    if scale_type == "currency_millions":
        if abs(value) < 1e-12:
            return "£0"
        if abs(value) < 0.01:
            return f"£{value:,.3f}m"
        if abs(value) < 1:
            return f"£{value:,.2f}m"
        if abs(value - round(value)) < 1e-9:
            return f"£{value:,.0f}m"
        return f"£{value:,.1f}m"
    if scale_type == "currency":
        return f"£{value:,.2f}"
    if scale_type == "number_decimal":
        return f"{value:,.2f}"
    return f"{value:,.0f}"


def _format_pounds_label(pounds):
    """Format a raw pound amount for a map label."""
    return _format_map_value(float(pounds) / 1_000_000, "currency_millions")


_RANK_N = 10


def _rank_label_text(row, scale_type):
    """Label the plotted indicator: £ for money, the value otherwise."""
    if scale_type in {"currency", "currency_millions"}:
        pounds = row.get("_award_pounds")
        if pd.notna(pounds):
            return _format_pounds_label(pounds)
        return _format_map_value(row["_plot_value"], "currency_millions")
    return _format_map_value(row["_plot_value"], scale_type)


def _boxes_overlap(box_a, box_b, pad=0):
    left_a, right_a, bottom_a, top_a = box_a
    left_b, right_b, bottom_b, top_b = box_b
    return not (
        right_a + pad < left_b
        or left_a - pad > right_b
        or top_a + pad < bottom_b
        or bottom_a - pad > top_b
    )


def _label_box(text_x, text_y, ha, va, width, height):
    left = text_x if ha == "left" else text_x - width
    right = text_x + width if ha == "left" else text_x
    bottom = text_y if va == "bottom" else text_y - height
    top = text_y + height if va == "bottom" else text_y
    return (left, right, bottom, top)


def _overlap_fraction(box, geom):
    from shapely.geometry import box as shapely_box

    if geom is None or geom.is_empty:
        return 0.0
    left, right, bottom, top = box
    area = (right - left) * (top - bottom)
    if area <= 0:
        return 0.0
    return shapely_box(left, bottom, right, top).intersection(geom).area / area


def _leader_hits_boxes(origin, dest, boxes):
    from shapely.geometry import LineString, box as shapely_box

    line = LineString([origin, dest])
    for left, right, bottom, top in boxes:
        if line.intersects(shapely_box(left, bottom, right, top)):
            return True
    return False


def _union_geoms(geoms):
    if geoms is None:
        return None
    items = list(geoms)
    if not items:
        return None
    try:
        from shapely import union_all

        return union_all(items)
    except (ImportError, TypeError, AttributeError):
        from shapely.ops import unary_union

        return unary_union(items)


def _anchor_on_polygon(geom, text_x, text_y):
    from shapely.geometry import Point
    from shapely.ops import nearest_points

    return nearest_points(geom, Point(text_x, text_y))[0]


def _away_from_neighbours(origin, text_x, text_y, other_points, span):
    """Penalise labels that sit towards another nearby ranked area."""
    lx = text_x - origin[0]
    ly = text_y - origin[1]
    label_len = hypot(lx, ly) or 1.0
    reach = 0.12 * span
    penalty = 0.0
    for other_x, other_y in other_points:
        dx = other_x - origin[0]
        dy = other_y - origin[1]
        dist = hypot(dx, dy)
        if dist < 1 or dist > reach:
            continue
        cosine = (lx * dx + ly * dy) / (label_len * dist)
        if cosine > 0:
            penalty += 0.35 * span * cosine * (1.0 - dist / reach)
    return penalty


def _leader_crosses_geom(origin, dest, geom, slack):
    from shapely.geometry import LineString

    if geom is None or geom.is_empty:
        return False
    hit = LineString([origin, dest]).intersection(geom)
    return getattr(hit, "length", 0.0) > slack


def _label_ranked_areas(
    ax, top, background_color, text_color, scale_type, land=None
):
    """Place name-and-value notes close to each area, without stacking."""
    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()
    x_span = x_max - x_min
    y_span = y_max - y_min
    margin_x = 0.02 * x_span
    margin_y = 0.02 * y_span
    span = max(x_span, y_span)
    pad = 0.02 * x_span

    land_union = _union_geoms(land) if land is not None and len(land) > 0 else None
    ranked_geoms = list(top.geometry)

    placed = [
        (
            x_min,
            x_min + 0.26 * x_span,
            y_max - 0.22 * y_span,
            y_max,
        )
    ]

    angles = np.deg2rad(np.arange(0, 360, 15))
    radii_near = np.array([0.04, 0.065, 0.09, 0.12]) * span
    radii_far = np.array([0.16, 0.21]) * span

    points = np.array(
        [
            (geom.representative_point().x, geom.representative_point().y)
            for geom in ranked_geoms
        ]
    )
    crowd = np.array(
        [
            np.sum(np.hypot(points[:, 0] - x, points[:, 1] - y) < 0.18 * span)
            for x, y in points
        ]
    )
    areas = np.array([float(geom.area) for geom in ranked_geoms])
    draw_order = np.lexsort((areas, -crowd))

    for index in draw_order:
        row = top.iloc[index]
        geom = ranked_geoms[index]
        point = geom.representative_point()
        centre = (point.x, point.y)
        neighbour_pts = [tuple(pt) for j, pt in enumerate(points) if j != index]
        others = _union_geoms(
            [other for j, other in enumerate(ranked_geoms) if j != index]
        )
        name = str(row.get("region_name") or "").strip()
        if len(name) > 22:
            name = _wrap_label_two_lines(name)
        amount = _rank_label_text(row, scale_type)
        label = f"{name}\n{amount}"
        lines = label.split("\n")
        longest = max(len(line) for line in lines)
        box_w = (longest + 5) * 0.0078 * x_span
        box_h = (0.034 + 0.024 * len(lines)) * y_span

        def _candidates(
            radii, require_clear_leader, max_ranked_overlap, block_crossing=True
        ):
            found = []
            for radius in radii:
                for angle in angles:
                    text_x = point.x + radius * np.cos(angle)
                    text_y = point.y + radius * np.sin(angle)
                    ha = "left" if text_x >= point.x else "right"
                    va = "bottom" if text_y >= point.y else "top"
                    box = _label_box(text_x, text_y, ha, va, box_w, box_h)
                    left, right, bottom, top_edge = box
                    if (
                        left < x_min + margin_x
                        or right > x_max - margin_x
                        or bottom < y_min + margin_y
                        or top_edge > y_max - margin_y
                    ):
                        continue
                    if any(_boxes_overlap(box, taken, pad=pad) for taken in placed):
                        continue
                    ranked_frac = _overlap_fraction(box, others)
                    if ranked_frac > max_ranked_overlap:
                        continue
                    anchor = _anchor_on_polygon(geom, text_x, text_y)
                    origin = (anchor.x, anchor.y)
                    dest = (text_x, text_y)
                    if require_clear_leader and _leader_hits_boxes(
                        origin, dest, placed
                    ):
                        continue
                    if block_crossing and _leader_crosses_geom(
                        origin, dest, others, 0.008 * span
                    ):
                        continue
                    score = (
                        radius
                        + 0.04 * span * _overlap_fraction(box, land_union)
                        + 0.18 * span * ranked_frac
                        + _away_from_neighbours(
                            centre, text_x, text_y, neighbour_pts, span
                        )
                    )
                    found.append((score, text_x, text_y, ha, va, box, anchor))
            return found

        candidates = (
            _candidates(radii_near, True, 0.12)
            or _candidates(radii_near, False, 0.12)
            or _candidates(radii_far, True, 0.25)
            or _candidates(radii_far, False, 0.5, False)
        )
        if not candidates:
            text_x = x_min + margin_x
            text_y = y_min + margin_y + (len(placed) * 0.06 * y_span)
            ha, va = "left", "bottom"
            box = _label_box(text_x, text_y, ha, va, box_w, box_h)
            anchor = _anchor_on_polygon(geom, text_x, text_y)
        else:
            _, text_x, text_y, ha, va, box, anchor = min(
                candidates, key=lambda item: item[0]
            )

        placed.append(box)
        ax.annotate(
            label,
            xy=(anchor.x, anchor.y),
            xytext=(text_x, text_y),
            textcoords="data",
            fontsize=7.5,
            color=text_color,
            ha=ha,
            va=va,
            linespacing=1.15,
            bbox={
                "boxstyle": "round,pad=0.22",
                "facecolor": background_color,
                "edgecolor": "#cccccc",
                "linewidth": 0.7,
            },
            arrowprops={
                "arrowstyle": "-",
                "color": "#4a4a4a",
                "lw": 0.6,
                "shrinkA": 0,
                "shrinkB": 1,
            },
            zorder=6,
        )


def plot_england_heatmap(
    data,
    indicator,
    regional_breakdown=None,
    source=None,
    bins=5,
    title=None,
    save_path=None,
    region_name=None,
    style="coverage",
    rank=False,
    transparent="F",
):
    """Map one indicator onto England boundaries.

    Default `style="coverage"` is a presence map: areas with a value
    get one fill colour, areas without stay blank. Use this when the
    indicator is not comparable across places (raw totals, ENG).

    `style="choropleth"` shades by value. Only use that when the
    indicator is normalised or otherwise meant to be compared.

    If `rank=True`, the highest-value areas are filled in a darker red
    and labelled. That is the top 10 UTLAs, or the top 3 English regions
    (with all nine regions labelled). Money indicators show pounds;
    ENG shows the ENG figure.

    `regional_breakdown` chooses the border layer:
    - upper tier local authority → ONS counties and unitary authorities
    - best available region → the nine English regions

    Values are joined on `region_name`. Pass source when the indicator
    exists in more than one source. Pass save_path to write a PNG in
    outputs/. transparent="T" saves that PNG with no background.
    """
    if not isinstance(indicator, str):
        raise ValueError("indicator must be a single indicator name.")

    region_col = "region_name"
    indicator_col = "indicator"
    value_col = "value"
    source_col = "source"

    indicator_scales = {
        "Total amount awarded": "currency_millions",
        "Total amount awarded per head": "currency",
        "Funding per child aged 5 to 20": "currency",
        "Funding per child aged 5 to 20 (excluding largest grant)": "currency",
        "Effective number of grants (ENG)": "number_decimal",
        "Number of grants": "number",
        "% of creative and cultural employees": "percentage",
        "% of the population engaged with the arts online in the last 12 months": "percentage",
        "% of the population engaged with the arts in person in the last 12 months": "percentage",
        "Number of unique total learners in scope": "number",
        "Number of total enrichment activities": "number",
        "Average number of enrichment activities per learner": "number_decimal",
        "Total hours of enrichment provision": "number_decimal",
        "Average number of enrichment hours per learner": "number_decimal",
    }

    required_cols = {region_col, indicator_col, value_col, "regional_breakdown"}
    if source is not None:
        required_cols.add(source_col)
    missing_cols = required_cols - set(data.columns)
    if missing_cols:
        raise ValueError(
            f"Missing required columns: {', '.join(sorted(missing_cols))}"
        )

    plot_data = _apply_geography_filter(
        data,
        regional_breakdown=regional_breakdown,
        region_name=region_name,
    )
    if source is not None:
        plot_data = _apply_column_value_filter(plot_data, source_col, source)

    source_data = plot_data.copy()
    plot_data = plot_data.loc[plot_data[indicator_col] == indicator].copy()
    if plot_data.empty:
        raise ValueError(f"No rows found for indicator: {indicator}")

    if source is None and source_col in plot_data.columns:
        sources_present = sorted(plot_data[source_col].dropna().unique())
        if len(sources_present) > 1:
            raise ValueError(
                f"Indicator '{indicator}' appears in multiple sources: "
                f"{', '.join(sources_present)}. "
                "Pass source=... to choose which source to plot."
            )

    breakdowns = list(plot_data["regional_breakdown"].dropna().unique())
    if regional_breakdown is None:
        if len(breakdowns) != 1:
            raise ValueError(
                "Pass regional_breakdown=... so the map can pick one "
                "border layer. Found: "
                f"{', '.join(sorted(str(item) for item in breakdowns))}"
            )
        breakdown_used = breakdowns[0]
    elif isinstance(regional_breakdown, str):
        breakdown_used = regional_breakdown
    else:
        chosen = list(regional_breakdown)
        if len(chosen) != 1:
            raise ValueError(
                "plot_england_heatmap needs one regional_breakdown so it "
                "can pick a single border layer."
            )
        breakdown_used = chosen[0]

    layer_key = _BREAKDOWN_TO_LAYER.get(str(breakdown_used).strip())
    if layer_key is None:
        raise ValueError(
            f"No England border layer for regional_breakdown="
            f"'{breakdown_used}'. Use 'upper tier local authority' or "
            "'best available region'."
        )
    is_region = layer_key == "region"
    boundary_lw = 0.7 if is_region else 0.25
    top_lw = 0.9 if is_region else 0.35

    values = (
        plot_data.groupby(region_col, as_index=False)[value_col]
        .mean()
    )
    values[value_col] = pd.to_numeric(values[value_col], errors="coerce")
    values["name_key"] = values[region_col].map(_normalise_area_name)

    boundaries = _load_england_boundaries(layer_key)
    mapped = boundaries.merge(
        values[["name_key", value_col, region_col]],
        on="name_key",
        how="left",
    )

    unmatched = sorted(
        set(values["name_key"]) - set(boundaries["name_key"])
    )
    if unmatched:
        print(
            "These region names did not match a boundary and are omitted: "
            + ", ".join(unmatched)
        )

    scale_type = indicator_scales.get(indicator, "number")
    plot_values = mapped[value_col].astype(float)
    if scale_type == "percentage" and plot_values.dropna().abs().max() <= 1:
        plot_values = plot_values * 100
    elif scale_type == "currency_millions":
        plot_values = plot_values / 1_000_000
    mapped["_plot_value"] = plot_values

    award_rows = source_data.loc[
        source_data[indicator_col] == "Total amount awarded"
    ].copy()
    if not award_rows.empty:
        award_rows[value_col] = pd.to_numeric(
            award_rows[value_col], errors="coerce"
        )
        award_by_area = award_rows.groupby(region_col)[value_col].mean()
        mapped["_award_pounds"] = mapped[region_col].map(award_by_area)
    elif scale_type in {"currency", "currency_millions"}:
        mapped["_award_pounds"] = mapped[value_col]
    else:
        mapped["_award_pounds"] = np.nan

    present = mapped.loc[mapped["_plot_value"].notna()].copy()
    missing = mapped.loc[mapped["_plot_value"].isna()].copy()
    if present.empty:
        raise ValueError("No values could be joined to England boundaries.")

    if style not in {"coverage", "choropleth"}:
        raise ValueError("style must be 'coverage' or 'choropleth'.")

    rank_n = min(3 if is_region else _RANK_N, len(present))
    top = present.nlargest(rank_n, "_plot_value")
    if rank:
        top_keys = set(top["name_key"])
        rest = present.loc[~present["name_key"].isin(top_keys)].copy()
        top = present.loc[present["name_key"].isin(top_keys)].copy()
        top = top.sort_values("_plot_value", ascending=False)
        labels = present if is_region else top
    else:
        rest = present
        labels = top.iloc[0:0]

    plt.rcParams["font.family"] = ["Roboto Slab", "serif"]
    background_color = "#ffffff"
    text_color = "#222222"
    axis_color = "#000000"
    has_data_color = "#f6b8c0"
    top_color = "#d93649"
    chart_title = title if title is not None else indicator

    fig, ax = plt.subplots(figsize=(8, 9))
    fig.patch.set_facecolor(background_color)
    ax.set_facecolor(background_color)

    if not missing.empty:
        missing.plot(
            ax=ax,
            facecolor="none",
            edgecolor="#666666",
            linewidth=0.45,
            zorder=2,
        )

    coverage_handles = [
        Patch(
            facecolor=has_data_color,
            edgecolor="#4a4a4a",
            linewidth=1.0,
            label="Has data",
        ),
    ]
    if rank:
        coverage_handles.append(
            Patch(
                facecolor=top_color,
                edgecolor="#4a4a4a",
                linewidth=1.0,
                label=f"Top {rank_n}",
            )
        )
    coverage_handles.append(
        Patch(
            facecolor="none",
            edgecolor="#666666",
            linewidth=1.0,
            label="No data",
        )
    )
    missing_handles = [
        Patch(
            facecolor="none",
            edgecolor="#666666",
            linewidth=1.0,
            label="No data",
        )
    ]

    if style == "coverage":
        if rank:
            if not rest.empty:
                rest.plot(
                    ax=ax,
                    facecolor=has_data_color,
                    edgecolor="#4a4a4a",
                    linewidth=boundary_lw,
                    zorder=3,
                )
            top.plot(
                ax=ax,
                facecolor=top_color,
                edgecolor="#4a4a4a",
                linewidth=top_lw,
                zorder=4,
            )
        else:
            present.plot(
                ax=ax,
                facecolor=has_data_color,
                edgecolor="#4a4a4a",
                linewidth=boundary_lw,
                zorder=3,
            )
    else:
        if isinstance(bins, (list, tuple, np.ndarray)):
            edges = np.asarray(bins, dtype=float)
        else:
            n_bins = int(bins)
            if n_bins < 2:
                raise ValueError("bins must be at least 2.")
            edges = np.unique(
                np.quantile(
                    present["_plot_value"], np.linspace(0, 1, n_bins + 1)
                )
            )
            if len(edges) < 3:
                vmin = float(present["_plot_value"].min())
                vmax = float(present["_plot_value"].max())
                if vmin == vmax:
                    edges = np.array([vmin, vmax + 1e-9])
                else:
                    edges = np.linspace(vmin, vmax, 3)

        cmap = LinearSegmentedColormap.from_list(
            "england_heatmap",
            ["#f6b8c0", "#d93649"],
        )
        norm = BoundaryNorm(edges, cmap.N, clip=True)

        colorbar_label = {
            "currency_millions": "£ millions",
            "currency": "£",
            "percentage": "%",
            "number_decimal": indicator,
            "number": indicator,
        }.get(scale_type, indicator)

        present.plot(
            ax=ax,
            column="_plot_value",
            cmap=cmap,
            norm=norm,
            edgecolor="#4a4a4a",
            linewidth=boundary_lw,
            zorder=3,
        )
        colorbar = fig.colorbar(
            plt.cm.ScalarMappable(norm=norm, cmap=cmap),
            ax=ax,
            fraction=0.035,
            pad=0.02,
        )
        colorbar.set_ticks(edges)
        colorbar.set_ticklabels(
            [_format_map_value(edge, scale_type) for edge in edges]
        )
        colorbar.ax.tick_params(labelsize=9, colors=text_color)
        colorbar.set_label(colorbar_label, color=text_color, fontsize=10)
        colorbar.outline.set_edgecolor(axis_color)

    ax.set_axis_off()
    ax.set_aspect("equal")
    ax.set_title(
        chart_title,
        color=text_color,
        fontsize=14,
        fontweight="bold",
        pad=14,
    )

    if style == "coverage":
        _add_map_note(
            ax,
            coverage_handles,
            mapped.geometry,
            background_color,
            text_color,
        )
    elif not missing.empty:
        _add_map_note(
            ax,
            missing_handles,
            mapped.geometry,
            background_color,
            text_color,
        )

    if rank:
        if style != "coverage":
            top.plot(
                ax=ax,
                facecolor=top_color,
                edgecolor="#4a4a4a",
                linewidth=top_lw,
                zorder=4,
            )
        _label_ranked_areas(
            ax,
            labels,
            background_color,
            text_color,
            scale_type,
            land=mapped.geometry,
        )

    print(f"England {style} map: {indicator}")
    print(f"Border layer: {breakdown_used}")
    print(f"Areas with data: {len(present)}")
    print(f"Areas with no data: {len(missing)}")
    if rank:
        print(f"Top {rank_n}:")
        for _, row in top.iterrows():
            amount = _rank_label_text(row, scale_type)
            print(f"  {row['region_name']}: {amount}")

    if save_path is not None:
        if transparent not in {"T", "F"}:
            raise ValueError("transparent must be 'T' or 'F'.")
        output_dir = Path("outputs")
        output_dir.mkdir(exist_ok=True)
        output_path = output_dir / Path(save_path).name
        if output_path.suffix.lower() != ".png":
            output_path = output_path.with_suffix(".png")
        clear = transparent == "T"
        fig.savefig(
            output_path,
            dpi=300,
            bbox_inches="tight",
            facecolor="none" if clear else background_color,
            edgecolor="none",
            transparent=clear,
        )

    return ax, mapped


# Two-tier counties in GrantNav. Districts are IoD 2025 LADs in imd_all_la.
_COUNTY_DISTRICTS = {
    "Cambridgeshire": [
        "Cambridge", "East Cambridgeshire", "Fenland",
        "Huntingdonshire", "South Cambridgeshire",
    ],
    "Derbyshire": [
        "Amber Valley", "Bolsover", "Chesterfield", "Derbyshire Dales",
        "Erewash", "High Peak", "North East Derbyshire", "South Derbyshire",
    ],
    "Devon": [
        "East Devon", "Exeter", "Mid Devon", "North Devon",
        "South Hams", "Teignbridge", "Torridge", "West Devon",
    ],
    "East Sussex": ["Eastbourne", "Hastings", "Lewes", "Rother", "Wealden"],
    "Essex": [
        "Basildon", "Braintree", "Brentwood", "Castle Point", "Chelmsford",
        "Colchester", "Epping Forest", "Harlow", "Maldon", "Rochford",
        "Tendring", "Uttlesford",
    ],
    "Gloucestershire": [
        "Cheltenham", "Cotswold", "Forest of Dean",
        "Gloucester", "Stroud", "Tewkesbury",
    ],
    "Hampshire": [
        "Basingstoke and Deane", "East Hampshire", "Eastleigh", "Fareham",
        "Gosport", "Hart", "Havant", "New Forest", "Rushmoor",
        "Test Valley", "Winchester",
    ],
    "Hertfordshire": [
        "Broxbourne", "Dacorum", "East Hertfordshire", "Hertsmere",
        "North Hertfordshire", "St Albans", "Stevenage", "Three Rivers",
        "Watford", "Welwyn Hatfield",
    ],
    "Kent": [
        "Ashford", "Canterbury", "Dartford", "Dover",
        "Folkestone and Hythe", "Gravesham", "Maidstone", "Sevenoaks",
        "Swale", "Thanet", "Tonbridge and Malling", "Tunbridge Wells",
    ],
    "Lancashire": [
        "Burnley", "Chorley", "Fylde", "Hyndburn", "Lancaster", "Pendle",
        "Preston", "Ribble Valley", "Rossendale", "South Ribble",
        "West Lancashire", "Wyre",
    ],
    "Leicestershire": [
        "Blaby", "Charnwood", "Harborough", "Hinckley and Bosworth",
        "Melton", "North West Leicestershire", "Oadby and Wigston",
    ],
    "Lincolnshire": [
        "Boston", "East Lindsey", "Lincoln", "North Kesteven",
        "South Holland", "South Kesteven", "West Lindsey",
    ],
    "Norfolk": [
        "Breckland", "Broadland", "Great Yarmouth",
        "King's Lynn and West Norfolk", "North Norfolk", "Norwich",
        "South Norfolk",
    ],
    "Nottinghamshire": [
        "Ashfield", "Bassetlaw", "Broxtowe", "Gedling", "Mansfield",
        "Newark and Sherwood", "Rushcliffe",
    ],
    "Oxfordshire": [
        "Cherwell", "Oxford", "South Oxfordshire",
        "Vale of White Horse", "West Oxfordshire",
    ],
    "Staffordshire": [
        "Cannock Chase", "East Staffordshire", "Lichfield",
        "Newcastle-under-Lyme", "South Staffordshire", "Stafford",
        "Staffordshire Moorlands", "Tamworth",
    ],
    "Suffolk": [
        "Babergh", "East Suffolk", "Ipswich", "Mid Suffolk", "West Suffolk",
    ],
    "Surrey": [
        "Elmbridge", "Epsom and Ewell", "Guildford", "Mole Valley",
        "Reigate and Banstead", "Runnymede", "Spelthorne", "Surrey Heath",
        "Tandridge", "Waverley", "Woking",
    ],
    "Warwickshire": [
        "North Warwickshire", "Nuneaton and Bedworth", "Rugby",
        "Stratford-on-Avon", "Warwick",
    ],
    "West Sussex": [
        "Adur", "Arun", "Chichester", "Crawley", "Horsham",
        "Mid Sussex", "Worthing",
    ],
    "Worcestershire": [
        "Bromsgrove", "Malvern Hills", "Redditch", "Worcester",
        "Wychavon", "Wyre Forest",
    ],
}

_GRANTNAV_INDICATORS = [
    "Total amount awarded",
    "Number of grants",
    "Effective number of grants (ENG)",
]

_DEPRIVATION_METADATA = [
    ("Scope", "GrantNav360 upper-tier local authorities only. Navigate is not in this file."),
    (
        "GrantNav extract",
        "From data_final/all_data_final.csv: source GrantNav360, "
        "regional_breakdown = upper tier local authority, "
        "indicators Total amount awarded, Number of grants, and "
        "Effective number of grants (ENG), "
        "reference_date 2025/26.",
    ),
    (
        "IMD and IDACI join",
        "Official MHCLG average scores from data_final/imd_all_la.xlsx "
        "(IMD tab 'IMD - Average score'; IDACI tab 'IDACI - Average score'). "
        "imd_all_la is 2024 local authority districts. GrantNav names are matched "
        "after stripping council suffixes. 107 of 128 UTLAs are themselves a district "
        "and join 1:1. The other 21 are two-tier counties, which File 10 does not publish.",
    ),
    (
        "County aggregation",
        "For those 21 counties, imd_avg_score and idaci_avg_score are the "
        "unweighted mean of the official district scores listed under County districts. "
        "This is not an MHCLG county figure. match_method = county_district_mean "
        "flags those rows; direct_lad flags the 107 one-to-one matches.",
    ),
    (
        "CYP join",
        "Not in imd_all_la.xlsx. Unweighted mean of Children and Young People "
        "Sub-domain Score from data_final/imd_scores_all.xlsx, by 2024 LAD. "
        "Direct UTLAs use that LAD mean. Counties use the unweighted mean of "
        "their districts' LAD means.",
    ),
    (
        "CSV format",
        "data/deprivation_analysis_data.csv is the same table in long format, "
        "source = GrantNav360.",
    ),
    (None, None),
    ("regional_breakdown", "Always upper tier local authority (GrantNav)."),
    ("region_name", "GrantNav UTLA name."),
    ("reference_date", "2025/26."),
    ("match_method", "direct_lad or county_district_mean. See County aggregation."),
    (
        "Total amount awarded",
        "GrantNav360 total awarded to the UTLA in 2025/26.",
    ),
    (
        "Number of grants",
        "GrantNav360 count of grants in the UTLA, 2025/26.",
    ),
    (
        "Effective number of grants (ENG)",
        "GrantNav360 effective number of grants in the UTLA, 2025/26.",
    ),
    (
        "idaci_avg_score",
        "Income Deprivation Affecting Children Index average score. "
        "Official MHCLG figure for direct_lad rows; county mean otherwise. "
        "Higher = more deprived.",
    ),
    (
        "imd_avg_score",
        "Index of Multiple Deprivation average score. "
        "Official MHCLG figure for direct_lad rows; county mean otherwise. "
        "Higher = more deprived.",
    ),
    (
        "cyp_avg_score",
        "Children and Young People sub-domain. Derived, not an official LA figure. "
        "Zero is the national neighbourhood average.",
    ),
    (None, None),
    (
        "County districts",
        "; ".join(
            f"{county}: {', '.join(districts)}"
            for county, districts in _COUNTY_DISTRICTS.items()
        ),
    ),
]


def _lad_score_lookup(imd, idaci, cyp):
    lad_code = "Local Authority District code (2024)"
    name_col = "Local Authority District name (2024)"
    scores = imd[[lad_code, name_col, "IMD - Average score"]].merge(
        idaci[[lad_code, "IDACI - Average score"]], on=lad_code
    )
    scores = scores.merge(cyp, left_on=lad_code, right_index=True, how="left")
    scores["name_key"] = scores[name_col].map(_normalise_area_name)
    return scores.rename(
        columns={
            "IMD - Average score": "imd_avg_score",
            "IDACI - Average score": "idaci_avg_score",
        }
    )


def build_deprivation_analysis_data(
    save_csv="data/deprivation_analysis_data.csv",
    save_xlsx="data/deprivation_analysis_data.xlsx",
):
    """GrantNav360 UTLAs plus IMD/IDACI from imd_all_la and derived CYP."""
    lad_code = "Local Authority District code (2024)"
    grantnav = pd.read_csv("data_final/all_data_final.csv")
    grantnav = grantnav.loc[
        (grantnav["source"] == "GrantNav360")
        & (grantnav["regional_breakdown"] == "upper tier local authority")
        & grantnav["indicator"].isin(_GRANTNAV_INDICATORS)
    ].copy()
    grantnav["value"] = pd.to_numeric(grantnav["value"], errors="coerce")
    wide = grantnav.pivot_table(
        index=["regional_breakdown", "region_name", "reference_date"],
        columns="indicator",
        values="value",
        aggfunc="first",
    ).reset_index()
    wide.columns.name = None
    wide["name_key"] = wide["region_name"].map(_normalise_area_name)

    imd = pd.read_excel("data_final/imd_all_la.xlsx", sheet_name="IMD")
    idaci = pd.read_excel("data_final/imd_all_la.xlsx", sheet_name="IDACI")
    imd.columns = imd.columns.str.strip()
    idaci.columns = idaci.columns.str.strip()
    lsoa = pd.read_excel("data_final/imd_scores_all.xlsx", sheet_name="IoD2025 Scores")
    cyp = (
        lsoa.groupby(lad_code)["Children and Young People Sub-domain Score"]
        .mean()
        .rename("cyp_avg_score")
        .round(4)
    )
    scores = _lad_score_lookup(imd, idaci, cyp)

    direct = wide.merge(
        scores[
            ["name_key", "imd_avg_score", "idaci_avg_score", "cyp_avg_score"]
        ],
        on="name_key",
        how="left",
    )
    direct_hit = direct["imd_avg_score"].notna()
    direct.loc[direct_hit, "match_method"] = "direct_lad"

    district_to_county = {
        _normalise_area_name(district): county
        for county, districts in _COUNTY_DISTRICTS.items()
        for district in districts
    }
    scores["county"] = scores["name_key"].map(district_to_county)
    county_means = (
        scores.dropna(subset=["county"])
        .groupby("county", as_index=False)[
            ["imd_avg_score", "idaci_avg_score", "cyp_avg_score"]
        ]
        .mean()
        .rename(columns={"county": "region_name"})
    )

    missing = direct.loc[~direct_hit, ["regional_breakdown", "region_name", "reference_date", *_GRANTNAV_INDICATORS]].copy()
    missing = missing.merge(county_means, on="region_name", how="left")
    missing["match_method"] = "county_district_mean"

    wide = pd.concat(
        [
            direct.loc[
                direct_hit,
                [
                    "regional_breakdown",
                    "region_name",
                    "reference_date",
                    "match_method",
                    *_GRANTNAV_INDICATORS,
                    "idaci_avg_score",
                    "imd_avg_score",
                    "cyp_avg_score",
                ],
            ],
            missing[
                [
                    "regional_breakdown",
                    "region_name",
                    "reference_date",
                    "match_method",
                    *_GRANTNAV_INDICATORS,
                    "idaci_avg_score",
                    "imd_avg_score",
                    "cyp_avg_score",
                ]
            ],
        ],
        ignore_index=True,
    )
    if wide[["imd_avg_score", "idaci_avg_score", "cyp_avg_score"]].isna().any().any():
        unmatched = wide.loc[wide["imd_avg_score"].isna(), "region_name"].tolist()
        raise ValueError(f"GrantNav UTLAs with no IMD match: {unmatched}")

    long = wide.melt(
        id_vars=["regional_breakdown", "region_name", "reference_date"],
        value_vars=[
            *_GRANTNAV_INDICATORS,
            "idaci_avg_score",
            "imd_avg_score",
            "cyp_avg_score",
        ],
        var_name="indicator",
        value_name="value",
    )
    long["source"] = "GrantNav360"
    long["value"] = pd.to_numeric(long["value"], errors="coerce")

    if save_csv is not None:
        Path(save_csv).parent.mkdir(parents=True, exist_ok=True)
        long.to_csv(save_csv, index=False)

    if save_xlsx is not None:
        metadata = pd.DataFrame(
            _DEPRIVATION_METADATA, columns=["Field", "Definition / source"]
        )
        with pd.ExcelWriter(save_xlsx, engine="openpyxl") as writer:
            wide.sort_values("region_name").to_excel(
                writer, sheet_name="deprivation_analysis_data", index=False
            )
            metadata.to_excel(writer, sheet_name="metadata", index=False)

    return long


##### FOI SAMPLE DESCRIPTIVES


def _foi_expenditure_total(data, questions, scopes):
    """Sum expenditure amounts for the given questions and scopes.

    Blank amounts are skipped. Scope is matched on in/out as stored.
    """
    rows = data[
        data["unit"].astype(str).str.strip().eq("expenditure")
        & data["question"].isin(questions)
        & data["scope"].astype(str).str.strip().str.lower().isin(scopes)
    ]
    numeric = pd.to_numeric(
        rows["value"].astype(str).str.strip().replace({"": pd.NA}),
        errors="coerce",
    )
    return numeric.sum(min_count=1)


def _foi_la_expenditure(data, questions, scopes):
    """One expenditure total per local authority. Authorities with no figure are left out."""
    rows = data[
        data["unit"].astype(str).str.strip().eq("expenditure")
        & data["question"].isin(questions)
        & data["scope"].astype(str).str.strip().str.lower().isin(scopes)
    ].copy()
    rows["_value"] = pd.to_numeric(
        rows["value"].astype(str).str.strip().replace({"": pd.NA}),
        errors="coerce",
    )
    return rows.groupby("LA")["_value"].sum(min_count=1).dropna()


_FOI_GROSS_INDICATORS = {
    "V9": (("Q1",), {"in", "out"}, "Total gross expenditure for Q1"),
    "V10": (("Q2",), {"in", "out"}, "Total gross expenditure for Q2"),
    "V11": (("Q1", "Q2"), {"in", "out"}, "Total gross expenditure for Q1 and Q2"),
    "V12": (("Q1",), {"in"}, "Total gross expenditure in scope for Q1"),
    "V13": (("Q2",), {"in"}, "Total gross expenditure in scope for Q2"),
    "V14": (("Q1", "Q2"), {"in"}, "Total gross expenditure in scope for Q1 and Q2"),
    "V15": (("Q1",), {"out"}, "Total gross expenditure out of scope for Q1"),
    "V16": (("Q2",), {"out"}, "Total gross expenditure out of scope for Q2"),
    "V17": (("Q1", "Q2"), {"out"}, "Total gross expenditure out of scope for Q1 and Q2"),
}


def _foi_gross_by_la(data, indicator):
    """Local-authority totals for one gross-expenditure indicator, V9-V17."""
    if indicator not in _FOI_GROSS_INDICATORS:
        raise ValueError("indicator must be one of: " + ", ".join(_FOI_GROSS_INDICATORS) + ".")
    questions, scopes, label = _FOI_GROSS_INDICATORS[indicator]
    values = _foi_la_expenditure(data, questions, scopes)
    if values.empty:
        raise ValueError(f"No local-authority totals found for {indicator}.")
    return values, label


def _foi_share(part, whole):
    """Part as a percentage of whole. Blank when the denominator is missing or zero."""
    if pd.isna(part) or pd.isna(whole) or whole == 0:
        return pd.NA
    return 100 * part / whole


def sample_FOI_descriptives(
    data,
    table_mode="ON",
    indicator_family="NUMBER_LAS",
    type="mixed",
    indicator=None,
    html="OFF",
):
    """Print FOI sample descriptives for one indicator family.

    table_mode ON prints the whole family for the chosen type, and indicator
    must be None. OFF prints the chosen indicator. NUMBER_LAS type is mixed,
    count, or prevalence. TOTAL_EXPENDITURE type is mixed, gross_exp, or
    pct_exp. For expenditure mixed with table_mode OFF, indicator is a pair
    of one gross code and one percentage code. html ON opens that table in
    the browser.
    """
    measure = type
    family_types = {
        "NUMBER_LAS": ("mixed", "count", "prevalence"),
        "TOTAL_EXPENDITURE": ("mixed", "gross_exp", "pct_exp"),
    }
    if table_mode not in {"ON", "OFF"}:
        raise ValueError("table_mode must be ON or OFF.")
    if html not in {"ON", "OFF"}:
        raise ValueError("html must be ON or OFF.")
    if indicator_family not in family_types:
        raise ValueError("indicator_family must be NUMBER_LAS or TOTAL_EXPENDITURE.")
    if measure not in family_types[indicator_family]:
        allowed = ", ".join(family_types[indicator_family])
        raise ValueError(f"When indicator_family is {indicator_family}, type must be one of: {allowed}.")
    if table_mode == "ON" and indicator is not None:
        raise ValueError("When table_mode is ON, indicator must be None.")
    if table_mode == "OFF" and indicator is None:
        raise ValueError("When table_mode is OFF, choose an indicator.")

    count_labels = {
        "V1": "Number of unique LAs in the sample",
        "V2": "Number of LAs that answered Q1 with a total expenditure figure, including zero",
        "V3": "Number of LAs that answered Q2 with a total expenditure figure, including zero",
        "V4": "Number of LAs that answered both Q1 and Q2 with total expenditure figures, including zero",
        "V5": "Number of LAs that answered N/A for Q1",
        "V6": "Number of LAs that answered N/A for Q2",
        "V7": "Number of LAs that answered N/A for both Q1 and Q2",
        "V8": "Number of LAs that answered zero expenditure (whole nil return) for Q1",
    }
    prevalence_labels = {
        "V1": "Prevalence of unique LAs in the sample",
        "V2": "Prevalence of unique LAs answering Q1 with a total expenditure figure, including zero",
        "V3": "Prevalence of unique LAs that answered Q2 with a total expenditure figure, including zero",
        "V4": "Prevalence of unique LAs that answered both Q1 and Q2 with total expenditure figures, including zero",
        "V5": "Prevalence of unique LAs that answered N/A for Q1",
        "V6": "Prevalence of unique LAs that answered N/A for Q2",
        "V7": "Prevalence of unique LAs that answered N/A for both Q1 and Q2",
        "V8": "Prevalence of unique LAs that answered zero expenditure (whole nil return) for Q1",
    }
    gross_labels = {
        "V9": "Total gross expenditure for Q1",
        "V10": "Total gross expenditure for Q2",
        "V11": "Total gross expenditure for Q1 and Q2",
        "V12": "Total gross expenditure in scope for Q1",
        "V13": "Total gross expenditure in scope for Q2",
        "V14": "Total gross expenditure in scope for Q1 and Q2",
        "V15": "Total gross expenditure out of scope for Q1",
        "V16": "Total gross expenditure out of scope for Q2",
        "V17": "Total gross expenditure out of scope for Q1 and Q2",
    }
    pct_labels = {
        "V18": "Percentage of total Q1 gross expenditure that is in scope",
        "V19": "Percentage of total Q2 gross expenditure that is in scope",
        "V20": "Percentage of total gross expenditure that is in scope",
        "V21": "Percentage of total Q1 gross expenditure that is out of scope",
        "V22": "Percentage of total Q2 gross expenditure that is out of scope",
        "V23": "Percentage of total gross expenditure that is out of scope",
    }
    la_codes = list(count_labels)
    gross_codes = list(gross_labels)
    pct_codes = list(pct_labels)

    if indicator_family == "NUMBER_LAS":
        if table_mode == "OFF" and indicator not in count_labels:
            raise ValueError("indicator must be one of: " + ", ".join(la_codes) + ".")
        chosen_codes = la_codes if table_mode == "ON" else [indicator]
        kinds = {"count", "prevalence"} if measure == "mixed" else {measure}
    elif measure == "gross_exp":
        if table_mode == "OFF" and indicator not in gross_labels:
            raise ValueError("indicator must be one of: " + ", ".join(gross_codes) + ".")
        chosen_codes = gross_codes if table_mode == "ON" else [indicator]
        kinds = {"gross_exp"}
    elif measure == "pct_exp":
        if table_mode == "OFF" and indicator not in pct_labels:
            raise ValueError("indicator must be one of: " + ", ".join(pct_codes) + ".")
        chosen_codes = pct_codes if table_mode == "ON" else [indicator]
        kinds = {"pct_exp"}
    else:
        kinds = {"gross_exp", "pct_exp"}
        if table_mode == "ON":
            chosen_codes = gross_codes + pct_codes
        else:
            pair = indicator
            if (
                not isinstance(pair, (tuple, list))
                or len(pair) != 2
                or not (
                    (pair[0] in gross_labels and pair[1] in pct_labels)
                    or (pair[0] in pct_labels and pair[1] in gross_labels)
                )
            ):
                raise ValueError(
                    "When type is mixed and table_mode is OFF, indicator must be a pair "
                    "of one code from V9-V17 and one code from V18-V23."
                )
            chosen_codes = list(pair)

    numeric = pd.to_numeric(
        data["value"].astype(str).str.strip().replace({"": pd.NA}),
        errors="coerce",
    )
    question_totals = (
        data.assign(_value=numeric)
        .groupby(["LA", "question"], as_index=False)["_value"]
        .sum(min_count=1)
    )

    def figure_las(question):
        rows = question_totals[
            question_totals["question"].eq(question) & question_totals["_value"].notna()
        ]
        return set(rows["LA"])

    def coded_las(question, code):
        rows = data[data["question"].eq(question) & data["code"].eq(code)]
        return set(rows["LA"])

    q1 = figure_las("Q1")
    q2 = figure_las("Q2")
    na_q1 = coded_las("Q1", "N/A FIGURE")
    na_q2 = coded_las("Q2", "N/A FIGURE")
    counts = {
        "V1": data["LA"].nunique(),
        "V2": len(q1),
        "V3": len(q2),
        "V4": len(q1 & q2),
        "V5": len(na_q1),
        "V6": len(na_q2),
        "V7": len(na_q1 & na_q2),
        "V8": len(coded_las("Q1", "ZERO EXPENDITURE (WHOLE NIL RETURN)")),
    }
    either = {"in", "out"}
    gross = {
        "V9": _foi_expenditure_total(data, ("Q1",), either),
        "V10": _foi_expenditure_total(data, ("Q2",), either),
        "V11": _foi_expenditure_total(data, ("Q1", "Q2"), either),
        "V12": _foi_expenditure_total(data, ("Q1",), {"in"}),
        "V13": _foi_expenditure_total(data, ("Q2",), {"in"}),
        "V14": _foi_expenditure_total(data, ("Q1", "Q2"), {"in"}),
        "V15": _foi_expenditure_total(data, ("Q1",), {"out"}),
        "V16": _foi_expenditure_total(data, ("Q2",), {"out"}),
        "V17": _foi_expenditure_total(data, ("Q1", "Q2"), {"out"}),
    }
    shares = {
        "V18": _foi_share(gross["V12"], gross["V9"]),
        "V19": _foi_share(gross["V13"], gross["V10"]),
        "V20": _foi_share(gross["V14"], gross["V11"]),
        "V21": _foi_share(gross["V15"], gross["V9"]),
        "V22": _foi_share(gross["V16"], gross["V10"]),
        "V23": _foi_share(gross["V17"], gross["V11"]),
    }

    def shown(value, kind):
        if pd.isna(value):
            return ""
        if kind in {"prevalence", "pct_exp"}:
            return f"{value:.1f}%"
        if kind == "gross_exp":
            return f"£{value:,.2f}"
        return int(value)

    catalogue = []
    for code in la_codes:
        catalogue.append(("count", code, count_labels[code], counts[code]))
        share = pd.NA if counts["V1"] == 0 else 100 * counts[code] / counts["V1"]
        catalogue.append(("prevalence", code, prevalence_labels[code], share))
    for code in gross_codes:
        catalogue.append(("gross_exp", code, gross_labels[code], gross[code]))
    for code in pct_codes:
        catalogue.append(("pct_exp", code, pct_labels[code], shares[code]))

    order = {code: index for index, code in enumerate(chosen_codes)}
    picked = [row for row in catalogue if row[0] in kinds and row[1] in order]
    picked.sort(key=lambda row: (order[row[1]], 0 if row[0] in {"count", "gross_exp"} else 1))
    table = pd.DataFrame({
        "Variable": [f"{code}. {label}" for _, code, label, _ in picked],
        "Value": [shown(value, kind) for kind, _, _, value in picked],
    })
    if table_mode == "ON":
        print(table.to_string(index=False))
    else:
        for variable, value in zip(table["Variable"], table["Value"]):
            print(f"{variable}: {value}")
    if html == "ON":
        import webbrowser
        page = Path("outputs") / "foi_descriptives.html"
        page.parent.mkdir(exist_ok=True)
        page.write_text(
            "<!DOCTYPE html><meta charset='utf-8'>"
            "<style>body{font-family:sans-serif}table{border-collapse:collapse}"
            "td,th{border:1px solid #ccc;padding:6px 10px;text-align:left}</style>"
            + table.to_html(index=False),
            encoding="utf-8",
        )
        webbrowser.open(page.resolve().as_uri())
    return table


def _foi_prevalence_counts(data):
    """Local-authority counts behind V1-V8. A question total ignores blank amounts."""
    numeric = pd.to_numeric(
        data["value"].astype(str).str.strip().replace({"": pd.NA}),
        errors="coerce",
    )
    totals = (
        data.assign(_value=numeric)
        .groupby(["LA", "question"], as_index=False)["_value"]
        .sum(min_count=1)
    )

    def figure_las(question):
        rows = totals[totals["question"].eq(question) & totals["_value"].notna()]
        return set(rows["LA"])

    def coded_las(question, code):
        rows = data[data["question"].eq(question) & data["code"].eq(code)]
        return set(rows["LA"])

    q1 = figure_las("Q1")
    q2 = figure_las("Q2")
    na_q1 = coded_las("Q1", "N/A FIGURE")
    na_q2 = coded_las("Q2", "N/A FIGURE")
    return {
        "V1": data["LA"].nunique(),
        "V2": len(q1),
        "V3": len(q2),
        "V4": len(q1 & q2),
        "V5": len(na_q1),
        "V6": len(na_q2),
        "V7": len(na_q1 & na_q2),
        "V8": len(coded_las("Q1", "ZERO EXPENDITURE (WHOLE NIL RETURN)")),
    }


def _foi_heading(ax, heading, y, bold_prefix=None):
    """Centered donut heading. bold_prefix is the leading words drawn in bold."""
    ink = "#222222"
    wrapped = heading if "\n" in heading else textwrap.fill(heading, 52)
    if not bold_prefix or "\n" in wrapped or not wrapped.startswith(bold_prefix):
        ax.text(0.5, y, wrapped, transform=ax.transAxes, ha="center", va="top", fontsize=13, color=ink, clip_on=False)
        return
    rest = wrapped[len(bold_prefix):]
    bold = FontProperties(family="Roboto Slab", weight="bold", size=13)
    regular = FontProperties(family="Roboto Slab", size=13)
    renderer = ax.figure.canvas.get_renderer()
    bold_width, _, _ = renderer.get_text_width_height_descent(bold_prefix, bold, False)
    rest_width, _, _ = renderer.get_text_width_height_descent(rest, regular, False)
    width = ax.get_position().width * ax.figure.get_figwidth() * ax.figure.dpi
    x = 0.5 - (bold_width + rest_width) / width / 2
    ax.text(x, y, bold_prefix, transform=ax.transAxes, ha="left", va="top", color=ink, clip_on=False, fontproperties=bold)
    ax.text(x + bold_width / width, y, rest, transform=ax.transAxes, ha="left", va="top", color=ink, clip_on=False, fontproperties=regular)


def _plot_foi_donut(ax, share, detail, heading, other_label=None, bold_prefix=None, heading_y=1.22, place_heading=True):
    """One donut. The share sits on the red slice when the slice is wide enough."""
    red = "#d93649"
    pink = "#f6b8c0"
    ink = "#222222"
    if share <= 0:
        sizes, colours = [1], [pink]
    elif share >= 100:
        sizes, colours = [1], [red]
    else:
        sizes, colours = [share, 100 - share], [red, pink]
    wedges, _ = ax.pie(
        sizes,
        colors=colours,
        startangle=90,
        counterclock=False,
        wedgeprops={"width": 0.42, "edgecolor": "white", "linewidth": 2},
    )
    ax.set_aspect("equal")
    wedge = wedges[0]
    mid = np.deg2rad((wedge.theta1 + wedge.theta2) / 2)
    # A slice of at least half the ring covers the bottom of the band.
    # If that point sits on the pink edge, move the label back into the red.
    on_slice = share >= 50
    long_detail = detail.startswith("£") or detail.endswith("[in scope]")
    if on_slice:
        theta1, theta2 = wedge.theta1, wedge.theta2
        gap_to_pink = -90 - theta1
        if theta1 <= -90 <= theta2 and gap_to_pink >= 40:
            label_x, label_y = 0, -0.79
        else:
            angle = np.deg2rad(theta1 + 55)
            label_x, label_y = 0.79 * np.cos(angle), 0.79 * np.sin(angle)
        ax.text(label_x, label_y, f"{share:.1f}%", ha="center", va="center", fontsize=13, fontweight="bold", color="#ffffff")
        ax.text(0, -1.32, detail, ha="center", va="top", fontsize=11, fontweight="bold" if "in scope" in detail else "normal", color=ink, clip_on=False)
    else:
        radius = 1.38
        x = radius * np.cos(mid)
        y = radius * np.sin(mid)
        ax.text(x, y + (0.02 if long_detail else 0.08), f"{share:.1f}%", ha="center", va="center", fontsize=17, fontweight="bold", color=ink)
        if long_detail:
            ax.text(0, -1.32, detail, ha="center", va="top", fontsize=11, fontweight="bold" if "in scope" in detail else "normal", color=ink, clip_on=False)
        else:
            ax.text(x, y - 0.16, detail, ha="center", va="center", fontsize=11, color=ink)
    if other_label and len(wedges) > 1:
        other = wedges[1]
        omid = np.deg2rad((other.theta1 + other.theta2) / 2)
        ox = 1.18 * np.cos(omid)
        oy = 1.18 * np.sin(omid)
        ha = "right" if ox < 0 else "left"
        ox += -0.12 if ox < 0 else 0.12
        ax.text(ox, oy + 0.1, f"{100 - share:.1f}%", ha=ha, va="center", fontsize=13, fontweight="bold", color=ink, clip_on=False)
        ax.text(ox, oy - 0.12, other_label, ha=ha, va="center", fontsize=10, color=ink, clip_on=False)
    if place_heading and other_label:
        _foi_heading(ax, heading, heading_y, bold_prefix)
    elif place_heading:
        ax.set_title(heading if "\n" in heading else textwrap.fill(heading, 52), color=ink, fontsize=13, pad=8)
    if other_label:
        ax.set_ylim(-1.52, 1.38)
    elif place_heading:
        ax.set_ylim(-1.55, 1.45)
    ax.set_xlim(-2.15, 2.15)
    ax.set_axis_off()


def _code_counts(data, scope, question):
    """Frequency of each code for one question and scope.

    A cell that lists several codes, separated by a semicolon, counts
    once for each of them. NEET support is counted with Careers support.
    A "not coded" label is left out.
    """
    rows = data[
        data["question"].astype(str).str.strip().eq(question)
        & data["scope"].astype(str).str.strip().str.lower().eq(scope)
    ]
    codes = rows["code"].astype(str).str.strip()
    codes = codes[codes.ne("")].str.split(";").explode().str.strip()
    codes = codes[codes.ne("") & ~codes.str.lower().str.startswith("not coded")]
    codes = codes.replace({"NEET support": "Careers support"})
    if codes.empty:
        raise ValueError(f"No coded rows for {question} with scope {scope}.")
    return codes.value_counts()


def plot_codes(
    data,
    scope,
    question,
    ascending=False,
    brand_font="OFF",
    title=None,
    save_path=None,
    transparent="F",
    rank_style="T",
):
    """Horizontal bars of FOI code frequency.

    scope is "in" or "out". question is "Q1", "Q2", or "both". "both"
    draws Q1 and Q2 side by side. ascending=False puts the highest
    frequency at the top. brand_font OFF uses Arial; ON uses Roboto Slab.
    rank_style T colours the three most frequent codes in the darker
    shade and the rest in a lighter shade. Q1 uses #FF4D5E. Q2 uses
    #B7D6EA.
    """
    if scope not in {"in", "out"}:
        raise ValueError("scope must be 'in' or 'out'.")
    if question not in {"Q1", "Q2", "both"}:
        raise ValueError("question must be 'Q1', 'Q2', or 'both'.")
    if brand_font not in {"ON", "OFF"}:
        raise ValueError("brand_font must be ON or OFF.")
    if transparent not in {"T", "F"}:
        raise ValueError("transparent must be 'T' or 'F'.")
    if rank_style not in {"T", "F"}:
        raise ValueError("rank_style must be 'T' or 'F'.")
    missing = {"code", "scope", "question"} - set(data.columns)
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")

    questions = ["Q1", "Q2"] if question == "both" else [question]
    counts = {q: _code_counts(data, scope, q) for q in questions}
    shades = {
        "Q1": ("#FF4D5E", _lighter_shade("#FF4D5E")),
        "Q2": (_darker_shade("#B7D6EA"), "#B7D6EA"),
    }
    plt.rcParams["font.family"] = (
        ["Roboto Slab", "serif"] if brand_font == "ON" else ["Arial", "sans-serif"]
    )
    ink = "#222222"
    height = max(4.6, max(len(series) for series in counts.values()) * 0.42)
    if question == "both":
        fig, axes = plt.subplots(1, 2, figsize=(16, height))
    else:
        fig, ax = plt.subplots(figsize=(10, height))
        axes = [ax]
    fig.patch.set_facecolor("#ffffff")
    for ax, q in zip(axes, questions):
        _draw_code_bars(ax, counts[q], shades[q], ascending, rank_style, ink)
        ax.set_title(q if question == "both" else (title or _code_chart_title(scope, q)), color=ink, fontsize=14, fontweight="bold", pad=12)
    if question == "both":
        fig.suptitle(title or _code_chart_title(scope, "Q1 and Q2"), color=ink, fontsize=16, fontweight="bold")
        fig.tight_layout(rect=(0, 0, 1, 0.94))
    _save_figure_to_outputs(fig, save_path, transparent)
    return None


def _code_chart_title(scope, question):
    place = "in scope" if scope == "in" else "out of scope"
    return f"{question} codes {place}"


def _darker_shade(color):
    """One step darker than color, same hue."""
    red, green, blue = to_rgb(color)
    hue, light, sat = colorsys.rgb_to_hls(red, green, blue)
    return to_hex(colorsys.hls_to_rgb(hue, light * 0.62, sat))


def _draw_code_bars(ax, counts, shades, ascending, rank_style, ink):
    ordered = counts.sort_values(ascending=ascending).iloc[::-1]
    dark, light = shades
    top = set(counts.sort_values(ascending=False).head(3).index)
    colors = [
        dark if rank_style == "F" or name in top else light
        for name in ordered.index
    ]
    ax.set_facecolor("#ffffff")
    ax.barh(
        [_format_region_axis_label(name, 36) for name in ordered.index],
        ordered.to_numpy(),
        color=colors,
        height=0.68,
    )
    ax.set_xlabel("Frequency", color=ink, fontsize=11)
    ax.tick_params(axis="both", colors=ink, labelsize=9)
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#000000")
    ax.spines["bottom"].set_color("#000000")
    ax.set_xlim(0, float(ordered.max()) * 1.12)
    ax.set_axisbelow(True)


def plotting_FOI_donut(data, canva=None, indicator=None, title=None, save_path=None):
    """Donut of one prevalence or expenditure percentage, or a three-donut canvas.

    canva "response_behaviour" draws V2, V3, and V4. canva "in_scope" draws
    V18, V19, and V20. canva None draws the chosen prevalence (V1-V8) or
    expenditure percentage (V18-V23).
    """
    prevalence_headings = {
        "V1": "Prevalence of unique LAs in the sample",
        "V2": "Answered Q1 with a total expenditure figure, including zero",
        "V3": "Answered Q2 with a total expenditure figure, including zero",
        "V4": "Answered both questions with a total expenditure figure, including zero",
        "V5": "Answered N/A for Q1",
        "V6": "Answered N/A for Q2",
        "V7": "Answered N/A for both Q1 and Q2",
        "V8": "Answered zero expenditure (whole nil return) for Q1",
    }
    if canva not in {None, "response_behaviour", "in_scope"}:
        raise ValueError("canva must be response_behaviour, in_scope, or None.")

    plt.rcParams["font.family"] = ["Roboto Slab", "serif"]

    def three_donuts(panels, canvas_title, other_label, heading_y=1.22):
        fig = plt.figure(figsize=(14.6, 8.0))
        fig.patch.set_facecolor("#ffffff")
        fig.subplots_adjust(top=0.84, bottom=0.02)
        grid = fig.add_gridspec(2, 4, hspace=0.22, wspace=0.08)
        axes = (
            fig.add_subplot(grid[0, 0:2]),
            fig.add_subplot(grid[0, 2:4]),
            fig.add_subplot(grid[1, 1:3]),
        )
        pending = []
        for ax, (share, detail, heading, bold_prefix) in zip(axes, panels):
            _plot_foi_donut(
                ax, share, detail, heading,
                other_label=other_label, bold_prefix=bold_prefix, heading_y=heading_y,
                place_heading=False,
            )
            pending.append((ax, heading, bold_prefix))
        fig.canvas.draw()
        for ax, heading, bold_prefix in pending:
            _foi_heading(ax, heading, heading_y, bold_prefix)
        fig.suptitle(
            textwrap.fill(title or canvas_title, 90),
            color="#222222",
            fontsize=16,
            fontweight="bold",
            y=1.0,
        )
        _save_figure_to_outputs(fig, save_path)
        return None

    if canva == "response_behaviour":
        counts = _foi_prevalence_counts(data)
        sample = counts["V1"]

        def prevalence(code):
            if sample == 0:
                return 0.0
            return 100 * counts[code] / sample

        return three_donuts(
            (
                (prevalence("V2"), f"{counts['V2']} of {sample} LAs", "Answered Q1: Universal Services for CYP", "Answered Q1"),
                (prevalence("V3"), f"{counts['V3']} of {sample} LAs", "Answered Q2: HAF Programme", "Answered Q2"),
                (prevalence("V4"), f"{counts['V4']} of {sample} LAs", "Answered both questions\nwith a figure, including £0", None),
            ),
            "Percentage of LAs answering the FOI questions with a figure, including £0",
            "other, like N/A",
        )

    pct_parts = {
        "V18": (("Q1",), {"in"}, ("Q1",), {"in", "out"}),
        "V19": (("Q2",), {"in"}, ("Q2",), {"in", "out"}),
        "V20": (("Q1", "Q2"), {"in"}, ("Q1", "Q2"), {"in", "out"}),
        "V21": (("Q1",), {"out"}, ("Q1",), {"in", "out"}),
        "V22": (("Q2",), {"out"}, ("Q2",), {"in", "out"}),
        "V23": (("Q1", "Q2"), {"out"}, ("Q1", "Q2"), {"in", "out"}),
    }
    pct_headings = {
        "V18": "Percentage of total Q1 gross expenditure that is in scope",
        "V19": "Percentage of total Q2 gross expenditure that is in scope",
        "V20": "Percentage of total gross expenditure that is in scope",
        "V21": "Percentage of total Q1 gross expenditure that is out of scope",
        "V22": "Percentage of total Q2 gross expenditure that is out of scope",
        "V23": "Percentage of total gross expenditure that is out of scope",
    }

    def expenditure_panel(code, heading, bold_prefix=None, compact=False):
        part_q, part_s, whole_q, whole_s = pct_parts[code]
        part = _foi_expenditure_total(data, part_q, part_s)
        whole = _foi_expenditure_total(data, whole_q, whole_s)
        share = _foi_share(part, whole)
        part = 0 if pd.isna(part) else part
        whole = 0 if pd.isna(whole) else whole
        if compact:
            detail = f"£{part / 1_000_000:.1f}M in scope out of £{whole / 1_000_000:.0f}M"
        else:
            detail = f"£{part:,.0f} of £{whole:,.0f}"
        return (0.0 if pd.isna(share) else float(share), detail, heading, bold_prefix)

    if canva == "in_scope":
        return three_donuts(
            (
                expenditure_panel("V18", "Q1: Universal Services for CYP", "Q1", compact=True),
                expenditure_panel("V19", "Q2: HAF Programme", "Q2", compact=True),
                expenditure_panel("V20", "Total gross expenditure", compact=True),
            ),
            "Percentage of total gross expenditure in scope",
            "out of scope",
            heading_y=1.12,
        )

    if indicator in prevalence_headings:
        counts = _foi_prevalence_counts(data)
        sample = counts["V1"]
        share = 0.0 if sample == 0 else 100 * counts[indicator] / sample
        detail = f"{counts[indicator]} of {sample} LAs"
        heading = title or prevalence_headings[indicator]
    elif indicator in pct_parts:
        share, detail, heading, _prefix = expenditure_panel(indicator, title or pct_headings[indicator])
    else:
        raise ValueError("When canva is None, indicator must be a prevalence V1-V8 or a percentage V18-V23.")

    fig, ax = plt.subplots(figsize=(6.5, 6))
    fig.patch.set_facecolor("#ffffff")
    _plot_foi_donut(ax, share, detail, heading)
    _save_figure_to_outputs(fig, save_path)
    return None

