from __future__ import annotations
import argparse
import glob
import json
import os
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Mapping,
    MutableMapping,
    Optional,
    Sequence,
    Tuple,
    Union,
)
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd  # Import pandas

try:
    from tabulate import tabulate
    TABULATE_AVAILABLE = True
except ImportError:
    TABULATE_AVAILABLE = False

# --- START MOCK DATA (for debugging) ---
# Replaced the original import with mock string data to make the script runnable.
# from utils.data_utils.autoguiv2.classify_region_types.classify_functional_regions import (
#     EXTRA_LEAF_TYPES,
#     TAXONOMY,
# )

# Mock TAXONOMY
from utils.data_utils.autoguiv2.classify_region_types.classify_functional_regions import (
    EXTRA_LEAF_TYPES,
    TAXONOMY,
) 
# --- END MOCK DATA ---

UNKNOWN_PARENT = "Other / Unknown"
ROOT_LABEL = '<b>3710 Functional Regions</b>'


@dataclass(frozen=True)
class TaxonomyInfo:
    """Convenience container for taxonomy lookups."""

    parents: Mapping[str, str]
    descriptions: Mapping[str, str]
    parent_order: Sequence[str]
    children_order: Mapping[str, Sequence[str]]


def parse_taxonomy(include_extra_leaf_types: bool = False) -> TaxonomyInfo:
    """Parse the TAXONOMY constant into structured mappings."""
    # This replacement is a bit different from the original,
    # as we are injecting JSON key-value pairs, not just a string.
    taxonomy_str = TAXONOMY
    if include_extra_leaf_types and EXTRA_LEAF_TYPES.strip():
        # Find the last '}' in the main taxonomy
        last_brace = taxonomy_str.rfind("}")
        if last_brace != -1:
            # Find the last '}' in the *last* child dictionary
            last_child_brace = taxonomy_str.rfind("}", 0, last_brace)
            if last_child_brace != -1:
                # Inject the extra leaf types as new key-value pairs
                # into the last child dictionary.
                injection_point = last_child_brace
                # We need to add a comma if the last child dict wasn't empty
                comma = ""
                # --- FIX: Corrected the slice index ---
                # Changed `injection_point-1_token_limit` to `injection_point - 1`
                prev_char = taxonomy_str[injection_point - 1 : injection_point].strip()
                if prev_char and prev_char != '{':
                    comma = ","
                
                taxonomy_str = (
                    taxonomy_str[:injection_point]
                    + comma
                    + EXTRA_LEAF_TYPES
                    + taxonomy_str[injection_point:]
                )
    else:
        taxonomy_str = taxonomy_str.replace("{extra_leaf_types}", "")
    try:
        taxonomy_raw: Dict[str, Dict[str, str]] = json.loads(taxonomy_str)
    except json.JSONDecodeError as exc:
        print(f"Failed to parse taxonomy string:\n{taxonomy_str}")
        raise RuntimeError("Failed to parse TAXONOMY as JSON") from exc
        
    parents: Dict[str, str] = {}
    descriptions: Dict[str, str] = {}
    parent_order: List[str] = []
    children_order: Dict[str, List[str]] = {}
    
    for parent, children in taxonomy_raw.items():
        parent_order.append(parent)
        ordered_children: List[str] = []
        for child, description in children.items():
            parents[child] = parent
            descriptions[child] = description
            ordered_children.append(child)
        children_order[parent] = ordered_children
        
    return TaxonomyInfo(
        parents=parents,
        descriptions=descriptions,
        parent_order=parent_order,
        children_order=children_order,
    )


def load_json(path: str) -> Dict[str, Any]:
    """Load JSON safely."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def iter_region_types_from_payload(
    payload: Mapping[str, Any],
) -> Iterable[str]:
    """Yield raw region type strings from a single classification payload entry."""
    region_types = payload.get("region_types")
    if not isinstance(region_types, Mapping):
        return []
    extracted: List[str] = []
    for node_payload in region_types.values():
        if isinstance(node_payload, str):
            continue
        # Check both 'type' and 'label' fields
        region_type = node_payload.get("type") or node_payload.get("label")
        # Handle None or empty string
        if not region_type:
            continue
        # Start cleaning
        region_type = (
            region_type.replace("Other", "")
            .replace("other", "")
            .strip(' ,:;.?"')
        )
        # Normalize common variations
        if "input field" in region_type.lower():
            region_type = "Input field"
        elif " link" in region_type.lower():
            region_type = "Link"
        elif " image" in region_type.lower():
            region_type = "Image"
        elif " button" in region_type.lower():
            region_type = "Button"
        if not region_type:
            continue
        cleaned = str(region_type).replace("Type:", "").strip()
        if cleaned:
            extracted.append(cleaned)
    return extracted


def aggregate_region_types(
    inputs: Sequence[str],
    taxonomy: TaxonomyInfo,
) -> Tuple[Counter, Counter, int, MutableMapping[str, int]]:
    """Aggregate counts of region types across multiple JSON result files.
    Returns:
        parent_counts: counts per top-level taxonomy category (inner ring).
        leaf_counts: counts per leaf type (outer ring).
        total: total number of nodes considered.
        diagnostics: additional counters (e.g., unknown labels).
    """
    parent_counts: Counter = Counter()
    leaf_counts: Counter = Counter()
    diagnostics: MutableMapping[str, int] = defaultdict(int)
    total = 0
    for path in inputs:
        if not os.path.exists(path):
            diagnostics[f"file_not_found_{path}"] += 1
            continue
        try:
            payload = load_json(path)
        except Exception as e:
            diagnostics[f"json_load_error_{path}"] += 1
            print(f"Error loading JSON from {path}: {e}")
            continue
        results = payload.get("results")
        if not isinstance(results, Mapping):
            diagnostics["missing_results"] += 1
            continue
        for entry in results.values():
            if not isinstance(entry, Mapping):
                continue
            for region_type in iter_region_types_from_payload(entry):
                parent = taxonomy.parents.get(region_type, UNKNOWN_PARENT)

                leaf_counts[region_type] += 1
                parent_counts[parent] += 1
                total += 1
    if total == 0:
        print("Warning: No region types were found in the provided inputs.")

    return parent_counts, leaf_counts, total, diagnostics


def save_region_type_stats(
    parent_counts: Counter,
    leaf_counts: Counter,
    taxonomy: TaxonomyInfo,
    total: int,
    output_path: str,
) -> None:
    """Save region type statistics (counts and proportions) to a JSON file.
    
    Args:
        parent_counts: Counts per top-level taxonomy category
        leaf_counts: Counts per leaf type
        taxonomy: Taxonomy information for organizing data
        total: Total number of nodes
        output_path: Path to save the JSON file
    """
    stats = {
        "summary": {
            "total_regions": total,
            "total_parent_categories": len(parent_counts),
            "total_leaf_types": len(leaf_counts),
        },
        "parent_categories": {},
        "leaf_types_by_parent": {},
        "all_leaf_types": {},
    }
    
    # Calculate proportions for parent categories
    for parent in taxonomy.parent_order:
        count = parent_counts.get(parent, 0)
        if count > 0:
            proportion = count / total if total > 0 else 0.0
            stats["parent_categories"][parent] = {
                "count": count,
                "proportion": proportion,
                "percentage": proportion * 100,
            }
    
    # Handle unknown parent category
    if UNKNOWN_PARENT in parent_counts:
        count = parent_counts[UNKNOWN_PARENT]
        proportion = count / total if total > 0 else 0.0
        stats["parent_categories"][UNKNOWN_PARENT] = {
            "count": count,
            "proportion": proportion,
            "percentage": proportion * 100,
        }
    
    # Organize leaf types by parent
    for parent in taxonomy.parent_order:
        children = taxonomy.children_order.get(parent, ())
        leaf_types = {}
        for child in children:
            count = leaf_counts.get(child, 0)
            if count > 0:
                proportion = count / total if total > 0 else 0.0
                leaf_types[child] = {
                    "count": count,
                    "proportion": proportion,
                    "percentage": proportion * 100,
                    "description": taxonomy.descriptions.get(child, ""),
                }
        if leaf_types:
            stats["leaf_types_by_parent"][parent] = leaf_types
    
    # Handle unknown parent leaf types
    unknown_leaves = {}
    for leaf, count in leaf_counts.items():
        if taxonomy.parents.get(leaf) is None and count > 0:
            proportion = count / total if total > 0 else 0.0
            unknown_leaves[leaf] = {
                "count": count,
                "proportion": proportion,
                "percentage": proportion * 100,
                "description": taxonomy.descriptions.get(leaf, ""),
            }
    if unknown_leaves:
        stats["leaf_types_by_parent"][UNKNOWN_PARENT] = unknown_leaves
    
    # Create a flat list of all leaf types sorted by count
    for leaf, count in leaf_counts.most_common():
        proportion = count / total if total > 0 else 0.0
        parent = taxonomy.parents.get(leaf, UNKNOWN_PARENT)
        stats["all_leaf_types"][leaf] = {
            "count": count,
            "proportion": proportion,
            "percentage": proportion * 100,
            "parent_category": parent,
            "description": taxonomy.descriptions.get(leaf, ""),
        }
    
    # Save to JSON file
    output_dir = os.path.dirname(os.path.abspath(output_path))
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    
    print(f"Saved region type statistics to {os.path.abspath(output_path)}")


def resolve_input_paths(patterns: Union[str, Sequence[str]]) -> List[str]:
    """Expand file path patterns (supports glob wildcards)."""
    if isinstance(patterns, str):
        patterns = [patterns]
    resolved: List[str] = []
    for pattern in patterns:
        matches = sorted(glob.glob(pattern, recursive=True))
        if matches:
            resolved.extend(matches)
            continue
        if os.path.exists(pattern):
            resolved.append(os.path.abspath(pattern))
            continue

        # Don't raise error, just warn and continue
        print(f"Warning: No files match the pattern and path does not exist: {pattern}")
    # Deduplicate while preserving order of first occurrence.
    seen = set()
    unique: List[str] = []
    for path in resolved:
        abs_path = os.path.abspath(path)
        if abs_path in seen:
            continue
        seen.add(abs_path)
        unique.append(abs_path)
    if not unique:
        print("Warning: No input files were resolved from the provided patterns.")
    return unique


def _parse_color(color: str) -> Tuple[float, float, float]:
    """Convert a Plotly color string (hex or rgb) into normalized RGB tuple."""
    color = color.strip()
    if color.startswith("#"):
        color = color.lstrip("#")
        if len(color) == 3:
            color = "".join(ch * 2 for ch in color)
        r = int(color[0:2], 16) / 255.0
        g = int(color[2:4], 16) / 255.0
        b = int(color[4:6], 16) / 255.0
        return r, g, b
    if color.lower().startswith("rgb"):
        start = color.find("(")
        end = color.find(")")
        if start == -1 or end == -1:
            raise ValueError(f"Unrecognized rgb color format: {color}")
        channels = color[start + 1 : end].split(",")
        r, g, b = [int(float(ch.strip())) / 255.0 for ch in channels[:3]]
        return r, g, b
    raise ValueError(f"Unsupported color format: {color}")


def _rgb_to_hex(rgb: Tuple[float, float, float]) -> str:
    r, g, b = (max(0, min(1, channel)) for channel in rgb)
    return f"#{int(r * 255):02x}{int(g * 255):02x}{int(b * 255):02x}"


def brighten_color(base_color: str, factor: float) -> str:
    """Brighten a color by interpolating towards white."""
    r, g, b = _parse_color(base_color)
    r = r + (1.0 - r) * factor
    g = g + (1.0 - g) * factor
    b = b + (1.0 - b) * factor
    return _rgb_to_hex((r, g, b))


def build_figure(
    parent_counts: Counter,
    leaf_counts: Counter,
    taxonomy: TaxonomyInfo,
    total: int,
    title: str,
    height: int,
    balance_factor: float,  # <-- Changed from boolean to float
) -> go.Figure:
    """Construct the Plotly figure using Plotly Express."""
    raw_total = total
    top_n_children = 3
    top_n_parents = 5  # <-- New: Define how many parents to show
    selected_children: Dict[str, List[Tuple[str, int]]] = {}
    display_parent_counts: Counter = Counter()
    display_leaf_counts: Counter = Counter()

    for parent in taxonomy.parent_order:
        children = taxonomy.children_order.get(parent, ())
        child_counts = [
            (child, leaf_counts.get(child, 0))
            for child in children
            if leaf_counts.get(child, 0) > 0
        ]
        child_counts.sort(key=lambda item: (-item[1], item[0]))
        top_children = child_counts[:top_n_children]
        if not top_children:
            continue
        selected_children[parent] = top_children
        display_parent_counts[parent] = sum(count for _, count in top_children)
        for child, count in top_children:
            display_leaf_counts[child] = count

    other_leaf_items = [
        (leaf, leaf_counts[leaf])
        for leaf in leaf_counts
        if taxonomy.parents.get(leaf) is None and leaf_counts[leaf] > 0
    ]
    other_leaf_items.sort(key=lambda item: (-item[1], item[0]))
    top_other_leaves = other_leaf_items[:top_n_children]
    if top_other_leaves:
        selected_children[UNKNOWN_PARENT] = top_other_leaves
        display_parent_counts[UNKNOWN_PARENT] = sum(
            count for _, count in top_other_leaves
        )
        for leaf, count in top_other_leaves:
            display_leaf_counts[leaf] = count

    # --- New Top 5 logic ---
    # Get the top N parents based on their aggregated counts
    top_5_parents = [p for p, c in display_parent_counts.most_common(top_n_parents)]

    display_total = 0  # Recalculate display_total based on top 5
    for parent in top_5_parents:
        if parent in selected_children:
            display_total += sum(
                count for _, count in selected_children[parent]
            )
    # --- End New Top 5 logic ---

    # --- New Plotly Express Logic ---

    if display_total == 0:
        print("No data to plot.")
        fig = go.Figure()
        fig.update_layout(
            title_text=f"{title}<br><sup>No data to display</sup>",
            height=height,
            xaxis={"visible": False},
            yaxis={"visible": False},
            annotations=[
                dict(
                    text="No data found",
                    xref="paper",
                    yref="paper",
                    showarrow=False,
                    font=dict(size=20),
                )
            ],
        )
        return fig

    # --- NATURE/SCIENCE PALETTE ---
    # Use elegant, muted colors from the Carto pack

    # OPTION 1 (Current Dim Palette):
    # palette = (
    #     px.colors.carto.Antique
    #     + px.colors.carto.Safe
    #     + px.colors.cartcarto.Vivid
    #     + px.colors.qualitative.G10
    # )

    # OPTION 2 (Brighter, Scientific):
    # palette = (
    #     px.colors.sequential.Viridis
    #     + px.colors.sequential.Plasma
    # )

    # OPTION 3 (Default Plotly - Bright & Bold):
    # palette = (
    #     px.colors.qualitative.Plotly
    #     + px.colors.qualitative.Bold
    #     + px.colors.qualitative.Vivid
    # )
    
    # OPTION 4 (High Contrast):
    palette = (
        px.colors.qualitative.Set3
        + px.colors.qualitative.T10
    )
    
    # palette = (
    #     ["#3C4F76"]
    #     + ["#999999", "#E69F00", "#B6D8EB", "#F5E76C", "#56B4E9", "#009E73", "#F0E442", "#6497B5", "#F1948A", "#0072B2", "#D55E00", "#CC79A7"]
    # )
    # --- END PALETTE SELECTION ---

    parent_color_map: Dict[str, str] = {}
    color_idx = 0
    parent_labels_in_chart: List[str] = []
    
    # Only iterate over the top 5 parents for the color map
    for parent in top_5_parents:
        display_parent = f'<b>{parent.split("/")[0].strip().replace(" ", "<br>")}</b>'  # <-- Add <br>
        if display_parent not in parent_color_map:
            parent_labels_in_chart.append(display_parent)
            parent_color_map[display_parent] = palette[color_idx % len(palette)]
            color_idx += 1

    # Build a DataFrame in "long format" for Plotly Express
    # This is similar to the approach in VerbNounStatistics.py
    data_for_df = []
    # Only iterate over the top 5 parents to build the dataframe
    for parent in top_5_parents:
        if parent not in selected_children:
            continue
        
        display_parent = f'<b>{parent.split("/")[0].strip().replace(" ", "<br>")}</b>'  # <-- Add <br>
        parent_color = parent_color_map[display_parent]

        for idx, (child, child_value) in enumerate(selected_children[parent]):
            
            # This logic is from the original script to get shaded leaf colors
            if parent != UNKNOWN_PARENT:
                brighten_factor = 0.25 + 0.15 * idx
            else:
                brighten_factor = 0.35 + 0.15 * idx
            leaf_color = brighten_color(parent_color, brighten_factor)
            
            # <-- FIX: Define display_child by splitting the name
            display_child = f"<b>{child.split('/')[0].strip().replace(' ', '<br>')}</b>" # child.split("/")[0].strip().replace(" ", "<br>") # <-- Add <br>

            # <-- Updated logic for balancing
            # Use real value for hover, but a balanced value for visual size
            real_value = child_value
            # Apply the balance_factor as an exponent.
            # 1.0 = no change, 0.0 = full balance (all 1), 0.5 = sqrt()
            display_value = real_value ** balance_factor

            data_for_df.append(
                {
                    "root": ROOT_LABEL,  # <-- FIX 1: Add the root column
                    "parent": display_parent, # Use display name
                    "leaf": display_child,    # Use display name
                    "value": display_value, # <-- Use the new display_value
                    "description": taxonomy.descriptions.get(child, display_child), # Lookup with original name
                    "color": leaf_color, # We'll color by leaves to get shades
                    "real_value": real_value, # <-- Store the real value for hover
                }
            )

    df = pd.DataFrame(data_for_df)

    # We need a color map for all leaves and parents to control everything
    final_color_map = parent_color_map.copy()
    for item in data_for_df:
        final_color_map[item["leaf"]] = item["color"]
    
    # --- REDUCE CENTER HOLE (Part 1) ---
    # Make the root node (the fake hole) transparent
    final_color_map[ROOT_LABEL] = "rgba(0,0,0,0)"
    # --- END ---

    fig = px.sunburst(
        df,
        path=["root", "parent", "leaf"],  # <-- FIX 2: Use the column name "root"
        values="value",
        custom_data=["description", "real_value"], # <-- Pass real_value to custom_data
        # We can't use color='parent' as it won't allow shaded leaves.
        # Instead, we build a full color map for all labels.
    )

    # Manually update colors for all segments
    labels = fig.data[0].labels
    colors = [final_color_map.get(label, "#cccccc") for label in labels]
    
    fig.update_traces(
        marker=dict(colors=colors, line=dict(color="#1f1f1f", width=1)),
        hovertemplate=(
            "<b>%{label}</b><br>"
            "%{custom_data[0]}<br>"
            "<b>Actual Count: %{custom_data[1]:,}</b><br>" # <-- Updated to show real count
            "Visual Share: %{percentRoot:.2%}" # This will be the *balanced* share
            "<extra></extra>"
        ),
        insidetextorientation="radial",
        maxdepth=3,
        # <-- FIX 3: Set Arial font for text
        textfont=dict(family="Arial", size=24),
        outsidetextfont=dict(family="Arial", size=20),
        leaf=dict(opacity=0.9),
        # --- FIX: Removed invalid 'hole=0.2' parameter ---
    )

    # --- End New Logic ---

    if raw_total != display_total:
        subtitle_text = (
            f"Top 3 (of Top 5 Parents): {display_total:,} (of {raw_total:,} total)"
        )
    else:
        subtitle_text = f"Total regions: {display_total:,}"
    
    # <-- Updated subtitle logic
    if balance_factor != 1.0:
        subtitle_text += f" | <b>Balance Factor: {balance_factor}</b>"

    subtitle = f"<span style='font-size:16px;'>{subtitle_text}</span>"
    
    fig.update_layout(
        # <-- FIX 3: Set Arial font for title
        title=dict(
            text=f"{title}<br><sup>{subtitle}</sup>",
            x=0.5,
            font=dict(family="Arial", size=20),
        ),
        margin=dict(t=100, l=10, r=10, b=10), # <-- Reduced margins
        height=height,
        uniformtext=dict(minsize=12, mode="show"),
        # <-- FIX 3: Set Arial font for hover and global
        hoverlabel=dict(font=dict(family="Arial", size=14)),
        font=dict(family="Arial"),
    )
    
    # fig.add_annotation(
    #     dict(
    #         x=0.5,
    #         y=0.5,
    #         xref="paper",
    #         yref="paper",
    #         # --- UPDATE: Changed center text and increased font size per user request ---
    #         text=f"<b>{display_total:,}</b><br>Functional Regions",
    #         showarrow=False,
    #         # <-- FIX 2 & 3: Center text to black and Arial
    #         font=dict(size=16, color="#000000", family="Arial"),
    #         # --- END ---
    #     )
    # )
    return fig


def _create_mock_json_files():
    """Create dummy JSON files for testing the script."""
    data1 = {
        "results": {
            "screen1": {
                "region_types": {
                    "node1": {"type": "Link"},
                    "node2": {"label": "Type: Button"},
                    "node3": {"type": " input field (search)"},
                    "node4": {"type": "Image"},
                    "node5": {"type": "Advertisement"},  # This will be known
                    "node6": {"type": "Totally Unknown"}, # This will be unknown
                }
            },
            "screen2": {
                "region_types": {
                    "node1": {"label": "Login button"},
                    "node2": {"type": "Input field"},
                    "node3": {"type": "Input field"},
                    "node4": {"type": "Checkbox"},
                    "node5": {"type": "Text"},
                }
            },
        }
    }

    data2 = {
        "results": {
            "screen1": {
                "region_types": {
                    "node1": {"type": "Link"},
                    "node2": {"type": "Link"},
                    "node3": {"type": "Dropdown"},
                    "node4": {"type": "Tab"},
                    "node5": {"type": "Other non-taxonomy link"},  # This will be unknown
                }
            }
        }
    }

    os.makedirs("testdata", exist_ok=True)
    path1 = os.path.join("testdata", "data1.json")
    path2 = os.path.join("testdata", "data2.json")

    with open(path1, "w", encoding="utf-8") as f:
        json.dump(data1, f)

    with open(path2, "w", encoding="utf-8") as f:
        json.dump(data2, f)
    print(f"Created mock {path1} and {path2}")
    return os.path.join("testdata", "data*.json")


def print_region_type_table(
    leaf_counts: Counter,
    taxonomy: TaxonomyInfo,
    total: int,
) -> None:
    """Print a beautiful table showing the count of each sub region type.
    
    Args:
        leaf_counts: Counter of leaf type counts
        taxonomy: Taxonomy information for organizing data
        total: Total number of regions
    """
    if total == 0 or not leaf_counts:
        print("\n" + "=" * 100)
        print("No region type data to display.")
        print("=" * 100 + "\n")
        return
    
    # Prepare data for the table
    table_data = []
    for leaf_type, count in leaf_counts.most_common():
        parent = taxonomy.parents.get(leaf_type, UNKNOWN_PARENT)
        description = taxonomy.descriptions.get(leaf_type, "")
        proportion = (count / total * 100) if total > 0 else 0.0
        
        # Truncate long strings for better table display
        # Keep original names but truncate for display
        leaf_type_display = leaf_type[:40] + "..." if len(leaf_type) > 40 else leaf_type
        parent_display = parent[:35] + "..." if len(parent) > 35 else parent
        description_display = description[:55] + "..." if len(description) > 55 else description
        
        table_data.append({
            "Sub Region Type": leaf_type_display,
            "Count": count,
            "Percentage": f"{proportion:.2f}%",
            "Parent Category": parent_display,
            "Description": description_display,
        })
    
    # Create DataFrame
    df = pd.DataFrame(table_data)
    
    # Print header
    print("\n" + "=" * 100)
    print(f"Sub Region Type Statistics (Total: {total:,} regions)")
    print("=" * 100)
    
    if TABULATE_AVAILABLE:
        # Use tabulate for beautiful formatting
        try:
            print(tabulate(
                df.values.tolist(),
                headers=df.columns.tolist(),
                tablefmt="grid",
                numalign="right",
                stralign="left",
            ))
        except Exception:
            # Fallback if tabulate has issues
            pd.set_option('display.max_columns', None)
            pd.set_option('display.width', None)
            pd.set_option('display.max_colwidth', 50)
            print(df.to_string(index=False))
    else:
        # Fallback to pandas string representation
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)
        pd.set_option('display.max_colwidth', 50)
        print(df.to_string(index=False))
    
    print("=" * 100 + "\n")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot a multi-level donut chart (inner ring: taxonomy parents, outer ring: leaf types)."
    )
    parser.add_argument(
        "-i",
        "--input",
        # Default changed to None, will trigger mock data creation
        default="/mnt/vdb1/hongxin_li/AutoGUIv2/*/gemini-2.5-pro-thinking/v2/*_region_types_gemini-2.5-pro-thinking.json",
        nargs="+",  # Allow multiple inputs
        help="Path or glob pattern for classification results JSON files (repeat for multiple inputs). (Default: creates and uses mock data)",
    )
    parser.add_argument(
        "-o",
        "--output",
        dest="output",
        default="chart.html",  # Add a default output
        help="Optional path to write an interactive HTML visualization.",
    )
    parser.add_argument(
        "--title",
        default="Functional Region Type Distribution",
        help="Main title for the chart.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=900,
        help="Height of the resulting figure in pixels.",
    )
    parser.add_argument(
        "--include-extra-leaf-types",
        action="store_true",
        help="Include the EXTRA_LEAF_TYPES additions from the taxonomy (default: enabled).",
    )
    parser.add_argument(
        "--no-extra-leaf-types",
        dest="include_extra_leaf_types",
        action="store_false", # Corrected to store_false
        help="Exclude the EXTRA_LEAF_TYPES additions from the taxonomy.",
    )
    # <-- START: Added arguments for high-res image export -->
    parser.add_argument(
        "--output-image",
        default=os.path.join(os.path.dirname(__file__), "region_type_donut_chart.png"),
        help="Optional path to save a high-resolution static PNG image. Requires 'kaleido'.",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=3.0,
        help="Image scale factor (controls DPI). 1=low, 3=good, 6=high-res. Used with --output-image. Default: 3.0",
    )
    parser.add_argument(
        "--output-stats",
        default=os.path.join(os.path.dirname(__file__), "region_type_statistics.json"),
        help="Optional path to save region type statistics (counts and proportions) as JSON.",
    )
    parser.set_defaults(include_extra_leaf_types=False)
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Skip rendering the interactive window (primarily for automated pipelines).",
    )
    # <-- Changed argument from --balance-sectors to --balance-factor
    parser.add_argument(
        "--balance-factor",
        type=float,
        default=0.5,
        help="Adjusts visual sector balance. 1.0 = real counts, 0.5 = sqrt(counts), 0.0 = all equal. Default: 0.5",
    )

    if argv is not None:
        args = parser.parse_args(argv)
    else:
        # Use parse_args() for robust default handling if no argv is given
        args, _ = parser.parse_known_args() 

    # Handle if a single string is passed for input
    if isinstance(args.input, str):
        args.input = [args.input]
        
    # If no input, use mock data
    if args.input is None:
        print("No --input specified, using mock data...")
        mock_pattern = _create_mock_json_files()
        args.input = [mock_pattern]

    return args


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)

    taxonomy = parse_taxonomy(
        include_extra_leaf_types=False #args.include_extra_leaf_types
    )

    # Pass the list of patterns to resolve_input_paths
    input_paths = resolve_input_paths(args.input)
    print(f"Aggregating data from {len(input_paths)} file(s)...")
    parent_counts, leaf_counts, total, diagnostics = aggregate_region_types(
        input_paths, taxonomy
    )
    if diagnostics:
        for key, value in diagnostics.items():
            print(f"[plot_region_type_donut] diagnostic {key}: {value}")

    # Print sub region type statistics table
    print_region_type_table(leaf_counts, taxonomy, total)

    # Save region type statistics to JSON if requested
    if args.output_stats:
        save_region_type_stats(
            parent_counts,
            leaf_counts,
            taxonomy,
            total,
            args.output_stats,
        )

    print("Building figure...")
    fig = build_figure(
        parent_counts,
        leaf_counts,
        taxonomy,
        total,
        args.title,
        args.height,
        args.balance_factor,  # <-- Pass the new float argument
    )

    # <-- START: Added logic to save high-res image -->
    if args.output_image:
        try:
            print(f"Saving high-resolution image (scale={args.scale}) to {args.output_image}...")
            # Ensure the output directory exists
            img_output_dir = os.path.dirname(os.path.abspath(args.output_image))
            if img_output_dir:
                os.makedirs(img_output_dir, exist_ok=True)
            
            fig.write_image(args.output_image, scale=args.scale)
            print(f"Successfully saved image to {os.path.abspath(args.output_image)}")
        except Exception as e:
            print(f"\n--- ERROR: Failed to save image ---")
            print(f"To save static images, you must first install 'kaleido':")
            print(f"  pip install kaleido")
            print(f"Error details: {e}\n")
    # <-- END: Added logic to save high-res image -->

    if args.output:
        output_dir = os.path.dirname(os.path.abspath(args.output))
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        fig.write_html(args.output, include_plotlyjs="cdn")
        print(f"Saved interactive chart to {os.path.abspath(args.output)}")
    if not args.no_show:
        print("Showing interactive chart...")
        fig.show()


if __name__ == "__main__":
    main()