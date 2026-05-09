import random
from colorsys import hsv_to_rgb
from typing import Any, Dict, List, Optional, Tuple

import folium
import geopandas as gpd
import numpy as np
from branca.colormap import LinearColormap
from branca.element import MacroElement, Template
from folium import GeoJson, Popup, PolyLine
from scipy.spatial import KDTree, distance
from shapely.geometry import Polygon

Coord = Tuple[float, float]
Node = int


# --------------------------------------------------------------------------- #
# Spatial helpers
# --------------------------------------------------------------------------- #

def generate_centroid_polygon_dict(
    grid_cells: gpd.GeoDataFrame,
) -> Dict[Coord, Polygon]:
    centroids = grid_cells["geometry"].centroid
    return {
        (centroid.y, centroid.x): polygon
        for centroid, polygon in zip(centroids, grid_cells["geometry"])
    }


def build_centroid_kdtree(
    centroid_polygon_dict: Dict[Coord, Polygon],
) -> Tuple[KDTree, List[Coord]]:
    centroids = list(centroid_polygon_dict.keys())
    return KDTree(centroids), centroids


def find_matching_cell(
    coords: Coord,
    polygon_dict: Dict[Coord, Polygon],
    kdtree: KDTree,
    centroids: List[Coord],
) -> Optional[Polygon]:
    _, idx = kdtree.query(coords)
    return polygon_dict.get(centroids[idx])


def get_polygon_coordinates(geometry: Polygon) -> List[List[float]]:
    if not geometry.is_valid or not hasattr(geometry, "exterior"):
        return []
    return [[p[1], p[0]] for p in geometry.exterior.coords]


# --------------------------------------------------------------------------- #
# Map primitives
# --------------------------------------------------------------------------- #

def create_map(center_coords: Coord, zoom_start: int = 10) -> folium.Map:
    return folium.Map(location=center_coords, zoom_start=zoom_start)


def calculate_center_coordinates(coord_labels: Dict[Node, Coord]) -> Coord:
    lons, lats = zip(*coord_labels.values())
    return float(np.mean(lats)), float(np.mean(lons))


def add_polygon_to_map(
    target: Any,
    lat_lon_coords: List[List[float]],
    color: str,
    popup_html: str,
) -> None:
    folium.Polygon(
        locations=lat_lon_coords,
        color=color,
        weight=0.5,
        fill=True,
        fill_color=color,
        fill_opacity=0.75,
        popup=folium.Popup(popup_html, max_width=300),
    ).add_to(target)


def add_polygon_with_popup(
    m: folium.Map,
    geometry: Polygon,
    popup_html: str,
    color: str = "red",
    fill_opacity: float = 0.7,
    weight: float = 0.5,
) -> None:
    geojson = GeoJson(
        geometry.__geo_interface__,
        style_function=lambda x, c=color, o=fill_opacity, w=weight: {
            "fillColor": c,
            "color": c,
            "fillOpacity": o,
            "weight": w,
        },
    )
    geojson.add_child(Popup(popup_html, max_width=200))
    geojson.add_to(m)


def add_marker_to_map(
    m: folium.Map,
    lat: float,
    lon: float,
    color: str,
    popup_html: str,
) -> None:
    folium.CircleMarker(
        location=[lat, lon],
        radius=5,
        color=color,
        fill=True,
        fill_color=color,
        fill_opacity=0.6,
        popup=folium.Popup(popup_html, max_width=300),
    ).add_to(m)


def add_polyline(
    m: folium.Map,
    start: Coord,
    end: Coord,
    color: str = "black",
    weight: float = 2.5,
    opacity: float = 0.1,
) -> None:
    PolyLine([start, end], color=color, weight=weight, opacity=opacity).add_to(m)


# --------------------------------------------------------------------------- #
# Color generation
# --------------------------------------------------------------------------- #

def generate_predefined_colors(
    num_colors: int,
    seed_value: int = 42,
    min_diff: float = 0.4,
) -> List[List[int]]:
    random.seed(seed_value)
    colors: List[List[int]] = []
    while len(colors) < num_colors:
        rgb = [int(255 * x) for x in hsv_to_rgb(random.random(), 0.9, 0.9)]
        if all(distance.euclidean(rgb, c) >= min_diff * 255 for c in colors):
            colors.append(rgb)
    return colors


# --------------------------------------------------------------------------- #
# High-level plot functions
# --------------------------------------------------------------------------- #

def plot_communities_on_map(
    community_nodes_dict: Dict[int, List[Node]],
    coord_labels: Dict[Node, Coord],
    polygon_dict: Dict[Coord, Polygon],
    cell_data_dict: Dict[Coord, Dict[str, Any]],
    kdtree: KDTree,
    centroids: List[Coord],
    zoom_level: int = 10,
    seed_value: int = 69,
) -> folium.Map:
    center = calculate_center_coordinates(coord_labels)
    m = create_map(center, zoom_start=zoom_level)

    multi_node = [c for c in community_nodes_dict if len(community_nodes_dict[c]) > 1]
    community_colors = generate_predefined_colors(len(multi_node), seed_value)

    for community_id in sorted(community_nodes_dict):
        nodes = community_nodes_dict[community_id]
        if len(nodes) > 1:
            rgb = community_colors[community_id % len(community_colors)]
            color_rgb = f"rgb({rgb[0]}, {rgb[1]}, {rgb[2]})"
        else:
            color_rgb = "rgb(0, 0, 0)"

        group = folium.FeatureGroup(name=f"Community {community_id}")
        for node in nodes:
            coords = coord_labels.get(node)
            if coords is None:
                continue
            cell_data = cell_data_dict.get(coords, {})
            popup_html = (
                f"<b>Community:</b> {community_id}<br>"
                f"<b>City:</b> {cell_data.get('CIDADE', 'N/A')}<br>"
                f"<b>Neighborhood:</b> {cell_data.get('BAIRRO', 'N/A')}"
            )
            geom = find_matching_cell(coords[::-1], polygon_dict, kdtree, centroids)
            if geom:
                add_polygon_to_map(group, get_polygon_coordinates(geom), color_rgb, popup_html)
        group.add_to(m)

    folium.LayerControl().add_to(m)
    return m


def plot_cell_and_neighbours(
    key: Coord,
    neighbours: List[Coord],
    cell_data_dict: Dict[Coord, Dict[str, Any]],
    polygon_dict: Dict[Coord, Polygon],
    kdtree: KDTree,
    centroids: List[Coord],
    zoom_level: int = 12,
    edge_opacity: float = 0.1,
    key_color: str = "red",
    neighbor_color: str = "blue",
) -> folium.Map:
    m = create_map(key[::-1], zoom_start=zoom_level)

    for nb in neighbours:
        add_polyline(m, key[::-1], nb[::-1], opacity=edge_opacity)

    def _add_cell(coords: Coord, color: str) -> None:
        data = cell_data_dict.get(coords, {})
        popup_html = (
            f"<b>City:</b> {data.get('CIDADE', 'N/A')}<br>"
            f"<b>Neighborhood:</b> {data.get('BAIRRO', 'N/A')}<br>"
            f"<b>Lat:</b> {coords[1]:.4f} <b>Lon:</b> {coords[0]:.4f}"
        )
        geom = find_matching_cell(coords[::-1], polygon_dict, kdtree, centroids)
        if geom:
            add_polygon_with_popup(m, geom, popup_html, color=color)

    _add_cell(key, key_color)
    for nb in neighbours:
        _add_cell(nb, neighbor_color)

    return m


def plot_metric_map(
    coord_labels: Dict[Node, Coord],
    metric_values: Dict[Node, float],
    polygon_dict: Dict[Coord, Polygon],
    cell_data_dict: Dict[Coord, Dict[str, Any]],
    kdtree: KDTree,
    centroids: List[Coord],
    visualization_type: str = "polygon",
    metric_name: str = "Metric",
    zoom_level: int = 10,
    metric_precision: int = 4,
) -> folium.Map:
    center = calculate_center_coordinates(coord_labels)
    m = create_map(center, zoom_level)

    values = list(set(metric_values.values()))
    colormap = LinearColormap(
        ["blue", "green", "yellow", "orange", "red"],
        vmin=min(values),
        vmax=max(values),
    )
    colormap.add_to(m)

    for node, metric_value in metric_values.items():
        coords = coord_labels.get(node)
        if coords is None:
            continue
        data = cell_data_dict.get(coords, {})
        popup_html = (
            f"<b>City:</b> {data.get('CIDADE', 'N/A')}<br>"
            f"<b>Neighborhood:</b> {data.get('BAIRRO', 'N/A')}<br>"
            f"<b>{metric_name}:</b> {metric_value:.{metric_precision}f}"
        )
        color = colormap(metric_value)

        if visualization_type == "polygon":
            geom = find_matching_cell(coords[::-1], polygon_dict, kdtree, centroids)
            if geom:
                add_polygon_to_map(m, get_polygon_coordinates(geom), color, popup_html)
        else:
            add_marker_to_map(m, coords[1], coords[0], color, popup_html)

    return m
