"""Shared utilities for clustered Overpass enrichment pipelines."""

from __future__ import annotations

import csv
import json
import logging
import math
import time
import zipfile
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence, Set, Tuple
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen, urlretrieve

# Mean Earth radius in meters (WGS84-compatible approximation for short distances).
EARTH_RADIUS_M = 6_371_008.8


@dataclass(frozen=True)
class GeoPoint:
    """Generic point record used by clustered enrichment scripts."""

    feature_id: str
    lon: float
    lat: float


@dataclass(frozen=True)
class Cluster:
    """A set of point indices grouped into a spatial cluster."""

    cluster_id: int
    point_indices: Tuple[int, ...]
    bbox: Tuple[float, float, float, float]  # south, west, north, east


@dataclass(frozen=True)
class RoadResult:
    """Nearest-road output for one point."""

    distance_m: Optional[float]
    highway: Optional[str]
    name: Optional[str]
    osm_id: Optional[int]


def setup_logging(level: str) -> None:
    """Configure standard console logging."""

    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s %(levelname)s %(message)s",
    )


def load_uk_geometry(cache_dir: Path):
    """Download and load the UK country geometry from Natural Earth."""

    try:
        import geopandas as gpd
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "GeoPandas is required for UK filtering. Install: pip install geopandas"
        ) from exc

    cache_dir.mkdir(parents=True, exist_ok=True)

    zip_path = cache_dir / "ne_10m_admin_0_countries.zip"
    extract_dir = cache_dir / "ne_10m_admin_0_countries"
    shp_path = extract_dir / "ne_10m_admin_0_countries.shp"

    if not zip_path.exists():
        url = "https://naturalearth.s3.amazonaws.com/10m_cultural/ne_10m_admin_0_countries.zip"
        logging.info("Downloading UK boundary dataset to %s", zip_path)
        urlretrieve(url, zip_path)

    if not shp_path.exists():
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(extract_dir)

    countries = gpd.read_file(shp_path).to_crs("EPSG:4326")

    name_columns = ["ADMIN", "NAME", "NAME_EN", "SOVEREIGNT", "BRK_NAME"]
    uk_rows = None

    for col in name_columns:
        if col in countries.columns:
            candidate = countries[countries[col] == "United Kingdom"]
            if not candidate.empty:
                uk_rows = candidate
                break

    if uk_rows is None or uk_rows.empty:
        raise LookupError("Could not find 'United Kingdom' in Natural Earth countries data")

    return uk_rows.union_all()


def is_in_uk(lat: float, lon: float, uk_geometry) -> bool:
    """Return True when point (lat, lon) is inside (or on boundary of) UK."""

    try:
        import geopandas as gpd
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "GeoPandas is required for UK filtering. Install: pip install geopandas"
        ) from exc

    point = gpd.points_from_xy([float(lon)], [float(lat)], crs="EPSG:4326")[0]
    return bool(uk_geometry.covers(point))


def latlon_to_global_meters(lat: float, lon: float, ref_lat: float) -> Tuple[float, float]:
    """Project WGS84 lat/lon to simple global meters for coarse clustering."""

    x = math.radians(lon) * EARTH_RADIUS_M * math.cos(math.radians(ref_lat))
    y = math.radians(lat) * EARTH_RADIUS_M
    return x, y


def bbox_expand_m(
    south: float,
    west: float,
    north: float,
    east: float,
    buffer_m: float,
) -> Tuple[float, float, float, float]:
    """Expand a lat/lon bbox by approximately buffer_m in all directions."""

    center_lat = (south + north) / 2.0
    dlat = math.degrees(buffer_m / EARTH_RADIUS_M)
    denom = EARTH_RADIUS_M * max(1e-9, math.cos(math.radians(center_lat)))
    dlon = math.degrees(buffer_m / denom)

    return south - dlat, west - dlon, north + dlat, east + dlon


def cluster_points(points: Sequence[GeoPoint], cell_m: float, bbox_buffer_m: float) -> list[Cluster]:
    """Cluster points by connected occupied grid cells and return per-cluster bboxes."""

    if not points:
        return []

    ref_lat = sum(p.lat for p in points) / len(points)
    cells: Dict[Tuple[int, int], list[int]] = defaultdict(list)

    for idx, point in enumerate(points):
        x_m, y_m = latlon_to_global_meters(point.lat, point.lon, ref_lat)
        cell_x = int(math.floor(x_m / cell_m))
        cell_y = int(math.floor(y_m / cell_m))
        cells[(cell_x, cell_y)].append(idx)

    visited_cells = set()
    clusters: list[Cluster] = []
    next_cluster_id = 1

    for start_cell in cells:
        if start_cell in visited_cells:
            continue

        queue = deque([start_cell])
        visited_cells.add(start_cell)
        component_cells = []

        while queue:
            current = queue.popleft()
            component_cells.append(current)

            cx, cy = current
            for nx in (cx - 1, cx, cx + 1):
                for ny in (cy - 1, cy, cy + 1):
                    neighbor = (nx, ny)
                    if neighbor in cells and neighbor not in visited_cells:
                        visited_cells.add(neighbor)
                        queue.append(neighbor)

        idxs: list[int] = []
        for cell in component_cells:
            idxs.extend(cells[cell])

        lats = [points[i].lat for i in idxs]
        lons = [points[i].lon for i in idxs]
        south, north = min(lats), max(lats)
        west, east = min(lons), max(lons)
        expanded_bbox = bbox_expand_m(south, west, north, east, bbox_buffer_m)

        clusters.append(
            Cluster(
                cluster_id=next_cluster_id,
                point_indices=tuple(sorted(idxs)),
                bbox=expanded_bbox,
            )
        )
        next_cluster_id += 1

    return clusters


def build_overpass_bbox_query(
    south: float,
    west: float,
    north: float,
    east: float,
    timeout_s: int,
    highway_filter: Optional[str] = None,
) -> str:
    """Construct an Overpass QL query for highway ways in a bbox."""

    highway_selector = '["highway"]' if not highway_filter else f'["highway"="{highway_filter}"]'
    return (
        f"[out:json][timeout:{int(timeout_s)}];\n"
        f"way{highway_selector}({south},{west},{north},{east});\n"
        "out tags geom;"
    )


def fetch_overpass_json(
    query: str,
    endpoint: str,
    timeout_s: int,
    max_retries: int,
    retry_wait_s: int,
) -> dict:
    """Execute an Overpass query with conservative retry behavior."""

    request_body = urlencode({"data": query}).encode("utf-8")
    request = Request(
        endpoint,
        data=request_body,
        headers={"User-Agent": "openbenches-batch-enrichment/1.0"},
    )

    attempt = 0
    while True:
        try:
            with urlopen(request, timeout=timeout_s) as response:
                return json.loads(response.read().decode("utf-8"))
        except HTTPError as err:
            retryable = err.code in {429, 500, 502, 503, 504}
            if not retryable or attempt >= max_retries:
                raise
            attempt += 1
            wait_s = retry_wait_s * attempt
            logging.warning(
                "Overpass HTTP %s (attempt %s/%s). Waiting %ss before retry.",
                err.code,
                attempt,
                max_retries,
                wait_s,
            )
            time.sleep(wait_s)
        except URLError:
            if attempt >= max_retries:
                raise
            attempt += 1
            wait_s = retry_wait_s * attempt
            logging.warning(
                "Overpass connection error (attempt %s/%s). Waiting %ss before retry.",
                attempt,
                max_retries,
                wait_s,
            )
            time.sleep(wait_s)


def project_local_meters(
    lat: float,
    lon: float,
    lat0: float,
    lon0: float,
) -> Tuple[float, float]:
    """Project lat/lon to local x/y meters around reference point (lat0, lon0)."""

    x = math.radians(lon - lon0) * EARTH_RADIUS_M * math.cos(math.radians(lat0))
    y = math.radians(lat - lat0) * EARTH_RADIUS_M
    return x, y


def point_to_segment_distance_m(
    ax: float,
    ay: float,
    bx: float,
    by: float,
) -> float:
    """Distance from local origin point (0,0) to segment A->B in meters."""

    dx = bx - ax
    dy = by - ay
    seg_len_sq = (dx * dx) + (dy * dy)

    if seg_len_sq == 0.0:
        return math.sqrt((ax * ax) + (ay * ay))

    t = max(0.0, min(1.0, -((ax * dx) + (ay * dy)) / seg_len_sq))
    cx = ax + (t * dx)
    cy = ay + (t * dy)
    return math.sqrt((cx * cx) + (cy * cy))


def nearest_road_for_point(lat: float, lon: float, overpass_payload: dict) -> RoadResult:
    """Find nearest road segment for one point from pre-fetched cluster ways."""

    best_distance: Optional[float] = None
    best_tags: Dict[str, str] = {}
    best_osm_id: Optional[int] = None

    for element in overpass_payload.get("elements", []):
        geometry = element.get("geometry", [])
        if len(geometry) < 2:
            continue

        for start, end in zip(geometry, geometry[1:]):
            ax, ay = project_local_meters(start["lat"], start["lon"], lat, lon)
            bx, by = project_local_meters(end["lat"], end["lon"], lat, lon)
            d_m = point_to_segment_distance_m(ax, ay, bx, by)

            if best_distance is None or d_m < best_distance:
                best_distance = d_m
                best_tags = element.get("tags", {})
                best_osm_id = element.get("id")

    if best_distance is None:
        return RoadResult(distance_m=None, highway=None, name=None, osm_id=None)

    return RoadResult(
        distance_m=round(best_distance, 2),
        highway=best_tags.get("highway"),
        name=best_tags.get("name"),
        osm_id=int(best_osm_id) if best_osm_id is not None else None,
    )


def write_csv(output_path: Path, rows: Iterable[dict], fieldnames: Sequence[str]) -> None:
    """Write rows to CSV with explicit, caller-provided field order."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def read_existing_csv(output_path: Path, key_column: str) -> Dict[str, dict]:
    """Load existing CSV rows keyed by a stable identifier column."""

    if not output_path.exists():
        return {}

    existing_rows: Dict[str, dict] = {}
    with output_path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            key = (row.get(key_column) or "").strip()
            if not key:
                continue
            existing_rows[key] = row
    return existing_rows


def cluster_is_complete(
    cluster: Cluster,
    kept_points: Sequence[GeoPoint],
    row_by_feature_id: Dict[str, dict],
    required_values: Dict[str, Set[str]],
) -> bool:
    """Check whether every point in a cluster already has complete values.

    For a column mapped to an empty set in required_values, completion means the
    column contains a non-empty value.
    """

    for kept_idx in cluster.point_indices:
        feature_id = kept_points[kept_idx].feature_id
        row = row_by_feature_id.get(feature_id)
        if row is None:
            return False
        for column_name, valid_values in required_values.items():
            value = row.get(column_name)
            if not valid_values:
                if value is None:
                    return False
                text_value = str(value).strip()
                if not text_value or text_value.lower() == "nan":
                    return False
            elif value not in valid_values:
                return False
    return True