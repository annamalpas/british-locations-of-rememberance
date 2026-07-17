"""Batch bench-to-road enrichment using clustered bbox Overpass queries.

Pipeline overview:
1. Load bench points from a GeoJSON FeatureCollection.
2. Keep only points inside the UK boundary (optional, enabled by default).
3. Cluster benches spatially (grid-cell connected components).
4. Query Overpass once per cluster bbox for all highway ways.
5. Query Overpass once per cluster bbox for settlement features (city/town/village/hamlet).
6. Compute nearest road segment per bench locally.
7. Determine settlement name/type for each bench.
8. Write results to CSV.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import time
import zipfile
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen, urlretrieve
from shapely.geometry import Point

# Mean Earth radius in meters (WGS84-compatible approximation for short distances).
EARTH_RADIUS_M = 6_371_008.8


@dataclass(frozen=True)
class BenchPoint:
    """Single bench location extracted from input GeoJSON."""

    feature_id: int
    lon: float
    lat: float


@dataclass(frozen=True)
class Cluster:
    """A set of bench indices grouped into a spatial cluster."""

    cluster_id: int
    bench_indices: Tuple[int, ...]
    bbox: Tuple[float, float, float, float]  # south, west, north, east


@dataclass(frozen=True)
class RoadResult:
    """Nearest-road output for one bench."""

    distance_m: Optional[float]
    highway: Optional[str]
    name: Optional[str]
    osm_id: Optional[int]


@dataclass(frozen=True)
class SettlementResult:
    """Settlement lookup output for one bench."""

    settlement_name: Optional[str]
    settlement_type: Optional[str]


def parse_args() -> argparse.Namespace:
    """Define CLI arguments and parse command line input."""

    parser = argparse.ArgumentParser(
        description=(
            "Batch-enrich bench points with nearest OSM road details using "
            "clustered bbox Overpass requests."
        )
    )

    # Input/output paths.
    parser.add_argument(
        "--input",
        default="data_extraction/data.json",
        help="Path to input GeoJSON FeatureCollection (default: data_extraction/data.json)",
    )
    parser.add_argument(
        "--output",
        default="data_extraction/bench_road_enrichment.csv",
        help="Path to output CSV file",
    )

    # Clustering controls.
    parser.add_argument(
        "--cluster-cell-m",
        type=float,
        default=300.0,
        help="Grid cell size in meters used to form bench clusters",
    )
    parser.add_argument(
        "--bbox-buffer-m",
        type=float,
        default=120.0,
        help="Extra meter buffer added around each cluster bbox",
    )

    # Overpass controls.
    parser.add_argument(
        "--overpass-endpoint",
        default="https://overpass-api.de/api/interpreter",
        help="Overpass interpreter endpoint URL",
    )
    parser.add_argument(
        "--timeout-s",
        type=int,
        default=25,
        help="Overpass query timeout hint in seconds",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Maximum retry attempts for HTTP 429/5xx errors",
    )
    parser.add_argument(
        "--retry-wait-s",
        type=int,
        default=30,
        help="Base wait time in seconds before retrying failed requests",
    )

    parser.add_argument(
        "--settlement-fallback-distance-m",
        type=float,
        default=0.0,
        help=(
            "Optional max distance in meters to nearest settlement center when no containing "
            "settlement polygon is found (0 disables fallback)"
        ),
    )

    # UK filtering and boundary dataset cache.
    parser.add_argument(
        "--skip-uk-filter",
        action="store_true",
        help="If set, do not filter points by UK boundary",
    )
    parser.add_argument(
        "--shapefile-cache-dir",
        default="data_extraction/shapefiles",
        help="Directory used to cache UK boundary shapefile files",
    )

    # Logging verbosity.
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )

    return parser.parse_args()


def setup_logging(level: str) -> None:
    """Configure standard console logging."""

    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s %(levelname)s %(message)s",
    )


def load_geojson_points(input_path: Path) -> List[BenchPoint]:
    """Load bench points from a GeoJSON FeatureCollection.

    Expected shape:
    {
      "type": "FeatureCollection",
      "features": [
        {"id": 1, "geometry": {"type": "Point", "coordinates": [lon, lat]}},
        ...
      ]
    }
    """

    with input_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    features = payload.get("features", [])
    points: List[BenchPoint] = []

    # Iterate with fallback index in case an explicit feature id is missing.
    for idx, feature in enumerate(features):
        geometry = feature.get("geometry", {})
        if geometry.get("type") != "Point":
            continue

        coordinates = geometry.get("coordinates", [])
        if not isinstance(coordinates, list) or len(coordinates) < 2:
            continue

        lon = float(coordinates[0])
        lat = float(coordinates[1])

        # Use OSM feature id when available; otherwise use 1-based row number.
        feature_id = int(feature.get("id", idx + 1))
        points.append(BenchPoint(feature_id=feature_id, lon=lon, lat=lat))

    return points


def load_uk_geometry(cache_dir: Path):
    """Download and load the UK country geometry from Natural Earth.

    The function imports GeoPandas lazily so the script can still show a clear
    error message if geospatial dependencies are missing.
    """

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

    # Download Natural Earth country boundaries once and reuse local cache.
    if not zip_path.exists():
        url = "https://naturalearth.s3.amazonaws.com/10m_cultural/ne_10m_admin_0_countries.zip"
        logging.info("Downloading UK boundary dataset to %s", zip_path)
        urlretrieve(url, zip_path)

    # Extract only when needed to avoid repeated filesystem work.
    if not shp_path.exists():
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(extract_dir)

    countries = gpd.read_file(shp_path).to_crs("EPSG:4326")

    # Natural Earth naming columns can differ by version.
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

    # union_all returns one geometry object representing the whole UK boundary.
    return uk_rows.union_all()


def is_in_uk(lat: float, lon: float, uk_geometry) -> bool:
    """Return True when point (lat, lon) is inside (or on boundary of) UK."""

    point = Point(float(lon), float(lat))
    return bool(uk_geometry.covers(point))


def latlon_to_global_meters(lat: float, lon: float, ref_lat: float) -> Tuple[float, float]:
    """Project WGS84 lat/lon to simple global meters for coarse clustering.

    This is an equirectangular projection approximation suitable for local grouping.
    """

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

    # Convert meters to degrees latitude.
    dlat = math.degrees(buffer_m / EARTH_RADIUS_M)

    # Convert meters to degrees longitude, adjusted by latitude.
    denom = EARTH_RADIUS_M * max(1e-9, math.cos(math.radians(center_lat)))
    dlon = math.degrees(buffer_m / denom)

    return south - dlat, west - dlon, north + dlat, east + dlon


def cluster_points(points: Sequence[BenchPoint], cell_m: float, bbox_buffer_m: float) -> List[Cluster]:
    """Cluster points by connected occupied grid cells and return per-cluster bboxes.

    Method summary:
    1. Convert points to a coarse meter grid.
    2. Mark occupied cells.
    3. Find connected components across 8-neighbor cells.
    4. Build one bbox per connected component (with configurable buffer).
    """

    if not points:
        return []

    ref_lat = sum(p.lat for p in points) / len(points)

    # Mapping from (cell_x, cell_y) -> list of point indices inside that cell.
    cells: Dict[Tuple[int, int], List[int]] = defaultdict(list)

    for idx, point in enumerate(points):
        x_m, y_m = latlon_to_global_meters(point.lat, point.lon, ref_lat)
        cell_x = int(math.floor(x_m / cell_m))
        cell_y = int(math.floor(y_m / cell_m))
        cells[(cell_x, cell_y)].append(idx)

    visited_cells = set()
    clusters: List[Cluster] = []
    next_cluster_id = 1

    # Visit each occupied cell once and flood-fill to find connected components.
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
            # 8-neighbor adjacency includes diagonal contact.
            for nx in (cx - 1, cx, cx + 1):
                for ny in (cy - 1, cy, cy + 1):
                    neighbor = (nx, ny)
                    if neighbor in cells and neighbor not in visited_cells:
                        visited_cells.add(neighbor)
                        queue.append(neighbor)

        # Flatten all point indices from all cells in this component.
        idxs: List[int] = []
        for cell in component_cells:
            idxs.extend(cells[cell])

        # Compute raw bbox from clustered points.
        lats = [points[i].lat for i in idxs]
        lons = [points[i].lon for i in idxs]
        south, north = min(lats), max(lats)
        west, east = min(lons), max(lons)

        # Expand bbox so nearby roads just outside the tight bench box are still found.
        expanded_bbox = bbox_expand_m(south, west, north, east, bbox_buffer_m)

        clusters.append(
            Cluster(
                cluster_id=next_cluster_id,
                bench_indices=tuple(sorted(idxs)),
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
) -> str:
    """Construct an Overpass QL query for all highway ways in a bbox."""

    return (
        f"[out:json][timeout:{int(timeout_s)}];\n"
        f"way[\"highway\"]({south},{west},{north},{east});\n"
        "out tags geom;"
    )


def fetch_overpass_json(
    query: str,
    endpoint: str,
    timeout_s: int,
    max_retries: int,
    retry_wait_s: int,
) -> dict:
    """Execute an Overpass query with conservative retry behavior.

    Retries are intentionally limited and sequential to remain friendly to public
    Overpass rate limits.
    """

    request_body = urlencode({"data": query}).encode("utf-8")

    # Identify this client in User-Agent per Overpass instance best practices.
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
        # Degenerate segment: distance to a single point.
        return math.sqrt((ax * ax) + (ay * ay))

    # Closest point on finite segment to origin.
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

        # Compare each road segment in this way.
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


def build_overpass_settlement_query(
    south: float,
    west: float,
    north: float,
    east: float,
    timeout_s: int,
) -> str:
    """Construct an Overpass QL query for settlement features in a bbox."""

    return (
        f"[out:json][timeout:{int(timeout_s)}];\n"
        "(\n"
        f"  way[\"place\"~\"^(city|town|village|hamlet)$\"]({south},{west},{north},{east});\n"
        f"  rel[\"place\"~\"^(city|town|village|hamlet)$\"]({south},{west},{north},{east});\n"
        f"  node[\"place\"~\"^(city|town|village|hamlet)$\"]({south},{west},{north},{east});\n"
        ");\n"
        "out tags geom center;"
    )


def extract_settlement_candidates(overpass_payload: dict):
    """Extract settlement polygons and center points from Overpass payload."""

    allowed = {"city", "town", "village", "hamlet"}

    try:
        from shapely.geometry import Polygon
    except ModuleNotFoundError:
        Polygon = None  # type: ignore[assignment]

    polygons = []
    centers = []

    for element in overpass_payload.get("elements", []):
        tags = element.get("tags", {})
        place_type = tags.get("place")
        settlement_name = tags.get("name")

        if place_type not in allowed or not settlement_name:
            continue

        geometry = element.get("geometry", [])
        if Polygon is not None and len(geometry) >= 4:
            # A closed ring can represent the settlement confines.
            first = geometry[0]
            last = geometry[-1]
            if first.get("lat") == last.get("lat") and first.get("lon") == last.get("lon"):
                coords = [(p["lon"], p["lat"]) for p in geometry if "lat" in p and "lon" in p]
                if len(coords) >= 4:
                    polygon = Polygon(coords)
                    if not polygon.is_valid:
                        polygon = polygon.buffer(0)
                    if not polygon.is_empty:
                        polygons.append(
                            {
                                "name": settlement_name,
                                "type": place_type,
                                "polygon": polygon,
                                "area": float(polygon.area),
                            }
                        )

        if element.get("type") == "node" and "lat" in element and "lon" in element:
            centers.append(
                {
                    "name": settlement_name,
                    "type": place_type,
                    "lat": float(element["lat"]),
                    "lon": float(element["lon"]),
                }
            )
        elif "center" in element and "lat" in element["center"] and "lon" in element["center"]:
            centers.append(
                {
                    "name": settlement_name,
                    "type": place_type,
                    "lat": float(element["center"]["lat"]),
                    "lon": float(element["center"]["lon"]),
                }
            )

    return polygons, centers


def settlement_for_point(
    lat: float,
    lon: float,
    settlement_polygons: Sequence[dict],
    settlement_centers: Sequence[dict],
    fallback_distance_m: float,
) -> SettlementResult:
    """Determine settlement name/type for one point.

    Priority:
    1. A containing settlement polygon (best match: smallest containing polygon).
    2. Optional nearest settlement center fallback within fallback_distance_m.
    """

    try:
        from shapely.geometry import Point
    except ModuleNotFoundError:
        Point = None  # type: ignore[assignment]

    if Point is not None:
        point = Point(float(lon), float(lat))
        containing = [poly for poly in settlement_polygons if poly["polygon"].covers(point)]
        if containing:
            best = min(containing, key=lambda poly: poly["area"])
            return SettlementResult(
                settlement_name=best["name"],
                settlement_type=best["type"],
            )

    # Optional fallback for areas where only point/center settlement data exists.
    if fallback_distance_m > 0 and settlement_centers:
        nearest = None
        nearest_distance = None
        for center in settlement_centers:
            dx, dy = project_local_meters(center["lat"], center["lon"], lat, lon)
            distance_m = math.sqrt((dx * dx) + (dy * dy))
            if nearest_distance is None or distance_m < nearest_distance:
                nearest_distance = distance_m
                nearest = center

        if nearest is not None and nearest_distance is not None and nearest_distance <= fallback_distance_m:
            return SettlementResult(
                settlement_name=nearest["name"],
                settlement_type=nearest["type"],
            )

    return SettlementResult(settlement_name=None, settlement_type=None)


def write_csv(output_path: Path, rows: Iterable[dict]) -> None:
    """Write enrichment rows to CSV with a stable, explicit column order."""

    fieldnames = [
        "feature_id",
        "lat",
        "lon",
        "is_in_uk",
        "cluster_id",
        "distance_m",
        "highway",
        "name",
        "osm_id",
        "settlement_name",
        "settlement_type",
        "road_status",
        "settlement_status",
    ]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def read_existing_csv(output_path: Path) -> Dict[int, dict]:
    """Load previously written rows so the script can resume without re-querying.

    The CSV is keyed by feature_id, which is stable across runs and lets us check
    whether a cluster already has all of its bench rows populated.
    """

    if not output_path.exists():
        return {}

    existing_rows: Dict[int, dict] = {}
    with output_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            feature_id_raw = row.get("feature_id")
            if not feature_id_raw:
                continue

            try:
                feature_id = int(feature_id_raw)
            except ValueError:
                continue

            existing_rows[feature_id] = row

    return existing_rows


def row_has_result(row: dict) -> bool:
    """Return True when a CSV row already contains complete enrichment status."""

    road_status = row.get("road_status")
    settlement_status = row.get("settlement_status")
    return road_status in {"found", "not_found"} and settlement_status in {"found", "not_found"}


def cluster_is_complete(
    cluster: Cluster,
    kept_points: Sequence[BenchPoint],
    row_by_feature_id: Dict[int, dict],
) -> bool:
    """Check whether every bench in a cluster already has enrichment data."""

    # cluster.bench_indices stores indexes into kept_points, not feature IDs.
    for kept_idx in cluster.bench_indices:
        feature_id = kept_points[kept_idx].feature_id
        row = row_by_feature_id.get(feature_id)
        if row is None or not row_has_result(row):
            return False
    return True


def main() -> int:
    """Program entrypoint."""

    args = parse_args()
    setup_logging(args.log_level)

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        logging.error("Input file not found: %s", input_path)
        return 1

    logging.info("Loading bench points from %s", input_path)
    all_points = load_geojson_points(input_path)
    logging.info("Loaded %s point features", len(all_points))

    if not all_points:
        logging.error("No valid point features found in input file")
        return 1

    uk_geometry = None
    if not args.skip_uk_filter:
        logging.info("Loading UK boundary geometry for point filtering")
        uk_geometry = load_uk_geometry(Path(args.shapefile_cache_dir))

    # If a previous run already wrote part or all of the CSV, load it now so we
    # can skip clusters that are already complete.
    existing_rows = read_existing_csv(output_path)
    if existing_rows:
        logging.info("Loaded %s existing rows from %s", len(existing_rows), output_path)

    # Build an output row skeleton for every input point, then fill enrichment fields.
    rows: List[dict] = []
    kept_points: List[BenchPoint] = []
    row_by_feature_id: Dict[int, dict] = {}

    for point in all_points:
        in_uk = True
        if uk_geometry is not None:
            in_uk = is_in_uk(point.lat, point.lon, uk_geometry)

        row = {
            "feature_id": point.feature_id,
            "lat": point.lat,
            "lon": point.lon,
            "is_in_uk": in_uk,
            "cluster_id": None,
            "distance_m": None,
            "highway": None,
            "name": None,
            "osm_id": None,
            "settlement_name": None,
            "settlement_type": None,
            "road_status": None,
            "settlement_status": None,
        }

        # Restore previous enrichment values when they already exist in the CSV.
        existing_row = existing_rows.get(point.feature_id)
        if existing_row is not None:
            row["cluster_id"] = existing_row.get("cluster_id") or None
            row["distance_m"] = existing_row.get("distance_m") or None
            row["highway"] = existing_row.get("highway") or None
            row["name"] = existing_row.get("name") or None
            row["osm_id"] = existing_row.get("osm_id") or None
            row["settlement_name"] = existing_row.get("settlement_name") or None
            row["settlement_type"] = existing_row.get("settlement_type") or None
            row["road_status"] = existing_row.get("road_status") or None
            row["settlement_status"] = existing_row.get("settlement_status") or None

        rows.append(row)
        row_by_feature_id[point.feature_id] = row

        if in_uk:
            kept_points.append(point)

    logging.info("Points inside UK: %s / %s", len(kept_points), len(all_points))

    if not kept_points:
        logging.warning("No UK points found. Writing output with empty enrichment fields.")
        write_csv(output_path, rows)
        logging.info("CSV written to %s", output_path)
        return 0

    # Build clusters and run one Overpass request per cluster.
    clusters = cluster_points(
        points=kept_points,
        cell_m=max(1.0, float(args.cluster_cell_m)),
        bbox_buffer_m=max(0.0, float(args.bbox_buffer_m)),
    )
    logging.info("Built %s clusters from UK points", len(clusters))

    for c_idx, cluster in enumerate(clusters, start=1):
        if cluster_is_complete(cluster, kept_points, row_by_feature_id):
            logging.info(
                "Skipping cluster %s/%s (id=%s) because all rows already exist in CSV",
                c_idx,
                len(clusters),
                cluster.cluster_id,
            )
            continue

        south, west, north, east = cluster.bbox
        logging.info(
            "Cluster %s/%s (id=%s): %s benches, bbox=(%.6f, %.6f, %.6f, %.6f)",
            c_idx,
            len(clusters),
            cluster.cluster_id,
            len(cluster.bench_indices),
            south,
            west,
            north,
            east,
        )

        query = build_overpass_bbox_query(
            south=south,
            west=west,
            north=north,
            east=east,
            timeout_s=args.timeout_s,
        )

        settlement_query = build_overpass_settlement_query(
            south=south,
            west=west,
            north=north,
            east=east,
            timeout_s=args.timeout_s,
        )

        try:
            payload = fetch_overpass_json(
                query=query,
                endpoint=args.overpass_endpoint,
                timeout_s=args.timeout_s,
                max_retries=max(0, int(args.max_retries)),
                retry_wait_s=max(1, int(args.retry_wait_s)),
            )
        except HTTPError as err:
            # If Overpass times out (504) or rate-limits too aggressively (429),
            # skip this cluster and keep the run moving forward.
            if err.code in {429, 504}:
                logging.warning(
                    "Skipping cluster id=%s due to Overpass HTTP %s after retries.",
                    cluster.cluster_id,
                    err.code,
                )
                write_csv(output_path, rows)
                logging.info(
                    "Flushed progress to %s after skipping cluster %s",
                    output_path,
                    cluster.cluster_id,
                )
                continue
            raise
        except (URLError, TimeoutError) as err:
            # Network timeout/connection issues should not abort the full run.
            logging.warning(
                "Skipping cluster id=%s due to network timeout/error: %s",
                cluster.cluster_id,
                err,
            )
            write_csv(output_path, rows)
            logging.info(
                "Flushed progress to %s after skipping cluster %s",
                output_path,
                cluster.cluster_id,
            )
            continue

        try:
            settlement_payload = fetch_overpass_json(
                query=settlement_query,
                endpoint=args.overpass_endpoint,
                timeout_s=args.timeout_s,
                max_retries=max(0, int(args.max_retries)),
                retry_wait_s=max(1, int(args.retry_wait_s)),
            )
        except HTTPError as err:
            if err.code in {429, 504}:
                logging.warning(
                    "Skipping cluster id=%s due to settlement HTTP %s after retries.",
                    cluster.cluster_id,
                    err.code,
                )
                write_csv(output_path, rows)
                logging.info(
                    "Flushed progress to %s after skipping cluster %s",
                    output_path,
                    cluster.cluster_id,
                )
                continue
            raise
        except (URLError, TimeoutError) as err:
            logging.warning(
                "Skipping cluster id=%s due to settlement network timeout/error: %s",
                cluster.cluster_id,
                err,
            )
            write_csv(output_path, rows)
            logging.info(
                "Flushed progress to %s after skipping cluster %s",
                output_path,
                cluster.cluster_id,
            )
            continue

        settlement_polygons, settlement_centers = extract_settlement_candidates(settlement_payload)

        # Enrich each bench in this cluster using local nearest-segment computation.
        for kept_idx in cluster.bench_indices:
            point = kept_points[kept_idx]
            road = nearest_road_for_point(point.lat, point.lon, payload)

            # Update the output row directly using feature_id as key.
            row = row_by_feature_id[point.feature_id]
            row["cluster_id"] = cluster.cluster_id
            row["distance_m"] = road.distance_m
            row["highway"] = road.highway
            row["name"] = road.name
            row["osm_id"] = road.osm_id

            settlement = settlement_for_point(
                lat=point.lat,
                lon=point.lon,
                settlement_polygons=settlement_polygons,
                settlement_centers=settlement_centers,
                fallback_distance_m=max(0.0, float(args.settlement_fallback_distance_m)),
            )

            row["settlement_name"] = settlement.settlement_name
            row["settlement_type"] = settlement.settlement_type
            row["road_status"] = "found" if road.distance_m is not None else "not_found"
            row["settlement_status"] = (
                "found" if settlement.settlement_name and settlement.settlement_type else "not_found"
            )

        # Flush progress after each processed cluster so reruns can resume from CSV.
        write_csv(output_path, rows)
        logging.info("Flushed progress to %s after cluster %s", output_path, cluster.cluster_id)

    write_csv(output_path, rows)
    logging.info("CSV written to %s", output_path)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt as exc:
        logging.error("Interrupted by user")
        raise SystemExit(130) from exc
