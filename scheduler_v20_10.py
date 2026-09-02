from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime, timedelta, time
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple
import math
import re
from difflib import SequenceMatcher

import pandas as pd

from coordinate_clustering import (
    NO_GOOGLE_RADIUS_KM,
    estimate_local_transfer_minutes,
    haversine_km,
    should_bypass_google_between_sites,
)


# ============================================================================
# SAME-ROAD TEAM TRAVEL LEEWAY RULE
# ============================================================================
SAME_ROAD_COORDINATE_FALLBACK_KM = 0.20

# Google-specific routing guardrail.
#
# There is deliberately NO cap on surveyable/candidate sites. The whole
# eligible shortlist may remain available. At each routing decision, however,
# coordinates are used to pre-rank disconnected local groups and only this
# small number of representatives is sent to Google for precise TRANSIT
# validation.
GOOGLE_GROUPS_PER_DECISION = 8
GOOGLE_FALLBACK_GROUPS_PER_DECISION = 2

# Far strategic-cluster transition efficiency rule.
#
# Normal/local cluster changes are untouched. Only a move to a DIFFERENT
# strategic cluster with >=30 minutes of travel must unlock at least twice as
# much feasible remaining survey time as the travel required.
FAR_CLUSTER_TRANSITION_MINUTES = 30.0
FAR_CLUSTER_MIN_SURVEY_TO_TRAVEL_RATIO = 2.0

_ROAD_SUFFIX_CANONICAL = {
    "road":"road", "rd":"road", "street":"street", "st":"street",
    "avenue":"avenue", "ave":"avenue", "lane":"lane", "ln":"lane",
    "way":"way", "close":"close", "crescent":"crescent", "cres":"crescent",
    "drive":"drive", "dr":"drive", "gardens":"gardens", "garden":"gardens",
    "grove":"grove", "hill":"hill", "park":"park", "place":"place", "pl":"place",
    "square":"square", "terrace":"terrace", "walk":"walk", "row":"row",
    "mews":"mews", "parade":"parade", "rise":"rise", "vale":"vale", "view":"view",
    "boulevard":"boulevard", "approach":"approach", "green":"green", "common":"common",
    "broadway":"broadway", "circus":"circus", "causeway":"causeway", "bank":"bank",
    "quay":"quay", "wharf":"wharf", "yard":"yard",
}
_UK_POSTCODE_AT_END_RE = re.compile(r"\b(?:GIR\s?0AA|(?:[A-Z]{1,2}\d[A-Z\d]?)\s?\d[A-Z]{2})\s*$", re.IGNORECASE)
_ROAD_PHRASE_RE = re.compile(
    r"\b([A-Za-z][A-Za-z'&.\-]*(?:\s+[A-Za-z][A-Za-z'&.\-]*){0,5})\s+(" +
    "|".join(sorted((re.escape(v) for v in _ROAD_SUFFIX_CANONICAL), key=len, reverse=True)) + r")\b",
    re.IGNORECASE,
)
def _normalise_road_text(value: str) -> str:
    value=str(value or "").lower().replace("&"," and ")
    value=re.sub(r"[^a-z0-9'\s-]+"," ",value)
    return re.sub(r"\s+"," ",value).strip()
def infer_road_name_from_building_name(building_name: str):
    raw=str(building_name or "").strip()
    if not raw: return None
    parts=[part.strip() for part in re.split(r"[|,]",raw) if part and part.strip()]
    for part in reversed(parts):
        candidate=_UK_POSTCODE_AT_END_RE.sub("",part).strip()
        candidate=re.sub(r"^\s*(?:[A-Z](?:-[A-Z])?\s*)?(?:\d+[A-Z]?(?:\s*[-/]\s*\d+[A-Z]?)?\s*)+","",candidate,flags=re.IGNORECASE).strip()
        matches=list(_ROAD_PHRASE_RE.finditer(candidate))
        if not matches: continue
        m=matches[-1]; root=_normalise_road_text(m.group(1)); suffix=_ROAD_SUFFIX_CANONICAL.get(m.group(2).lower())
        if not root or not suffix: continue
        if root in {"block","house","court","building","community","estate"}: continue
        return f"{root} {suffix}"
    return None
def should_remove_team_travel_leeway(current_site: dict, next_site: dict, fallback_radius_km: float=SAME_ROAD_COORDINATE_FALLBACK_KM) -> bool:
    current_road=infer_road_name_from_building_name(current_site.get("building_name",""))
    next_road=infer_road_name_from_building_name(next_site.get("building_name",""))
    if current_road is not None and next_road is not None:
        return current_road == next_road
    distance_km=haversine_km(current_site.get("latitude"),current_site.get("longitude"),next_site.get("latitude"),next_site.get("longitude"))
    return distance_km is not None and distance_km <= float(fallback_radius_km)


# ============================================================================
# LOCAL PROXIMITY GRAPH ORDERING
# ============================================================================
# The strategic Planning Cluster is unchanged. Within it, sites connected by
# successive <= NO_GOOGLE_RADIUS_KM coordinate hops form one local group.


def _planning_cluster_key(site: dict) -> str:
    return str(
        site.get("planning_cluster", "")
        or postcode_district(site.get("postcode", ""))
    )



def _remaining_cluster_survey_minutes(
    sites: Sequence[dict],
    target_cluster: str,
    current_day,
) -> float:
    """
    Total currently usable predicted/planning survey minutes remaining in one
    strategic cluster. Future-dated special-request sites are excluded because
    the normal scheduler would not allow them to be consumed today.
    """
    total = 0.0

    for site in sites:
        if _planning_cluster_key(site) != target_cluster:
            continue

        preferred_date = site.get("special_request_date")
        if preferred_date:
            try:
                if hasattr(preferred_date, "date"):
                    preferred_date = preferred_date.date()
                if current_day < preferred_date:
                    continue
            except Exception:
                pass

        try:
            minutes = float(site.get("planning_minutes", 0) or 0)
        except Exception:
            minutes = 0.0

        if math.isfinite(minutes) and minutes > 0:
            total += minutes

    return float(total)


def _far_cluster_transition_is_efficient(
    travel_minutes: float,
    feasible_survey_minutes: float,
    far_threshold_minutes: float = FAR_CLUSTER_TRANSITION_MINUTES,
    minimum_ratio: float = FAR_CLUSTER_MIN_SURVEY_TO_TRAVEL_RATIO,
) -> bool:
    """
    Apply the efficiency rule only to a genuinely far inter-cluster jump.

    Example with the defaults:
      - 25 min travel -> rule does not activate
      - 40 min travel + 100 min feasible survey work -> allow (2.5:1)
      - 40 min travel + 50 min feasible survey work -> reject (1.25:1)
    """
    try:
        travel = float(travel_minutes)
        survey = max(0.0, float(feasible_survey_minutes))
    except Exception:
        return False

    if not math.isfinite(travel) or travel <= 0:
        return True

    if travel < float(far_threshold_minutes):
        return True

    return (
        survey / travel
        >= float(minimum_ratio)
    )


def _coordinate_edge_distance_km(
    site_a: dict,
    site_b: dict,
    radius_km: float = NO_GOOGLE_RADIUS_KM,
):
    distance = haversine_km(
        site_a.get("latitude"), site_a.get("longitude"),
        site_b.get("latitude"), site_b.get("longitude"),
    )
    if distance is None or distance > float(radius_km):
        return None
    return float(distance)


def _connected_components_for_indices(
    sites: List[dict],
    indices: Sequence[int],
    radius_km: float = NO_GOOGLE_RADIUS_KM,
) -> List[List[int]]:
    """Connected local groups inside the existing strategic cluster."""
    indices = list(indices)
    adjacency = {idx: set() for idx in indices}

    for pos, idx_a in enumerate(indices):
        cluster_a = _planning_cluster_key(sites[idx_a])
        for idx_b in indices[pos + 1:]:
            if cluster_a != _planning_cluster_key(sites[idx_b]):
                continue

            linked = (
                _coordinate_edge_distance_km(
                    sites[idx_a], sites[idx_b], radius_km
                )
                is not None
            )

            # Preserve the old exact-postcode / legacy-campus collapse signals.
            if not linked:
                linked = bool(
                    should_bypass_google_between_sites(
                        sites[idx_a], sites[idx_b], radius_km=radius_km
                    )[0]
                )
            if not linked:
                try:
                    linked = same_campus(
                        sites[idx_a].get("building_name", ""),
                        sites[idx_a].get("postcode", ""),
                        sites[idx_b].get("building_name", ""),
                        sites[idx_b].get("postcode", ""),
                    )
                except Exception:
                    linked = False

            if linked:
                adjacency[idx_a].add(idx_b)
                adjacency[idx_b].add(idx_a)

    components = []
    unseen = set(indices)
    while unseen:
        start = min(unseen)
        unseen.remove(start)
        stack = [start]
        component = []
        while stack:
            node = stack.pop()
            component.append(node)
            for neighbour in sorted(adjacency[node], reverse=True):
                if neighbour in unseen:
                    unseen.remove(neighbour)
                    stack.append(neighbour)
        components.append(sorted(component))

    components.sort(key=lambda values: values[0])
    return components


def _coordinate_shortest_paths_from_current(
    current_site: dict,
    sites: List[dict],
    candidate_indices: Sequence[int],
    radius_km: float = NO_GOOGLE_RADIUS_KM,
) -> Dict[int, float]:
    """Shortest chained local distance from current site to connected sites."""
    current_cluster = _planning_cluster_key(current_site)
    if not current_cluster:
        return {}

    eligible = [
        idx
        for idx in candidate_indices
        if _planning_cluster_key(sites[idx]) == current_cluster
    ]
    nodes = [-1] + eligible
    node_site = {-1: current_site}
    node_site.update({idx: sites[idx] for idx in eligible})
    adjacency = {node: [] for node in nodes}

    for pos, node_a in enumerate(nodes):
        for node_b in nodes[pos + 1:]:
            edge = _coordinate_edge_distance_km(
                node_site[node_a], node_site[node_b], radius_km
            )
            if edge is not None:
                adjacency[node_a].append((node_b, edge))
                adjacency[node_b].append((node_a, edge))

    distances = {-1: 0.0}
    visited = set()
    while True:
        choices = [
            (distance, node)
            for node, distance in distances.items()
            if node not in visited
        ]
        if not choices:
            break
        distance, node = min(choices)
        visited.add(node)
        for neighbour, edge in adjacency[node]:
            candidate_distance = distance + edge
            if candidate_distance < distances.get(neighbour, float("inf")):
                distances[neighbour] = candidate_distance

    return {
        idx: float(distance)
        for idx, distance in distances.items()
        if idx >= 0
    }


def _same_confident_road(site_a: dict, site_b: dict) -> bool:
    road_a = infer_road_name_from_building_name(
        site_a.get("building_name", "")
    )
    road_b = infer_road_name_from_building_name(
        site_b.get("building_name", "")
    )
    return (
        road_a is not None
        and road_b is not None
        and road_a == road_b
    )


# ============================================================================
# DAILY ROUTE-SEQUENCING OPTIMISATION
# ============================================================================
#
# This layer does NOT change the strategic Planning Cluster, eligibility, or
# surveyor/day allocation. It only supplies a preferred visit order within the
# daily candidate set already handed to this scheduler.
#
# Sequencing micro-clusters may be connected by:
#   - the existing coordinate no-Google radius;
#   - a confidently matching road name; or
#   - a stricter version of the existing same-campus/development check.
#
# These are sequencing signals only. The existing Google-bypass rules remain
# the authority on whether a journey is actually sent to Google.


def _site_coordinate(site: dict):
    try:
        latitude = float(site.get("latitude"))
        longitude = float(site.get("longitude"))
    except (TypeError, ValueError):
        return None

    if not (
        math.isfinite(latitude)
        and math.isfinite(longitude)
        and -90.0 <= latitude <= 90.0
        and -180.0 <= longitude <= 180.0
    ):
        return None

    return (latitude, longitude)


def _sequencing_components_for_indices(
    sites: List[dict],
    indices: Sequence[int],
    radius_km: float = NO_GOOGLE_RADIUS_KM,
) -> List[List[int]]:
    """
    Build sequencing-only local micro-clusters inside the existing strategic
    Planning Cluster.

    The coordinate relation is a graph relation rather than a representative
    radius relation. Therefore A-B-C-D remains one local group when each
    successive link is <= radius, even if A-D is farther than radius.
    """
    indices = list(indices)
    if not indices:
        return []

    adjacency = {idx: set() for idx in indices}

    for position, idx_a in enumerate(indices):
        site_a = sites[idx_a]
        cluster_a = _planning_cluster_key(site_a)

        for idx_b in indices[position + 1:]:
            site_b = sites[idx_b]

            # Never merge sequencing groups across a strategic cluster boundary.
            if cluster_a != _planning_cluster_key(site_b):
                continue

            linked = (
                _coordinate_edge_distance_km(
                    site_a,
                    site_b,
                    radius_km=radius_km,
                )
                is not None
            )

            # A confidently matching road is a sequencing micro-cluster signal,
            # even when the two ends of the road are farther apart than radius.
            if not linked:
                linked = _same_confident_road(site_a, site_b)

            # Preserve estate/development continuity, but use a deliberately
            # stricter threshold than the old generic campus fallback so vague
            # name similarity cannot join separate areas.
            if not linked:
                try:
                    linked = same_campus(
                        site_a.get("building_name", ""),
                        site_a.get("postcode", ""),
                        site_b.get("building_name", ""),
                        site_b.get("postcode", ""),
                        similarity_threshold=0.80,
                    )
                except Exception:
                    linked = False

            if linked:
                adjacency[idx_a].add(idx_b)
                adjacency[idx_b].add(idx_a)

    components = []
    unseen = set(indices)

    while unseen:
        start = min(unseen)
        unseen.remove(start)
        stack = [start]
        component = []

        while stack:
            node = stack.pop()
            component.append(node)

            for neighbour in sorted(adjacency[node], reverse=True):
                if neighbour in unseen:
                    unseen.remove(neighbour)
                    stack.append(neighbour)

        components.append(sorted(component))

    components.sort(key=lambda values: values[0])
    return components


def _site_identity_for_sequence(site: dict) -> str:
    reference = str(site.get("customer_reference", "") or "").strip()
    if reference:
        return reference

    return (
        str(site.get("building_name", "") or "").strip()
        + "|"
        + str(site.get("postcode", "") or "").strip()
    )


def _assign_stable_sequence_component_keys(
    sites: List[dict],
    radius_km: float = NO_GOOGLE_RADIUS_KM,
) -> None:
    """
    Attach stable sequencing-only micro-cluster keys.

    The key is calculated once before the day starts so a chain group does not
    change identity when its bridge buildings are completed and removed.
    """
    components = _sequencing_components_for_indices(
        sites,
        range(len(sites)),
        radius_km=radius_km,
    )

    for component in components:
        identities = tuple(
            sorted(
                _site_identity_for_sequence(sites[idx])
                for idx in component
            )
        )
        strategic_cluster = (
            _planning_cluster_key(sites[component[0]])
            if component
            else ""
        )
        key = (strategic_cluster, identities)

        for idx in component:
            sites[idx]["_sequence_component_key"] = key


def _component_centroid(
    sites: List[dict],
    component: Sequence[int],
):
    coordinates = [
        _site_coordinate(sites[idx])
        for idx in component
    ]
    coordinates = [
        coordinate
        for coordinate in coordinates
        if coordinate is not None
    ]

    if not coordinates:
        return None

    return (
        sum(value[0] for value in coordinates) / len(coordinates),
        sum(value[1] for value in coordinates) / len(coordinates),
    )


def _coordinate_distance_between_points(point_a, point_b):
    if point_a is None or point_b is None:
        return None

    return haversine_km(
        point_a[0],
        point_a[1],
        point_b[0],
        point_b[1],
    )


def _open_route_distance(
    order,
    point_by_id,
    start_point=None,
) -> float:
    """Straight-line length of an open route with an optional fixed start."""
    if not order:
        return 0.0

    total = 0.0
    previous = start_point

    for node in order:
        point = point_by_id.get(node)

        # Missing-coordinate sites retain their existing routing fallback rather
        # than being assigned a fabricated geographic position.
        if point is None:
            previous = None
            continue

        if previous is not None:
            distance = _coordinate_distance_between_points(
                previous,
                point,
            )
            if distance is not None:
                total += float(distance)

        previous = point

    return total


def _two_opt_open_route(
    order,
    point_by_id,
    start_point=None,
):
    """
    Lightweight 2-opt improvement pass.

    The nearest-neighbour route is the seed. Reversals are accepted only when
    they reduce total coordinate distance, so the pass removes obvious
    crossovers/backtracking without introducing any Google calls.
    """
    best = list(order)

    if len(best) < 3:
        return best

    best_distance = _open_route_distance(
        best,
        point_by_id,
        start_point=start_point,
    )

    improved = True
    passes = 0
    max_passes = max(2, min(6, len(best)))

    while improved and passes < max_passes:
        improved = False
        passes += 1

        for i in range(len(best) - 1):
            for j in range(i + 1, len(best)):
                candidate = (
                    best[:i]
                    + list(reversed(best[i:j + 1]))
                    + best[j + 1:]
                )
                candidate_distance = _open_route_distance(
                    candidate,
                    point_by_id,
                    start_point=start_point,
                )

                if candidate_distance + 1e-9 < best_distance:
                    best = candidate
                    best_distance = candidate_distance
                    improved = True
                    break

            if improved:
                break

    return best


def _nearest_neighbour_order(
    node_ids,
    point_by_id,
    start_point=None,
):
    """Deterministic coordinate nearest-neighbour seed."""
    remaining = list(node_ids)
    ordered = []
    current = start_point

    while remaining:
        def key(node):
            point = point_by_id.get(node)
            distance = _coordinate_distance_between_points(
                current,
                point,
            )

            if distance is None:
                return (1, float("inf"), str(node))

            return (0, float(distance), str(node))

        chosen = min(remaining, key=key)
        remaining.remove(chosen)
        ordered.append(chosen)

        chosen_point = point_by_id.get(chosen)
        if chosen_point is not None:
            current = chosen_point

    return ordered


def _optimised_component_order(
    sites: List[dict],
    components: Sequence[Sequence[int]],
    start_site: Optional[dict] = None,
):
    """
    Determine a continuous sweep through disconnected micro-clusters using
    nearest-neighbour followed by 2-opt.
    """
    component_ids = list(range(len(components)))
    point_by_id = {
        component_id: _component_centroid(
            sites,
            component,
        )
        for component_id, component in enumerate(components)
    }
    start_point = _site_coordinate(start_site or {})

    seed = _nearest_neighbour_order(
        component_ids,
        point_by_id,
        start_point=start_point,
    )

    return _two_opt_open_route(
        seed,
        point_by_id,
        start_point=start_point,
    )


def _optimised_site_order_within_component(
    sites: List[dict],
    component: Sequence[int],
    start_site: Optional[dict] = None,
):
    """
    Determine the preferred order inside one micro-cluster.

    Same-road continuity is a strong seed preference, then coordinate distance
    determines the nearest next building. A 2-opt pass removes residual
    backtracking.
    """
    point_by_id = {
        idx: _site_coordinate(sites[idx])
        for idx in component
    }
    start_point = _site_coordinate(start_site or {})

    remaining = list(component)
    seed = []
    current_site = start_site
    current_point = start_point

    while remaining:
        def key(idx):
            candidate = sites[idx]
            candidate_point = point_by_id.get(idx)
            distance = _coordinate_distance_between_points(
                current_point,
                candidate_point,
            )
            same_road = (
                current_site is not None
                and _same_confident_road(
                    current_site,
                    candidate,
                )
            )

            return (
                0 if same_road else 1,
                0 if distance is not None else 1,
                float(distance)
                if distance is not None
                else float("inf"),
                idx,
            )

        chosen = min(remaining, key=key)
        remaining.remove(chosen)
        seed.append(chosen)

        current_site = sites[chosen]
        chosen_point = point_by_id.get(chosen)
        if chosen_point is not None:
            current_point = chosen_point

    return _two_opt_open_route(
        seed,
        point_by_id,
        start_point=start_point,
    )


def _daily_route_sequence_ranks(
    sites: List[dict],
    current_site: Optional[dict] = None,
):
    """
    Build the separate Stage-2 route sequence for the current strategic cluster.

    Only ordering ranks are returned. No site is added, removed, made eligible
    or moved to another strategic cluster/day by this function.
    """
    if not sites:
        return {}, {}, {}

    current_cluster = _planning_cluster_key(
        current_site or {}
    )

    # After the first survey, optimise only the strategic cluster currently
    # being worked. Other strategic clusters retain the existing Google logic.
    if current_cluster:
        indices = [
            idx
            for idx, site in enumerate(sites)
            if _planning_cluster_key(site) == current_cluster
        ]
    else:
        indices = list(range(len(sites)))

    if not indices:
        return {}, {}, {}

    # Use stable micro-cluster keys assigned before the day starts.
    grouped = {}
    for idx in indices:
        key = sites[idx].get("_sequence_component_key")
        if key is None:
            key = ("dynamic", idx)
        grouped.setdefault(key, []).append(idx)

    components = list(grouped.values())
    components.sort(key=lambda values: min(values))

    if not components:
        return {}, {}, {}

    component_order = _optimised_component_order(
        sites,
        components,
        start_site=current_site,
    )

    site_rank = {}
    component_rank = {}
    component_id_by_index = {}
    running_rank = 0
    previous_site = current_site

    for rank, component_id in enumerate(component_order):
        component = components[component_id]
        ordered_sites = _optimised_site_order_within_component(
            sites,
            component,
            start_site=previous_site,
        )

        for idx in ordered_sites:
            component_rank[idx] = rank
            component_id_by_index[idx] = component_id
            site_rank[idx] = running_rank
            running_rank += 1
            previous_site = sites[idx]

    return (
        site_rank,
        component_rank,
        component_id_by_index,
    )



def _remaining_cluster_proximity_km(
    current_site: dict,
    sites: List[dict],
) -> Dict[str, float]:
    """
    Minimum straight-line distance from the current site to each remaining
    strategic Planning Cluster.

    Used only to rank which already-assigned cluster should be visited next.
    Google remains the actual public-transport authority and no API calls are
    added here.
    """
    current_coordinate = _site_coordinate(current_site)
    if current_coordinate is None:
        return {}

    result: Dict[str, float] = {}

    for site in sites:
        cluster = _planning_cluster_key(site)
        if not cluster:
            continue

        site_coordinate = _site_coordinate(site)
        if site_coordinate is None:
            continue

        distance = haversine_km(
            current_coordinate[0],
            current_coordinate[1],
            site_coordinate[0],
            site_coordinate[1],
        )
        if distance is None:
            continue

        previous = result.get(cluster)
        if previous is None or float(distance) < previous:
            result[cluster] = float(distance)

    return result



def _select_google_groups_for_decision(
    proximity_groups: Sequence[dict],
    sites: List[dict],
    current_site: Optional[dict],
    current_planning_cluster: str,
    remaining_cluster_proximity: Dict[str, float],
    route_site_rank: Dict[int, int],
    max_groups: int = GOOGLE_GROUPS_PER_DECISION,
    fallback_groups: int = GOOGLE_FALLBACK_GROUPS_PER_DECISION,
) -> List[dict]:
    """
    Choose the small set of disconnected local-group representatives that
    actually need Google TRANSIT validation for this scheduling decision.

    This is a Google-workload limit, NOT a site/candidate limit. Groups omitted
    from this particular Google call stay in `remaining` and are reconsidered
    on later iterations.

    Priority:
      1) finish the current strategic cluster;
      2) then prefer the geographically nearest next strategic cluster;
      3) reserve a couple of fallback groups so one infeasible cluster cannot
         prematurely end the day;
      4) before the first survey, preserve existing shortlist/priority order.
    """
    groups = list(proximity_groups)
    if not groups:
        return []

    max_groups = max(1, int(max_groups))
    fallback_groups = max(
        0,
        min(int(fallback_groups), max_groups - 1),
    )

    def group_cluster(group):
        idx = int(group["representative_idx"])
        return _planning_cluster_key(sites[idx])

    def representative_distance(group):
        if current_site is None:
            return float("inf")
        idx = int(group["representative_idx"])
        representative = sites[idx]
        distance = haversine_km(
            current_site.get("latitude"),
            current_site.get("longitude"),
            representative.get("latitude"),
            representative.get("longitude"),
        )
        return (
            float(distance)
            if distance is not None
            else float("inf")
        )

    def group_route_rank(group):
        idx = int(group["representative_idx"])
        return int(route_site_rank.get(idx, 10**6))

    # First survey: no home coordinates are stored in this scheduler, so retain
    # strategic shortlist order but avoid sending the whole shortlist to Google.
    if current_site is None:
        cluster_order = []
        by_cluster = {}

        for position, group in enumerate(groups):
            cluster = group_cluster(group)
            if cluster not in by_cluster:
                by_cluster[cluster] = []
                cluster_order.append(cluster)
            by_cluster[cluster].append((position, group))

        if len(cluster_order) <= 1:
            return groups[:max_groups]

        primary_budget = max_groups - fallback_groups
        selected = [
            group
            for _, group in by_cluster[cluster_order[0]][:primary_budget]
        ]

        fallbacks = []
        for cluster in cluster_order[1:]:
            fallbacks.extend(by_cluster[cluster])
        fallbacks.sort(key=lambda item: item[0])

        selected.extend(
            group
            for _, group in fallbacks[
                : max_groups - len(selected)
            ]
        )
        return selected[:max_groups]

    current_groups = []
    other_groups = []

    for position, group in enumerate(groups):
        cluster = group_cluster(group)
        item = (
            group_route_rank(group),
            representative_distance(group),
            position,
            group,
        )

        if (
            current_planning_cluster
            and cluster == current_planning_cluster
        ):
            current_groups.append(item)
        else:
            other_groups.append(item)

    current_groups.sort(
        key=lambda item: (
            item[0],
            item[1],
            item[2],
        )
    )

    other_groups.sort(
        key=lambda item: (
            remaining_cluster_proximity.get(
                group_cluster(item[3]),
                float("inf"),
            ),
            item[1],
            item[0],
            item[2],
        )
    )

    # While the current cluster still has Google-required groups, keep most
    # Google slots there and reserve only a small fallback allowance.
    if current_groups:
        local_budget = max_groups - fallback_groups
        selected = [
            item[3]
            for item in current_groups[:local_budget]
        ]
        selected.extend(
            item[3]
            for item in other_groups[
                : max_groups - len(selected)
            ]
        )
        return selected[:max_groups]

    # Current cluster exhausted: concentrate on the nearest next cluster while
    # keeping a couple of next-nearest fallbacks.
    by_cluster = {}
    cluster_order = []

    for item in other_groups:
        group = item[3]
        cluster = group_cluster(group)
        if cluster not in by_cluster:
            by_cluster[cluster] = []
            cluster_order.append(cluster)
        by_cluster[cluster].append(item)

    if not cluster_order:
        return []

    primary_cluster = cluster_order[0]
    primary_budget = max_groups - fallback_groups
    selected = [
        item[3]
        for item in by_cluster[primary_cluster][:primary_budget]
    ]

    fallbacks = []
    for cluster in cluster_order[1:]:
        fallbacks.extend(by_cluster[cluster])

    fallbacks.sort(
        key=lambda item: (
            remaining_cluster_proximity.get(
                group_cluster(item[3]),
                float("inf"),
            ),
            item[1],
            item[0],
            item[2],
        )
    )

    selected.extend(
        item[3]
        for item in fallbacks[
            : max_groups - len(selected)
        ]
    )

    return selected[:max_groups]


def postcode_district(postcode: str) -> str:
    text = str(postcode or "").strip().upper()
    if not text:
        return ""
    # UK outward code, e.g. HA8, NW9, SW1V, SG14.
    # Standard formatted UK postcodes place a space before the inward code.
    parts = text.split()
    if len(parts) >= 2:
        return parts[0]
    # Fallback for compact postcodes: remove the final three-character inward code.
    compact = re.sub(r"\s+", "", text)
    return compact[:-3] if len(compact) > 3 else compact


def _normalise_postcode(postcode: str) -> str:
    return re.sub(r"\s+", "", str(postcode or "").upper())


def _normalise_site_name(name: str, postcode: str = "") -> str:
    """
    Reduce Salesforce building/address text to a comparable campus signature.

    Examples:
      "102831 | A-B, 155 Cambridge Street SW1V 4QB"
      "101100 | 157 Cambridge Street SW1V 4QB"

    both become close to "cambridge street".
    """
    text = str(name or "").lower()

    # Remove a leading Salesforce / asset reference before the pipe.
    if "|" in text:
        text = text.split("|", 1)[1]

    pc = _normalise_postcode(postcode).lower()
    compact_text = re.sub(r"\s+", "", text)
    if pc and pc in compact_text:
        # Remove postcode in a spacing-insensitive way by first replacing the
        # normally formatted postcode variants.
        formatted = str(postcode or "").lower().strip()
        if formatted:
            text = text.replace(formatted, " ")

    # Remove flat numbers, street numbers and punctuation so nearby blocks with
    # the same underlying estate/address wording compare well.
    text = re.sub(r"\b\d+[a-z]?\b", " ", text)
    text = re.sub(r"\b[a-z]\s*-\s*[a-z]\b", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)

    # These tokens commonly vary between blocks on the same campus but do not
    # materially identify a different site.
    weak_tokens = {
        "block", "blocks", "building", "buildings",
        "flat", "flats", "car", "park", "parking",
        "entrance", "wing", "tower",
    }

    tokens = [
        token
        for token in text.split()
        if len(token) > 1 and token not in weak_tokens
    ]
    return " ".join(tokens)


def same_campus(
    name_a: str,
    postcode_a: str,
    name_b: str,
    postcode_b: str,
    similarity_threshold: float = 0.62,
) -> bool:
    """
    Treat two buildings as the same campus/site group when:
      1) their full postcodes match; and
      2) their normalised building/address names are similar.

    This is intentionally conservative: postcode match is mandatory.
    """
    pc_a = _normalise_postcode(postcode_a)
    pc_b = _normalise_postcode(postcode_b)
    if not pc_a or pc_a != pc_b:
        return False

    a = _normalise_site_name(name_a, postcode_a)
    b = _normalise_site_name(name_b, postcode_b)
    if not a or not b:
        return False

    tokens_a = set(a.split())
    tokens_b = set(b.split())

    # Strong signal: one address signature is contained within the other.
    if tokens_a.issubset(tokens_b) or tokens_b.issubset(tokens_a):
        if len(tokens_a & tokens_b) >= 2:
            return True

    jaccard = (
        len(tokens_a & tokens_b) / len(tokens_a | tokens_b)
        if (tokens_a | tokens_b)
        else 0.0
    )
    sequence = SequenceMatcher(None, a, b).ratio()

    return max(jaccard, sequence) >= similarity_threshold


@dataclass
class ScheduledSurvey:
    sequence: int
    customer_reference: str
    building_name: str
    postcode: str
    cluster: str
    depart_previous: datetime
    travel_minutes: float
    arrive_site: datetime
    survey_minutes: float
    survey_start: datetime
    survey_end: datetime
    predicted_minutes: float
    confidence: str
    model_used: str
    building_height: Optional[float] = None
    flats: Optional[float] = None
    ground_floor_area: Optional[float] = None


@dataclass
class DailyScheduleResult:
    items: List[ScheduledSurvey]
    start_location: str
    start_time: datetime
    return_location: str
    return_departure: datetime
    return_travel_minutes: float
    return_time: datetime
    latest_return: datetime
    unscheduled_count: int
    first_survey_target: Optional[datetime] = None
    latest_survey_finish: Optional[datetime] = None
    lunch_start: Optional[datetime] = None
    lunch_end: Optional[datetime] = None
    lunch_location: str = ""
    lunch_after_sequence: Optional[int] = None
    routing_stats: Optional[dict] = None

    @property
    def survey_minutes(self) -> float:
        return round(sum(i.survey_minutes for i in self.items), 1)

    @property
    def travel_minutes(self) -> int:
        return round(
            sum(i.travel_minutes for i in self.items)
            + self.return_travel_minutes
        )

    def to_dataframe(self) -> pd.DataFrame:
        rows = []

        def lunch_row():
            return {
                "Sequence": "LUNCH",
                "Customer Reference": "",
                "Building Name": (
                    f"Lunch break — {self.lunch_location}"
                    if self.lunch_location
                    else "Lunch break"
                ),
                "Postcode": "",
                "Cluster": "",
                "Depart Previous": "",
                "Travel (Minutes)": "",
                "Arrive": "",
                "Survey Start": (
                    self.lunch_start.strftime("%H:%M")
                    if self.lunch_start
                    else ""
                ),
                "Survey End": (
                    self.lunch_end.strftime("%H:%M")
                    if self.lunch_end
                    else ""
                ),
                "Planning Survey Duration (Minutes)": 30,
                "Raw Predicted Duration (Minutes)": "",
                "Building Height": "",
                "Sovereign Flat": "",
                "Internal Ground Floor Area (m2)": "",
                "Prediction Confidence": "",
                "Prediction Model": "Lunch break",
            }

        lunch_inserted = False
        if (
            self.lunch_start is not None
            and self.lunch_after_sequence == 0
        ):
            rows.append(lunch_row())
            lunch_inserted = True

        for item in self.items:
            rows.append({
                "Sequence": item.sequence,
                "Customer Reference": item.customer_reference,
                "Building Name": item.building_name,
                "Postcode": item.postcode,
                "Cluster": item.cluster,
                "Depart Previous": item.depart_previous.strftime("%H:%M"),
                "Travel (Minutes)": round(item.travel_minutes),
                "Arrive": item.arrive_site.strftime("%H:%M"),
                "Survey Start": item.survey_start.strftime("%H:%M"),
                "Survey End": item.survey_end.strftime("%H:%M"),
                "Planning Survey Duration (Minutes)": item.survey_minutes,
                "Raw Predicted Duration (Minutes)": item.predicted_minutes,
                "Building Height": item.building_height,
                "Sovereign Flat": item.flats,
                "Internal Ground Floor Area (m2)": item.ground_floor_area,
                "Prediction Confidence": item.confidence,
                "Prediction Model": item.model_used,
            })

            if (
                self.lunch_start is not None
                and not lunch_inserted
                and self.lunch_after_sequence == item.sequence
            ):
                rows.append(lunch_row())
                lunch_inserted = True

        if self.lunch_start is not None and not lunch_inserted:
            rows.append(lunch_row())

        rows.append({
            "Sequence": "RETURN",
            "Customer Reference": "",
            "Building Name": self.return_location,
            "Postcode": "",
            "Cluster": "",
            "Depart Previous": self.return_departure.strftime("%H:%M"),
            "Travel (Minutes)": round(self.return_travel_minutes),
            "Arrive": self.return_time.strftime("%H:%M"),
            "Survey Start": "",
            "Survey End": "",
            "Planning Survey Duration (Minutes)": "",
            "Raw Predicted Duration (Minutes)": "",
            "Building Height": "",
            "Sovereign Flat": "",
            "Internal Ground Floor Area (m2)": "",
            "Prediction Confidence": "",
            "Prediction Model": "",
        })
        return pd.DataFrame(rows)



@dataclass
class WeeklyScheduleResult:
    days: List[DailyScheduleResult]
    unscheduled_sites: List[dict]

    @property
    def total_surveys(self) -> int:
        return sum(len(day.items) for day in self.days)

    @property
    def total_survey_minutes(self) -> float:
        return sum(day.survey_minutes for day in self.days)

    @property
    def total_travel_minutes(self) -> int:
        return sum(day.travel_minutes for day in self.days)

    @property
    def routing_stats(self) -> dict:
        """Aggregate reporting-only routing counters across the final week."""
        keys = {
            "candidate_evaluations",
            "google_matrix_logical_calls",
            "google_matrix_destinations",
            "same_postcode_bypasses",
            "coordinate_radius_bypasses",
            "legacy_campus_bypasses",
            "collapsed_nearby_destinations",
        }
        totals = {key: 0 for key in keys}
        for day in self.days:
            for key, value in (day.routing_stats or {}).items():
                if key in totals:
                    totals[key] += int(value or 0)
        totals["local_bypasses"] = (
            totals["same_postcode_bypasses"]
            + totals["coordinate_radius_bypasses"]
            + totals["legacy_campus_bypasses"]
        )
        totals["routing_calculations_avoided"] = (
            totals["local_bypasses"]
            + totals["collapsed_nearby_destinations"]
        )
        denominator = (
            totals["google_matrix_destinations"]
            + totals["routing_calculations_avoided"]
        )
        totals["site_to_site_avoidance_pct"] = (
            100.0 * totals["routing_calculations_avoided"] / denominator
            if denominator
            else 0.0
        )
        return totals

    def summary_dataframe(self) -> pd.DataFrame:
        rows = []
        for day in self.days:
            clusters = []
            for item in day.items:
                if item.cluster and item.cluster not in clusters:
                    clusters.append(item.cluster)

            service_dt = day.first_survey_target or day.start_time
            first_actual = day.items[0].survey_start if day.items else None
            last_actual = day.items[-1].survey_end if day.items else None
            rows.append({
                "Date": service_dt.strftime("%A %d %B %Y"),
                "Surveys": len(day.items),
                "Clusters": " → ".join(clusters),
                "Leave Harpenden": day.start_time.strftime("%H:%M"),
                "First Survey": first_actual.strftime("%H:%M") if first_actual else "",
                "Last Survey Finish": last_actual.strftime("%H:%M") if last_actual else "",
                "Return Harpenden": day.return_time.strftime("%H:%M"),
                "Survey Time (Minutes)": day.survey_minutes,
                "Travel Time (Minutes)": day.travel_minutes,
            })

        return pd.DataFrame(rows)

    def full_schedule_dataframe(self) -> pd.DataFrame:
        frames = []
        for day in self.days:
            df = day.to_dataframe().copy()
            service_dt = day.first_survey_target or day.start_time
            df.insert(0, "Date", service_dt.date().isoformat())
            df.insert(1, "Day", service_dt.strftime("%A"))
            frames.append(df)

        if not frames:
            return pd.DataFrame()

        return pd.concat(frames, ignore_index=True)


class DailyTransitScheduler:
    """
    Greedy, time-dependent public-transport scheduler.

    Latitude/Longitude provide the cheap geographic layer before Google.
    Buildings within the configured no-Google radius, or with the same full
    postcode, bypass Google for their site-to-site move and use a conservative
    local walking/transfer estimate instead. Inside each existing strategic
    Planning Cluster, successive local coordinate links form connected groups,
    which are traversed nearest-next before leaving for another local group.

    A separate Stage-2 coordinate sequencing layer orders candidates already
    supplied to the day by micro-cluster, nearest-neighbour and a lightweight
    2-opt improvement pass, with a strong penalty for leaving and later
    re-entering a local area.

    Google remains the final public-transport authority for meaningful journeys,
    first-site travel and return-home feasibility. Hard time/lunch rules are
    unchanged.
    """

    def __init__(
        self,
        router,
        home_location: str = "Harpenden Station",
        max_candidate_checks: int = 8,
        same_postcode_transfer_minutes: int = 5,
        travel_leeway_minutes: int = 5,
        pre_survey_buffer_minutes: int = 5,
        post_survey_buffer_minutes: int = 5,
        ai_priority_weight_minutes: float = 20.0,
        lunch_minutes: int = 30,
        lunch_window_start_clock=time(11, 45),
        lunch_latest_start_clock=time(13, 0),
    ):
        self.router = router
        self.home_location = home_location
        self.max_candidate_checks = max_candidate_checks
        self.same_postcode_transfer_minutes = same_postcode_transfer_minutes
        self.travel_leeway_minutes = travel_leeway_minutes
        self.pre_survey_buffer_minutes = pre_survey_buffer_minutes
        self.post_survey_buffer_minutes = post_survey_buffer_minutes
        self.ai_priority_weight_minutes = ai_priority_weight_minutes
        self.lunch_minutes = lunch_minutes
        self.lunch_window_start_clock = lunch_window_start_clock
        self.lunch_latest_start_clock = lunch_latest_start_clock

    @staticmethod
    def _same_postcode(a: str, b: str) -> bool:
        normalise = lambda x: re.sub(r"\s+", "", str(x or "").upper())
        return bool(normalise(a)) and normalise(a) == normalise(b)

    def _site_to_site_leeway_minutes(
        self,
        current_site: dict,
        next_site: dict,
    ) -> float:
        """
        Return Team Travel Leeway for direct site-to-site travel only.

        Rules:
          - same confidently identified road -> 0 minutes
          - <= 0.20 km -> 0 minutes
          - >0.20 to 0.50 km -> 2 minutes
          - >0.50 to 1.00 km -> 3 minutes
          - >1.00 to 2.00 km -> 5 minutes
          - >2.00 km -> existing UI Team Travel Leeway
          - missing coordinates -> existing UI Team Travel Leeway

        This changes only the additional leeway. It does not change Google
        journey times, local-transfer estimates, the no-Google threshold,
        clustering, ordering, home-to-first-site or return-home behaviour.
        """
        current_road = infer_road_name_from_building_name(
            current_site.get("building_name", "")
        )
        next_road = infer_road_name_from_building_name(
            next_site.get("building_name", "")
        )

        if (
            current_road is not None
            and next_road is not None
            and current_road == next_road
        ):
            return 0.0

        distance_km = haversine_km(
            current_site.get("latitude"),
            current_site.get("longitude"),
            next_site.get("latitude"),
            next_site.get("longitude"),
        )

        if distance_km is None:
            return float(self.travel_leeway_minutes)

        if distance_km <= 0.20:
            return 0.0
        if distance_km <= 0.50:
            return 2.0
        if distance_km <= 1.00:
            return 3.0
        if distance_km <= 2.00:
            return 5.0

        return float(self.travel_leeway_minutes)

    def build_day(
        self,
        sites: List[dict],
        first_survey_start: datetime,
        latest_survey_finish: datetime,
        latest_return: datetime,
    ) -> DailyScheduleResult:
        remaining = [dict(site) for site in sites]

        # Stage 2 route-sequencing metadata only. This does not change Stage 1
        # cluster membership, eligibility, or the candidate sites supplied to
        # the day.
        _assign_stable_sequence_component_keys(
            remaining,
            radius_km=NO_GOOGLE_RADIUS_KM,
        )

        scheduled: List[ScheduledSurvey] = []

        # Reporting only. Counts the routing work the existing algorithm performs.
        routing_stats = {
            "candidate_evaluations": 0,
            "google_matrix_logical_calls": 0,
            "google_matrix_destinations": 0,
            "same_postcode_bypasses": 0,
            "coordinate_radius_bypasses": 0,
            "legacy_campus_bypasses": 0,
            "collapsed_nearby_destinations": 0,
        }

        current_location = self.home_location
        current_postcode = ""
        current_building_name = ""
        current_latitude = None
        current_longitude = None
        current_planning_cluster = ""
        current_sequence_component_key = None

        # Route-sequencing state. A completed/left micro-cluster is remembered
        # so A -> B -> A receives a strong re-entry penalty.
        last_sequence_component_key = None
        closed_sequence_component_keys = set()

        # For the first leg, Google needs a departure time to price/rank transit.
        # Probe shortly before the requested first-survey start, then back-calculate
        # the actual home departure for the chosen first site. This stops a long
        # commute from consuming the user's survey-work window.
        current_time = first_survey_start - timedelta(minutes=90)
        home_departure_time = current_time

        lunch_taken = False
        lunch_start = None
        lunch_end = None
        lunch_location = ""
        lunch_after_sequence = None
        lunch_window_start = datetime.combine(
            first_survey_start.date(),
            self.lunch_window_start_clock,
            tzinfo=first_survey_start.tzinfo,
        )
        lunch_latest_start = datetime.combine(
            first_survey_start.date(),
            self.lunch_latest_start_clock,
            tzinfo=first_survey_start.tzinfo,
        )

        while remaining:
            # Take lunch at the first natural between-survey boundary from 11:45.
            # The break must START by 13:00. It is a hard constraint, not an AI
            # preference, and therefore cannot be skipped to fit another survey.
            if (
                not lunch_taken
                and lunch_window_start <= current_time <= lunch_latest_start
            ):
                lunch_start = current_time
                lunch_end = lunch_start + timedelta(minutes=self.lunch_minutes)
                lunch_location = current_location
                lunch_after_sequence = len(scheduled)
                current_time = lunch_end
                lunch_taken = True

            # This should only be reachable if no feasible pre-13:00 lunch could
            # be protected. Do not schedule additional work after missing lunch.
            if not lunch_taken and current_time > lunch_latest_start:
                break
            # Coordinate-first Google reduction.
            #
            # After the first site is reached, a remaining building is NOT sent
            # to the Google Route Matrix for the site-to-site move when:
            #   - it shares the exact full postcode; OR
            #   - its coordinates are within NO_GOOGLE_RADIUS_KM of the current site.
            # The old same-campus name check remains as an additional fallback.
            #
            # External destinations are also collapsed when they are physically
            # close to one another. Google therefore sees one representative for
            # a micro-area instead of many nearly identical nearby buildings.
            routing_stats["candidate_evaluations"] += len(remaining)
            travel_by_index = {}
            external_indices = []

            current_site = {
                "building_name": current_building_name,
                "postcode": current_postcode,
                "latitude": current_latitude,
                "longitude": current_longitude,
                "planning_cluster": current_planning_cluster,
                "_sequence_component_key": current_sequence_component_key,
            }

            connected_path_km = (
                _coordinate_shortest_paths_from_current(
                    current_site,
                    remaining,
                    range(len(remaining)),
                    radius_km=NO_GOOGLE_RADIUS_KM,
                )
                if scheduled
                else {}
            )

            # Separate Stage-2 route-order optimisation. The first site retains
            # the existing Google-based behaviour; once inside a strategic
            # cluster, the already-supplied candidates are given a local
            # coordinate sequence before the normal feasibility checks.
            if scheduled:
                (
                    route_site_rank,
                    route_component_rank,
                    route_component_id,
                ) = _daily_route_sequence_ranks(
                    remaining,
                    current_site=current_site,
                )
            else:
                route_site_rank = {}
                route_component_rank = {}
                route_component_id = {}

            current_component_keys = {
                idx: site.get("_sequence_component_key")
                for idx, site in enumerate(remaining)
            }

            remaining_cluster_proximity = (
                _remaining_cluster_proximity_km(
                    current_site,
                    remaining,
                )
                if scheduled
                else {}
            )

            for idx, site in enumerate(remaining):
                bypass = False
                distance = None
                bypass_reason = ""
                if scheduled:
                    bypass, distance, bypass_reason = (
                        should_bypass_google_between_sites(
                            current_site,
                            site,
                            radius_km=NO_GOOGLE_RADIUS_KM,
                        )
                    )
                    if not bypass and current_postcode:
                        try:
                            bypass = same_campus(
                                current_building_name,
                                current_postcode,
                                site.get("building_name", ""),
                                site.get("postcode", ""),
                            )
                            if bypass:
                                bypass_reason = "legacy same-campus rule"
                        except Exception:
                            bypass = False

                if bypass:
                    if bypass_reason == "same full postcode":
                        routing_stats["same_postcode_bypasses"] += 1
                    elif bypass_reason == "legacy same-campus rule":
                        routing_stats["legacy_campus_bypasses"] += 1
                    else:
                        routing_stats["coordinate_radius_bypasses"] += 1

                    travel_by_index[idx] = estimate_local_transfer_minutes(
                        distance,
                        minimum_minutes=float(self.same_postcode_transfer_minutes),
                    )
                elif idx in connected_path_km:
                    # Reachable through a chain of <=radius local coordinate hops.
                    routing_stats["collapsed_nearby_destinations"] += 1
                    travel_by_index[idx] = estimate_local_transfer_minutes(
                        connected_path_km[idx],
                        minimum_minutes=float(self.same_postcode_transfer_minutes),
                    )
                else:
                    external_indices.append(idx)

            # Google now sees one representative per CONNECTED local component
            # rather than one representative-radius pocket.
            proximity_groups = []
            for component in _connected_components_for_indices(
                remaining,
                external_indices,
                radius_km=NO_GOOGLE_RADIUS_KM,
            ):
                representative_idx = component[0]
                proximity_groups.append({
                    "indices": component,
                    "representative_idx": representative_idx,
                    "route_location": remaining[representative_idx]["route_location"],
                })

            # All candidate sites stay available, but only the most relevant
            # disconnected local groups enter Google on this iteration.
            google_groups = _select_google_groups_for_decision(
                proximity_groups=proximity_groups,
                sites=remaining,
                current_site=(current_site if scheduled else None),
                current_planning_cluster=current_planning_cluster,
                remaining_cluster_proximity=(
                    remaining_cluster_proximity
                ),
                route_site_rank=route_site_rank,
                max_groups=GOOGLE_GROUPS_PER_DECISION,
                fallback_groups=(
                    GOOGLE_FALLBACK_GROUPS_PER_DECISION
                ),
            )

            google_representative_indices = {
                group["representative_idx"]
                for group in google_groups
            }
            google_group_member_indices = {
                idx
                for group in google_groups
                for idx in group["indices"]
            }

            if google_groups:
                # Count collapse savings only for groups that actually reached
                # this Google decision.
                routing_stats["collapsed_nearby_destinations"] += sum(
                    max(0, len(group["indices"]) - 1)
                    for group in google_groups
                )

                google_destinations = [
                    group["route_location"]
                    for group in google_groups
                ]
                routing_stats["google_matrix_logical_calls"] += 1
                routing_stats["google_matrix_destinations"] += len(
                    google_destinations
                )

                google_matrix = self.router.one_to_many(
                    current_location,
                    google_destinations,
                    current_time,
                )

                for group, minutes in zip(
                    google_groups,
                    google_matrix,
                ):
                    for idx in group["indices"]:
                        travel_by_index[idx] = minutes

            ranked = []
            for idx, site in enumerate(remaining):
                minutes = travel_by_index.get(idx)

                if minutes is None:
                    continue

                # Strategic cluster first; inside it, finish the connected local
                # group nearest-next and favour a continuous same-road sequence.
                cluster_tier = 0
                next_cluster_distance_sort = float("inf")
                local_group_tier = 1

                route_component_order = (
                    route_component_rank.get(idx, 10**6)
                    if scheduled
                    else 10**6
                )
                route_site_order = (
                    route_site_rank.get(idx, 10**6)
                    if scheduled
                    else 10**6
                )

                component_key = current_component_keys.get(idx)
                reentry_tier = (
                    1
                    if (
                        component_key is not None
                        and component_key
                        in closed_sequence_component_keys
                    )
                    else 0
                )

                same_road_tier = 1
                local_distance_sort = float("inf")
                postcode_tier = 0
                google_representative_tier = 0

                if scheduled:
                    next_cluster = str(
                        site.get("planning_cluster", "")
                        or postcode_district(site.get("postcode"))
                    )
                    cluster_tier = (
                        0
                        if current_planning_cluster
                        and current_planning_cluster == next_cluster
                        else 1
                    )

                    if cluster_tier == 0:
                        next_cluster_distance_sort = 0.0
                    else:
                        next_cluster_distance_sort = (
                            remaining_cluster_proximity.get(
                                next_cluster,
                                float("inf"),
                            )
                        )

                    postcode_tier = (
                        0
                        if self._same_postcode(
                            current_postcode, site.get("postcode", "")
                        )
                        else 1
                    )

                    if idx in connected_path_km:
                        local_group_tier = 0
                        direct_distance = haversine_km(
                            current_latitude,
                            current_longitude,
                            site.get("latitude"),
                            site.get("longitude"),
                        )
                        local_distance_sort = (
                            float(direct_distance)
                            if direct_distance is not None
                            else float(connected_path_km[idx])
                        )
                        if _same_confident_road(current_site, site):
                            same_road_tier = 0
                    else:
                        direct_bypass, direct_distance, _ = (
                            should_bypass_google_between_sites(
                                current_site,
                                site,
                                radius_km=NO_GOOGLE_RADIUS_KM,
                            )
                        )
                        if direct_bypass:
                            local_group_tier = 0
                            if direct_distance is not None:
                                local_distance_sort = float(direct_distance)
                            if _same_confident_road(current_site, site):
                                same_road_tier = 0
                elif idx in google_group_member_indices:
                    google_representative_tier = (
                        0 if idx in google_representative_indices else 1
                    )

                # AI priority is advisory: Google transit remains dominant.
                # A 100-point AI priority can improve the candidate score by at
                # most ai_priority_weight_minutes; a 0-point priority adds no bonus.
                try:
                    ai_priority = float(site.get("ai_priority", 50))
                except Exception:
                    ai_priority = 50.0
                ai_priority = max(0.0, min(100.0, ai_priority))
                # Centre the AI effect at 50. High-priority sites become
                # somewhat more attractive; low-priority/deferred sites become
                # somewhat less attractive. The absolute effect remains capped
                # so AI cannot overpower real transit geography.
                ai_adjustment = (
                    (ai_priority - 50.0) / 50.0
                ) * self.ai_priority_weight_minutes

                defer_penalty = (
                    15.0
                    if str(site.get("ai_decision", "")).lower() == "defer"
                    else 0.0
                )

                # Weekly notes are a preference layer only. A preferred site is
                # fully reserved until its requested date, then strongly favoured
                # on that date. The normal feasibility/return-home checks below
                # still decide whether it can actually be scheduled.
                special_request_adjustment = 0.0
                preferred_date = site.get("special_request_date")
                if preferred_date:
                    try:
                        if hasattr(preferred_date, "date"):
                            preferred_date = preferred_date.date()
                        current_day = first_survey_start.date()
                        if current_day < preferred_date:
                            # Do not consume a request-area site on an earlier day.
                            # If the note trial later fails, the entire trial is
                            # discarded and the baseline schedule is restored.
                            continue
                        elif current_day == preferred_date:
                            special_request_adjustment -= float(
                                site.get("special_request_bonus_minutes", 75.0)
                            )
                    except Exception:
                        pass

                score = (
                    float(minutes)
                    - ai_adjustment
                    + defer_penalty
                    + special_request_adjustment
                )
                ranked.append(
                    (
                        cluster_tier,
                        next_cluster_distance_sort,
                        local_group_tier,
                        reentry_tier,
                        route_component_order,
                        route_site_order,
                        same_road_tier,
                        local_distance_sort,
                        postcode_tier,
                        google_representative_tier,
                        score,
                        idx,
                        float(minutes),
                    )
                )

            if not ranked:
                break

            # Finish the current strategic cluster first. Once it is exhausted,
            # prefer the geographically nearest remaining assigned cluster, then
            # let the existing Google travel data and feasibility rules validate
            # the actual move.
            #
            # To preserve robustness, always keep up to two out-of-cluster
            # fallback candidates in the feasibility pool. That means an
            # impossible local job cannot prematurely end the day, while a
            # distant hop cannot beat feasible nearby work merely because its
            # transit time happens to be a few minutes shorter.
            ranked.sort(
                key=lambda x: (
                    x[0],  # finish current strategic cluster first
                    x[1],  # then move to the nearest next strategic cluster
                    x[2],  # finish connected local group first
                    x[3],  # strongly penalise A -> B -> A re-entry
                    x[4],  # optimised micro-cluster order
                    x[5],  # optimised site order within micro-cluster
                    x[6],  # same-road continuity
                    x[7],  # nearest-next coordinate distance
                    x[8],  # same full postcode
                    x[9],  # Google representative before group members
                    x[10], # existing Google/AI/special-request score
                )
            )

            if scheduled:
                same_cluster = [r for r in ranked if r[0] == 0]
                other_cluster = [r for r in ranked if r[0] != 0]

                fallback_slots = (
                    min(2, len(other_cluster))
                    if same_cluster
                    else min(self.max_candidate_checks, len(other_cluster))
                )
                local_slots = max(
                    0,
                    self.max_candidate_checks - fallback_slots,
                )
                candidate_pool = (
                    same_cluster[:local_slots]
                    + other_cluster[:fallback_slots]
                )
            else:
                candidate_pool = ranked[:self.max_candidate_checks]

            chosen = None
            lunch_blocked_candidate = False

            for (
                _,
                _,
                _,
                _,
                _,
                _,
                _,
                _,
                _,
                _,
                _,
                idx,
                travel_minutes,
            ) in candidate_pool:
                site = remaining[idx]

                is_first_survey = len(scheduled) == 0

                if is_first_survey:
                    # Back-calculate home departure so the first survey begins at
                    # the selected target rather than after an arbitrary fixed
                    # "leave home" time. Use one exact Compute Routes call only for
                    # the shortlisted candidate being feasibility-tested.
                    estimated_departure = first_survey_start - timedelta(
                        minutes=(
                            travel_minutes
                            + self.travel_leeway_minutes
                            + self.pre_survey_buffer_minutes
                        )
                    )
                    try:
                        outbound_route = self.router.compute_route(
                            self.home_location,
                            site["route_location"],
                            estimated_departure,
                        )
                    except Exception:
                        continue

                    buffered_travel_minutes = (
                        float(outbound_route.duration_minutes)
                        + self.travel_leeway_minutes
                    )
                    # Re-anchor the displayed/operational home departure using
                    # the exact route duration we just received so the first
                    # survey starts at the selected time rather than merely
                    # "not before" it. This does not add another Google call.
                    depart_previous = first_survey_start - timedelta(
                        minutes=(
                            buffered_travel_minutes
                            + self.pre_survey_buffer_minutes
                        )
                    )
                    arrive = depart_previous + timedelta(
                        minutes=buffered_travel_minutes
                    )
                    survey_start = first_survey_start
                else:
                    site_to_site_leeway = self._site_to_site_leeway_minutes(
                        current_site, site
                    )
                    buffered_travel_minutes = travel_minutes + site_to_site_leeway
                    depart_previous = current_time
                    arrive = current_time + timedelta(
                        minutes=buffered_travel_minutes
                    )
                    survey_start = arrive + timedelta(
                        minutes=self.pre_survey_buffer_minutes
                    )

                # Efficiency gate for FAR moves between strategic clusters only.
                #
                # This does not affect:
                #   - work inside the current strategic cluster;
                #   - local/short cluster changes under 30 minutes;
                #   - candidate ranking;
                #   - Google journey times;
                #   - any existing hard feasibility rule.
                #
                # "Feasible survey minutes" is the remaining survey workload in
                # the destination cluster, capped by the survey-time window still
                # available after arriving there. The existing detailed scheduler
                # continues to validate every individual survey and the return
                # journey home afterwards.
                if not is_first_survey and current_planning_cluster:
                    target_cluster = _planning_cluster_key(site)
                    is_inter_cluster_jump = (
                        target_cluster
                        and target_cluster
                        != current_planning_cluster
                    )

                    if (
                        is_inter_cluster_jump
                        and float(travel_minutes)
                        >= FAR_CLUSTER_TRANSITION_MINUTES
                    ):
                        destination_work_minutes = (
                            _remaining_cluster_survey_minutes(
                                remaining,
                                target_cluster,
                                first_survey_start.date(),
                            )
                        )

                        remaining_survey_window_minutes = max(
                            0.0,
                            (
                                latest_survey_finish
                                - survey_start
                            ).total_seconds()
                            / 60.0,
                        )

                        # If lunch is still due during the remaining survey
                        # window, do not count those protected 30 minutes as
                        # productive survey capacity for this ratio.
                        if (
                            not lunch_taken
                            and survey_start <= lunch_latest_start
                            and latest_survey_finish > lunch_window_start
                        ):
                            remaining_survey_window_minutes = max(
                                0.0,
                                remaining_survey_window_minutes
                                - float(self.lunch_minutes),
                            )

                        feasible_destination_survey_minutes = min(
                            destination_work_minutes,
                            remaining_survey_window_minutes,
                        )

                        if not _far_cluster_transition_is_efficient(
                            travel_minutes=travel_minutes,
                            feasible_survey_minutes=(
                                feasible_destination_survey_minutes
                            ),
                        ):
                            continue

                survey_end = survey_start + timedelta(
                    minutes=float(site["planning_minutes"])
                )

                # Independent hard deadline: the final survey itself cannot end
                # after the user's selected cut-off, regardless of the journey home.
                if survey_end > latest_survey_finish:
                    continue

                ready_to_leave = survey_end + timedelta(
                    minutes=self.post_survey_buffer_minutes
                )

                # If we have not yet had lunch and taking this job would keep the
                # surveyor occupied beyond the latest allowed lunch START (13:00),
                # do not commit it. If no shorter feasible candidate exists, the
                # scheduler will reserve lunch at 11:45 and then re-route.
                if (
                    not lunch_taken
                    and current_time < lunch_window_start
                    and ready_to_leave > lunch_latest_start
                ):
                    lunch_blocked_candidate = True
                    continue

                # If this survey finishes during the lunch-start window, include
                # the 30-minute break in the return-home feasibility calculation.
                # This prevents a site being accepted only because lunch was
                # silently omitted from the hard return deadline.
                return_departure_for_check = ready_to_leave
                if (
                    not lunch_taken
                    and lunch_window_start <= ready_to_leave <= lunch_latest_start
                ):
                    return_departure_for_check = (
                        ready_to_leave
                        + timedelta(minutes=self.lunch_minutes)
                    )

                # Hard feasibility check using a real public-transport route home
                # at the time this particular survey (and, where required, lunch)
                # would finish.
                try:
                    return_route = self.router.compute_route(
                        site["route_location"],
                        self.home_location,
                        return_departure_for_check,
                    )
                except Exception:
                    continue

                return_time = return_departure_for_check + timedelta(
                    minutes=return_route.duration_minutes
                    + self.travel_leeway_minutes
                )

                if return_time <= latest_return:
                    chosen = (
                        idx,
                        site,
                        buffered_travel_minutes,
                        arrive,
                        survey_start,
                        survey_end,
                        ready_to_leave,
                        depart_previous,
                    )
                    break

            if chosen is None:
                # All otherwise-attractive candidates would cause lunch to be
                # missed. Reserve lunch at 11:45, then re-run routing at 12:15.
                if (
                    lunch_blocked_candidate
                    and not lunch_taken
                    and current_time < lunch_window_start
                ):
                    lunch_start = lunch_window_start
                    lunch_end = lunch_start + timedelta(minutes=self.lunch_minutes)
                    lunch_location = current_location
                    lunch_after_sequence = len(scheduled)
                    current_time = lunch_end
                    lunch_taken = True
                    continue
                break

            (
                idx,
                site,
                travel_minutes,
                arrive,
                survey_start,
                survey_end,
                ready_to_leave,
                depart_previous,
            ) = chosen

            if not scheduled:
                home_departure_time = depart_previous

            scheduled.append(
                ScheduledSurvey(
                    sequence=len(scheduled) + 1,
                    customer_reference=str(site.get("customer_reference", "")),
                    building_name=str(site.get("building_name", "")),
                    postcode=str(site.get("postcode", "")),
                    cluster=str(
                        site.get("planning_cluster", "")
                        or postcode_district(site.get("postcode", ""))
                    ),
                    depart_previous=depart_previous,
                    travel_minutes=travel_minutes,
                    arrive_site=arrive,
                    survey_minutes=float(site["planning_minutes"]),
                    survey_start=survey_start,
                    survey_end=survey_end,
                    predicted_minutes=float(site["predicted_minutes"]),
                    confidence=str(site.get("confidence", "")),
                    model_used=str(site.get("model_used", "")),
                    building_height=site.get("building_height"),
                    flats=site.get("flats"),
                    ground_floor_area=site.get("ground_floor_area"),
                )
            )

            current_location = site["route_location"]
            current_postcode = site.get("postcode", "")
            current_building_name = site.get("building_name", "")
            current_latitude = site.get("latitude")
            current_longitude = site.get("longitude")
            current_planning_cluster = str(
                site.get("planning_cluster", "")
                or postcode_district(site.get("postcode", ""))
            )
            current_time = ready_to_leave

            chosen_component_key = current_component_keys.get(idx)

            if (
                last_sequence_component_key is not None
                and chosen_component_key is not None
                and chosen_component_key
                != last_sequence_component_key
            ):
                closed_sequence_component_keys.add(
                    last_sequence_component_key
                )

            if chosen_component_key is not None:
                last_sequence_component_key = chosen_component_key
                current_sequence_component_key = chosen_component_key

            remaining.pop(idx)

        # If the final survey finishes during the lunch window, take the break
        # before the final journey home. A day that genuinely finishes before
        # 11:45 does not need a lunch break inserted.
        if (
            scheduled
            and not lunch_taken
            and lunch_window_start <= current_time <= lunch_latest_start
        ):
            lunch_start = current_time
            lunch_end = lunch_start + timedelta(minutes=self.lunch_minutes)
            lunch_location = current_location
            lunch_after_sequence = len(scheduled)
            current_time = lunch_end
            lunch_taken = True

        # Always calculate the final journey home from the last site.
        if scheduled:
            final_route = self.router.compute_route(
                current_location,
                self.home_location,
                current_time,
            )
            return_minutes = (
                final_route.duration_minutes + self.travel_leeway_minutes
            )
            return_time = current_time + timedelta(minutes=return_minutes)
        else:
            return_minutes = 0.0
            return_time = first_survey_start
            home_departure_time = first_survey_start

        return DailyScheduleResult(
            items=scheduled,
            start_location=self.home_location,
            start_time=home_departure_time,
            return_location=self.home_location,
            return_departure=current_time,
            return_travel_minutes=return_minutes,
            return_time=return_time,
            latest_return=latest_return,
            unscheduled_count=len(remaining),
            first_survey_target=first_survey_start,
            latest_survey_finish=latest_survey_finish,
            lunch_start=lunch_start,
            lunch_end=lunch_end,
            lunch_location=lunch_location,
            lunch_after_sequence=lunch_after_sequence,
            routing_stats=routing_stats,
        )

    def build_week(
        self,
        sites: List[dict],
        dates: Sequence,
        first_survey_start_clock,
        latest_survey_finish_clock,
        latest_return_clock,
        timezone,
    ) -> WeeklyScheduleResult:
        """
        Build a multi-day schedule.

        Each day uses the same live transit-routing logic as build_day.
        Once a site is scheduled it is removed from the candidate pool before
        the next day is planned.
        """
        remaining = [dict(site) for site in sites]
        days: List[DailyScheduleResult] = []

        for day_date in dates:
            if not remaining:
                break

            first_survey_dt = datetime.combine(
                day_date,
                first_survey_start_clock,
                tzinfo=timezone,
            )
            latest_survey_finish_dt = datetime.combine(
                day_date,
                latest_survey_finish_clock,
                tzinfo=timezone,
            )
            return_deadline_dt = datetime.combine(
                day_date,
                latest_return_clock,
                tzinfo=timezone,
            )

            day_result = self.build_day(
                sites=remaining,
                first_survey_start=first_survey_dt,
                latest_survey_finish=latest_survey_finish_dt,
                latest_return=return_deadline_dt,
            )
            days.append(day_result)

            scheduled_refs = {
                item.customer_reference
                for item in day_result.items
                if item.customer_reference
            }

            # Use reference code where available, otherwise building+postcode.
            scheduled_fallback = {
                (item.building_name, item.postcode)
                for item in day_result.items
            }

            new_remaining = []
            for site in remaining:
                ref = str(site.get("customer_reference", ""))
                fallback = (
                    str(site.get("building_name", "")),
                    str(site.get("postcode", "")),
                )

                if ref and ref in scheduled_refs:
                    continue
                if fallback in scheduled_fallback:
                    continue
                new_remaining.append(site)

            # If no progress is possible on this date, retain all sites and move on.
            remaining = new_remaining

        return WeeklyScheduleResult(
            days=days,
            unscheduled_sites=remaining,
        )

