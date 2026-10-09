from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime, timedelta, time
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple
import math
import re
from difflib import SequenceMatcher

import pandas as pd
from google_routes import GoogleNoRouteError, GoogleRoutesError

from coordinate_clustering import (
    NO_GOOGLE_RADIUS_KM,
    GEOGRAPHIC_CLUSTER_MAX_DIAMETER_KM,
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

# Every site-to-site move of >=30 minutes must unlock at least twice as much
# feasible local survey time, including moves inside one strategic cluster.
FAR_CLUSTER_TRANSITION_MINUTES = 30.0
FAR_CLUSTER_MIN_SURVEY_TO_TRAVEL_RATIO = 2.0


class CachedRunRouter:
    """Reuse successful routing answers only for identical trips/times this run."""
    def __init__(self, router):
        self.router = router
        self.routes = {}
        self.matrix = {}

    def __getattr__(self, name):
        return getattr(self.router, name)

    def compute_route(self, origin, destination, departure_time):
        key = (origin, destination, departure_time.isoformat())
        if key not in self.routes:
            self.routes[key] = self.router.compute_route(origin, destination, departure_time)
        return self.routes[key]

    def one_to_many(self, origin, destinations, departure_time):
        keys = [(origin, d, departure_time.isoformat()) for d in destinations]
        missing = list(dict.fromkeys(key for key in keys if key not in self.matrix))
        if missing:
            values = self.router.one_to_many(origin, [key[1] for key in missing], departure_time)
            if len(values) != len(missing):
                raise GoogleRoutesError("Google returned an incomplete routing matrix.")
            self.matrix.update(zip(missing, values))
        return [self.matrix[key] for key in keys]


def survey_clocks_for_date(day_date, first_survey_clock, latest_survey_clock,
                           latest_return_clock, saturday_time_window=None):
    """Use the optional Saturday window without changing weekday hours."""
    if day_date.weekday() == 5 and saturday_time_window is not None:
        return tuple(saturday_time_window)
    return first_survey_clock, latest_survey_clock, latest_return_clock


def _site_allowed_today(site, day_date):
    # Saturday is an optional retry day, never a fresh-building working day.
    if day_date.weekday() == 5 and not _is_retry_site(site):
        return False
    if _retry_forbidden_on_date(site, day_date):
        return False
    required = site.get("retry_required_weekdays") or []
    if isinstance(required, str):
        required = [value.strip() for value in required.split(",") if value.strip()]
    if not isinstance(required, (list, tuple, set)):
        required = []
    if required and day_date.strftime("%A") not in required:
        return False
    preferred = site.get("special_request_date")
    if preferred is not None and not pd.isna(preferred):
        preferred = pd.to_datetime(preferred).date()
        if day_date < preferred:
            return False
    return True


def _is_retry_site(site):
    return str(site.get("is_retry", False)).strip().lower() in {"true", "1", "1.0"}


def _site_allowed_for_surveyor(site, surveyor_name):
    original = str(site.get("retry_review_original_surveyor") or "").strip().casefold()
    if original in {"nan", "none", "<na>"}:
        original = ""
    return not original or bool(surveyor_name and str(surveyor_name).strip().casefold() != original)


def _requested_retry_day(site, day_date):
    if not _is_retry_site(site):
        return False
    wanted = (_retry_preferred_weekdays(site.get("retry_preferred_weekdays"))
              | _retry_preferred_weekdays(site.get("retry_required_weekdays")))
    return day_date.strftime("%A").lower() in wanted and _site_allowed_today(site, day_date)


def _day_area_workloads(sites, available_minutes, buffer_minutes):
    """Cheap packing estimate for area selection, not a feasibility certificate."""
    by_area = {}
    for site in sites:
        minutes = float(site["planning_minutes"])
        if math.isfinite(minutes) and minutes > 0:
            by_area.setdefault(_planning_cluster_key(site), []).append(minutes)
    workloads = {}
    for area, durations in by_area.items():
        used = survey = 0.0
        for duration in sorted(durations):
            if used + duration + buffer_minutes <= available_minutes:
                used += duration + buffer_minutes
                survey += duration
        workloads[area] = survey
    return workloads


def _local_work_candidates(anchor, sites):
    """Keep a bounded geographic area around an anchor; retain missing-data fallback."""
    result = []
    for site in sites:
        distance = haversine_km(anchor.get("latitude"), anchor.get("longitude"),
                                site.get("latitude"), site.get("longitude"))
        if ((distance is not None and distance <= GEOGRAPHIC_CLUSTER_MAX_DIAMETER_KM)
                or (distance is None and _planning_cluster_key(site) == _planning_cluster_key(anchor))):
            result.append(site)
    return result

# Retry scheduling preferences. The different-weekday rule is hard and is
# enforced in build_week before any day routing. Morning/afternoon remains a
# bounded soft preference so it cannot destroy an otherwise efficient route.
RETRY_PERIOD_GROUP_RADIUS_KM = NO_GOOGLE_RADIUS_KM
RETRY_PERIOD_MISMATCH_DISTANCE_PENALTY_KM = 0.50
RETRY_PERIOD_MISMATCH_SCORE_PENALTY_MINUTES = 20.0
RETRY_WEEKDAY_MISMATCH_SCORE_PENALTY_MINUTES = 15.0

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



def _normalise_retry_period(value) -> str:
    text = str(value or "").strip().lower()
    if text == "morning":
        return "morning"
    if text == "afternoon":
        return "afternoon"
    return ""


def _time_period_for_datetime(value: datetime) -> str:
    return "morning" if int(value.hour) < 12 else "afternoon"


def _retry_preferred_weekdays(value) -> set:
    if value is None:
        return set()
    if isinstance(value, (list, tuple, set)):
        values = value
    else:
        values = str(value).split(",")
    valid = {
        "monday", "tuesday", "wednesday", "thursday",
        "friday", "saturday", "sunday",
    }
    return {
        str(item).strip().lower()
        for item in values
        if str(item).strip().lower() in valid
    }


def _assign_retry_period_neighbourhood_preferences(
    sites: List[dict],
    radius_km: float = RETRY_PERIOD_GROUP_RADIUS_KM,
) -> None:
    """
    Propagate a retry's morning/afternoon preference to nearby buildings so the
    optimiser can move a coherent local group with the revisit instead of making
    an isolated time-of-day detour.

    This is deliberately sequencing-only and bounded to the existing local
    no-Google radius (plus exact full-postcode matches). It can cross strategic
    cluster boundaries because geographic proximity, not the cluster label, is
    what matters for keeping the local route coherent.
    """
    anchors = []
    for anchor in sites:
        period = _normalise_retry_period(
            anchor.get("retry_preferred_period")
        )
        if not period:
            continue
        anchors.append((anchor, period))

    if not anchors:
        return

    for site in sites:
        preferences = set()
        for anchor, period in anchors:
            same_postcode = False
            try:
                a_pc = re.sub(r"\s+", "", str(anchor.get("postcode", "")).upper())
                s_pc = re.sub(r"\s+", "", str(site.get("postcode", "")).upper())
                same_postcode = bool(a_pc) and a_pc == s_pc
            except Exception:
                same_postcode = False

            distance = haversine_km(
                anchor.get("latitude"),
                anchor.get("longitude"),
                site.get("latitude"),
                site.get("longitude"),
            )
            nearby = (
                distance is not None
                and float(distance) <= float(radius_km)
            )

            if same_postcode or nearby:
                preferences.add(period)

        # Conflicting nearby retry anchors cancel out rather than forcing an
        # arbitrary morning/afternoon choice onto the local area.
        site["_retry_group_preferred_period"] = (
            next(iter(preferences)) if len(preferences) == 1 else ""
        )


def _retry_forbidden_on_date(site: dict, day_date) -> bool:
    value = site.get("retry_forbidden_weekday")
    if value is None or value == "":
        return False
    try:
        forbidden = int(float(value))
    except Exception:
        return False
    try:
        return int(day_date.weekday()) == forbidden
    except Exception:
        return False


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
    # Without a coordinate origin, a nearest-neighbour rank is only an index
    # order. It must not outrank measured Google journey times.
    if not sites or _site_coordinate(current_site or {}) is None:
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
            and _site_coordinate(site) is not None
        ]
    else:
        indices = [idx for idx, site in enumerate(sites) if _site_coordinate(site) is not None]

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




def _remaining_sequence_component_proximity_km(
    current_site: dict,
    sites: List[dict],
) -> Dict[object, float]:
    """
    Minimum straight-line distance from the current site to each remaining
    stable sequencing component.

    This is sequencing-only. It deliberately ignores strategic-cluster
    preference so a genuinely nearby local area can be visited before a much
    farther part of the current strategic cluster. Google still validates the
    actual journey where required.
    """
    current_coordinate = _site_coordinate(current_site)
    if current_coordinate is None:
        return {}

    result: Dict[object, float] = {}

    for site in sites:
        component_key = site.get("_sequence_component_key")
        if component_key is None:
            continue

        site_coordinate = _site_coordinate(site)
        if site_coordinate is None:
            continue

        distance = _coordinate_distance_between_points(
            current_coordinate,
            site_coordinate,
        )
        if distance is None:
            continue

        previous = result.get(component_key)
        if previous is None or float(distance) < previous:
            result[component_key] = float(distance)

    return result


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
    first_area_scores: Optional[Dict[str, float]] = None,
    requested_areas: Optional[set] = None,
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
      4) before the first survey, compare area workload and home travel, with
         requested-date areas taking priority.
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

    # Compare productive working areas before choosing the first building.
    if current_site is None:
        if first_area_scores is not None:
            ordered = sorted(enumerate(groups), key=lambda item: (
                0 if group_cluster(item[1]) in (requested_areas or set()) else 1,
                -first_area_scores.get(group_cluster(item[1]), 0.0), item[0],
            ))
            selected, seen = [], set()
            # Compare one entrance per working area before spending additional
            # matrix slots on alternative entrances to the same area.
            for _, group in ordered:
                area = group_cluster(group)
                if area not in seen:
                    selected.append(group)
                    seen.add(area)
                if len(selected) == max_groups:
                    return selected
            selected_ids = {id(group) for group in selected}
            selected.extend(group for _, group in ordered if id(group) not in selected_ids)
            return selected[:max_groups]
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
        minimum_survey_to_travel_ratio=FAR_CLUSTER_MIN_SURVEY_TO_TRAVEL_RATIO,
        surveyor_name="",
    ):
        self.router = router if isinstance(router, CachedRunRouter) else CachedRunRouter(router)
        self.home_location = home_location
        self.surveyor_name = surveyor_name
        self.max_candidate_checks = max_candidate_checks
        self.same_postcode_transfer_minutes = same_postcode_transfer_minutes
        self.travel_leeway_minutes = travel_leeway_minutes
        self.pre_survey_buffer_minutes = pre_survey_buffer_minutes
        self.post_survey_buffer_minutes = post_survey_buffer_minutes
        self.ai_priority_weight_minutes = ai_priority_weight_minutes
        self.minimum_survey_to_travel_ratio = float(minimum_survey_to_travel_ratio)
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
        *,
        resume_from: Optional[DailyScheduleResult] = None,
        resume_site: Optional[dict] = None,
        local_continuation_only: bool = False,
    ) -> DailyScheduleResult:
        remaining = [dict(site) for site in sites
                     if _site_allowed_today(site, first_survey_start.date())
                     and _site_allowed_for_surveyor(site, self.surveyor_name)]

        # Stage 2 route-sequencing metadata only. This does not change Stage 1
        # cluster membership, eligibility, or the candidate sites supplied to
        # the day.
        _assign_stable_sequence_component_keys(
            remaining,
            radius_km=NO_GOOGLE_RADIUS_KM,
        )
        _assign_retry_period_neighbourhood_preferences(
            remaining,
            radius_km=RETRY_PERIOD_GROUP_RADIUS_KM,
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
            "fallback_search_passes": 0,
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

        # Road-level continuity sits inside the existing micro-cluster logic.
        # Once a confidently identified road has been left, returning to that
        # road later is strongly penalised. This prevents sequences such as
        # Pember Road -> Warfield Road -> Pember Road when the remaining Pember
        # sites could have been completed before leaving.
        last_sequence_road_name = None
        closed_sequence_road_names = set()

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

        if resume_from is not None and resume_from.items:
            if resume_site is None:
                raise ValueError("The last site's location is required when extending a day.")
            scheduled = list(resume_from.items)
            for key, value in (resume_from.routing_stats or {}).items():
                routing_stats[key] = routing_stats.get(key, 0) + value
            current_location = resume_site["route_location"]
            current_postcode = resume_site.get("postcode", "")
            current_building_name = resume_site.get("building_name", "")
            current_latitude = resume_site.get("latitude")
            current_longitude = resume_site.get("longitude")
            current_planning_cluster = _planning_cluster_key(resume_site)
            current_time = resume_from.return_departure
            home_departure_time = resume_from.start_time
            lunch_start, lunch_end = resume_from.lunch_start, resume_from.lunch_end
            lunch_taken = lunch_start is not None
            lunch_location = resume_from.lunch_location
            lunch_after_sequence = resume_from.lunch_after_sequence
            last_sequence_road_name = infer_road_name_from_building_name(current_building_name)
            closed_sequence_road_names = {
                infer_road_name_from_building_name(item.building_name) for item in scheduled[:-1]
            } - {None, last_sequence_road_name}

        # Rejected candidates are exhausted only at this location/time. Explore
        # the next small Google batch before ending a day; never re-query an
        # exhausted batch at the same state. A survey or lunch resets the state.
        search_state = None
        exhausted_indices = set()
        lunch_blocked_candidate = False

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
            state = (len(scheduled), current_time)
            if state != search_state:
                search_state = state
                exhausted_indices = set()
                lunch_blocked_candidate = False
            day_area_workloads = {}
            first_area_scores = None
            requested_areas = set()
            if not scheduled:
                work_start = max(first_survey_start, current_time)
                work_window = max(0.0, (latest_survey_finish - work_start).total_seconds() / 60)
                if not lunch_taken and latest_survey_finish >= lunch_window_start:
                    work_window = max(0.0, work_window - self.lunch_minutes)
                day_area_workloads = _day_area_workloads(
                    remaining, work_window,
                    self.pre_survey_buffer_minutes + self.post_survey_buffer_minutes,
                )
                first_area_scores = dict(day_area_workloads)
                for site in remaining:
                    area = _planning_cluster_key(site)
                    home_minutes = site.get("home_to_cluster_minutes")
                    if home_minutes is not None and math.isfinite(float(home_minutes)):
                        first_area_scores[area] = day_area_workloads.get(area, 0.0) - 2 * float(home_minutes)
                    preferred = site.get("special_request_date")
                    if preferred is not None and not pd.isna(preferred) and pd.to_datetime(preferred).date() == first_survey_start.date():
                        requested_areas.add(area)
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

            remaining_component_proximity = (
                _remaining_sequence_component_proximity_km(
                    current_site,
                    remaining,
                )
                if scheduled
                else {}
            )

            remaining_cluster_proximity = (
                _remaining_cluster_proximity_km(
                    current_site,
                    remaining,
                )
                if scheduled
                else {}
            )

            for idx, site in enumerate(remaining):
                if idx in exhausted_indices:
                    continue
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
                first_area_scores=first_area_scores,
                requested_areas=requested_areas,
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
                if len(google_matrix) != len(google_groups):
                    raise GoogleRoutesError("Google returned an incomplete routing matrix.")

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
                local_group_tier = float("inf")

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

                # Sequencing preference: after the current road/development is
                # finished, favour the geographically nearest remaining local
                # component. This can be in another strategic cluster; the
                # normal Google and far-cluster feasibility checks still decide
                # whether the move can actually be made.
                if scheduled and component_key is not None:
                    local_group_tier = (
                        remaining_component_proximity.get(
                            component_key,
                            float("inf"),
                        )
                    )

                candidate_road_name = (
                    infer_road_name_from_building_name(
                        site.get("building_name", "")
                    )
                )
                component_reentry = (
                    component_key is not None
                    and component_key in closed_sequence_component_keys
                )
                road_reentry = (
                    candidate_road_name is not None
                    and candidate_road_name
                    in closed_sequence_road_names
                )
                reentry_tier = (
                    1
                    if component_reentry or road_reentry
                    else 0
                )

                same_road_tier = 1
                local_distance_sort = float("inf")
                postcode_tier = 0
                google_representative_tier = 0
                retry_period_mismatch = False
                retry_weekday_mismatch = False

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
                            if direct_distance is not None:
                                local_distance_sort = float(direct_distance)
                            if _same_confident_road(current_site, site):
                                same_road_tier = 0
                elif idx in google_group_member_indices:
                    google_representative_tier = (
                        0 if idx in google_representative_indices else 1
                    )

                # Retry morning/afternoon is a bounded SOFT preference. Nearby
                # buildings inherit the revisit period, but geography still wins
                # when forcing the period would create an inefficient detour.
                group_period = _normalise_retry_period(
                    site.get("_retry_group_preferred_period")
                    or site.get("retry_preferred_period")
                )
                if group_period:
                    if scheduled:
                        estimated_leeway = self._site_to_site_leeway_minutes(
                            current_site, site
                        )
                        estimated_start = current_time + timedelta(
                            minutes=(
                                float(minutes)
                                + float(estimated_leeway)
                                + float(self.pre_survey_buffer_minutes)
                            )
                        )
                    else:
                        estimated_start = first_survey_start

                    retry_period_mismatch = (
                        _time_period_for_datetime(estimated_start)
                        != group_period
                    )
                    if (
                        retry_period_mismatch
                        and math.isfinite(float(local_group_tier))
                    ):
                        local_group_tier = (
                            float(local_group_tier)
                            + RETRY_PERIOD_MISMATCH_DISTANCE_PENALTY_KM
                        )

                preferred_weekdays = _retry_preferred_weekdays(
                    site.get("retry_preferred_weekdays")
                )
                if preferred_weekdays:
                    retry_weekday_mismatch = (
                        first_survey_start.strftime("%A").lower()
                        not in preferred_weekdays
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
                    + (
                        RETRY_PERIOD_MISMATCH_SCORE_PENALTY_MINUTES
                        if retry_period_mismatch
                        else 0.0
                    )
                    + (
                        RETRY_WEEKDAY_MISMATCH_SCORE_PENALTY_MINUTES
                        if retry_weekday_mismatch
                        else 0.0
                    )
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

            # Route sequencing is local-first rather than strategic-cluster-first.
            #
            # Once the current road/development is finished, choose the nearest
            # remaining local sequencing component by straight-line distance.
            # Strategic-cluster membership remains a tie-break only. This prevents
            # a distant part of the current strategic cluster from pulling the
            # route past a much closer neighbouring area.
            #
            # Google still validates meaningful journeys, and the existing
            # >=30-minute / 2:1 survey-to-travel efficiency gate applies to
            # every substantial move, regardless of strategic-cluster labels.
            ranked.sort(
                key=lambda x: (
                    x[3],  # avoid component/road A -> B -> A re-entry
                    x[6],  # finish the current confident road before leaving it
                    x[2],  # nearest remaining local component
                    x[0],  # current strategic cluster only as a tie-break
                    x[1],  # then nearest strategic-cluster proximity
                    x[4],  # existing optimised micro-cluster order
                    x[5],  # existing optimised site order
                    x[7],  # nearest-next coordinate distance
                    x[8],  # same full postcode
                    x[9],  # Google representative before group members
                    x[10], # existing Google/AI/special-request score
                )
            )
            if not scheduled:
                ranked.sort(key=lambda row: (
                    0 if _planning_cluster_key(remaining[row[11]]) in requested_areas else 1,
                    -(day_area_workloads.get(_planning_cluster_key(remaining[row[11]]), 0.0)
                      - 2 * (row[12] + self.travel_leeway_minutes)),
                    row[9], row[10], row[11],
                ))

            # Do not rebuild the feasibility pool by strategic cluster after the
            # local-first sort; doing so would undo the sequencing decision above.
            # Check all already-routed candidates, not just the first eight.
            # Cheap duration/lunch checks still precede return-home routing.
            # max_candidate_checks is retained in the constructor for callers
            # using the old API; Google group batches remain bounded separately.
            candidate_pool = ranked

            chosen = None
            approved_block = None

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
                first_site_start = max(first_survey_start, current_time)
                earliest_start = first_site_start if is_first_survey else current_time
                if earliest_start + timedelta(minutes=float(site["planning_minutes"])) > latest_survey_finish:
                    continue

                if is_first_survey:
                    # Back-calculate home departure so the first survey begins at
                    # the selected target rather than after an arbitrary fixed
                    # "leave home" time. Use one exact Compute Routes call only for
                    # the shortlisted candidate being feasibility-tested.
                    estimated_departure = first_site_start - timedelta(
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
                    except GoogleNoRouteError:
                        continue

                    buffered_travel_minutes = (
                        float(outbound_route.duration_minutes)
                        + self.travel_leeway_minutes
                    )
                    # Re-anchor the displayed/operational home departure using
                    # the exact route duration we just received so the first
                    # survey starts at the selected time rather than merely
                    # "not before" it. This does not add another Google call.
                    depart_previous = first_site_start - timedelta(
                        minutes=(
                            buffered_travel_minutes
                            + self.pre_survey_buffer_minutes
                        )
                    )
                    arrive = depart_previous + timedelta(
                        minutes=buffered_travel_minutes
                    )
                    survey_start = first_site_start
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

                # Every substantial site-to-site journey is checked, even
                # inside the same strategic cluster. Proof routes cannot make
                # another substantial move to inflate their claimed workload.
                substantial_move = (
                    not is_first_survey
                    and buffered_travel_minutes >= FAR_CLUSTER_TRANSITION_MINUTES
                )
                if substantial_move and local_continuation_only:
                    continue
                # Final day filling accepts any positive survey work that fits.
                # It must also be free to make another long move afterwards;
                # the local productivity-proof route would otherwise stop it.
                needs_work_proof = substantial_move and self.minimum_survey_to_travel_ratio > 0
                proof_candidates = []
                if needs_work_proof:
                    proof_candidates = _local_work_candidates(site, remaining)
                    potential_work = sum(float(s["planning_minutes"]) for s in proof_candidates)
                    available_work = max(0.0, (latest_survey_finish - survey_start).total_seconds() / 60)
                    if not lunch_taken and latest_survey_finish >= lunch_window_start:
                        available_work = max(0.0, available_work - self.lunch_minutes)
                    if not _far_cluster_transition_is_efficient(
                        buffered_travel_minutes, min(potential_work, available_work),
                        minimum_ratio=self.minimum_survey_to_travel_ratio
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
                except GoogleNoRouteError:
                    continue

                return_time = return_departure_for_check + timedelta(
                    minutes=return_route.duration_minutes
                    + self.travel_leeway_minutes
                )

                if return_time <= latest_return:
                    if needs_work_proof:
                        first_item = ScheduledSurvey(
                            sequence=len(scheduled) + 1,
                            customer_reference=str(site.get("customer_reference", "")),
                            building_name=str(site.get("building_name", "")),
                            postcode=str(site.get("postcode", "")), cluster=_planning_cluster_key(site),
                            depart_previous=depart_previous, travel_minutes=buffered_travel_minutes,
                            arrive_site=arrive, survey_minutes=float(site["planning_minutes"]),
                            survey_start=survey_start, survey_end=survey_end,
                            predicted_minutes=float(site["predicted_minutes"]),
                            confidence=str(site.get("confidence", "")), model_used=str(site.get("model_used", "")),
                            building_height=site.get("building_height"), flats=site.get("flats"),
                            ground_floor_area=site.get("ground_floor_area"),
                        )
                        prefix = DailyScheduleResult(
                            items=scheduled + [first_item], start_location=self.home_location,
                            start_time=home_departure_time, return_location=self.home_location,
                            return_departure=ready_to_leave,
                            return_travel_minutes=float(return_route.duration_minutes) + self.travel_leeway_minutes,
                            return_time=return_time, latest_return=latest_return,
                            unscheduled_count=0, first_survey_target=first_survey_start,
                            latest_survey_finish=latest_survey_finish,
                            lunch_start=lunch_start, lunch_end=lunch_end,
                            lunch_location=lunch_location, lunch_after_sequence=lunch_after_sequence,
                        )
                        proof = self.build_day(
                            [candidate for candidate in proof_candidates if candidate is not site],
                            first_survey_start, latest_survey_finish, latest_return,
                            resume_from=prefix, resume_site=site, local_continuation_only=True,
                        )
                        achieved_work = sum(item.survey_minutes for item in proof.items[len(scheduled):])
                        if (proof.return_time > latest_return
                                or not _far_cluster_transition_is_efficient(buffered_travel_minutes, achieved_work, minimum_ratio=self.minimum_survey_to_travel_ratio)):
                            continue
                        # Commit the proven local block together, so subsequent
                        # greedy choices cannot discard the work justifying it.
                        approved_block = proof
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
                exhausted_indices.update(travel_by_index)
                if len(exhausted_indices) < len(remaining):
                    routing_stats["fallback_search_passes"] += 1
                    continue
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

            if approved_block is not None:
                new_items = approved_block.items[len(scheduled):]
                by_identity = {_site_identity_for_sequence(site): site for site in remaining}
                last = new_items[-1]
                last_key = _site_identity_for_sequence(vars(last))
                last_site = by_identity[last_key]
                used = {_site_identity_for_sequence(vars(item)) for item in new_items}
                for item in new_items:
                    component = by_identity[_site_identity_for_sequence(vars(item))].get("_sequence_component_key")
                    if last_sequence_component_key is not None and component != last_sequence_component_key:
                        closed_sequence_component_keys.add(last_sequence_component_key)
                    last_sequence_component_key = component
                remaining = [site for site in remaining if _site_identity_for_sequence(site) not in used]
                scheduled = list(approved_block.items)
                current_location = last_site["route_location"]
                current_postcode, current_building_name = last.postcode, last.building_name
                current_latitude, current_longitude = last_site.get("latitude"), last_site.get("longitude")
                current_planning_cluster = _planning_cluster_key(last_site)
                current_sequence_component_key = last_site.get("_sequence_component_key")
                last_sequence_component_key = current_sequence_component_key
                closed_sequence_road_names.update(
                    infer_road_name_from_building_name(item.building_name) for item in scheduled[:-1]
                )
                last_sequence_road_name = infer_road_name_from_building_name(last.building_name)
                closed_sequence_road_names.discard(last_sequence_road_name)
                closed_sequence_road_names.discard(None)
                current_time = approved_block.return_departure
                lunch_start, lunch_end = approved_block.lunch_start, approved_block.lunch_end
                lunch_taken = lunch_start is not None
                lunch_location, lunch_after_sequence = approved_block.lunch_location, approved_block.lunch_after_sequence
                for key, value in (approved_block.routing_stats or {}).items():
                    routing_stats[key] = routing_stats.get(key, 0) + value
                continue

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

            chosen_road_name = infer_road_name_from_building_name(
                site.get("building_name", "")
            )
            if (
                last_sequence_road_name is not None
                and chosen_road_name != last_sequence_road_name
            ):
                closed_sequence_road_names.add(
                    last_sequence_road_name
                )
            last_sequence_road_name = chosen_road_name

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
        saturday_time_window=None,
    ) -> WeeklyScheduleResult:
        """
        Build a multi-day schedule.

        Each day uses the same live transit-routing logic as build_day.
        Once a site is scheduled it is removed from the candidate pool before
        the next day is planned.
        """
        remaining = [dict(site) for site in sites]
        days: List[DailyScheduleResult] = []

        # Try requested retry days before other days can consume their sites.
        # A dedicated first pass also keeps geographic ranking from burying a
        # requested retry underneath a full day of ordinary work.
        remaining = [site for site in remaining if _site_allowed_for_surveyor(site, self.surveyor_name)]
        pending_dates = sorted(set(dates), key=lambda d: (
            not any(_requested_retry_day(site, d) for site in remaining), d))
        for day_date in list(pending_dates):
            pending_dates.remove(day_date)
            if not remaining:
                break

            first_clock, finish_clock, return_clock = survey_clocks_for_date(
                day_date, first_survey_start_clock, latest_survey_finish_clock,
                latest_return_clock, saturday_time_window,
            )
            first_survey_dt = datetime.combine(
                day_date,
                first_clock,
                tzinfo=timezone,
            )
            latest_survey_finish_dt = datetime.combine(
                day_date,
                finish_clock,
                tzinfo=timezone,
            )
            return_deadline_dt = datetime.combine(
                day_date,
                return_clock,
                tzinfo=timezone,
            )

            # No-answer retries and known access days are hard exclusions.
            # Filter before build_day so forbidden retries do not create Google
            # routing calls on that date. They remain in the weekly candidate
            # pool for the later available dates.
            day_sites = [
                site
                for site in remaining
                if _site_allowed_today(site, day_date)
                and (_requested_retry_day(site, day_date)
                     or not any(_requested_retry_day(site, later) for later in pending_dates))
            ]
            if not day_sites:
                continue

            priority_sites = [site for site in day_sites if _requested_retry_day(site, day_date)]
            priority_result = None
            if priority_sites:
                old_ratio = self.minimum_survey_to_travel_ratio
                try:
                    self.minimum_survey_to_travel_ratio = 0.0
                    priority_result = self.build_day(
                        priority_sites, first_survey_dt,
                        latest_survey_finish_dt, return_deadline_dt)
                finally:
                    self.minimum_survey_to_travel_ratio = old_ratio
            prefix = priority_result if priority_result and priority_result.items else None
            placed = {_site_identity_for_sequence(vars(item)) for item in prefix.items} if prefix else set()
            resume_site = next((site for site in day_sites
                                if prefix and _site_identity_for_sequence(site)
                                == _site_identity_for_sequence(vars(prefix.items[-1]))), None)
            day_result = self.build_day(
                sites=[site for site in day_sites if _site_identity_for_sequence(site) not in placed],
                first_survey_start=first_survey_dt,
                latest_survey_finish=latest_survey_finish_dt,
                latest_return=return_deadline_dt,
                resume_from=prefix,
                resume_site=resume_site,
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
            days=sorted(days, key=lambda day: day.first_survey_target or day.start_time),
            unscheduled_sites=remaining,
        )
