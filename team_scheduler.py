from __future__ import annotations

from dataclasses import dataclass, asdict, replace
from datetime import datetime
import math
from typing import Dict, List, Sequence, Tuple

import pandas as pd

from portfolio_clusterer import _site_sort_frame
from coordinate_clustering import NO_GOOGLE_RADIUS_KM, haversine_km
from scheduler_v20_10 import (
    WeeklyScheduleResult,
    _day_area_workloads,
    _local_work_candidates,
    _planning_cluster_key,
    _site_allowed_today,
    _site_identity_for_sequence,
    _requested_retry_day,
    _retry_preferred_weekdays,
    _site_allowed_for_surveyor,
    survey_clocks_for_date,
)


# ============================================================================
# SURVEYOR HOME-FIT RULES
# ============================================================================
#
# These are the station coordinates supplied for the five regular surveyors.
# They are used only as a cheap geographic sanity check during cluster
# allocation. Google remains the public-transport source of truth.
#
# No extra Google requests are introduced by this logic.
SURVEYOR_HOME_COORDINATES = {
    "conor birch": (51.81489, -0.35194),       # Harpenden Train Station
    "harrison grice": (51.441346, 0.366643),  # Gravesend Station
    "joe reynolds": (51.742, -0.491),          # Hemel Hempstead
    "rod harrison": (52.379, -1.250),          # Rugby
    "toby lawal": (51.567956, 0.129558),       # Chadwell Heath
}

# If an available surveyor is more than this many Google-transit minutes worse
# than the best available surveyor for a cluster, workload balancing cannot
# steal the cluster from the better-positioned surveyor.
HOME_FIT_TRAVEL_BAND_MINUTES = 30.0

# Geographic sanity check. A surveyor more than this many straight-line km
# farther from the cluster centre than the nearest available surveyor is not
# allowed to win purely because of workload balancing while a natural-fit
# surveyor still has capacity.
HOME_FIT_DISTANCE_BAND_KM = 25.0

# Whole-week post-allocation repair threshold. The repair reuses only the
# home->cluster Google matrix already calculated above, so it creates no new
# Google calls.
TEAM_ALLOCATION_SWAP_MIN_SAVING_MINUTES = 20.0


def _normalise_name(value: str) -> str:
    return " ".join(str(value or "").strip().lower().split())


def _surveyor_home_coordinates(surveyor: "SurveyorConfig"):
    return SURVEYOR_HOME_COORDINATES.get(
        _normalise_name(surveyor.name)
    )


@dataclass
class SurveyorConfig:
    name: str
    start_location: str
    available_dates: object = None


@dataclass
class ClusterAllocation:
    surveyor_name: str
    cluster: str
    target_sites: int
    home_to_cluster_minutes: float
    cluster_priority: int
    cluster_reason: str
    estimated_candidate_minutes: float


def representative_sites(
    portfolio: pd.DataFrame,
    cluster_choices: Sequence[dict],
    target_week_start,
) -> List[dict]:
    """
    Pick one real site to represent each selected geographic planning cluster.

    Google is used only for these tiny home -> cluster-representative comparisons
    before the more detailed per-surveyor routing begins.
    """
    eligible = portfolio[
        portfolio["Eligible for Selected Week"] == True
    ].copy()

    reps = []
    for choice in cluster_choices:
        cluster = str(choice.get("cluster", "")).strip()
        group = eligible[
            eligible["Planning Cluster"].astype(str) == cluster
        ].copy()
        if group.empty:
            continue

        ranked = _site_sort_frame(group, target_week_start)
        row = ranked.iloc[0]
        building = str(row.get("Building Name", "")).strip()
        postcode = str(row.get("Postcode", "")).strip()
        route_location = (
            f"{building}, {postcode}" if building else postcode
        )

        centre_latitude = row.get(
            "Cluster Centre Latitude",
            row.get("Latitude Clean", row.get("Latitude")),
        )
        centre_longitude = row.get(
            "Cluster Centre Longitude",
            row.get("Longitude Clean", row.get("Longitude")),
        )

        reps.append({
            "cluster": cluster,
            "route_location": route_location,
            "building_name": building or postcode,
            "postcode": postcode,
            "latitude": centre_latitude,
            "longitude": centre_longitude,
            "priority": int(choice.get("priority", 50)),
            "target_sites": int(choice.get("target_sites", 1)),
            "reason": str(choice.get("reason", "")),
        })

    return reps


def home_to_cluster_matrix(
    router,
    surveyors: Sequence[SurveyorConfig],
    representatives: Sequence[dict],
    departure_time: datetime,
) -> Dict[Tuple[str, str], float]:
    """
    Very small Google matrix: one origin per active surveyor and one destination
    per selected strategic cluster.
    """
    if not representatives:
        return {}

    destinations = [r["route_location"] for r in representatives]
    result = {}

    # Cheap geographic metadata for the allocator. These calculations happen
    # locally and do not create any Google API usage.
    for surveyor in surveyors:
        home_coordinates = _surveyor_home_coordinates(surveyor)
        if home_coordinates is None:
            continue

        home_latitude, home_longitude = home_coordinates
        for rep in representatives:
            distance_km = haversine_km(
                home_latitude,
                home_longitude,
                rep.get("latitude"),
                rep.get("longitude"),
            )
            if distance_km is not None:
                result[
                    ("__home_distance_km__", surveyor.name, rep["cluster"])
                ] = float(distance_km)

    for surveyor in surveyors:
        surveyor_departure = departure_time
        if surveyor.available_dates:
            first_available = min(surveyor.available_dates)
            surveyor_departure = departure_time.replace(
                year=first_available.year,
                month=first_available.month,
                day=first_available.day,
            )

        durations = router.one_to_many(
            surveyor.start_location,
            destinations,
            surveyor_departure,
        )
        for rep, minutes in zip(representatives, durations):
            if minutes is not None:
                result[(surveyor.name, rep["cluster"])] = float(minutes)

    return result


def _cluster_average_minutes(
    cluster_summary: pd.DataFrame,
    cluster: str,
) -> float:
    match = cluster_summary[
        cluster_summary["Cluster"].astype(str) == str(cluster)
    ]
    if match.empty:
        return 75.0
    value = match.iloc[0].get("Average Planning Minutes")
    try:
        value = float(value)
    except Exception:
        value = 75.0
    if pd.isna(value) or value <= 0:
        value = 75.0
    return value



def _travel_minutes_for_allocation(
    travel_matrix: Dict[Tuple[str, str], float],
    surveyor_name: str,
    cluster: str,
):
    value = travel_matrix.get((surveyor_name, cluster))
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def _home_distance_for_allocation(
    travel_matrix: Dict[Tuple[str, str], float],
    surveyor_name: str,
    cluster: str,
):
    value = travel_matrix.get(
        ("__home_distance_km__", surveyor_name, cluster)
    )
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def _copy_allocation_with_surveyor(
    allocation: ClusterAllocation,
    surveyor_name: str,
    travel_matrix: Dict[Tuple[str, str], float],
) -> ClusterAllocation:
    travel = _travel_minutes_for_allocation(
        travel_matrix,
        surveyor_name,
        allocation.cluster,
    )
    if travel is None:
        travel = float(allocation.home_to_cluster_minutes)

    return ClusterAllocation(
        surveyor_name=surveyor_name,
        cluster=allocation.cluster,
        target_sites=int(allocation.target_sites),
        home_to_cluster_minutes=round(float(travel), 1),
        cluster_priority=int(allocation.cluster_priority),
        cluster_reason=str(allocation.cluster_reason),
        estimated_candidate_minutes=float(
            allocation.estimated_candidate_minutes
        ),
    )


def _repair_team_allocations(
    allocations: Sequence[ClusterAllocation],
    surveyors: Sequence[SurveyorConfig],
    travel_matrix: Dict[Tuple[str, str], float],
    capacities: Dict[str, int],
) -> List[ClusterAllocation]:
    """
    Repair the initial greedy weekly allocation using the existing Google
    home->cluster matrix.

    This addresses cases where an early allocation fills the naturally placed
    surveyor and later forces a geographically poor cluster onto somebody else.

    The repair can:
      - move a clearly misallocated chunk when the better surveyor has spare
        capacity; or
      - swap two chunks when doing so materially reduces total home->cluster
        commute and remains within both surveyor capacities.

    No additional Google calls are made.
    """
    working = [
        ClusterAllocation(**asdict(allocation))
        for allocation in allocations
    ]
    if not working:
        return working

    surveyor_names = [surveyor.name for surveyor in surveyors]
    loads = {name: 0 for name in surveyor_names}
    for allocation in working:
        loads.setdefault(allocation.surveyor_name, 0)
        loads[allocation.surveyor_name] += int(allocation.target_sites)

    def clear_home_fit_improvement(
        current_name: str,
        alternative_name: str,
        cluster: str,
    ) -> bool:
        current_travel = _travel_minutes_for_allocation(
            travel_matrix,
            current_name,
            cluster,
        )
        alternative_travel = _travel_minutes_for_allocation(
            travel_matrix,
            alternative_name,
            cluster,
        )
        if current_travel is None or alternative_travel is None:
            return False

        travel_improvement = current_travel - alternative_travel

        current_distance = _home_distance_for_allocation(
            travel_matrix,
            current_name,
            cluster,
        )
        alternative_distance = _home_distance_for_allocation(
            travel_matrix,
            alternative_name,
            cluster,
        )
        distance_improvement = (
            current_distance - alternative_distance
            if (
                current_distance is not None
                and alternative_distance is not None
            )
            else None
        )

        return (
            travel_improvement
            >= float(HOME_FIT_TRAVEL_BAND_MINUTES)
            or (
                distance_improvement is not None
                and distance_improvement
                >= float(HOME_FIT_DISTANCE_BAND_KM)
            )
        )

    for _ in range(max(2, min(12, len(working) * 2))):
        improved = False

        # A) Move a clearly poor allocation if the better surveyor has capacity.
        for idx, allocation in enumerate(list(working)):
            current_name = allocation.surveyor_name
            chunk = int(allocation.target_sites)

            alternatives = []
            current_travel = _travel_minutes_for_allocation(
                travel_matrix,
                current_name,
                allocation.cluster,
            )
            if current_travel is None:
                continue

            for alternative_name in surveyor_names:
                if alternative_name == current_name:
                    continue
                if (
                    loads.get(alternative_name, 0) + chunk
                    > capacities.get(alternative_name, 0)
                ):
                    continue

                alternative_travel = _travel_minutes_for_allocation(
                    travel_matrix,
                    alternative_name,
                    allocation.cluster,
                )
                if alternative_travel is None:
                    continue

                if not clear_home_fit_improvement(
                    current_name,
                    alternative_name,
                    allocation.cluster,
                ):
                    continue

                alternatives.append(
                    (
                        current_travel - alternative_travel,
                        alternative_travel,
                        alternative_name,
                    )
                )

            if not alternatives:
                continue

            alternatives.sort(
                key=lambda item: (-item[0], item[1], item[2])
            )
            saving, _, chosen_name = alternatives[0]
            if saving <= 0:
                continue

            loads[current_name] -= chunk
            loads[chosen_name] = loads.get(chosen_name, 0) + chunk
            working[idx] = _copy_allocation_with_surveyor(
                allocation,
                chosen_name,
                travel_matrix,
            )
            improved = True

        # B) Swap chunks if both better-positioned surveyors are already full.
        best_swap = None

        for i in range(len(working)):
            allocation_a = working[i]
            surveyor_a = allocation_a.surveyor_name
            chunk_a = int(allocation_a.target_sites)

            for j in range(i + 1, len(working)):
                allocation_b = working[j]
                surveyor_b = allocation_b.surveyor_name
                chunk_b = int(allocation_b.target_sites)

                if surveyor_a == surveyor_b:
                    continue
                if allocation_a.cluster == allocation_b.cluster:
                    continue

                a_current = _travel_minutes_for_allocation(
                    travel_matrix,
                    surveyor_a,
                    allocation_a.cluster,
                )
                b_current = _travel_minutes_for_allocation(
                    travel_matrix,
                    surveyor_b,
                    allocation_b.cluster,
                )
                a_swapped = _travel_minutes_for_allocation(
                    travel_matrix,
                    surveyor_b,
                    allocation_a.cluster,
                )
                b_swapped = _travel_minutes_for_allocation(
                    travel_matrix,
                    surveyor_a,
                    allocation_b.cluster,
                )

                if any(
                    value is None
                    for value in (
                        a_current,
                        b_current,
                        a_swapped,
                        b_swapped,
                    )
                ):
                    continue

                new_load_a = (
                    loads.get(surveyor_a, 0)
                    - chunk_a
                    + chunk_b
                )
                new_load_b = (
                    loads.get(surveyor_b, 0)
                    - chunk_b
                    + chunk_a
                )
                if (
                    new_load_a > capacities.get(surveyor_a, 0)
                    or new_load_b > capacities.get(surveyor_b, 0)
                ):
                    continue

                current_total = a_current + b_current
                swapped_total = a_swapped + b_swapped
                saving = current_total - swapped_total

                if (
                    saving
                    < float(
                        TEAM_ALLOCATION_SWAP_MIN_SAVING_MINUTES
                    )
                ):
                    continue

                if not (
                    clear_home_fit_improvement(
                        surveyor_a,
                        surveyor_b,
                        allocation_a.cluster,
                    )
                    or clear_home_fit_improvement(
                        surveyor_b,
                        surveyor_a,
                        allocation_b.cluster,
                    )
                ):
                    continue

                candidate = (
                    saving,
                    i,
                    j,
                    surveyor_a,
                    surveyor_b,
                    new_load_a,
                    new_load_b,
                )
                if best_swap is None or candidate[0] > best_swap[0]:
                    best_swap = candidate

        if best_swap is not None:
            (
                _,
                i,
                j,
                surveyor_a,
                surveyor_b,
                new_load_a,
                new_load_b,
            ) = best_swap

            allocation_a = working[i]
            allocation_b = working[j]

            working[i] = _copy_allocation_with_surveyor(
                allocation_a,
                surveyor_b,
                travel_matrix,
            )
            working[j] = _copy_allocation_with_surveyor(
                allocation_b,
                surveyor_a,
                travel_matrix,
            )

            loads[surveyor_a] = new_load_a
            loads[surveyor_b] = new_load_b
            improved = True

        if not improved:
            break

    return working



def _ensure_available_surveyors_have_work(
    allocations: Sequence[ClusterAllocation],
    surveyors: Sequence[SurveyorConfig],
    travel_matrix: Dict[Tuple[str, str], float],
    capacities: Dict[str, int],
    chunk_size: int,
) -> List[ClusterAllocation]:
    """
    Prevent an explicitly available surveyor from finishing with zero candidate
    sites solely because home-fit protection kept all selected clusters with
    better-positioned colleagues.

    This runs after the existing whole-week home-fit repair, so normal
    geographic allocation is unchanged unless an available surveyor has zero
    work. It reuses the existing home->cluster Google matrix and never makes
    another Google request.

    The zero-work surveyor receives at most one normal allocation chunk,
    limited to their proportional share of the selected candidate pool.
    """
    working = [
        ClusterAllocation(**asdict(allocation))
        for allocation in allocations
    ]
    if not working:
        return working

    available_surveyors = [
        surveyor
        for surveyor in surveyors
        if len(surveyor.available_dates or []) > 0
    ]
    if not available_surveyors:
        return working

    total_available_days = max(
        1,
        sum(
            min(6, len(surveyor.available_dates or []))
            for surveyor in available_surveyors
        ),
    )

    total_selected_sites = sum(
        int(allocation.target_sites)
        for allocation in working
    )

    def current_loads():
        loads = {
            surveyor.name: 0
            for surveyor in available_surveyors
        }
        for allocation in working:
            loads.setdefault(allocation.surveyor_name, 0)
            loads[allocation.surveyor_name] += int(
                allocation.target_sites
            )
        return loads

    for surveyor in available_surveyors:
        loads = current_loads()
        if loads.get(surveyor.name, 0) > 0:
            continue

        available_days = min(
            6,
            len(surveyor.available_dates or []),
        )
        proportional_share = max(
            1,
            round(
                total_selected_sites
                * available_days
                / total_available_days
            ),
        )

        target_transfer = min(
            max(1, int(chunk_size)),
            proportional_share,
            max(1, capacities.get(surveyor.name, 1)),
        )
        remaining_needed = target_transfer

        candidate_allocations = []
        for idx, allocation in enumerate(working):
            donor_name = allocation.surveyor_name
            if donor_name == surveyor.name:
                continue

            donor_load = loads.get(donor_name, 0)
            if donor_load <= 1:
                continue

            travel = _travel_minutes_for_allocation(
                travel_matrix,
                surveyor.name,
                allocation.cluster,
            )
            if travel is None:
                continue

            candidate_allocations.append(
                (
                    float(travel),
                    -int(donor_load),
                    idx,
                )
            )

        candidate_allocations.sort()

        for travel, _, idx in candidate_allocations:
            if remaining_needed <= 0:
                break

            allocation = working[idx]
            donor_name = allocation.surveyor_name
            loads = current_loads()
            donor_load = loads.get(donor_name, 0)

            transfer = min(
                int(allocation.target_sites),
                max(0, donor_load - 1),
                remaining_needed,
            )
            if transfer <= 0:
                continue

            original_sites = int(allocation.target_sites)
            per_site_minutes = (
                float(allocation.estimated_candidate_minutes)
                / original_sites
                if original_sites > 0
                else 0.0
            )
            donor_remaining = original_sites - transfer

            working[idx] = ClusterAllocation(
                surveyor_name=allocation.surveyor_name,
                cluster=allocation.cluster,
                target_sites=donor_remaining,
                home_to_cluster_minutes=float(
                    allocation.home_to_cluster_minutes
                ),
                cluster_priority=int(allocation.cluster_priority),
                cluster_reason=str(allocation.cluster_reason),
                estimated_candidate_minutes=round(
                    donor_remaining * per_site_minutes,
                    1,
                ),
            )

            working.append(
                ClusterAllocation(
                    surveyor_name=surveyor.name,
                    cluster=allocation.cluster,
                    target_sites=transfer,
                    home_to_cluster_minutes=round(
                        float(travel),
                        1,
                    ),
                    cluster_priority=int(
                        allocation.cluster_priority
                    ),
                    cluster_reason=(
                        str(allocation.cluster_reason)
                        + " Availability repair: reassigned a small "
                        "candidate share because this explicitly available "
                        "surveyor otherwise had zero work."
                    ).strip(),
                    estimated_candidate_minutes=round(
                        transfer * per_site_minutes,
                        1,
                    ),
                )
            )

            remaining_needed -= transfer

    return [
        allocation
        for allocation in working
        if int(allocation.target_sites) > 0
    ]


def _balance_candidate_workload(
    allocations: Sequence[ClusterAllocation],
    surveyors: Sequence[SurveyorConfig],
    travel_matrix: Dict[Tuple[str, str], float],
    capacities: Dict[str, int],
    candidate_minutes_per_day: float = None,
) -> List[ClusterAllocation]:
    """Supply underloaded people from colleagues' surplus candidate workload.

    The existing home-fit allocation runs first. Repair only workload below a
    proportional share of predicted minutes, capped at the selected day window
    plus candidate reserve when provided. Donors keep their own target. This
    uses the existing home matrix and moves candidates, not booked visits;
    detailed routing still enforces every daily constraint.
    """
    working = [ClusterAllocation(**asdict(a)) for a in allocations]
    days = {s.name: len(set(s.available_dates or [])) for s in surveyors
            if s.available_dates}
    if not working or not days:
        return working
    minutes_per_day = sum(a.estimated_candidate_minutes for a in working) / sum(days.values())
    if candidate_minutes_per_day is not None:
        minutes_per_day = min(minutes_per_day, max(0.0, float(candidate_minutes_per_day)))
    targets = {name: count * minutes_per_day for name, count in days.items()}

    def loads():
        minutes = {name: 0.0 for name in days}
        counts = {name: 0 for name in days}
        for a in working:
            minutes[a.surveyor_name] += a.estimated_candidate_minutes
            counts[a.surveyor_name] += a.target_sites
        return minutes, counts

    # Every transfer lowers the recipient's deficit without taking a donor
    # below target, so a transferred site cannot bounce between surveyors.
    while True:
        minutes, counts = loads()
        recipients = sorted(days, key=lambda name: (
            minutes[name] / max(targets[name], 1.0), name,
        ))
        moved = False
        for recipient in recipients:
            needed = targets[recipient] - minutes[recipient]
            capacity_left = capacities.get(recipient, 0) - counts[recipient]
            if needed <= 0.1 or capacity_left <= 0:
                continue
            recipient_clusters = {a.cluster for a in working
                                  if a.surveyor_name == recipient and a.target_sites > 0}
            options = []
            for idx, a in enumerate(working):
                donor = a.surveyor_name
                if donor == recipient or a.target_sites <= 0:
                    continue
                per_site = a.estimated_candidate_minutes / a.target_sites
                surplus = minutes[donor] - targets[donor]
                travel = _travel_minutes_for_allocation(travel_matrix, recipient, a.cluster)
                if per_site <= 0 or not math.isfinite(per_site) or travel is None:
                    continue
                take = min(a.target_sites, capacity_left,
                           max(0, math.floor((surplus + 1e-7) / per_site)),
                           max(1, math.ceil(needed / per_site)))
                if take <= 0:
                    continue
                # Avoid adding more workload than it resolves for a nearly
                # supplied recipient (e.g. one huge survey for a tiny deficit).
                transfer_minutes = take * per_site
                if abs(needed - transfer_minutes) >= needed - 1e-7:
                    continue
                extra_commute = max(0.0, travel - a.home_to_cluster_minutes)
                score = travel + extra_commute - (12.0 if a.cluster in recipient_clusters else 0.0)
                options.append((score, -transfer_minutes, idx, take, travel))
            if not options:
                continue
            _, _, idx, take, travel = min(options)
            a = working[idx]
            transfer_minutes = a.estimated_candidate_minutes * take / a.target_sites
            a.target_sites -= take
            a.estimated_candidate_minutes -= transfer_minutes
            working.append(ClusterAllocation(
                surveyor_name=recipient, cluster=a.cluster, target_sites=take,
                home_to_cluster_minutes=round(travel, 1),
                cluster_priority=a.cluster_priority,
                cluster_reason=(a.cluster_reason + " Workload repair: transferred surplus "
                                "candidate minutes to cover selected available days."),
                estimated_candidate_minutes=transfer_minutes,
            ))
            moved = True
            break
        if not moved:
            break
    return [a for a in working if a.target_sites > 0]


def allocate_cluster_targets(
    cluster_choices: Sequence[dict],
    cluster_summary: pd.DataFrame,
    surveyors: Sequence[SurveyorConfig],
    travel_matrix: Dict[Tuple[str, str], float],
    max_sites_per_surveyor: int,
    candidate_minutes_per_day: float = None,
) -> List[ClusterAllocation]:
    """
    Split selected cluster capacity across the active team.

    The assignment uses:
    - actual Google home -> cluster representative transit time;
    - supplied surveyor station coordinates as a cheap geographic sanity check;
    - workload balancing;
    - a small bonus for keeping a surveyor in a cluster already allocated to them.

    Clear home-fit advantages are protected before workload balancing. A large
    cluster can still be shared. After the initial assignment, a whole-week
    repair pass reuses the same home->cluster Google matrix to move or swap
    clearly inefficient allocations before detailed scheduling begins.
    """
    surveyors = [s for s in surveyors if s.available_dates]
    if not surveyors:
        return []

    capacities = {
        s.name: max(
            1,
            round(
                int(max_sites_per_surveyor)
                * min(6, len(s.available_dates or []))
                / 5
            ),
        )
        for s in surveyors
    }
    assigned_sites = {s.name: 0 for s in surveyors}
    assigned_minutes = {s.name: 0.0 for s in surveyors}
    clusters_by_surveyor = {s.name: set() for s in surveyors}

    # Use manageable chunks so one large cluster can feed more than one person.
    chunk_size = max(6, min(15, int(max_sites_per_surveyor) // 3 or 6))

    chunks = []
    for choice in sorted(
        list(cluster_choices),
        key=lambda c: int(c.get("priority", 50)),
        reverse=True,
    ):
        cluster = str(choice.get("cluster", "")).strip()
        if not cluster:
            continue
        try:
            total = int(choice.get("target_sites", 1))
        except Exception:
            total = 1
        total = max(1, total)

        while total > 0:
            take = min(chunk_size, total)
            chunks.append((choice, take))
            total -= take

    allocations: List[ClusterAllocation] = []

    for choice, requested_chunk in chunks:
        cluster = str(choice.get("cluster", "")).strip()
        avg_minutes = _cluster_average_minutes(
            cluster_summary,
            cluster,
        )

        candidates = []
        for surveyor in surveyors:
            capacity_left = (
                capacities[surveyor.name]
                - assigned_sites[surveyor.name]
            )
            if capacity_left <= 0:
                continue

            travel = travel_matrix.get(
                (surveyor.name, cluster)
            )
            if travel is None:
                # If Google could not route this home -> cluster pair,
                # do not force the allocation to this surveyor.
                continue

            home_distance_km = travel_matrix.get(
                (
                    "__home_distance_km__",
                    surveyor.name,
                    cluster,
                )
            )

            chunk = min(requested_chunk, capacity_left)
            projected_minutes = (
                assigned_minutes[surveyor.name]
                + chunk * avg_minutes
            )

            # Rough target only for balancing candidate workload. Detailed
            # time feasibility is still enforced later by each weekly router.
            total_selected_minutes = 0.0
            for c in cluster_choices:
                c_cluster = str(c.get("cluster", "")).strip()
                total_selected_minutes += (
                    max(1, int(c.get("target_sites", 1)))
                    * _cluster_average_minutes(
                        cluster_summary,
                        c_cluster,
                    )
                )
            total_available_days = max(
                1,
                sum(
                    min(6, len(s.available_dates or []))
                    for s in surveyors
                ),
            )
            surveyor_day_share = (
                min(6, len(surveyor.available_dates or []))
                / total_available_days
            )
            target_minutes = max(
                1.0,
                total_selected_minutes * surveyor_day_share,
            )

            load_penalty = (
                projected_minutes / target_minutes
            ) * 50.0

            continuity_bonus = (
                12.0
                if cluster in clusters_by_surveyor[surveyor.name]
                else 0.0
            )

            # Give commute efficiency slightly more weight than before. This
            # does not reduce candidate capacity; it simply makes a cluster more
            # likely to go to the surveyor whose home journey is cheaper, leaving
            # more of the fixed survey window available for actual jobs.
            commute_efficiency_penalty = float(travel) * 0.50
            score = (
                float(travel)
                + commute_efficiency_penalty
                + load_penalty
                - continuity_bonus
            )

            candidates.append(
                (
                    score,
                    surveyor,
                    chunk,
                    float(travel),
                    projected_minutes,
                    (
                        float(home_distance_km)
                        if home_distance_km is not None
                        else None
                    ),
                )
            )

        if not candidates:
            continue

        # Home-fit protection comes BEFORE workload balancing.
        #
        # If one or more available surveyors are clearly well positioned for
        # this cluster, only that natural-fit group competes for the chunk.
        # Workload balancing still operates normally inside the group.
        #
        # If those natural-fit surveyors later run out of capacity they are no
        # longer present in `candidates`, so the cluster can still fall back to
        # another available surveyor rather than being left unscheduled.
        best_google_travel = min(candidate[3] for candidate in candidates)
        geo_distances = [
            candidate[5]
            for candidate in candidates
            if candidate[5] is not None
        ]
        best_geo_distance = (
            min(geo_distances)
            if geo_distances
            else None
        )

        natural_fit_candidates = []
        for candidate in candidates:
            google_fit = (
                candidate[3]
                <= best_google_travel
                + float(HOME_FIT_TRAVEL_BAND_MINUTES)
            )
            geo_distance = candidate[5]
            geographic_fit = (
                best_geo_distance is None
                or geo_distance is None
                or geo_distance
                <= best_geo_distance
                + float(HOME_FIT_DISTANCE_BAND_KM)
            )
            if google_fit and geographic_fit:
                natural_fit_candidates.append(candidate)

        if natural_fit_candidates:
            candidates = natural_fit_candidates

        candidates.sort(key=lambda x: x[0])
        _, chosen, chunk, travel, _, _ = candidates[0]

        estimated_minutes = chunk * avg_minutes
        allocations.append(
            ClusterAllocation(
                surveyor_name=chosen.name,
                cluster=cluster,
                target_sites=int(chunk),
                home_to_cluster_minutes=round(travel, 1),
                cluster_priority=int(choice.get("priority", 50)),
                cluster_reason=str(choice.get("reason", "")),
                estimated_candidate_minutes=round(
                    estimated_minutes, 1
                ),
            )
        )

        assigned_sites[chosen.name] += int(chunk)
        assigned_minutes[chosen.name] += estimated_minutes
        clusters_by_surveyor[chosen.name].add(cluster)

    repaired_allocations = _repair_team_allocations(
        allocations=allocations,
        surveyors=surveyors,
        travel_matrix=travel_matrix,
        capacities=capacities,
    )

    return _balance_candidate_workload(
        allocations=repaired_allocations,
        surveyors=surveyors,
        travel_matrix=travel_matrix,
        capacities=capacities,
        candidate_minutes_per_day=candidate_minutes_per_day,
    )


def build_team_shortlists(
    portfolio: pd.DataFrame,
    allocations: Sequence[ClusterAllocation],
    surveyors: Sequence[SurveyorConfig],
    target_week_start,
    max_sites_per_surveyor: int,
) -> Dict[str, pd.DataFrame]:
    """
    Hand distinct sites from each selected cluster to the allocated surveyors.
    No site can appear in two surveyor shortlists.
    """
    eligible = portfolio[
        portfolio["Eligible for Selected Week"] == True
    ].copy()

    by_cluster = {}
    for cluster, group in eligible.groupby("Planning Cluster"):
        by_cluster[str(cluster)] = _site_sort_frame(
            group.copy(),
            target_week_start,
        )

    used_indices = set()
    output = {
        s.name: eligible.head(0).copy() for s in surveyors
    }

    # Preserve strategic priority, then lower home travel.
    ordered = sorted(
        list(allocations),
        key=lambda a: (
            -int(a.cluster_priority),
            float(a.home_to_cluster_minutes),
        ),
    )

    parts = {s.name: [] for s in surveyors}
    counts = {s.name: 0 for s in surveyors}

    for allocation in ordered:
        surveyor_cfg = next(
            s for s in surveyors
            if s.name == allocation.surveyor_name
        )
        effective_capacity = max(
            1,
            round(
                int(max_sites_per_surveyor)
                * min(6, len(surveyor_cfg.available_dates or []))
                / 5
            ),
        )
        remaining_capacity = (
            effective_capacity
            - counts[allocation.surveyor_name]
        )
        if remaining_capacity <= 0:
            continue

        group = by_cluster.get(allocation.cluster)
        if group is None or group.empty:
            continue

        group = group[~group.index.isin(used_indices)]
        if group.empty:
            continue

        take = min(
            int(allocation.target_sites),
            len(group),
            remaining_capacity,
        )
        chosen = group.head(take).copy()

        chosen["Assigned Surveyor"] = (
            allocation.surveyor_name
        )
        chosen["Home to Cluster (Minutes)"] = (
            allocation.home_to_cluster_minutes
        )
        chosen["AI Cluster Priority"] = (
            allocation.cluster_priority
        )
        chosen["AI Cluster Reason"] = (
            allocation.cluster_reason
        )

        parts[allocation.surveyor_name].append(chosen)
        used_indices.update(chosen.index)
        counts[allocation.surveyor_name] += len(chosen)

    for surveyor in surveyors:
        if parts[surveyor.name]:
            df = pd.concat(
                parts[surveyor.name],
                ignore_index=False,
            ).drop_duplicates()
            helper = [
                c for c in df.columns if c.startswith("_")
            ]
            effective_capacity = max(
                1,
                round(
                    int(max_sites_per_surveyor)
                    * min(6, len(surveyor.available_dates or []))
                    / 5
                ),
            )
            output[surveyor.name] = df.drop(
                columns=helper,
                errors="ignore",
            ).head(effective_capacity)

    return output


def allocations_dataframe(
    allocations: Sequence[ClusterAllocation],
) -> pd.DataFrame:
    if not allocations:
        return pd.DataFrame()
    return pd.DataFrame([asdict(a) for a in allocations])


def prioritise_retry_day_assignments(shortlists, portfolio, surveyors, travel_matrix=None):
    """Put dated retries with people available on the requested days before routing."""
    updated = {name: frame.copy() for name, frame in shortlists.items()}
    audit, priority_load = [], {s.name: 0.0 for s in surveyors}
    travel_matrix = travel_matrix or {}
    for _, row in portfolio.iterrows():
        if not bool(row.get("Eligible for Selected Week", False)) or str(row.get("Is Retry", False)).lower() not in {"true", "1", "1.0"}:
            continue
        site = {"is_retry": True,
                "retry_preferred_weekdays": row.get("Retry Preferred Weekdays"),
                "retry_required_weekdays": row.get("Retry Required Weekdays"),
                "retry_forbidden_weekday": row.get("Retry Forbidden Weekday Number"),
                "retry_review_original_surveyor": row.get("Retry Review Original Surveyor")}
        wanted = (_retry_preferred_weekdays(site["retry_preferred_weekdays"])
                  | _retry_preferred_weekdays(site["retry_required_weekdays"]))
        if not wanted and not str(site["retry_review_original_surveyor"] or "").strip():
            continue
        reference = str(row.get("Customer Reference", "") or "").strip()
        wo = str(row.get("Work Order Number", "") or "").strip()
        options = []
        for surveyor in surveyors:
            dates = [d for d in surveyor.available_dates or []
                     if _site_allowed_today(site, d) and _site_allowed_for_surveyor(site, surveyor.name)
                     and (not wanted or d.strftime("%A").lower() in wanted)]
            if not dates:
                continue
            commute = travel_matrix.get((surveyor.name, str(row.get("Planning Cluster", ""))))
            commute = float(commute) if commute is not None and pd.notna(commute) else 120.0
            options.append((priority_load[surveyor.name] / len(dates) + commute, surveyor.name, dates))
        record = {"Work Order Number": wo, "Customer Reference": reference,
                  "Requested Weekdays": ", ".join(sorted(wanted)), "Surveyor": "",
                  "Available Requested Dates": "", "Assignment Outcome": "No suitable surveyor available for the requested days/review."}
        if options:
            _, target, dates = min(options)
            for name, frame in updated.items():
                if frame.empty:
                    continue
                if wo and "Work Order Number" in frame:
                    match = frame["Work Order Number"].fillna("").astype(str).str.strip().eq(wo)
                elif reference and "Customer Reference" in frame:
                    match = frame["Customer Reference"].fillna("").astype(str).str.strip().eq(reference)
                else:
                    match = frame["Building Name"].eq(row.get("Building Name")) & frame["Postcode"].eq(row.get("Postcode"))
                updated[name] = frame.loc[~match].copy()
            chosen = row.to_frame().T.copy()
            chosen["Assigned Surveyor"] = target
            chosen["Home to Cluster (Minutes)"] = travel_matrix.get((target, str(row.get("Planning Cluster", ""))))
            updated[target] = pd.concat([updated.get(target, portfolio.head(0)), chosen], ignore_index=True)
            priority_load[target] += float(row.get("Planning Duration (Minutes)", 0) or 0)
            record.update({"Surveyor": target, "Available Requested Dates": ", ".join(d.isoformat() for d in dates),
                           "Assignment Outcome": "Reserved in the candidate pool; route/time feasibility still required."})
        audit.append(record)
    return updated, pd.DataFrame(audit)


def fill_team_gaps(
    team_results, surveyors, sites_by_surveyor, scheduler_factory,
    first_survey_clock, latest_survey_clock, latest_return_clock, timezone,
    travel_matrix=None, reserve_sites=None, maximise_days=False,
    expand_underfilled_days=False, expansion_gap_minutes=30, diagnostics=None,
    saturday_time_window=None,
):
    """Append globally unbooked work to available days without moving bookings.

    Maximise-days filling first searches within 15 km without a minimum
    survey-to-travel ratio. Optional expansion then searches all remaining
    candidates if the local passes leave at least expansion_gap_minutes before
    the survey cut-off. Normal planning keeps its efficiency preference.
    Existing days first consider the final site's local area. Empty days compare
    up to three productive areas using the already-paid home/cluster matrix.
    All trials use the normal scheduler, including lunch, retry and return rules.
    Accepted weekly-note candidates remain reserved for their original owner.
    """
    results = {
        name: replace(result, days=list(result.days), unscheduled_sites=list(result.unscheduled_sites))
        if result is not None else None
        for name, result in team_results.items()
    }
    travel_matrix = travel_matrix or {}
    site_by_id, owner_by_id = {}, {}
    for name, sites in sites_by_surveyor.items():
        for site in sites:
            identity = _site_identity_for_sequence(site)
            # A note trial can introduce a site already in another shortlist.
            # Prefer its explicit reservation when choosing the pool owner.
            reserved = site.get("special_request_date")
            if identity not in site_by_id or (reserved is not None and not pd.isna(reserved)):
                site_by_id[identity] = site
                owner_by_id[identity] = name

    for site in reserve_sites or []:
        identity = _site_identity_for_sequence(site)
        if identity not in site_by_id:
            site_by_id[identity] = site
            owner_by_id[identity] = ""

    def item_id(item):
        return _site_identity_for_sequence(vars(item))

    booked = {
        item_id(item) for result in results.values() if result is not None
        for day in result.days for item in day.items
    }
    slots = []
    for surveyor in surveyors:
        result = results.get(surveyor.name)
        by_date = {
            (day.first_survey_target or day.start_time).date(): day
            for day in result.days
        } if result is not None else {}
        for day_date in sorted(set(surveyor.available_dates or [])):
            day = by_date.get(day_date)
            first_clock, finish_clock, _ = survey_clocks_for_date(
                day_date, first_survey_clock, latest_survey_clock,
                latest_return_clock, saturday_time_window,
            )
            first = datetime.combine(day_date, first_clock, tzinfo=timezone)
            finish = datetime.combine(day_date, finish_clock, tzinfo=timezone)
            has_work = day is not None and bool(day.items)
            gap = (finish - (day.return_departure if has_work else first)).total_seconds() / 60
            if gap > 0:
                slots.append((not has_work, -gap, day_date, surveyor.name, surveyor, day))

    available_slots = {(row[3], row[2]) for row in slots}
    untried_requested_slots = {
        identity: {(s.name, d) for s in surveyors for d in s.available_dates or []
                   if (s.name, d) in available_slots and _requested_retry_day(site, d)
                   and _site_allowed_for_surveyor(site, s.name)}
        for identity, site in site_by_id.items() if identity not in booked
    }
    slots.sort(key=lambda row: (
        not any((row[3], row[2]) in choices for choices in untried_requested_slots.values()),
        *row[:4]))
    additions = []
    for _, _, day_date, name, surveyor, baseline in slots:
        for choices in untried_requested_slots.values():
            choices.discard((name, day_date))
        first_clock, finish_clock, return_clock = survey_clocks_for_date(
            day_date, first_survey_clock, latest_survey_clock,
            latest_return_clock, saturday_time_window,
        )
        first = datetime.combine(day_date, first_clock, tzinfo=timezone)
        finish = datetime.combine(day_date, finish_clock, tzinfo=timezone)
        deadline = datetime.combine(day_date, return_clock, tzinfo=timezone)
        scheduler = scheduler_factory(surveyor)
        has_work = baseline is not None and bool(baseline.items)
        review = {
            "Surveyor": name, "Date": day_date.isoformat(),
            "Before Filling Survey End": baseline.items[-1].survey_end.strftime("%H:%M") if has_work else "",
            "After Local Filling Survey End": "",
            "Minutes Left After Local Filling": None,
            "Wider Search Candidates": 0, "Candidates Beyond 15 km": 0,
            "Wider Search": "Disabled", "Wider Search Additions": 0,
            "Wider Search Survey Minutes": 0.0,
        }

        def record_review(final_day, outcome):
            if diagnostics is not None:
                review.update({
                    "Final Survey End": final_day.items[-1].survey_end.strftime("%H:%M") if final_day and final_day.items else "",
                    "Return Home": final_day.return_time.strftime("%H:%M") if final_day and final_day.items else "",
                    "Outcome": outcome,
                })
                diagnostics.append(review)

        available_minutes = (finish - (baseline.return_departure if has_work else first)).total_seconds() / 60
        candidates = []
        for identity, site in site_by_id.items():
            if (identity in booked or not _site_allowed_today(site, day_date)
                    or not _site_allowed_for_surveyor(site, name)):
                continue
            if not _requested_retry_day(site, day_date) and untried_requested_slots.get(identity):
                continue
            reserved = site.get("special_request_date")
            if reserved is not None and not pd.isna(reserved) and owner_by_id[identity] != name:
                continue
            if float(site["planning_minutes"]) + scheduler.pre_survey_buffer_minutes > available_minutes:
                continue
            candidate = dict(site)
            candidate["home_to_cluster_minutes"] = travel_matrix.get((name, _planning_cluster_key(site)))
            candidates.append(candidate)

        all_candidates = list(candidates)
        anchor = None
        if has_work:
            anchor = site_by_id.get(item_id(baseline.items[-1]))
            if anchor is None:
                record_review(baseline, "Last scheduled building missing from the candidate data; cannot extend its route.")
                continue
            candidates = _local_work_candidates(anchor, candidates)
        else:
            # No added matrix: use existing home-fit measurements to shortlist
            # whole working areas before asking Google about an empty day.
            workloads = _day_area_workloads(
                candidates, max(0, available_minutes - scheduler.lunch_minutes),
                scheduler.pre_survey_buffer_minutes + scheduler.post_survey_buffer_minutes,
            )
            area_scores = {}
            for area, workload in workloads.items():
                commute = travel_matrix.get((name, area))
                if commute is not None and math.isfinite(float(commute)):
                    score = workload - 2 * float(commute)
                    if score > 0:
                        area_scores[area] = score
            if maximise_days:
                # Missing representative routes must not hide eligible reserve
                # areas. Coordinates rank them; real transit still decides fit.
                home = _surveyor_home_coordinates(surveyor)
                for site in candidates:
                    area = _planning_cluster_key(site)
                    if area not in area_scores:
                        distance = (haversine_km(*home, site.get("latitude"), site.get("longitude"))
                                    if home else None)
                        area_scores[area] = workloads.get(area, 0) - (2 * distance if distance is not None else 60)
            selected_areas = sorted(area_scores, key=lambda area: (-area_scores[area], area))[:3]
            candidates = [site for site in candidates if _planning_cluster_key(site) in selected_areas]
        requested_today = [site for site in all_candidates if _requested_retry_day(site, day_date)]
        if not candidates and not requested_today and not (maximise_days and (has_work or expand_underfilled_days)):
            record_review(baseline, "No candidates in the normal search after availability, reservation and duration checks.")
            continue

        if has_work:
            # New jobs on a previously visited road must not lose to a wider
            # detour solely because the baseline already left that road.
            immediate = []
            for site in candidates:
                distance = haversine_km(anchor.get("latitude"), anchor.get("longitude"),
                                        site.get("latitude"), site.get("longitude"))
                if ((distance is not None and distance <= NO_GOOGLE_RADIUS_KM)
                        or scheduler._same_postcode(anchor.get("postcode", ""), site.get("postcode", ""))):
                    immediate.append(site)
            same_area = [site for site in candidates
                         if _planning_cluster_key(site) == _planning_cluster_key(anchor)]
            batches = [immediate, same_area, candidates]
        else:
            batches = [candidates]

        priority_index = None
        if requested_today:
            priority_index = 0
            batches.insert(0, requested_today)

        wider_index = None
        if has_work and maximise_days:
            wider = []
            for site in all_candidates:
                distance = haversine_km(anchor.get("latitude"), anchor.get("longitude"),
                                        site.get("latitude"), site.get("longitude"))
                if distance is not None and distance <= 15.0:
                    wider.append(site)
            if wider:
                wider_index = len(batches)
                batches.append(wider)
        best = baseline if has_work else None
        expansion_index = None
        if maximise_days and expand_underfilled_days:
            expansion_index = len(batches)
            batches.append(all_candidates)
        addition_pass = {}
        for batch_index, batch in enumerate(batches):
            expanded_search = batch_index == expansion_index
            if expanded_search:
                remaining_minutes = (finish - (best.return_departure if best is not None else first)).total_seconds() / 60
                review["After Local Filling Survey End"] = best.items[-1].survey_end.strftime("%H:%M") if best is not None and best.items else ""
                review["Minutes Left After Local Filling"] = round(max(0, remaining_minutes), 1)
                if remaining_minutes < expansion_gap_minutes:
                    review["Wider Search"] = f"Not triggered: less than {expansion_gap_minutes:g} minutes left"
                    continue
            wider_search = maximise_days and (expanded_search or batch_index == priority_index or batch_index == wider_index or not has_work)
            scheduler.minimum_survey_to_travel_ratio = 0.0 if maximise_days else 2.0
            used_here = {item_id(item) for item in best.items} if best is not None else set()
            batch = [site for site in batch if _site_identity_for_sequence(site) not in used_here]
            if expanded_search:
                # Cheap duration pruning is safe; do not use current home travel
                # as a veto because a new stop may bring the surveyor nearer home.
                batch = [site for site in batch if float(site["planning_minutes"])
                         + scheduler.pre_survey_buffer_minutes <= remaining_minutes]
                review["Wider Search Candidates"] = len(batch)
                if best is not None:
                    current_anchor = site_by_id[item_id(best.items[-1])]
                    distances = [haversine_km(current_anchor.get("latitude"), current_anchor.get("longitude"),
                                             site.get("latitude"), site.get("longitude")) for site in batch]
                    review["Candidates Beyond 15 km"] = sum(d is not None and d > 15 for d in distances)
                review["Wider Search"] = "Searched all distances" if batch else "No eligible candidates remain that could fit before travel"
            if not batch:
                continue
            trial = scheduler.build_day(
                batch, first, finish, deadline,
                resume_from=best,
                resume_site=site_by_id[item_id(best.items[-1])] if best is not None else None,
                local_continuation_only=has_work and not wider_search,
            )
            prefix = best.items if best is not None else []
            added = trial.items[len(prefix):]
            added_ids = [item_id(item) for item in added]
            # Keep a productive close-by extension even if a later, wider
            # trial fails its travel or return-home check.
            extra_travel = trial.travel_minutes - (best.travel_minutes if best is not None else 0)
            added_work = sum(item.survey_minutes for item in added)
            if (not added or trial.items[:len(prefix)] != prefix
                    or len(set(added_ids)) != len(added_ids) or booked.intersection(added_ids)
                    or trial.return_time > deadline or any(item.survey_end > finish for item in added)
                    or added_work <= 0
                    or (not maximise_days and added_work < 2.0 * max(0, extra_travel))):
                continue
            best = trial
            for identity in added_ids:
                addition_pass[identity] = ("Expanded search" if expanded_search else
                                           "Requested retry day" if batch_index == priority_index else "Normal search")
            if expanded_search:
                review["Wider Search Additions"] = len(added)
                review["Wider Search Survey Minutes"] = round(added_work, 1)

        if review["Wider Search"] == "Searched all distances":
            outcome = ("Added work using the wider search within the survey, lunch and return-home limits."
                       if review["Wider Search Additions"] else
                       "Wider search found no additional feasible extension of this route within the time and lunch limits.")
        else:
            outcome = review["Wider Search"]
        record_review(best or baseline, outcome)

        if best is None or best is baseline:
            continue
        trial = best
        prefix = baseline.items if has_work else []
        added = trial.items[len(prefix):]
        added_ids = [item_id(item) for item in added]

        result = results.get(name)
        days = list(result.days) if result is not None else []
        if baseline is not None:
            days = [trial if day is baseline else day for day in days]
        else:
            days.append(trial)
        days.sort(key=lambda day: day.first_survey_target or day.start_time)
        results[name] = WeeklyScheduleResult(days, [])
        booked.update(added_ids)
        for item, identity in zip(added, added_ids):
            additions.append({
                "Customer Reference": item.customer_reference,
                "Building Name": item.building_name,
                "Postcode": item.postcode,
                "Original Surveyor": owner_by_id[identity],
                "Surveyor": name,
                "Date": day_date.isoformat(),
                "Survey Minutes Added": item.survey_minutes,
                "Fill Pass": addition_pass[identity],
            })

    for name, result in list(results.items()):
        if result is not None:
            results[name] = replace(result, unscheduled_sites=[
                site for site in sites_by_surveyor.get(name, [])
                if _site_identity_for_sequence(site) not in booked
            ])
    return results, pd.DataFrame(additions)


def apply_gap_assignments(shortlists, additions, travel_matrix, reserve_portfolio=None):
    """Keep exported shortlists and downstream resource assignment in sync."""
    updated = {name: frame.copy() for name, frame in shortlists.items()}
    for addition in additions.to_dict(orient="records"):
        source, target = addition["Original Surveyor"], addition["Surveyor"]
        if source == target:
            continue
        frame = updated[source] if source else reserve_portfolio
        if frame is None:
            raise ValueError("Missing reserve portfolio for an added gap-fill site.")
        reference = str(addition["Customer Reference"] or "").strip()
        if reference:
            match = frame["Customer Reference"].fillna("").astype(str).str.strip().eq(reference)
        else:
            match = (frame["Building Name"].astype(str).eq(addition["Building Name"])
                     & frame["Postcode"].astype(str).eq(addition["Postcode"]))
        moved = frame.loc[match].head(1).copy()
        if moved.empty:
            raise ValueError(f"Cannot find gap-fill source row for {addition['Building Name']}.")
        if source:
            updated[source] = frame.loc[~match].copy()
        moved["Assigned Surveyor"] = target
        moved["Home to Cluster (Minutes)"] = moved["Planning Cluster"].map(
            lambda area: travel_matrix.get((target, str(area)))
        )
        updated[target] = pd.concat([updated.get(target, pd.DataFrame()), moved], ignore_index=True)
    return updated


def capacity_review(portfolio, surveyors, results, survey_minutes_per_day,
                    survey_minutes_by_date=None):
    """Distinguish a workload shortage from work left unplaced by routing."""
    eligible = portfolio.loc[portfolio["Eligible for Selected Week"].eq(True)]
    available = float(pd.to_numeric(eligible["Planning Duration (Minutes)"], errors="coerce").fillna(0).sum())
    days = sum(len(s.available_dates or []) for s in surveyors)
    survey_minutes_by_date = survey_minutes_by_date or {}
    window = sum(float(survey_minutes_by_date.get(day, survey_minutes_per_day))
                 for surveyor in surveyors for day in surveyor.available_dates or [])
    scheduled = sum(r.total_survey_minutes for r in results.values() if r is not None)
    return pd.DataFrame([
        {"Measure": "Eligible buildings", "Value": len(eligible), "Meaning": "After status, access and booking gates."},
        {"Measure": "Eligible survey hours", "Value": round(available / 60, 2), "Meaning": "All eligible work, including outside initial shortlists."},
        {"Measure": "Selected person-days", "Value": days, "Meaning": "Surveyor availability selected for this run."},
        {"Measure": "Survey window hours before travel", "Value": round(window / 60, 2), "Meaning": "Working window less lunch; travel will reduce this."},
        {"Measure": "Scheduled survey hours", "Value": round(scheduled / 60, 2), "Meaning": "Survey time actually placed in the final routes."},
        {"Measure": "Unplaced eligible survey hours", "Value": round(max(0, available - scheduled) / 60, 2), "Meaning": "Check remaining locations, access days and journey feasibility."},
        {"Measure": "Supply assessment", "Value": "Eligible work remains unplaced" if available - scheduled > 0.1 else "All eligible survey work placed",
         "Meaning": "Survey hours alone do not measure full days because travel also uses time. See Day Filling Review for the wider-search results."},
    ])
