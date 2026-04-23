"""
Ontology-guided affordance reasoning benchmark for robot navigation explanations.

How to run:
    python affordance_ontology_experiments.py

What the script does:
    This script implements a lightweight, fully Python-based benchmark that
    evaluates whether reasoning over an affordance ontology helps identify
    relevant explanation factors for robot navigation. The motivating scenario
    is a robot librarian that has to fetch a book in a library. The robot can
    either take a direct route through a shortcut corridor or a longer route
    through a detour corridor. Objects such as chairs, doors, cabinets, carts,
    and people may appear in the environment. The key question is whether the
    system can correctly identify which object and which affordance-state change
    would most improve or restore the route.

Compared methods:
    1. Ontology-guided affordance reasoning (our method).
    2. A semantic-only baseline that only considers rough corridor relevance.

Generated results:
    All outputs are exported into a folder named ``results`` next to this script.
    The script writes tabular results as CSV files, aggregate metrics as JSON,
    and figures as PNG files. Concretely, it generates:

    - benchmark_cases.csv: per-environment summary statistics.
    - method_comparison.csv: Precision@2 and Recall@2 per environment and method.
    - ontology_noise_results.csv: per-environment robustness results under
      incomplete ontology knowledge.
    - ontology_noise_summary.csv: mean robustness scores grouped by miss rate.
    - distractor_robustness_results.csv: per-environment robustness results
      under increasing semantic clutter.
    - distractor_robustness_summary.csv: mean clutter-robustness scores grouped
      by distractor level and method.
    - summary.json: aggregated main benchmark results.
    - precision_recall_barplot.png: bar plot comparing the main methods.
    - ontology_noise_robustness.png: line plot showing robustness to missing
      affordances in the ontology.
"""

import json
import math
import random
from dataclasses import dataclass
from heapq import heappop, heappush
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# The robot may move in four-neighborhood grid directions.
MOVES = [(1, 0), (-1, 0), (0, 1), (0, -1)]

# Mapping from semantic object type to its available affordances.
# This acts as the core affordance knowledge used by the ontology.
AFFORDANCE_MAP: Dict[str, List[str]] = {
    "chair": ["movable"],
    "door": ["openable"],
    "cabinet": ["movable", "openable"],
    "person": ["movable"],
    "cart": ["movable"],
}


@dataclass
class ObjectNode:
    """
    Single object instance in the environment.

    Attributes:
        obj_id: Unique identifier of the object instance.
        obj_type: Semantic type such as chair, door, or person.
        pos: Grid position of the object.
        affordances: List of affordances associated with the object.
        state: Current binary state of each affordance (0 = unresolved, 1 = satisfied).
        role: Whether the object is a true shortcut factor or only a distractor.
    """

    obj_id: str
    obj_type: str
    pos: Tuple[int, int]
    affordances: List[str]
    state: Dict[str, int]
    role: str  # "shortcut" or "distractor"


@dataclass
class Environment:
    """
    Procedural navigation environment used in the benchmark.

    Attributes:
        size: Grid side length.
        start: Robot start position.
        goal: Robot goal position.
        free_cells: Set of traversable cells that define the corridor structure.
        objects: All semantic objects currently present in the environment.
    """

    size: int
    start: Tuple[int, int]
    goal: Tuple[int, int]
    free_cells: Set[Tuple[int, int]]
    objects: List[ObjectNode]


def dijkstra(cost_grid: List[List[float]], start: Tuple[int, int], goal: Tuple[int, int]) -> Tuple[float, Optional[List[Tuple[int, int]]]]:
    """
    Compute the shortest path on a grid with non-negative traversal costs.

    Args:
        cost_grid: 2D cost map. Infinite values represent blocked cells.
        start: Start cell.
        goal: Goal cell.

    Returns:
        Tuple containing:
        - total path cost (infinity if unreachable),
        - reconstructed path as a list of cells, or None if unreachable.
    """

    
    height, width = len(cost_grid), len(cost_grid[0])
    
    # Priority queue for Dijkstra search.
    pq: List[Tuple[float, Tuple[int, int]]] = [(0.0, start)]
    
    # Previous-pointer structure for path reconstruction.    
    prev: Dict[Tuple[int, int], Optional[Tuple[int, int]]] = {start: None}
    
    # Best currently known distance to each visited cell.
    dist: Dict[Tuple[int, int], float] = {start: 0.0}

    while pq:
        current_dist, current = heappop(pq)

        # Ignore outdated queue entries.
        if current_dist != dist[current]:
            continue

        # If we reached the goal, reconstruct the path.
        if current == goal:
            path: List[Tuple[int, int]] = []
            node: Optional[Tuple[int, int]] = current
            while node is not None:
                path.append(node)
                node = prev[node]
            return current_dist, path[::-1]

        # Expand four-neighborhood successors.
        for dx, dy in MOVES:
            nx, ny = current[0] + dx, current[1] + dy
            if not (0 <= nx < height and 0 <= ny < width):
                continue
            step_cost = cost_grid[nx][ny]
            if math.isinf(step_cost):
                continue
            new_dist = current_dist + step_cost
            neighbor = (nx, ny)
            if new_dist < dist.get(neighbor, math.inf):
                dist[neighbor] = new_dist
                prev[neighbor] = current
                heappush(pq, (new_dist, neighbor))

    # Goal was unreachable.
    return math.inf, None


def make_environment(seed: int, size: int = 15) -> Environment:
    """
    Create one procedural benchmark environment.

    The environment contains two route structures:
    - a direct shortcut corridor along the center row,
    - a longer detour corridor along the top row and right side.

    Objects placed on the shortcut corridor are the true semantically relevant
    factors for explanation. Additional objects are placed as distractors.

    Args:
        seed: Random seed for reproducibility.
        size: Side length of the square grid.

    Returns:
        Procedurally generated Environment instance.
    """

    rnd = random.Random(seed)
    start = (size // 2, 0)
    goal = (size // 2, size - 1)

    # Two semantic route structures:
    # - direct shortcut through the center row
    # - long detour via top row
    
    # Direct route through the middle of the grid.
    shortcut = [(size // 2, y) for y in range(size)]
    # Longer detour: up to the top row, across, then down to the goal column.
    detour: List[Tuple[int, int]] = []
    for x in range(size // 2, -1, -1):
        detour.append((x, 0))
    for y in range(1, size):
        detour.append((0, y))
    for x in range(1, size // 2 + 1):
        detour.append((x, size - 1))

    # Only cells that belong to one of the two corridors are traversable.
    free_cells = set(shortcut) | set(detour)

    objects: List[ObjectNode] = []
    obj_counter = 0

    # Shortcut blockers: these are the semantically important explanatory candidates.
    shortcut_positions = rnd.sample(shortcut[2:-2], rnd.randint(2, 4))
    for pos in shortcut_positions:
        obj_type = rnd.choice(list(AFFORDANCE_MAP.keys()))
        affordances = AFFORDANCE_MAP[obj_type]

        # Shortcut objects start unresolved so that they can genuinely matter.
        state = {aff: 0 for aff in affordances}
        objects.append(
            ObjectNode(
                obj_id=f"o{obj_counter}",
                obj_type=obj_type,
                pos=pos,
                affordances=affordances,
                state=state,
                role="shortcut",
            )
        )
        obj_counter += 1
    
    # Distractors occupy other cells but are not central to shortcut recovery.
    occupied = {start, goal, *shortcut_positions}
    candidate_distractors = [cell for cell in free_cells if cell not in occupied]
    for pos in rnd.sample(candidate_distractors, rnd.randint(3, 6)):
        obj_type = rnd.choice(list(AFFORDANCE_MAP.keys()))
        affordances = AFFORDANCE_MAP[obj_type]
        
        # Distractors are sometimes unresolved and sometimes already satisfied.
        state = {aff: 0 if rnd.random() < 0.35 else 1 for aff in affordances}
        objects.append(
            ObjectNode(
                obj_id=f"o{obj_counter}",
                obj_type=obj_type,
                pos=pos,
                affordances=affordances,
                state=state,
                role="distractor",
            )
        )
        obj_counter += 1

    return Environment(size=size, start=start, goal=goal, free_cells=free_cells, objects=objects)


def cell_cost(obj: ObjectNode) -> float:
    """
    Convert object semantics and affordance state into a traversal cost.

    This simple cost model captures the idea that not all unresolved objects are
    equally problematic:
    - satisfied object states are cheap to traverse,
    - a closed door is expensive but still passable,
    - a person still in the way is even more expensive,
    - hard obstructions such as unmoved chairs or carts are impassable.
    """

    if all(value == 1 for value in obj.state.values()):
        return 1.0
    if obj.obj_type == "door":
        return 5.0
    if obj.obj_type == "person":
        return 8.0
    return math.inf


def qualitative_relation(robot: Tuple[int, int], obj_pos: Tuple[int, int]) -> str:
    """
    Return a simple qualitative spatial relation of an object to the robot.

    This mirrors the human-friendly spatial style used in your affordance-based
    explanation paper, where objects are verbalized as being in front, back,
    left, or right of the robot.
    """

    rx, ry = robot
    ox, oy = obj_pos
    dx, dy = ox - rx, oy - ry

    if abs(dx) >= abs(dy):
        return "front" if dx > 0 else "back"
    return "right" if dy > 0 else "left"


def cost_grid(env: Environment) -> List[List[float]]:
    """
    Build the traversal cost grid induced by the current ontology state.
    """

    grid = [[math.inf] * env.size for _ in range(env.size)]

    # Corridor cells are traversable by default.
    for x, y in env.free_cells:
        grid[x][y] = 1.0

    # Objects modify the cost of the cells they occupy.
    for obj in env.objects:
        x, y = obj.pos
        grid[x][y] = cell_cost(obj)

    # Ensure start and goal always remain traversable.
    sx, sy = env.start
    gx, gy = env.goal
    grid[sx][sy] = 1.0
    grid[gx][gy] = 1.0
    return grid


def clone_environment(env: Environment) -> Environment:
    """
    Deep-copy an environment so that hypothetical interventions are isolated.
    """

    return Environment(
        size=env.size,
        start=env.start,
        goal=env.goal,
        free_cells=set(env.free_cells),
        objects=[
            ObjectNode(
                obj_id=obj.obj_id,
                obj_type=obj.obj_type,
                pos=obj.pos,
                affordances=list(obj.affordances),
                state=dict(obj.state),
                role=obj.role,
            )
            for obj in env.objects
        ],
    )


def reason_over_affordances(env: Environment) -> Tuple[float, List[Dict[str, object]]]:
    """
    Rank object-affordance pairs by their utility for route recovery/improvement.

    For each unresolved affordance of each object, we simulate a hypothetical
    state change from 0 to 1, re-plan the route, and measure how much this
    improves the outcome.

    Returns:
        - the baseline path cost in the original environment,
        - a list of candidate explanation factors sorted by decreasing utility.
    """

    base_cost, _ = dijkstra(cost_grid(env), env.start, env.goal)
    results: List[Dict[str, object]] = []

    for obj in env.objects:
        for aff in obj.affordances:
            # Only unresolved affordances are meaningful intervention candidates.
            if obj.state[aff] != 0:
                continue

            # Apply a single hypothetical state change.
            intervened = clone_environment(env)
            for candidate in intervened.objects:
                if candidate.obj_id == obj.obj_id:
                    candidate.state[aff] = 1
                    break

            # Re-plan after the intervention.
            new_cost, _ = dijkstra(cost_grid(intervened), intervened.start, intervened.goal)

            # Utility definition:
            # - large reward if infeasible becomes feasible,
            # - otherwise cost improvement if route becomes shorter/safer,
            # - zero if nothing improves.
            if math.isinf(base_cost) and not math.isinf(new_cost):
                delta = 1000.0
            elif not math.isinf(base_cost) and not math.isinf(new_cost):
                delta = max(0.0, base_cost - new_cost)
            else:
                delta = 0.0

            results.append(
                {
                    "obj_id": obj.obj_id,
                    "obj_type": obj.obj_type,
                    "position": obj.pos,
                    "affordance": aff,
                    "role": obj.role,
                    "spatial_relation": qualitative_relation(env.start, obj.pos),
                    "delta": delta,
                    "new_cost": new_cost,
                }
            )

    # Highest-utility explanation factors should come first.
    results.sort(key=lambda item: (-item["delta"], item["obj_id"], item["affordance"]))
    return base_cost, results



def make_environment_with_extra_distractors(seed: int, extra_distractors: int, size: int = 15) -> Environment:
    """Create an environment and inject additional distractor objects.

    This helper is used for a clutter-robustness experiment. Starting from the
    standard benchmark environment, it adds more distractor objects to free cells
    that are not already occupied. The shortcut factors remain unchanged, so the
    experiment isolates the effect of semantic clutter.

    Args:
        seed: Random seed for reproducibility.
        extra_distractors: Number of extra distractor objects to add.
        size: Grid side length.

    Returns:
        Environment with additional distractor objects.
    """

    env = make_environment(seed, size=size)
    rnd = random.Random(50000 + seed + 17 * extra_distractors)
    occupied = {env.start, env.goal, *[obj.pos for obj in env.objects]}
    candidate_cells = [cell for cell in env.free_cells if cell not in occupied]
    num_to_add = min(extra_distractors, len(candidate_cells))
    obj_counter = len(env.objects)

    for pos in rnd.sample(candidate_cells, num_to_add):
        obj_type = rnd.choice(list(AFFORDANCE_MAP.keys()))
        affordances = AFFORDANCE_MAP[obj_type]
        state = {aff: 0 if rnd.random() < 0.35 else 1 for aff in affordances}
        env.objects.append(
            ObjectNode(
                obj_id=f"o{obj_counter}",
                obj_type=obj_type,
                pos=pos,
                affordances=affordances,
                state=state,
                role="distractor",
            )
        )
        obj_counter += 1

    return env


def evaluate_distractor_robustness(
    num_envs: int = 1000,
    extra_distractor_levels: Optional[List[int]] = None,
) -> pd.DataFrame:
    """
    Evaluate robustness when the environment contains more semantic clutter.

    The experiment keeps the shortcut explanation factors fixed but increases the
    number of distractor objects in the corridors. This tests whether ontology-
    guided reasoning remains selective when the environment becomes semantically
    busier.

    Args:
        num_envs: Number of procedural environments per clutter level.
        extra_distractor_levels: Numbers of extra distractors to add.

    Returns:
        Data frame with precision and recall values per seed, method, and level.
    """

    if extra_distractor_levels is None:
        extra_distractor_levels = [0, 2, 4, 6, 8]

    rows: List[Dict[str, object]] = []
    for extra in extra_distractor_levels:
        for seed in range(num_envs):
            # Build a cluttered environment and compute ranked explanation candidates.
            env = make_environment_with_extra_distractors(seed, extra_distractors=extra)
            _, ontology_results = reason_over_affordances(env)
            ontology_top2 = [entry for entry in ontology_results[:2] if entry["delta"] > 0]
            baseline_top2 = semantic_only_baseline(env, top_k=2)

            # Ground truth remains the set of shortcut-related candidates that actually help.
            ground_truth = {
                (entry["obj_id"], entry["affordance"])
                for entry in ontology_results
                if entry["role"] == "shortcut" and entry["delta"] > 0
            }
            if not ground_truth:
                continue

            # Compute retrieval metrics for both competing methods.
            for method_name, preds in [
                ("semantic_only", baseline_top2),
                ("ontology_guided", ontology_top2),
            ]:
                pred_set = {(entry["obj_id"], entry["affordance"]) for entry in preds}
                precision = len(pred_set & ground_truth) / max(1, len(pred_set))
                recall = len(pred_set & ground_truth) / len(ground_truth)
                rows.append(
                    {
                        "seed": seed,
                        "extra_distractors": extra,
                        "method": method_name,
                        "precision_at_2": precision,
                        "recall_at_2": recall,
                    }
                )

    return pd.DataFrame(rows)


def corrupt_environment_affordances(env: Environment, miss_rate: float, seed: int) -> Environment:
    """
    Create a copy of the environment with incomplete ontology knowledge.

    The corruption model simulates imperfect semantic knowledge by removing
    unresolved affordances from some objects with probability ``miss_rate``.
    This means the reasoner may fail to consider a truly helpful intervention
    because the ontology does not expose that affordance.

    Args:
        env: Original clean environment.
        miss_rate: Probability of hiding each unresolved affordance.
        seed: Random seed for reproducibility.

    Returns:
        A copied environment with partially missing affordance knowledge.
    """

    rnd = random.Random(seed)
    corrupted = clone_environment(env)
    for obj in corrupted.objects:
        kept_affordances = []
        for aff in obj.affordances:
            if obj.state.get(aff, 0) == 0 and rnd.random() < miss_rate:
                continue
            kept_affordances.append(aff)

        # Keep at least one affordance whenever the original object had any.
        # This avoids degenerate objects with no semantic description at all.
        if not kept_affordances and obj.affordances:
            kept_affordances = [rnd.choice(obj.affordances)]
        obj.affordances = kept_affordances
        obj.state = {aff: obj.state[aff] for aff in obj.affordances}
    return corrupted


def evaluate_ontology_noise(num_envs: int = 1000, miss_rates: Optional[List[float]] = None) -> pd.DataFrame:
    """
    Evaluate robustness under incomplete ontology knowledge.

    Ground truth is always defined on the clean environment. The reasoning step,
    however, is executed on a corrupted copy in which some helpful affordances are
    hidden from the ontology. This isolates how strongly explanation quality depends
    on complete semantic knowledge.

    Args:
        num_envs: Number of procedural environments.
        miss_rates: List of affordance-missing probabilities to evaluate.

    Returns:
        Data frame with precision and recall values for every seed and miss rate.
    """

    if miss_rates is None:
        miss_rates = [0.0, 0.1, 0.2, 0.3]

    rows: List[Dict[str, object]] = []
    for seed in range(num_envs):
        # Generate a clean environment whose helpful shortcut factors define ground truth.
        clean_env = make_environment(seed)
        _, clean_results = reason_over_affordances(clean_env)
        ground_truth = {
            (entry["obj_id"], entry["affordance"])
            for entry in clean_results
            if entry["role"] == "shortcut" and entry["delta"] > 0
        }

        if not ground_truth:
            continue

        for miss_rate in miss_rates:
            # Corrupt only the ontology knowledge, not the underlying environment itself.
            corrupted_env = corrupt_environment_affordances(clean_env, miss_rate=miss_rate, seed=10000 + seed)
            _, noisy_results = reason_over_affordances(corrupted_env)
            top2 = [entry for entry in noisy_results[:2] if entry["delta"] > 0]
            pred_set = {(entry["obj_id"], entry["affordance"]) for entry in top2}
            precision = len(pred_set & ground_truth) / max(1, len(pred_set))
            recall = len(pred_set & ground_truth) / len(ground_truth)
            rows.append(
                {
                    "seed": seed,
                    "miss_rate": miss_rate,
                    "precision_at_2": precision,
                    "recall_at_2": recall,
                }
            )

    return pd.DataFrame(rows)


def semantic_only_baseline(env: Environment, top_k: int = 2) -> List[Dict[str, object]]:
    """Baseline that ignores affordance reasoning.

    The baseline only ranks unresolved objects by how close they are to the
    center shortcut corridor. It does *not* simulate affordance-state changes.
    This makes it a useful comparison against the ontology-guided method.
    """

    center_row = env.start[0]
    candidates: List[Dict[str, object]] = []

    for obj in env.objects:
        unresolved = [aff for aff, value in obj.state.items() if value == 0]
        if not unresolved:
            continue

        # Higher score means more central to the nominal shortcut corridor.
        score = -abs(obj.pos[0] - center_row)
        candidates.append(
            {
                "obj_id": obj.obj_id,
                "obj_type": obj.obj_type,
                "position": obj.pos,
                "affordance": unresolved[0],
                "role": obj.role,
                "score": score,
            }
        )

    candidates.sort(key=lambda item: (-item["score"], item["obj_id"]))
    return candidates[:top_k]


def textualize(factor: Dict[str, object], mode: str = "suggestive") -> str:
    """
    Turn a ranked explanation factor into a natural-language explanation.

    Args:
        factor: Ranked explanation factor returned by the reasoner.
        mode: One of "descriptive", "suggestive", or "counterfactual".

    Returns:
        Human-readable explanation string.
    """

    obj_type = str(factor["obj_type"])
    affordance = str(factor["affordance"])
    relation = str(factor["spatial_relation"])

    if mode == "descriptive":
        return (
            f"I cannot proceed because the {obj_type} {relation} of me "
            f"is not in the required {affordance} state."
        )
    if mode == "counterfactual":
        return (
            f"If the {obj_type} {relation} of me satisfied the {affordance} "
            f"affordance, I could improve or restore my route."
        )
    return (
        f"Please make the {obj_type} {relation} of me satisfy the {affordance} "
        f"affordance so that I can continue safely."
    )


def evaluate_methods(num_envs: int = 1000) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, float]]:
    """
    Run the main benchmark comparison between the ontology-guided method and
    the semantic-only baseline.

    For each procedural environment, the function stores:
    - case-level information such as baseline feasibility and an example text,
    - per-method precision/recall values whenever shortcut ground truth exists,
    - aggregate summary statistics across all generated environments.

    Args:
        num_envs: Number of procedural environments to generate.

    Returns:
        Tuple containing:
        - case-level results as a data frame,
        - method comparison results as a data frame,
        - aggregated summary statistics as a dictionary.
    """

    cases: List[Dict[str, object]] = []
    comparisons: List[Dict[str, object]] = []

    for seed in range(num_envs):
        # Generate one environment and compute ranked explanation candidates.
        env = make_environment(seed)
        base_cost, ontology_results = reason_over_affordances(env)

        # Evaluate only the top two candidates, matching the paper metrics.
        ontology_top2 = [entry for entry in ontology_results[:2] if entry["delta"] > 0]
        baseline_top2 = semantic_only_baseline(env, top_k=2)

        ground_truth = {
            (entry["obj_id"], entry["affordance"])
            for entry in ontology_results
            if entry["role"] == "shortcut" and entry["delta"] > 0
        }

        # Store per-environment diagnostics and one example natural-language explanation.
        cases.append(
            {
                "seed": seed,
                "base_cost": base_cost,
                "base_feasible": int(not math.isinf(base_cost)),
                "num_positive_candidates": sum(1 for entry in ontology_results if entry["delta"] > 0),
                "top_delta": ontology_top2[0]["delta"] if ontology_top2 else 0.0,
                "gt_size": len(ground_truth),
                "example_explanation": textualize(ontology_top2[0], "suggestive") if ontology_top2 else "",
            }
        )

        if not ground_truth:
            continue

        for method_name, preds in [
            ("semantic_only", baseline_top2),
            ("ontology_guided", ontology_top2),
        ]:
            pred_set = {(entry["obj_id"], entry["affordance"]) for entry in preds}
            precision = len(pred_set & ground_truth) / max(1, len(pred_set))
            recall = len(pred_set & ground_truth) / len(ground_truth)
            comparisons.append(
                {
                    "seed": seed,
                    "method": method_name,
                    "precision_at_2": precision,
                    "recall_at_2": recall,
                }
            )

    # Convert collected rows to tabular results.
    case_df = pd.DataFrame(cases)
    comparison_df = pd.DataFrame(comparisons)

    # Aggregate the most important benchmark-level statistics.
    summary = {
        "num_envs": int(num_envs),
        "finite_base_fraction": float((case_df["base_feasible"] == 1).mean()),
        "cases_with_ground_truth_fraction": float((case_df["gt_size"] > 0).mean()),
        "mean_positive_candidates": float(case_df["num_positive_candidates"].mean()),
        "mean_top_delta_positive_cases": float(case_df.loc[case_df["top_delta"] > 0, "top_delta"].mean()),
        "ontology_precision_at_2": float(comparison_df.loc[comparison_df["method"] == "ontology_guided", "precision_at_2"].mean()),
        "ontology_recall_at_2": float(comparison_df.loc[comparison_df["method"] == "ontology_guided", "recall_at_2"].mean()),
        "baseline_precision_at_2": float(comparison_df.loc[comparison_df["method"] == "semantic_only", "precision_at_2"].mean()),
        "baseline_recall_at_2": float(comparison_df.loc[comparison_df["method"] == "semantic_only", "recall_at_2"].mean()),
    }

    return case_df, comparison_df, summary


def save_plots(comparison_df: pd.DataFrame, noise_df: pd.DataFrame, output_dir: Path) -> None:
    """
    Save the publication-style plots produced by the benchmark.

    Args:
        comparison_df: Data frame containing main precision/recall results for
            the ontology-guided method and the semantic-only baseline.
        noise_df: Data frame containing robustness results under incomplete
            ontology knowledge.
        output_dir: Directory in which all PNG figures should be saved.
    """

    sns.set_theme(style="whitegrid", context="paper")

    # Main bar plot comparing the two methods on precision and recall.
    long_df = comparison_df.melt(
        id_vars=["method"],
        value_vars=["precision_at_2", "recall_at_2"],
        var_name="metric",
        value_name="score",
    )

    plt.figure(figsize=(6, 4))
    ax = sns.barplot(data=long_df, x="metric", y="score", hue="method", errorbar="sd")
    ax.set_xlabel("")
    ax.set_ylabel("Score")
    ax.set_title("Explanation factor retrieval performance")
    plt.tight_layout()
    plt.savefig(output_dir / "precision_recall_barplot.png", dpi=300)
    plt.close()

    # Robustness plot: how performance changes when the ontology misses affordances.
    noise_long_df = noise_df.melt(
        id_vars=["miss_rate"],
        value_vars=["precision_at_2", "recall_at_2"],
        var_name="metric",
        value_name="score",
    )

    plt.figure(figsize=(6, 4))
    ax = sns.lineplot(data=noise_long_df, x="miss_rate", y="score", hue="metric", marker="o", errorbar="sd")
    ax.set_xlabel("Missing-affordance rate")
    ax.set_ylabel("Score")
    ax.set_title("Robustness to incomplete ontology knowledge")
    plt.tight_layout()
    plt.savefig(output_dir / "ontology_noise_robustness.png", dpi=300)
    plt.close()


def main() -> None:
    """
    Execute all benchmark experiments, export the results, and print summaries.

    The function runs three experiment blocks:
    1. the main method comparison,
    2. robustness to incomplete ontology knowledge,
    3. robustness to increasing semantic clutter.

    All tabular results are exported as CSV, aggregate statistics as JSON, and
    plots as PNG files inside the local ``results`` directory.
    """

    output_dir = Path(__file__).resolve().parent / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Run the main benchmark and both robustness experiments.
    case_df, comparison_df, summary = evaluate_methods(num_envs=1000)
    noise_df = evaluate_ontology_noise(num_envs=1000, miss_rates=[0.0, 0.1, 0.2, 0.3])
    noise_summary = (
        noise_df.groupby("miss_rate")[["precision_at_2", "recall_at_2"]]
        .mean()
        .reset_index()
    )
    clutter_df = evaluate_distractor_robustness(num_envs=1000, extra_distractor_levels=[0, 2, 4, 6, 8])
    clutter_summary = (
        clutter_df.groupby(["extra_distractors", "method"])[["precision_at_2", "recall_at_2"]]
        .mean()
        .reset_index()
    )

    # Export all machine-readable outputs for analysis and paper tables/figures.
    case_df.to_csv(output_dir / "benchmark_cases.csv", index=False)
    comparison_df.to_csv(output_dir / "method_comparison.csv", index=False)
    noise_df.to_csv(output_dir / "ontology_noise_results.csv", index=False)
    noise_summary.to_csv(output_dir / "ontology_noise_summary.csv", index=False)
    clutter_df.to_csv(output_dir / "distractor_robustness_results.csv", index=False)
    clutter_summary.to_csv(output_dir / "distractor_robustness_summary.csv", index=False)
    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Generate the publication-style figures after all tables are available.
    save_plots(comparison_df, noise_df, output_dir)

    # Print concise summaries to the terminal for quick inspection.
    print("Summary:")
    print(json.dumps(summary, indent=2))
    print("\nOntology noise summary:")
    print(noise_summary.to_string(index=False))
    print("\nDistractor robustness summary:")
    print(clutter_summary.to_string(index=False))
    print("\nSaved files:")
    print(output_dir / "benchmark_cases.csv")
    print(output_dir / "method_comparison.csv")
    print(output_dir / "ontology_noise_results.csv")
    print(output_dir / "ontology_noise_summary.csv")
    print(output_dir / "distractor_robustness_results.csv")
    print(output_dir / "distractor_robustness_summary.csv")
    print(output_dir / "summary.json")
    print(output_dir / "precision_recall_barplot.png")
    print(output_dir / "ontology_noise_robustness.png")

if __name__ == "__main__":
    main()
