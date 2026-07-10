from __future__ import annotations

import heapq
import json
import math
import os
import random
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pygame


ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
ASSET_DIR = ROOT / "assets" / "images"
FONT_CANDIDATES = (
    "/System/Library/Fonts/STHeiti Medium.ttc",
    "/System/Library/Fonts/STHeiti Light.ttc",
    "/System/Library/Fonts/PingFang.ttc",
    "/System/Library/Fonts/LanguageSupport/PingFang.ttc",
    "/Library/Fonts/Arial Unicode.ttf",
)


@dataclass
class Vec:
    x: float
    y: float

    def copy(self) -> "Vec":
        return Vec(self.x, self.y)

    def distance_to(self, other: "Vec") -> float:
        return math.hypot(self.x - other.x, self.y - other.y)

    def toward(self, target: "Vec", max_distance: float) -> bool:
        dist = self.distance_to(target)
        if dist <= max_distance or dist == 0:
            self.x = target.x
            self.y = target.y
            return True
        scale = max_distance / dist
        self.x += (target.x - self.x) * scale
        self.y += (target.y - self.y) * scale
        return False


@dataclass
class CircleBlocker:
    x: float
    y: float
    radius: float
    entity: Optional[object] = None


@dataclass
class PathResult:
    found: bool
    points: List[Vec] = field(default_factory=list)
    reason: str = ""


class GamePhase(str, Enum):
    PREP = "準備"
    ATTACK = "進攻"
    WIN = "勝利"
    LOSE = "失敗"


class AntState(str, Enum):
    TO_SWEET = "to_sweet"
    HARVESTING = "harvesting"
    LEAVING = "leaving"
    SIEGE = "siege"
    SIEGE_LEAVING = "siege_leaving"


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def load_ui_font(size: int) -> pygame.font.Font:
    for font_path in FONT_CANDIDATES:
        if Path(font_path).exists():
            return pygame.font.Font(font_path, size)
    return pygame.font.SysFont("Arial Unicode MS,Arial", size)


class DataStore:
    def __init__(self) -> None:
        self.config = load_json(DATA_DIR / "game_config.json")
        self.ants = load_json(DATA_DIR / "ants.json")
        self.sweets = load_json(DATA_DIR / "sweets.json")
        self.towers = load_json(DATA_DIR / "towers.json")
        self.blockers = load_json(DATA_DIR / "blockers.json")
        self.level_paths = sorted((DATA_DIR / "levels").glob("level_*.json"))
        self.levels = [load_json(path) for path in self.level_paths]
        self.level_numbers = [int(path.stem.split("_")[-1]) for path in self.level_paths]
        if not self.levels:
            raise RuntimeError("No level data found in data/levels")


class GridAStarPathfinder:
    def __init__(self, width: int, height: int, cell_size: int, margin: int) -> None:
        self.width = width
        self.height = height
        self.cell_size = cell_size
        self.margin = margin
        self.cols = math.ceil(width / cell_size)
        self.rows = math.ceil(height / cell_size)

    def has_path(self, start: Vec, goal: Vec, blockers: Iterable[CircleBlocker], unit_radius: float) -> bool:
        return self.find_path(start, goal, blockers, unit_radius).found

    def find_path(self, start: Vec, goal: Vec, blockers: Iterable[CircleBlocker], unit_radius: float) -> PathResult:
        blockers_list = list(blockers)
        start_cell = self._to_cell(start)
        goal_cell = self._to_cell(goal)
        if not self._in_bounds_cell(start_cell) or not self._in_bounds_cell(goal_cell):
            return PathResult(False, reason="start_or_goal_out_of_bounds")

        antenna_path = self._find_antenna_path(start, goal, blockers_list, unit_radius)
        if antenna_path.found:
            return antenna_path
        return self._find_astar_path(start, goal, blockers_list, unit_radius)

    def _find_antenna_path(self, start: Vec, goal: Vec, blockers: List[CircleBlocker], unit_radius: float) -> PathResult:
        # Ants steer like they are sensing with antennae: go straight until blocked,
        # pick a nearby side waypoint, then sense the next straight segment again.
        points = [start.copy()]
        current = start.copy()

        for _ in range(48):
            if current.distance_to(goal) <= 1:
                points[-1] = goal.copy()
                return PathResult(True, self._dedupe_points(points))

            hit = self._first_blocker_on_segment(current, goal, blockers, unit_radius)
            if hit is None:
                points.append(goal.copy())
                return PathResult(True, self._dedupe_points(points))

            waypoint = self._pick_detour_waypoint(current, goal, hit, blockers, unit_radius, points)
            if waypoint is None:
                return PathResult(False, reason="no_local_detour")

            points.append(waypoint)
            current = waypoint

        return PathResult(False, reason="too_many_local_detours")

    def _find_astar_path(self, start: Vec, goal: Vec, blockers: Iterable[CircleBlocker], unit_radius: float) -> PathResult:
        blocked = self._blocked_cells(blockers, unit_radius)
        start_cell = self._to_cell(start)
        goal_cell = self._to_cell(goal)
        for cell in (start_cell, goal_cell):
            if cell in blocked:
                blocked.remove(cell)
        if not self._in_bounds_cell(start_cell) or not self._in_bounds_cell(goal_cell):
            return PathResult(False, reason="start_or_goal_out_of_bounds")

        frontier: List[Tuple[float, Tuple[int, int]]] = []
        heapq.heappush(frontier, (0, start_cell))
        came_from: Dict[Tuple[int, int], Optional[Tuple[int, int]]] = {start_cell: None}
        cost_so_far: Dict[Tuple[int, int], float] = {start_cell: 0}

        while frontier:
            _, current = heapq.heappop(frontier)
            if current == goal_cell:
                cells = self._reconstruct(came_from, current)
                points = [self._to_world(cell) for cell in cells]
                if points:
                    points[0] = start.copy()
                    points[-1] = goal.copy()
                return PathResult(True, self._smooth(points, blocked))

            for neighbor, step_cost in self._neighbors(current):
                if neighbor in blocked:
                    continue
                new_cost = cost_so_far[current] + step_cost
                if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                    cost_so_far[neighbor] = new_cost
                    priority = new_cost + self._heuristic(neighbor, goal_cell)
                    heapq.heappush(frontier, (priority, neighbor))
                    came_from[neighbor] = current

        return PathResult(False, reason="no_path")

    def _first_blocker_on_segment(
        self,
        start: Vec,
        goal: Vec,
        blockers: Iterable[CircleBlocker],
        unit_radius: float,
    ) -> Optional[CircleBlocker]:
        hits = []
        for blocker in blockers:
            hit_at = self._segment_blocker_hit_t(start, goal, blocker, unit_radius)
            if hit_at is not None:
                hits.append((hit_at, blocker))
        if not hits:
            return None
        return min(hits, key=lambda item: item[0])[1]

    def _segment_blocker_hit_t(self, start: Vec, goal: Vec, blocker: CircleBlocker, unit_radius: float) -> Optional[float]:
        dx = goal.x - start.x
        dy = goal.y - start.y
        length_sq = dx * dx + dy * dy
        if length_sq == 0:
            return None

        center = Vec(blocker.x, blocker.y)
        t = ((center.x - start.x) * dx + (center.y - start.y) * dy) / length_sq
        if t <= 0.02 or t >= 0.98:
            return None

        closest = Vec(start.x + dx * t, start.y + dy * t)
        radius = blocker.radius + unit_radius + self.cell_size * 0.25
        if closest.distance_to(center) <= radius:
            return t
        return None

    def _pick_detour_waypoint(
        self,
        start: Vec,
        goal: Vec,
        blocker: CircleBlocker,
        blockers: List[CircleBlocker],
        unit_radius: float,
        existing_points: List[Vec],
    ) -> Optional[Vec]:
        center = Vec(blocker.x, blocker.y)
        dx = goal.x - start.x
        dy = goal.y - start.y
        distance = math.hypot(dx, dy)
        if distance == 0:
            return None

        perp_x = -dy / distance
        perp_y = dx / distance
        forward_x = dx / distance
        forward_y = dy / distance
        clearance = blocker.radius + unit_radius + self.cell_size * 1.35
        candidates: List[Vec] = []

        for side in (-1, 1):
            for forward_bias in (-0.35, 0, 0.45, 0.9):
                candidates.append(Vec(
                    center.x + perp_x * side * clearance + forward_x * clearance * forward_bias,
                    center.y + perp_y * side * clearance + forward_y * clearance * forward_bias,
                ))

        angle_from_center = math.atan2(start.y - center.y, start.x - center.x)
        for side in (-1, 1):
            for turn in (math.pi / 3, math.pi / 2, math.pi * 2 / 3):
                angle = angle_from_center + side * turn
                candidates.append(Vec(
                    center.x + math.cos(angle) * clearance,
                    center.y + math.sin(angle) * clearance,
                ))

        valid = [
            candidate for candidate in candidates
            if self._valid_detour_segment(start, candidate, blockers, unit_radius)
            and all(candidate.distance_to(point) > self.cell_size * 0.75 for point in existing_points)
        ]
        if not valid:
            return None

        return min(valid, key=lambda candidate: (
            candidate.distance_to(goal),
            start.distance_to(candidate),
            abs(self._turn_amount(start, goal, candidate)),
        ))

    def _valid_detour_segment(
        self,
        start: Vec,
        candidate: Vec,
        blockers: Iterable[CircleBlocker],
        unit_radius: float,
    ) -> bool:
        if not self._in_bounds_cell(self._to_cell(candidate)):
            return False
        for blocker in blockers:
            center = Vec(blocker.x, blocker.y)
            if candidate.distance_to(center) <= blocker.radius + unit_radius + self.cell_size * 0.35:
                return False
            if self._segment_blocker_hit_t(start, candidate, blocker, unit_radius) is not None:
                return False
        return True

    def _turn_amount(self, start: Vec, goal: Vec, candidate: Vec) -> float:
        goal_angle = math.atan2(goal.y - start.y, goal.x - start.x)
        candidate_angle = math.atan2(candidate.y - start.y, candidate.x - start.x)
        return math.atan2(math.sin(candidate_angle - goal_angle), math.cos(candidate_angle - goal_angle))

    def _dedupe_points(self, points: List[Vec]) -> List[Vec]:
        deduped: List[Vec] = []
        for point in points:
            if not deduped or point.distance_to(deduped[-1]) > 1:
                deduped.append(point)
        return deduped

    def _blocked_cells(self, blockers: Iterable[CircleBlocker], unit_radius: float) -> set[Tuple[int, int]]:
        blocked: set[Tuple[int, int]] = set()
        for blocker in blockers:
            radius = blocker.radius + unit_radius
            min_col = max(0, int((blocker.x - radius) // self.cell_size))
            max_col = min(self.cols - 1, int((blocker.x + radius) // self.cell_size))
            min_row = max(0, int((blocker.y - radius) // self.cell_size))
            max_row = min(self.rows - 1, int((blocker.y + radius) // self.cell_size))
            center = Vec(blocker.x, blocker.y)
            for col in range(min_col, max_col + 1):
                for row in range(min_row, max_row + 1):
                    if self._to_world((col, row)).distance_to(center) <= radius + self.cell_size * 0.72:
                        blocked.add((col, row))
        return blocked

    def _to_cell(self, point: Vec) -> Tuple[int, int]:
        return int(point.x // self.cell_size), int(point.y // self.cell_size)

    def _to_world(self, cell: Tuple[int, int]) -> Vec:
        return Vec((cell[0] + 0.5) * self.cell_size, (cell[1] + 0.5) * self.cell_size)

    def _in_bounds_cell(self, cell: Tuple[int, int]) -> bool:
        return 0 <= cell[0] < self.cols and 0 <= cell[1] < self.rows

    def _neighbors(self, cell: Tuple[int, int]) -> Iterable[Tuple[Tuple[int, int], float]]:
        for dx, dy, cost in (
            (-1, 0, 1), (1, 0, 1), (0, -1, 1), (0, 1, 1),
            (-1, -1, 1.414), (1, -1, 1.414), (-1, 1, 1.414), (1, 1, 1.414),
        ):
            next_cell = (cell[0] + dx, cell[1] + dy)
            if self._in_bounds_cell(next_cell):
                yield next_cell, cost

    def _heuristic(self, a: Tuple[int, int], b: Tuple[int, int]) -> float:
        return math.hypot(a[0] - b[0], a[1] - b[1])

    def _reconstruct(self, came_from: Dict[Tuple[int, int], Optional[Tuple[int, int]]], current: Tuple[int, int]) -> List[Tuple[int, int]]:
        cells = [current]
        while came_from[current] is not None:
            current = came_from[current]  # type: ignore[assignment]
            cells.append(current)
        cells.reverse()
        return cells

    def _smooth(self, points: List[Vec], blocked: set[Tuple[int, int]]) -> List[Vec]:
        if len(points) <= 2:
            return points
        smoothed = [points[0]]
        anchor = 0
        probe = 2
        while probe < len(points):
            if self._line_hits_blocked(points[anchor], points[probe], blocked):
                smoothed.append(points[probe - 1])
                anchor = probe - 1
            probe += 1
        smoothed.append(points[-1])
        return smoothed

    def _line_hits_blocked(self, a: Vec, b: Vec, blocked: set[Tuple[int, int]]) -> bool:
        dist = max(1, int(a.distance_to(b) / (self.cell_size * 0.5)))
        for i in range(dist + 1):
            t = i / dist
            cell = self._to_cell(Vec(a.x + (b.x - a.x) * t, a.y + (b.y - a.y) * t))
            if cell in blocked:
                return True
        return False


@dataclass
class Sweet:
    id: str
    kind: str
    pos: Vec
    label: str
    color: Tuple[int, int, int]
    radius: int
    max_radius: int
    mode: str
    units: int
    max_units: int
    chew_time: float
    value: int

    @property
    def depleted(self) -> bool:
        return self.units <= 0


@dataclass
class Building:
    id: str
    kind: str
    pos: Vec
    label: str
    color: Tuple[int, int, int]
    radius: int
    hp: float
    max_hp: float
    cost: int
    blocks_movement: bool
    destructible: bool = True

    @property
    def alive(self) -> bool:
        return not self.destructible or self.hp > 0


@dataclass
class Tower(Building):
    attack_range: float = 0
    damage: float = 0
    shots_per_second: float = 1
    cooldown: float = 0
    target_priority: str = "nearest"


@dataclass
class Projectile:
    pos: Vec
    target: "Ant"
    speed: float
    damage: float
    alive: bool = True

    def update(self, dt: float) -> None:
        if not self.target.alive:
            self.alive = False
            return
        if self.pos.toward(self.target.pos, self.speed * dt):
            self.target.take_damage(self.damage)
            self.alive = False


@dataclass
class Ant:
    id: str
    kind: str
    pos: Vec
    label: str
    color: Tuple[int, int, int]
    radius: int
    hp: float
    max_hp: float
    speed: float
    oil_reward: int
    score_reward: int
    role: str
    can_damage_buildings: bool
    building_damage_per_second: float
    state: AntState
    entry: Vec
    target_sweet: Optional[Sweet] = None
    path: List[Vec] = field(default_factory=list)
    path_index: int = 0
    harvest_timer: float = 0
    carrying_value: int = 0
    alive: bool = True
    finished: bool = False
    attack_target: Optional[Building] = None
    path_bias_side: int = 1
    path_bias_strength: float = 0
    path_wobble_phase: float = 0
    penalty_soldier: bool = False
    destroyed_buildings: int = 0
    destroy_goal: int = 0
    facing_angle: float = -math.pi / 2

    def take_damage(self, damage: float) -> None:
        self.hp -= damage
        if self.hp <= 0:
            self.alive = False

    def set_path(self, points: List[Vec]) -> None:
        self.path = self._with_individual_path_bias(points)
        self.path_index = 1 if len(points) > 1 else 0

    def follow_path(self, dt: float) -> bool:
        if self.path_index >= len(self.path):
            return True
        target = self.path[self.path_index]
        if self.move_toward(target, self.speed * dt):
            self.path_index += 1
        return self.path_index >= len(self.path)

    def move_toward(self, target: Vec, max_distance: float) -> bool:
        dx = target.x - self.pos.x
        dy = target.y - self.pos.y
        if dx or dy:
            self.facing_angle = math.atan2(dy, dx)
        return self.pos.toward(target, max_distance)

    def _with_individual_path_bias(self, points: List[Vec]) -> List[Vec]:
        if len(points) <= 1 or self.path_bias_strength <= 0:
            return [point.copy() for point in points]

        biased = [points[0].copy()]
        is_soldier = self.role == "siege"
        min_segment = 32 if is_soldier else 70
        for index in range(1, len(points)):
            start = points[index - 1]
            end = points[index]
            dx = end.x - start.x
            dy = end.y - start.y
            distance = math.hypot(dx, dy)

            if distance > min_segment:
                normal_x = -dy / distance
                normal_y = dx / distance
                wave = math.sin(self.path_wobble_phase + index * (2.37 if is_soldier else 1.73))
                amount = self.path_bias_side * self.path_bias_strength * (0.65 + 0.35 * wave)
                t = 0.45 + 0.08 * math.sin(self.path_wobble_phase + index * 2.11)
                biased.append(Vec(
                    start.x + dx * t + normal_x * amount,
                    start.y + dy * t + normal_y * amount,
                ))
                if is_soldier and distance > min_segment * 1.8:
                    counter_wave = math.sin(self.path_wobble_phase + index * 3.91)
                    counter_amount = -self.path_bias_side * self.path_bias_strength * (0.45 + 0.55 * counter_wave)
                    t2 = 0.72 + 0.1 * math.sin(self.path_wobble_phase + index * 1.29)
                    biased.append(Vec(
                        start.x + dx * t2 + normal_x * counter_amount,
                        start.y + dy * t2 + normal_y * counter_amount,
                    ))

            biased.append(end.copy())

        return biased


class AssetStore:
    def __init__(self) -> None:
        self.images: Dict[str, pygame.Surface] = {}
        self.scaled_cache: Dict[Tuple[str, int, int], pygame.Surface] = {}
        self.load_images()

    def load_images(self) -> None:
        for asset_id, filename in {
            "worker_ant_1": "worker_ant_walk_1.png",
            "worker_ant_2": "worker_ant_walk_2.png",
            "soldier_ant_1": "soldier_ant_walk_1.png",
            "soldier_ant_2": "soldier_ant_walk_2.png",
            "basic_turret": "basic_turret.png",
            "block_wall": "stone_wall.png",
            "crumb_rock": "rock_obstacle.png",
            "sugar_pile": "sugar_pile.png",
            "hard_candy": "hard_candy.png",
            "grass_background": "grass_background.png",
        }.items():
            path = ASSET_DIR / filename
            if path.exists():
                self.images[asset_id] = pygame.image.load(str(path)).convert_alpha()

    def get(self, asset_id: str) -> Optional[pygame.Surface]:
        return self.images.get(asset_id)

    def scaled(self, asset_id: str, width: int, height: int) -> Optional[pygame.Surface]:
        image = self.get(asset_id)
        if image is None:
            return None
        key = (asset_id, width, height)
        if key not in self.scaled_cache:
            self.scaled_cache[key] = pygame.transform.smoothscale(image, (width, height))
        return self.scaled_cache[key]


class Game:
    def __init__(self) -> None:
        pygame.init()
        pygame.display.set_caption("迷你甜食防禦軍")
        self.data = DataStore()
        self.level_index = 0
        self.level = self.data.levels[self.level_index]
        cfg = self.data.config
        self.world_w = cfg["world"]["width"]
        self.world_h = cfg["world"]["height"]
        self.ui_h = cfg["ui"]["topBarHeight"]
        self.screen = pygame.display.set_mode((cfg["window"]["width"], cfg["window"]["height"]))
        self.clock = pygame.time.Clock()
        self.font = load_ui_font(18)
        self.small_font = load_ui_font(14)
        self.assets = AssetStore()
        self.pathfinder = GridAStarPathfinder(
            self.world_w,
            self.world_h,
            cfg["pathfinding"]["cellSize"],
            cfg["world"]["edgeBuildMargin"],
        )
        self.phase = GamePhase.PREP
        self.resources = {"oil": self.level["initialResources"]["oil"]}
        self.score = 0
        self.wave_index = 0
        self.wave_time = 0.0
        self.spawn_timers: Dict[int, float] = {}
        self.spawn_counts: Dict[int, int] = {}
        self.penalty_cooldown = 0.0
        self.selected_buildable = "basic_turret"
        self.build_button_rects: Dict[str, pygame.Rect] = {}
        self.status_message = "按 1 選砲塔、2 選牆；左鍵建造，Space 開始進攻。"
        self.sweets: List[Sweet] = []
        self.towers: List[Tower] = []
        self.blockers: List[Building] = []
        self.ants: List[Ant] = []
        self.projectiles: List[Projectile] = []
        self.load_level_entities()

    def load_level_entities(self) -> None:
        self.sweets.clear()
        self.towers.clear()
        self.blockers.clear()
        self.ants.clear()
        self.projectiles.clear()
        for item in self.level["sweets"]:
            definition = self.data.sweets[item["type"]]
            self.sweets.append(Sweet(
                id=item["id"],
                kind=item["type"],
                pos=Vec(item["x"], item["y"]),
                label=definition["label"],
                color=tuple(definition["color"]),
                radius=definition["radius"],
                max_radius=definition["radius"],
                mode=definition["carryMode"],
                units=item.get("units", definition["units"]),
                max_units=item.get("units", definition["units"]),
                chew_time=definition.get("chewTime", 0),
                value=definition["value"],
            ))
        self.load_level_blockers()

    def load_level_blockers(self) -> None:
        for item in self.level.get("blockers", []):
            self.blockers.append(self.make_blocker_from_level(item))
        random_blockers = self.level.get("randomBlockers")
        if random_blockers:
            self.generate_random_blockers(random_blockers)

    def make_blocker_from_level(self, item: dict) -> Building:
        definition = self.data.blockers[item["type"]]
        return Building(
            id=item["id"],
            kind=item["type"],
            pos=Vec(item["x"], item["y"]),
            label=definition["label"],
            color=tuple(definition["color"]),
            radius=definition["radius"],
            hp=definition["hp"],
            max_hp=definition["hp"],
            cost=definition.get("cost", {}).get("oil", 0),
            blocks_movement=definition["blocksMovement"],
            destructible=definition.get("destructible", True),
        )

    def generate_random_blockers(self, settings: dict) -> None:
        rng = random.Random(settings.get("seed", self.level_index + 1))
        blocker_type = settings["type"]
        definition = self.data.blockers[blocker_type]
        count = settings["count"]
        area = settings["area"]
        attempts = count * 60
        placed = 0
        while placed < count and attempts > 0:
            attempts -= 1
            pos = Vec(rng.randint(area["xMin"], area["xMax"]), rng.randint(area["yMin"], area["yMax"]))
            if self.overlaps_existing(pos, definition["radius"]):
                continue
            self.blockers.append(Building(
                id=f"random_blocker_{placed}",
                kind=blocker_type,
                pos=pos,
                label=definition["label"],
                color=tuple(definition["color"]),
                radius=definition["radius"],
                hp=definition["hp"],
                max_hp=definition["hp"],
                cost=definition.get("cost", {}).get("oil", 0),
                blocks_movement=definition["blocksMovement"],
                destructible=definition.get("destructible", True),
            ))
            placed += 1

    def reset_level(self, level_index: Optional[int] = None) -> None:
        if level_index is not None:
            self.level_index = max(0, min(level_index, len(self.data.levels) - 1))
        self.level = self.data.levels[self.level_index]
        self.phase = GamePhase.PREP
        self.resources = {"oil": self.level["initialResources"]["oil"]}
        self.score = 0
        self.wave_index = 0
        self.wave_time = 0.0
        self.spawn_timers.clear()
        self.spawn_counts.clear()
        self.penalty_cooldown = 0.0
        self.status_message = f"已載入：{self.level['name']}。Space 開始進攻。"
        self.load_level_entities()

    def reset_level_number(self, level_number: int) -> None:
        if level_number in self.data.level_numbers:
            self.reset_level(self.data.level_numbers.index(level_number))
        else:
            self.status_message = f"尚未建立第 {level_number} 關。"

    def world_to_screen(self, pos: Vec) -> Tuple[int, int]:
        return int(pos.x), int(pos.y + self.ui_h)

    def screen_to_world(self, pos: Tuple[int, int]) -> Vec:
        return Vec(pos[0], pos[1] - self.ui_h)

    def run(self) -> None:
        running = True
        while running:
            dt = self.clock.tick(self.data.config["gameplay"]["fps"]) / 1000.0
            dt *= self.data.config["gameplay"].get("timeScale", 1)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                self.handle_event(event)
            self.update(dt)
            self.draw()
        pygame.quit()

    def handle_event(self, event: pygame.event.Event) -> None:
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_1:
                self.selected_buildable = "basic_turret"
                self.status_message = "已選：基礎砲塔"
            elif event.key == pygame.K_2:
                self.selected_buildable = "block_wall"
                self.status_message = "已選：阻擋牆"
            elif event.key == pygame.K_SPACE and self.phase == GamePhase.PREP:
                self.start_wave()
            elif event.key == pygame.K_r and self.phase in (GamePhase.WIN, GamePhase.LOSE):
                self.reset_level()
            elif event.key in (pygame.K_F1, pygame.K_F2, pygame.K_F3):
                self.reset_level({pygame.K_F1: 0, pygame.K_F2: 1, pygame.K_F3: 2}[event.key])
            elif event.key == pygame.K_F9:
                self.reset_level_number(9)
            elif event.key == pygame.K_s and self.phase == GamePhase.ATTACK:
                self.spawn_manual_soldier()
            elif event.key == pygame.K_ESCAPE:
                pygame.event.post(pygame.event.Event(pygame.QUIT))
        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if event.pos[1] < self.ui_h:
                self.handle_build_button_click(event.pos)
                return
            if self.phase in (GamePhase.PREP, GamePhase.ATTACK):
                self.try_place(self.screen_to_world(event.pos))

    def handle_build_button_click(self, pos: Tuple[int, int]) -> bool:
        self.layout_build_button_rects()
        for buildable_id, rect in self.build_button_rects.items():
            if rect.collidepoint(pos):
                self.selected_buildable = buildable_id
                self.status_message = f"已選：{self.buildable_name(buildable_id)}"
                return True
        return False

    def start_wave(self) -> None:
        if self.wave_index >= len(self.level["waves"]):
            self.phase = GamePhase.WIN
            return
        self.phase = GamePhase.ATTACK
        self.wave_time = 0.0
        self.spawn_timers.clear()
        self.spawn_counts.clear()
        self.status_message = f"{self.level['waves'][self.wave_index]['name']} 開始"

    def try_place(self, pos: Vec) -> None:
        if not self.in_buildable_area(pos):
            self.status_message = "邊緣禁放區或場外不能建造。"
            return
        definition, is_tower = self.build_definition(self.selected_buildable)
        cost = definition["cost"]["oil"]
        radius = definition["radius"]
        if self.resources["oil"] < cost:
            self.status_message = "油量不足。"
            return
        if self.overlaps_existing(pos, radius):
            self.status_message = "建造位置與其他物件重疊。"
            return
        building: Building
        if is_tower:
            attack = definition["attack"]
            building = Tower(
                id=f"tower_{len(self.towers)}",
                kind=self.selected_buildable,
                pos=pos,
                label=definition["label"],
                color=tuple(definition["color"]),
                radius=radius,
                hp=definition["hp"],
                max_hp=definition["hp"],
                cost=cost,
                blocks_movement=definition["blocksMovement"],
                destructible=definition.get("destructible", True),
                attack_range=attack["range"],
                damage=attack["damage"],
                shots_per_second=attack["shotsPerSecond"],
                target_priority=attack.get("targetPriority", "nearest"),
            )
            self.towers.append(building)
        else:
            building = Building(
                id=f"blocker_{len(self.blockers)}",
                kind=self.selected_buildable,
                pos=pos,
                label=definition["label"],
                color=tuple(definition["color"]),
                radius=radius,
                hp=definition["hp"],
                max_hp=definition["hp"],
                cost=cost,
                blocks_movement=definition["blocksMovement"],
                destructible=definition.get("destructible", True),
            )
            self.blockers.append(building)
        self.resources["oil"] -= cost
        self.status_message = f"建造完成，花費 {cost} 油量。"
        if self.phase == GamePhase.ATTACK and building.blocks_movement:
            rerouted, blocked = self.reroute_ants_after_new_blocker()
            if blocked:
                self.status_message = f"建造完成，{rerouted} 隻螞蟻重新尋路；{blocked} 隻被封路。"
            elif rerouted:
                self.status_message = f"建造完成，{rerouted} 隻螞蟻重新尋路。"

    def build_definition(self, buildable_id: str) -> Tuple[dict, bool]:
        if buildable_id in self.data.towers:
            return self.data.towers[buildable_id], True
        return self.data.blockers[buildable_id], False

    def buildable_ids(self) -> List[str]:
        ids = [*self.data.towers.keys(), *self.data.blockers.keys()]
        return [buildable_id for buildable_id in ids if self.buildable_cost(buildable_id) > 0]

    def buildable_name(self, buildable_id: str) -> str:
        definition, _ = self.build_definition(buildable_id)
        return definition.get("name", definition["label"])

    def buildable_cost(self, buildable_id: str) -> int:
        definition, _ = self.build_definition(buildable_id)
        return definition.get("cost", {}).get("oil", 0)

    def affordable_count(self, buildable_id: str) -> int:
        cost = self.buildable_cost(buildable_id)
        if cost <= 0:
            return 0
        return self.resources["oil"] // cost

    def in_buildable_area(self, pos: Vec) -> bool:
        margin = self.data.config["world"]["edgeBuildMargin"]
        return margin <= pos.x <= self.world_w - margin and margin <= pos.y <= self.world_h - margin

    def overlaps_existing(self, pos: Vec, radius: float) -> bool:
        padding = self.data.config["world"].get("placementPadding", 0)
        for entity in [*self.towers, *self.blockers, *self.sweets, *self.ants]:
            if isinstance(entity, Ant) and (not entity.alive or entity.finished):
                continue
            if pos.distance_to(entity.pos) < radius + entity.radius + padding:
                return True
        return False

    def reroute_ants_after_new_blocker(self) -> Tuple[int, int]:
        rerouted = 0
        blocked = 0
        for ant in self.ants:
            if not ant.alive or ant.finished:
                continue
            if ant.state == AntState.TO_SWEET and ant.target_sweet is not None and not ant.target_sweet.depleted:
                if self.reroute_ant(ant, ant.target_sweet.pos):
                    rerouted += 1
                else:
                    self.trigger_soldier_penalty(ant.entry, ant.target_sweet)
                    ant.finished = True
                    blocked += 1
            elif ant.state == AntState.LEAVING:
                if self.reroute_ant(ant, self.nearest_exit(ant.pos)):
                    rerouted += 1
                else:
                    ant.set_path([ant.pos.copy(), self.nearest_exit(ant.pos)])
            elif ant.state == AntState.SIEGE_LEAVING:
                if self.reroute_ant(ant, self.nearest_exit(ant.pos)):
                    rerouted += 1
        return rerouted, blocked

    def reroute_ant(self, ant: Ant, target: Vec) -> bool:
        path = self.pathfinder.find_path(ant.pos, target, self.current_blockers(), ant.radius)
        if not path.found:
            return False
        ant.set_path(path.points)
        return True

    def update(self, dt: float) -> None:
        if self.phase == GamePhase.ATTACK:
            self.wave_time += dt
            self.penalty_cooldown = max(0, self.penalty_cooldown - dt)
            self.update_spawning(dt)
            self.update_ants(dt)
            self.update_towers(dt)
            self.update_projectiles(dt)
            self.cleanup_entities()
            self.check_wave_end()
            self.check_loss()

    def update_spawning(self, dt: float) -> None:
        wave = self.level["waves"][self.wave_index]
        for idx, spawn in enumerate(wave["spawns"]):
            self.spawn_timers[idx] = self.spawn_timers.get(idx, 0) - dt
            self.spawn_counts[idx] = self.spawn_counts.get(idx, 0)
            if self.spawn_counts[idx] >= spawn["count"]:
                continue
            if self.spawn_timers[idx] <= 0:
                self.spawn_ant(spawn)
                self.spawn_counts[idx] += 1
                self.spawn_timers[idx] = spawn["interval"]

    def spawn_ant(self, spawn: dict) -> None:
        ant_type = spawn["antType"]
        entry_id = spawn["entryId"]
        entry = self.entry_by_id(entry_id)
        sweet = self.pick_target_sweet(entry)
        if not sweet:
            return
        definition = self.data.ants[ant_type]
        state = AntState.SIEGE if definition["role"] == "siege" else AntState.TO_SWEET
        ant = self.make_ant(
            ant_type,
            definition,
            entry.copy(),
            state,
            entry.copy(),
            sweet,
            destroy_goal=spawn.get("destroyCount", self.soldier_destroy_goal(ant_type)),
        )
        path = self.pathfinder.find_path(ant.pos, sweet.pos, self.current_blockers(), ant.radius)
        if not path.found:
            if ant.role == "siege":
                self.ants.append(ant)
            else:
                self.trigger_soldier_penalty(entry.copy(), sweet)
            return
        ant.set_path(path.points)
        self.ants.append(ant)

    def spawn_manual_soldier(self) -> None:
        if self.data.level_numbers[self.level_index] != 9:
            return
        entries = self.level["entries"]
        if not entries:
            return
        entry = entries[len([ant for ant in self.ants if ant.role == "siege"]) % len(entries)]
        self.spawn_ant({"antType": "soldier_ant", "entryId": entry["id"], "count": 1, "interval": 0})
        self.status_message = "手動召喚 1 隻兵蟻。"

    def trigger_soldier_penalty(self, entry: Vec, sweet: Sweet) -> None:
        settings = self.data.config["blockedPathPenalty"]
        active = sum(1 for ant in self.ants if ant.role == "siege" and ant.alive)
        if self.penalty_cooldown > 0 or active >= settings["maxActivePenaltyAnts"]:
            self.status_message = "工蟻路徑被封死，兵蟻懲罰冷卻中。"
            return
        definition = self.data.ants[settings["triggerAntType"]]
        soldier = self.make_ant(
            settings["triggerAntType"],
            definition,
            entry.copy(),
            AntState.SIEGE,
            entry.copy(),
            sweet,
            penalty_soldier=True,
            destroy_goal=0,
        )
        self.ants.append(soldier)
        self.penalty_cooldown = settings["cooldownSeconds"]
        self.status_message = "路徑被封死：巨大兵蟻出現，開始拆除阻礙。"

    def make_ant(
        self,
        ant_type: str,
        definition: dict,
        pos: Vec,
        state: AntState,
        entry: Vec,
        sweet: Optional[Sweet],
        penalty_soldier: bool = False,
        destroy_goal: int = 0,
    ) -> Ant:
        return Ant(
            id=f"ant_{len(self.ants)}_{random.randint(1000, 9999)}",
            kind=ant_type,
            pos=pos,
            label=definition["label"],
            color=tuple(definition["color"]),
            radius=definition["radius"],
            hp=definition["hp"],
            max_hp=definition["hp"],
            speed=definition["speed"],
            oil_reward=definition["oilReward"],
            score_reward=definition["scoreReward"],
            role=definition["role"],
            can_damage_buildings=definition["canDamageBuildings"],
            building_damage_per_second=definition["buildingDamagePerSecond"],
            state=state,
            entry=entry,
            target_sweet=sweet,
            path_bias_side=random.choice((-1, 1)),
            path_bias_strength=random.uniform(14.0, 30.0) if definition["role"] == "siege" else random.uniform(3.0, 8.0),
            path_wobble_phase=random.uniform(0, math.tau),
            penalty_soldier=penalty_soldier,
            destroy_goal=destroy_goal,
        )

    def soldier_destroy_goal(self, ant_type: str) -> int:
        definition = self.data.ants[ant_type]
        if definition["role"] != "siege":
            return 0
        goals = self.level.get("soldierDestructionGoals", {})
        return int(goals.get(ant_type, self.level_index + 1))

    def update_ants(self, dt: float) -> None:
        for ant in self.ants:
            if not ant.alive or ant.finished:
                continue
            if ant.state == AntState.TO_SWEET:
                self.update_worker_to_sweet(ant, dt)
            elif ant.state == AntState.HARVESTING:
                self.update_harvesting(ant, dt)
            elif ant.state == AntState.LEAVING:
                if ant.follow_path(dt):
                    ant.finished = True
            elif ant.state == AntState.SIEGE:
                self.update_soldier(ant, dt)
            elif ant.state == AntState.SIEGE_LEAVING:
                if ant.follow_path(dt):
                    ant.finished = True
        self.resolve_ant_collisions()

    def ant_collision_radius(self, ant: Ant) -> float:
        scale = 0.75 if ant.role == "siege" else 0.65
        return max(5.0, ant.radius * scale)

    def resolve_ant_collisions(self) -> None:
        active = [ant for ant in self.ants if ant.alive and not ant.finished]
        if len(active) < 2:
            return
        for _ in range(2):
            for i, first in enumerate(active):
                first_radius = self.ant_collision_radius(first)
                for j in range(i + 1, len(active)):
                    second = active[j]
                    dx = second.pos.x - first.pos.x
                    dy = second.pos.y - first.pos.y
                    distance = math.hypot(dx, dy)
                    min_distance = first_radius + self.ant_collision_radius(second)
                    if distance >= min_distance:
                        continue
                    if distance == 0:
                        angle = (i * 1.73 + j * 2.41) % math.tau
                        nx = math.cos(angle)
                        ny = math.sin(angle)
                    else:
                        nx = dx / distance
                        ny = dy / distance
                    push = (min_distance - distance) * 0.5
                    first.pos.x -= nx * push
                    first.pos.y -= ny * push
                    second.pos.x += nx * push
                    second.pos.y += ny * push
                    self.keep_ant_in_world(first)
                    self.keep_ant_in_world(second)

    def keep_ant_in_world(self, ant: Ant) -> None:
        ant.pos.x = max(0, min(self.world_w, ant.pos.x))
        ant.pos.y = max(0, min(self.world_h, ant.pos.y))

    def update_worker_to_sweet(self, ant: Ant, dt: float) -> None:
        sweet = ant.target_sweet
        if sweet is None or sweet.depleted:
            replacement = self.pick_target_sweet(ant.pos)
            ant.target_sweet = replacement
            if replacement is None:
                ant.finished = True
                return
            path = self.pathfinder.find_path(ant.pos, replacement.pos, self.current_blockers(), ant.radius)
            if not path.found:
                self.trigger_soldier_penalty(ant.entry, replacement)
                ant.finished = True
                return
            ant.set_path(path.points)
        if ant.follow_path(dt):
            if sweet.mode == "instant_pickup":
                self.take_sweet_unit(ant, sweet)
                self.route_ant_to_exit(ant)
            else:
                ant.state = AntState.HARVESTING
                ant.harvest_timer = sweet.chew_time

    def update_harvesting(self, ant: Ant, dt: float) -> None:
        sweet = ant.target_sweet
        if sweet is None or sweet.depleted:
            self.route_ant_to_exit(ant)
            return
        ant.harvest_timer -= dt
        if ant.harvest_timer <= 0:
            self.take_sweet_unit(ant, sweet)
            self.route_ant_to_exit(ant)

    def take_sweet_unit(self, ant: Ant, sweet: Sweet) -> None:
        if sweet.units > 0:
            sweet.units -= 1
            ant.carrying_value = sweet.value

    def route_ant_to_exit(self, ant: Ant) -> None:
        ant.state = AntState.LEAVING
        exit_pos = self.nearest_exit(ant.pos)
        path = self.pathfinder.find_path(ant.pos, exit_pos, self.current_blockers(), ant.radius)
        ant.set_path(path.points if path.found else [ant.pos.copy(), exit_pos])

    def update_soldier(self, ant: Ant, dt: float) -> None:
        sweet = ant.target_sweet
        if sweet is None:
            ant.finished = True
            return
        if ant.penalty_soldier and self.pathfinder.has_path(ant.entry, sweet.pos, self.current_blockers(), ant.radius):
            ant.state = AntState.SIEGE_LEAVING
            exit_pos = self.nearest_exit(ant.pos)
            path = self.pathfinder.find_path(ant.pos, exit_pos, self.current_blockers(), ant.radius)
            ant.set_path(path.points if path.found else [ant.pos.copy(), exit_pos])
            self.status_message = "巨大兵蟻已打通路徑，正在離開。"
            return
        target = ant.attack_target if ant.attack_target and ant.attack_target.alive else self.find_siege_target(ant, sweet)
        ant.attack_target = target
        if not target:
            self.route_ant_to_exit(ant)
            return
        if ant.pos.distance_to(target.pos) <= ant.radius + target.radius + 6:
            target.hp -= ant.building_damage_per_second * dt
            if target.hp <= 0:
                ant.destroyed_buildings += 1
                ant.attack_target = None
                if not ant.penalty_soldier and ant.destroy_goal > 0 and ant.destroyed_buildings >= ant.destroy_goal:
                    self.status_message = f"兵蟻已破壞 {ant.destroyed_buildings} 個目標，正在離開。"
                    self.route_ant_to_exit(ant)
        else:
            ant.move_toward(self.soldier_approach_point(ant, target), ant.speed * dt)

    def soldier_approach_point(self, ant: Ant, target: Building) -> Vec:
        dx = target.pos.x - ant.pos.x
        dy = target.pos.y - ant.pos.y
        distance = math.hypot(dx, dy)
        if distance == 0:
            return target.pos.copy()
        normal_x = -dy / distance
        normal_y = dx / distance
        pulse = math.sin(self.wave_time * 4.6 + ant.path_wobble_phase)
        chop = math.sin(self.wave_time * 9.5 + ant.path_wobble_phase * 0.7)
        offset = ant.path_bias_side * (10 + ant.path_bias_strength * 0.55 * pulse + 7 * chop)
        offset *= min(1.0, max(0.15, distance / 140))
        return Vec(target.pos.x + normal_x * offset, target.pos.y + normal_y * offset)

    def find_siege_target(self, ant: Ant, sweet: Sweet) -> Optional[Building]:
        candidates = [b for b in [*self.towers, *self.blockers] if b.alive and b.destructible]
        if not candidates:
            return None
        return min(candidates, key=lambda b: self.distance_point_to_segment(b.pos, ant.entry, sweet.pos) + b.pos.distance_to(ant.pos) * 0.25)

    def distance_point_to_segment(self, p: Vec, a: Vec, b: Vec) -> float:
        length_sq = max(1.0, (b.x - a.x) ** 2 + (b.y - a.y) ** 2)
        t = max(0, min(1, ((p.x - a.x) * (b.x - a.x) + (p.y - a.y) * (b.y - a.y)) / length_sq))
        projection = Vec(a.x + t * (b.x - a.x), a.y + t * (b.y - a.y))
        return p.distance_to(projection)

    def update_towers(self, dt: float) -> None:
        for tower in self.towers:
            if not tower.alive:
                continue
            tower.cooldown = max(0, tower.cooldown - dt)
            if tower.cooldown > 0:
                continue
            target = self.pick_tower_target(tower)
            if target:
                self.projectiles.append(Projectile(tower.pos.copy(), target, self.data.towers[tower.kind]["attack"]["projectileSpeed"], tower.damage))
                tower.cooldown = 1 / tower.shots_per_second

    def pick_tower_target(self, tower: Tower) -> Optional[Ant]:
        candidates = [ant for ant in self.ants if ant.alive and not ant.finished and ant.pos.distance_to(tower.pos) <= tower.attack_range]
        if not candidates:
            return None
        return min(candidates, key=lambda ant: ant.pos.distance_to(tower.pos))

    def update_projectiles(self, dt: float) -> None:
        for projectile in self.projectiles:
            projectile.update(dt)

    def cleanup_entities(self) -> None:
        for ant in self.ants:
            if not ant.alive and not ant.finished:
                self.resources["oil"] += ant.oil_reward
                self.score += ant.score_reward
                ant.finished = True
        self.projectiles = [p for p in self.projectiles if p.alive]
        self.towers = [t for t in self.towers if t.alive]
        self.blockers = [b for b in self.blockers if b.alive]
        self.ants = [a for a in self.ants if not (a.finished and not a.alive) and not (a.finished and a.state in (AntState.LEAVING, AntState.SIEGE_LEAVING))]

    def check_wave_end(self) -> None:
        wave = self.level["waves"][self.wave_index]
        all_spawned = all(self.spawn_counts.get(i, 0) >= spawn["count"] for i, spawn in enumerate(wave["spawns"]))
        active_ants = any(ant.alive and not ant.finished for ant in self.ants)
        if self.wave_time >= wave["duration"] and all_spawned and not active_ants:
            self.wave_index += 1
            if self.wave_index >= len(self.level["waves"]):
                self.phase = GamePhase.WIN
                self.score += self.remaining_sweet_units() * self.level["scoring"]["sweetUnitBonus"]
                self.status_message = "勝利！按 R 重新開始。"
            else:
                self.phase = GamePhase.PREP
                bonus = self.level["wavePrepOilBonus"]
                self.resources["oil"] += bonus
                self.status_message = f"波段結束，獲得 {bonus} 油量。Space 開始下一波。"

    def check_loss(self) -> None:
        if self.remaining_sweet_units() <= 0:
            self.phase = GamePhase.LOSE
            self.status_message = "地圖上的甜食已全部被搬走，失敗。按 R 重新開始。"

    def current_blockers(self) -> List[CircleBlocker]:
        blockers = []
        for building in [*self.towers, *self.blockers]:
            if building.alive and building.blocks_movement:
                blockers.append(CircleBlocker(building.pos.x, building.pos.y, building.radius, building))
        return blockers

    def pick_target_sweet(self, source: Vec) -> Optional[Sweet]:
        active = [sweet for sweet in self.sweets if not sweet.depleted]
        if not active:
            return None
        return min(active, key=lambda sweet: source.distance_to(sweet.pos))

    def entry_by_id(self, entry_id: str) -> Vec:
        for entry in self.level["entries"]:
            if entry["id"] == entry_id:
                return Vec(entry["x"], entry["y"])
        raise ValueError(f"Unknown entry id: {entry_id}")

    def exits(self) -> List[Vec]:
        return [Vec(item["x"], item["y"]) for item in self.level["exits"]]

    def nearest_exit(self, pos: Vec) -> Vec:
        return min(self.exits(), key=lambda exit_pos: pos.distance_to(exit_pos))

    def total_sweet_units_start(self) -> int:
        total = 0
        for item in self.level["sweets"]:
            total += item.get("units", self.data.sweets[item["type"]]["units"])
        return total

    def remaining_sweet_units(self) -> int:
        return sum(sweet.units for sweet in self.sweets)

    def draw(self) -> None:
        self.screen.fill((238, 232, 220))
        self.draw_ui()
        self.draw_world()
        pygame.display.flip()

    def draw_ui(self) -> None:
        pygame.draw.rect(self.screen, (38, 43, 50), (0, 0, self.screen.get_width(), self.ui_h))
        wave_name = "完成" if self.wave_index >= len(self.level["waves"]) else self.level["waves"][self.wave_index]["name"]
        level_name = self.level["name"]
        level_number = self.data.level_numbers[self.level_index]
        text = f"關卡:{level_number} {level_name}  階段:{self.phase.value}  波:{wave_name}  油量:{self.resources['oil']}  分數:{self.score}  甜食:{self.remaining_sweet_units()}/{self.total_sweet_units_start()}"
        self.blit_text(text, 12, 10, (255, 255, 255), self.font)
        hint = f"左鍵建造  Space開始  R重開  F1-F3/F9切關  | {self.status_message}"
        self.blit_text(hint, 12, 40, (219, 225, 232), self.small_font)
        self.draw_build_buttons()

    def draw_build_buttons(self) -> None:
        self.layout_build_button_rects()
        for buildable_id, rect in self.build_button_rects.items():
            definition, _ = self.build_definition(buildable_id)
            selected = buildable_id == self.selected_buildable
            afford = self.affordable_count(buildable_id)
            bg = (78, 113, 168) if selected else (55, 62, 72)
            border = (241, 216, 116) if selected else (124, 133, 146)
            pygame.draw.rect(self.screen, bg, rect, border_radius=6)
            pygame.draw.rect(self.screen, border, rect, 2, border_radius=6)
            icon_center = (rect.x + 22, rect.y + rect.h // 2)
            icon = self.assets.scaled(buildable_id, 34, 34)
            if icon:
                self.screen.blit(icon, icon.get_rect(center=icon_center))
            else:
                pygame.draw.circle(self.screen, tuple(definition["color"]), icon_center, 14)
                pygame.draw.circle(self.screen, (28, 30, 34), icon_center, 14, 2)
                icon_surface = self.small_font.render(definition["label"], True, (255, 255, 255))
                self.screen.blit(icon_surface, icon_surface.get_rect(center=icon_center))
            self.blit_text(f"{self.buildable_name(buildable_id)}", rect.x + 44, rect.y + 7, (255, 255, 255), self.small_font)
            self.blit_text(f"可放:{afford}  油:{self.buildable_cost(buildable_id)}", rect.x + 44, rect.y + 27, (225, 232, 238), self.small_font)

    def layout_build_button_rects(self) -> None:
        self.build_button_rects.clear()
        button_w = 132
        button_h = 48
        gap = 10
        total_w = len(self.buildable_ids()) * button_w + max(0, len(self.buildable_ids()) - 1) * gap
        x = self.screen.get_width() - total_w - 12
        y = 12
        for buildable_id in self.buildable_ids():
            self.build_button_rects[buildable_id] = pygame.Rect(x, y, button_w, button_h)
            x += button_w + gap

    def draw_world(self) -> None:
        margin = self.data.config["world"]["edgeBuildMargin"]
        mouse = pygame.mouse.get_pos()
        mouse_world = self.screen_to_world(mouse)
        mouse_in_world = mouse[1] >= self.ui_h
        _, selected_is_tower = self.build_definition(self.selected_buildable)
        hovered_tower = self.hovered_tower(mouse_world) if mouse_in_world else None
        placing_tower = (
            self.phase in (GamePhase.PREP, GamePhase.ATTACK)
            and mouse_in_world
            and selected_is_tower
            and hovered_tower is None
        )
        self.draw_grass_background()
        pygame.draw.rect(self.screen, (78, 115, 64), (margin, self.ui_h + margin, self.world_w - margin * 2, self.world_h - margin * 2), 2)
        for entry in self.level["entries"]:
            self.draw_entry_exit_marker(Vec(entry["x"], entry["y"]), "入", (35, 35, 35), entering=True)
        for exit_pos in self.exits():
            self.draw_entry_exit_marker(exit_pos, "出", (76, 96, 65), entering=False)
        for sweet in self.sweets:
            if not sweet.depleted:
                self.draw_sweet(sweet)
        for tower in self.towers:
            if placing_tower or tower is hovered_tower:
                self.draw_contrast_circle(tower.pos, int(tower.attack_range))
            self.draw_square_entity(tower)
        for blocker in self.blockers:
            self.draw_square_entity(blocker)
        for projectile in self.projectiles:
            pygame.draw.circle(self.screen, (30, 30, 30), self.world_to_screen(projectile.pos), 4)
        for ant in self.ants:
            if ant.alive and not ant.finished:
                self.draw_ant(ant)
                self.draw_hp_bar(ant.pos, ant.radius, ant.hp, ant.max_hp)
        if self.phase in (GamePhase.PREP, GamePhase.ATTACK) and mouse_in_world:
            self.draw_build_preview(self.screen_to_world(mouse))
        if self.phase in (GamePhase.WIN, GamePhase.LOSE):
            self.draw_center_overlay()

    def hovered_tower(self, pos: Vec) -> Optional[Tower]:
        for tower in reversed(self.towers):
            if tower.pos.distance_to(pos) <= tower.radius:
                return tower
        return None

    def draw_entry_exit_marker(self, pos: Vec, label: str, color: Tuple[int, int, int], entering: bool) -> None:
        alpha_color = color if self.phase == GamePhase.PREP else tuple(max(40, int(c * 0.7)) for c in color)
        pygame.draw.circle(self.screen, color, self.world_to_screen(pos), 16, 2)
        self.blit_center(label, pos, alpha_color, self.font)
        direction = self.edge_direction(pos)
        if not entering:
            direction = Vec(-direction.x, -direction.y)
        start = Vec(pos.x - direction.x * 26, pos.y - direction.y * 26)
        end = Vec(pos.x + direction.x * 26, pos.y + direction.y * 26)
        self.draw_arrow(start, end, alpha_color)

    def edge_direction(self, pos: Vec) -> Vec:
        if pos.x <= 0:
            return Vec(1, 0)
        if pos.x >= self.world_w - 1:
            return Vec(-1, 0)
        if pos.y <= 0:
            return Vec(0, 1)
        if pos.y >= self.world_h - 1:
            return Vec(0, -1)
        center = Vec(self.world_w / 2, self.world_h / 2)
        dx = center.x - pos.x
        dy = center.y - pos.y
        dist = math.hypot(dx, dy) or 1
        return Vec(dx / dist, dy / dist)

    def draw_arrow(self, start: Vec, end: Vec, color: Tuple[int, int, int]) -> None:
        start_pos = self.world_to_screen(start)
        end_pos = self.world_to_screen(end)
        pygame.draw.line(self.screen, color, start_pos, end_pos, 3)
        angle = math.atan2(end.y - start.y, end.x - start.x)
        head = 10
        points = [
            end_pos,
            (int(end_pos[0] - math.cos(angle - 0.55) * head), int(end_pos[1] - math.sin(angle - 0.55) * head)),
            (int(end_pos[0] - math.cos(angle + 0.55) * head), int(end_pos[1] - math.sin(angle + 0.55) * head)),
        ]
        pygame.draw.polygon(self.screen, color, points)

    def draw_contrast_circle(self, pos: Vec, radius: int) -> None:
        center = self.world_to_screen(pos)
        pygame.draw.circle(self.screen, (255, 255, 255), center, radius, 2)

    def draw_grass_background(self) -> None:
        grass = self.assets.scaled("grass_background", self.world_w, self.world_h)
        if grass:
            self.screen.blit(grass, (0, self.ui_h))
        else:
            pygame.draw.rect(self.screen, (135, 176, 100), (0, self.ui_h, self.world_w, self.world_h))

    def draw_sweet(self, sweet: Sweet) -> None:
        ratio = max(0.35, sweet.units / max(1, sweet.max_units))
        radius = max(8, int(sweet.max_radius * (0.55 + 0.45 * ratio)))
        if not self.draw_sprite(sweet.kind, sweet.pos, radius * 2.6):
            self.draw_circle_entity(sweet.pos, radius, sweet.color, f"{sweet.label}{sweet.units}")
        self.blit_center(str(sweet.units), Vec(sweet.pos.x, sweet.pos.y + radius + 12), (40, 32, 26), self.small_font)

    def draw_square_entity(self, entity: Building) -> None:
        if not self.draw_sprite(entity.kind, entity.pos, entity.radius * 3.1):
            x, y = self.world_to_screen(entity.pos)
            pygame.draw.rect(self.screen, entity.color, (x - entity.radius, y - entity.radius, entity.radius * 2, entity.radius * 2))
            pygame.draw.rect(self.screen, (32, 32, 32), (x - entity.radius, y - entity.radius, entity.radius * 2, entity.radius * 2), 2)
            self.blit_center(entity.label, entity.pos, (255, 255, 255), self.font)
        self.draw_hp_bar(entity.pos, entity.radius, entity.hp, entity.max_hp)

    def draw_ant(self, ant: Ant) -> None:
        frame = 1 + ((pygame.time.get_ticks() // 180) % 2)
        asset_id = f"{ant.kind}_{frame}"
        size = ant.radius * (3.4 if ant.role == "siege" else 4.0)
        if not self.draw_sprite(asset_id, ant.pos, size, rotation_degrees=-math.degrees(ant.facing_angle) - 90):
            self.draw_circle_entity(ant.pos, ant.radius, ant.color, ant.label)

    def draw_sprite(self, asset_id: str, pos: Vec, size: float, rotation_degrees: float = 0) -> bool:
        pixel_size = max(8, int(size))
        image = self.assets.scaled(asset_id, pixel_size, pixel_size)
        if image is None:
            return False
        if rotation_degrees:
            image = pygame.transform.rotozoom(image, rotation_degrees, 1.0)
        rect = image.get_rect(center=self.world_to_screen(pos))
        self.screen.blit(image, rect)
        return True

    def draw_circle_entity(self, pos: Vec, radius: int, color: Tuple[int, int, int], label: str) -> None:
        pygame.draw.circle(self.screen, color, self.world_to_screen(pos), radius)
        pygame.draw.circle(self.screen, (32, 32, 32), self.world_to_screen(pos), radius, 2)
        self.blit_center(label, pos, (255, 255, 255), self.small_font)

    def draw_hp_bar(self, pos: Vec, radius: int, hp: float, max_hp: float) -> None:
        if hp >= max_hp:
            return
        x, y = self.world_to_screen(pos)
        width = radius * 2
        ratio = max(0, hp / max_hp)
        pygame.draw.rect(self.screen, (90, 30, 30), (x - radius, y - radius - 8, width, 4))
        pygame.draw.rect(self.screen, (68, 180, 91), (x - radius, y - radius - 8, int(width * ratio), 4))

    def draw_build_preview(self, pos: Vec) -> None:
        definition, is_tower = self.build_definition(self.selected_buildable)
        radius = definition["radius"]
        valid = self.in_buildable_area(pos) and not self.overlaps_existing(pos, radius) and self.resources["oil"] >= definition["cost"]["oil"]
        preview = self.assets.scaled(self.selected_buildable, int(radius * 3.1), int(radius * 3.1))
        if preview:
            image = preview.copy()
            image.set_alpha(255 if valid else 95)
            self.screen.blit(image, image.get_rect(center=self.world_to_screen(pos)))
        if is_tower:
            self.draw_contrast_circle(pos, definition["attack"]["range"])

    def draw_center_overlay(self) -> None:
        rect = pygame.Rect(280, self.ui_h + 220, 450, 140)
        pygame.draw.rect(self.screen, (30, 34, 40), rect)
        pygame.draw.rect(self.screen, (255, 255, 255), rect, 2)
        title = "勝利" if self.phase == GamePhase.WIN else "失敗"
        self.blit_text(f"{title}  分數：{self.score}", rect.x + 30, rect.y + 36, (255, 255, 255), self.font)
        self.blit_text("按 R 重新開始，Esc 離開。", rect.x + 30, rect.y + 78, (230, 230, 230), self.font)

    def blit_text(self, text: str, x: int, y: int, color: Tuple[int, int, int], font: pygame.font.Font) -> None:
        self.screen.blit(font.render(text, True, color), (x, y))

    def blit_center(self, text: str, pos: Vec, color: Tuple[int, int, int], font: pygame.font.Font) -> None:
        surface = font.render(text, True, color)
        rect = surface.get_rect(center=self.world_to_screen(pos))
        self.screen.blit(surface, rect)


def main() -> None:
    os.chdir(ROOT)
    Game().run()


if __name__ == "__main__":
    main()
