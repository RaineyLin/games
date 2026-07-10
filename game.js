"use strict";

const DATA_PATHS = {
  config: "data/game_config.json",
  ants: "data/ants.json",
  sweets: "data/sweets.json",
  towers: "data/towers.json",
  blockers: "data/blockers.json",
  levels: [
    ["001", "data/levels/level_001.json"],
    ["002", "data/levels/level_002.json"],
    ["003", "data/levels/level_003.json"],
    ["009", "data/levels/level_009.json"],
  ],
};

const IMAGE_PATHS = {
  worker_ant_1: "assets/images/worker_ant_walk_1.png",
  worker_ant_2: "assets/images/worker_ant_walk_2.png",
  soldier_ant_1: "assets/images/soldier_ant_walk_1.png",
  soldier_ant_2: "assets/images/soldier_ant_walk_2.png",
  basic_turret: "assets/images/basic_turret.png",
  block_wall: "assets/images/stone_wall.png",
  crumb_rock: "assets/images/rock_obstacle.png",
  sugar_pile: "assets/images/sugar_pile.png",
  hard_candy: "assets/images/hard_candy.png",
  grass_background: "assets/images/grass_background.png",
};

const Phase = Object.freeze({ PREP: "準備", ATTACK: "進攻", WIN: "勝利", LOSE: "失敗" });
const AntState = Object.freeze({
  TO_SWEET: "to_sweet",
  HARVESTING: "harvesting",
  LEAVING: "leaving",
  SIEGE: "siege",
  SIEGE_LEAVING: "siege_leaving",
});

class Vec {
  constructor(x, y) {
    this.x = x;
    this.y = y;
  }

  copy() {
    return new Vec(this.x, this.y);
  }

  distanceTo(other) {
    return Math.hypot(this.x - other.x, this.y - other.y);
  }

  toward(target, maxDistance) {
    const dist = this.distanceTo(target);
    if (dist <= maxDistance || dist === 0) {
      this.x = target.x;
      this.y = target.y;
      return true;
    }
    const scale = maxDistance / dist;
    this.x += (target.x - this.x) * scale;
    this.y += (target.y - this.y) * scale;
    return false;
  }
}

class MinHeap {
  constructor() {
    this.items = [];
  }

  push(priority, value) {
    const item = { priority, value };
    this.items.push(item);
    let i = this.items.length - 1;
    while (i > 0) {
      const parent = Math.floor((i - 1) / 2);
      if (this.items[parent].priority <= item.priority) break;
      this.items[i] = this.items[parent];
      i = parent;
    }
    this.items[i] = item;
  }

  pop() {
    if (!this.items.length) return null;
    const first = this.items[0];
    const last = this.items.pop();
    if (this.items.length && last) {
      let i = 0;
      while (true) {
        const left = i * 2 + 1;
        const right = left + 1;
        if (left >= this.items.length) break;
        let child = left;
        if (right < this.items.length && this.items[right].priority < this.items[left].priority) {
          child = right;
        }
        if (this.items[child].priority >= last.priority) break;
        this.items[i] = this.items[child];
        i = child;
      }
      this.items[i] = last;
    }
    return first.value;
  }

  get length() {
    return this.items.length;
  }
}

class Pathfinder {
  constructor(width, height, cellSize) {
    this.width = width;
    this.height = height;
    this.cellSize = cellSize;
    this.cols = Math.ceil(width / cellSize);
    this.rows = Math.ceil(height / cellSize);
  }

  hasPath(start, goal, blockers, unitRadius) {
    return this.findPath(start, goal, blockers, unitRadius).found;
  }

  findPath(start, goal, blockers, unitRadius) {
    const local = this.findAntennaPath(start, goal, blockers, unitRadius);
    if (local.found) return local;
    return this.findAStarPath(start, goal, blockers, unitRadius);
  }

  findAntennaPath(start, goal, blockers, unitRadius) {
    const points = [start.copy()];
    let current = start.copy();
    for (let i = 0; i < 48; i += 1) {
      if (current.distanceTo(goal) <= 1) {
        points[points.length - 1] = goal.copy();
        return { found: true, points: this.dedupe(points) };
      }
      const hit = this.firstBlockerOnSegment(current, goal, blockers, unitRadius);
      if (!hit) {
        points.push(goal.copy());
        return { found: true, points: this.dedupe(points) };
      }
      const waypoint = this.pickDetourWaypoint(current, goal, hit, blockers, unitRadius, points);
      if (!waypoint) return { found: false, points: [], reason: "no_local_detour" };
      points.push(waypoint);
      current = waypoint;
    }
    return { found: false, points: [], reason: "too_many_local_detours" };
  }

  findAStarPath(start, goal, blockers, unitRadius) {
    const startCell = this.toCell(start);
    const goalCell = this.toCell(goal);
    if (!this.inBounds(startCell) || !this.inBounds(goalCell)) {
      return { found: false, points: [], reason: "out_of_bounds" };
    }
    const blocked = this.blockedCells(blockers, unitRadius);
    blocked.delete(this.key(startCell));
    blocked.delete(this.key(goalCell));
    const frontier = new MinHeap();
    frontier.push(0, startCell);
    const cameFrom = new Map([[this.key(startCell), null]]);
    const costSoFar = new Map([[this.key(startCell), 0]]);

    while (frontier.length) {
      const current = frontier.pop();
      const currentKey = this.key(current);
      if (current[0] === goalCell[0] && current[1] === goalCell[1]) {
        const cells = this.reconstruct(cameFrom, current);
        const points = cells.map((cell) => this.toWorld(cell));
        if (points.length) {
          points[0] = start.copy();
          points[points.length - 1] = goal.copy();
        }
        return { found: true, points: this.smooth(points, blocked) };
      }
      for (const [neighbor, stepCost] of this.neighbors(current)) {
        const neighborKey = this.key(neighbor);
        if (blocked.has(neighborKey)) continue;
        const newCost = costSoFar.get(currentKey) + stepCost;
        if (!costSoFar.has(neighborKey) || newCost < costSoFar.get(neighborKey)) {
          costSoFar.set(neighborKey, newCost);
          frontier.push(newCost + this.heuristic(neighbor, goalCell), neighbor);
          cameFrom.set(neighborKey, current);
        }
      }
    }
    return { found: false, points: [], reason: "no_path" };
  }

  firstBlockerOnSegment(start, goal, blockers, unitRadius) {
    let best = null;
    let bestT = Infinity;
    for (const blocker of blockers) {
      const hitAt = this.segmentBlockerHitT(start, goal, blocker, unitRadius);
      if (hitAt !== null && hitAt < bestT) {
        best = blocker;
        bestT = hitAt;
      }
    }
    return best;
  }

  segmentBlockerHitT(start, goal, blocker, unitRadius) {
    const dx = goal.x - start.x;
    const dy = goal.y - start.y;
    const lengthSq = dx * dx + dy * dy;
    if (!lengthSq) return null;
    const t = ((blocker.x - start.x) * dx + (blocker.y - start.y) * dy) / lengthSq;
    if (t <= 0.02 || t >= 0.98) return null;
    const closest = new Vec(start.x + dx * t, start.y + dy * t);
    const radius = blocker.radius + unitRadius + this.cellSize * 0.25;
    return closest.distanceTo(blocker) <= radius ? t : null;
  }

  pickDetourWaypoint(start, goal, blocker, blockers, unitRadius, existingPoints) {
    const dx = goal.x - start.x;
    const dy = goal.y - start.y;
    const distance = Math.hypot(dx, dy);
    if (!distance) return null;
    const perpX = -dy / distance;
    const perpY = dx / distance;
    const forwardX = dx / distance;
    const forwardY = dy / distance;
    const clearance = blocker.radius + unitRadius + this.cellSize * 1.35;
    const candidates = [];
    for (const side of [-1, 1]) {
      for (const bias of [-0.35, 0, 0.45, 0.9]) {
        candidates.push(new Vec(
          blocker.x + perpX * side * clearance + forwardX * clearance * bias,
          blocker.y + perpY * side * clearance + forwardY * clearance * bias,
        ));
      }
    }
    const angleFromCenter = Math.atan2(start.y - blocker.y, start.x - blocker.x);
    for (const side of [-1, 1]) {
      for (const turn of [Math.PI / 3, Math.PI / 2, Math.PI * 2 / 3]) {
        const angle = angleFromCenter + side * turn;
        candidates.push(new Vec(blocker.x + Math.cos(angle) * clearance, blocker.y + Math.sin(angle) * clearance));
      }
    }
    const valid = candidates.filter((candidate) => (
      this.validDetourSegment(start, candidate, blockers, unitRadius)
      && existingPoints.every((point) => candidate.distanceTo(point) > this.cellSize * 0.75)
    ));
    valid.sort((a, b) => (
      a.distanceTo(goal) - b.distanceTo(goal)
      || start.distanceTo(a) - start.distanceTo(b)
      || Math.abs(this.turnAmount(start, goal, a)) - Math.abs(this.turnAmount(start, goal, b))
    ));
    return valid[0] || null;
  }

  validDetourSegment(start, candidate, blockers, unitRadius) {
    if (!this.inBounds(this.toCell(candidate))) return false;
    for (const blocker of blockers) {
      if (candidate.distanceTo(blocker) <= blocker.radius + unitRadius + this.cellSize * 0.35) return false;
      if (this.segmentBlockerHitT(start, candidate, blocker, unitRadius) !== null) return false;
    }
    return true;
  }

  turnAmount(start, goal, candidate) {
    const goalAngle = Math.atan2(goal.y - start.y, goal.x - start.x);
    const candidateAngle = Math.atan2(candidate.y - start.y, candidate.x - start.x);
    return Math.atan2(Math.sin(candidateAngle - goalAngle), Math.cos(candidateAngle - goalAngle));
  }

  blockedCells(blockers, unitRadius) {
    const blocked = new Set();
    for (const blocker of blockers) {
      const radius = blocker.radius + unitRadius;
      const minCol = Math.max(0, Math.floor((blocker.x - radius) / this.cellSize));
      const maxCol = Math.min(this.cols - 1, Math.floor((blocker.x + radius) / this.cellSize));
      const minRow = Math.max(0, Math.floor((blocker.y - radius) / this.cellSize));
      const maxRow = Math.min(this.rows - 1, Math.floor((blocker.y + radius) / this.cellSize));
      for (let col = minCol; col <= maxCol; col += 1) {
        for (let row = minRow; row <= maxRow; row += 1) {
          if (this.toWorld([col, row]).distanceTo(blocker) <= radius + this.cellSize * 0.72) {
            blocked.add(this.key([col, row]));
          }
        }
      }
    }
    return blocked;
  }

  toCell(point) {
    return [Math.floor(point.x / this.cellSize), Math.floor(point.y / this.cellSize)];
  }

  toWorld(cell) {
    return new Vec((cell[0] + 0.5) * this.cellSize, (cell[1] + 0.5) * this.cellSize);
  }

  inBounds(cell) {
    return cell[0] >= 0 && cell[0] < this.cols && cell[1] >= 0 && cell[1] < this.rows;
  }

  neighbors(cell) {
    const result = [];
    for (const [dx, dy, cost] of [[-1, 0, 1], [1, 0, 1], [0, -1, 1], [0, 1, 1], [-1, -1, 1.414], [1, -1, 1.414], [-1, 1, 1.414], [1, 1, 1.414]]) {
      const next = [cell[0] + dx, cell[1] + dy];
      if (this.inBounds(next)) result.push([next, cost]);
    }
    return result;
  }

  heuristic(a, b) {
    return Math.hypot(a[0] - b[0], a[1] - b[1]);
  }

  reconstruct(cameFrom, current) {
    const cells = [current];
    let key = this.key(current);
    while (cameFrom.get(key) !== null) {
      current = cameFrom.get(key);
      key = this.key(current);
      cells.push(current);
    }
    return cells.reverse();
  }

  smooth(points, blocked) {
    if (points.length <= 2) return points;
    const smoothed = [points[0]];
    let anchor = 0;
    for (let probe = 2; probe < points.length; probe += 1) {
      if (this.lineHitsBlocked(points[anchor], points[probe], blocked)) {
        smoothed.push(points[probe - 1]);
        anchor = probe - 1;
      }
    }
    smoothed.push(points[points.length - 1]);
    return smoothed;
  }

  lineHitsBlocked(a, b, blocked) {
    const dist = Math.max(1, Math.floor(a.distanceTo(b) / (this.cellSize * 0.5)));
    for (let i = 0; i <= dist; i += 1) {
      const t = i / dist;
      const cell = this.toCell(new Vec(a.x + (b.x - a.x) * t, a.y + (b.y - a.y) * t));
      if (blocked.has(this.key(cell))) return true;
    }
    return false;
  }

  dedupe(points) {
    const out = [];
    for (const point of points) {
      if (!out.length || point.distanceTo(out[out.length - 1]) > 1) out.push(point);
    }
    return out;
  }

  key(cell) {
    return `${cell[0]},${cell[1]}`;
  }
}

class Game {
  constructor(canvas, data, images) {
    this.canvas = canvas;
    this.ctx = canvas.getContext("2d");
    this.data = data;
    this.images = images;
    this.levelIndex = 0;
    this.levelNumbers = data.levels.map((item) => item.number);
    this.level = data.levels[this.levelIndex].data;
    this.cfg = data.config;
    this.worldW = this.cfg.world.width;
    this.worldH = this.cfg.world.height;
    this.uiH = this.cfg.ui.topBarHeight;
    this.pathfinder = new Pathfinder(this.worldW, this.worldH, this.cfg.pathfinding.cellSize);
    this.mouse = new Vec(-999, -999);
    this.selectedBuildable = "basic_turret";
    this.buildButtonRects = new Map();
    this.statusMessage = "按 1 選砲塔、2 選牆；左鍵建造，Space 開始進攻。";
    this.resetLevel();
    this.bindEvents();
  }

  bindEvents() {
    window.addEventListener("keydown", (event) => this.handleKey(event));
    this.canvas.addEventListener("pointermove", (event) => {
      this.mouse = this.eventToCanvas(event);
    });
    this.canvas.addEventListener("pointerleave", () => {
      this.mouse = new Vec(-999, -999);
    });
    this.canvas.addEventListener("pointerdown", (event) => {
      const pos = this.eventToCanvas(event);
      this.mouse = pos;
      if (pos.y < this.uiH) {
        this.handleBuildButtonClick(pos);
      } else if (this.phase === Phase.PREP || this.phase === Phase.ATTACK) {
        this.tryPlace(this.screenToWorld(pos));
      }
    });
  }

  eventToCanvas(event) {
    const rect = this.canvas.getBoundingClientRect();
    return new Vec(
      (event.clientX - rect.left) * (this.canvas.width / rect.width),
      (event.clientY - rect.top) * (this.canvas.height / rect.height),
    );
  }

  handleKey(event) {
    if (event.key === "1") {
      this.selectedBuildable = "basic_turret";
      this.statusMessage = "已選：基礎砲塔";
    } else if (event.key === "2") {
      this.selectedBuildable = "block_wall";
      this.statusMessage = "已選：阻擋牆";
    } else if (event.code === "Space" && this.phase === Phase.PREP) {
      event.preventDefault();
      this.startWave();
    } else if ((event.key === "r" || event.key === "R") && [Phase.WIN, Phase.LOSE].includes(this.phase)) {
      this.resetLevel();
    } else if (event.key === "F1") {
      event.preventDefault();
      this.resetLevel(0);
    } else if (event.key === "F2") {
      event.preventDefault();
      this.resetLevel(1);
    } else if (event.key === "F3") {
      event.preventDefault();
      this.resetLevel(2);
    } else if (event.key === "F9") {
      event.preventDefault();
      this.resetLevelByNumber(9);
    } else if ((event.key === "s" || event.key === "S") && this.phase === Phase.ATTACK) {
      this.spawnManualSoldier();
    }
  }

  resetLevel(levelIndex = this.levelIndex) {
    this.levelIndex = Math.max(0, Math.min(levelIndex, this.data.levels.length - 1));
    this.level = this.data.levels[this.levelIndex].data;
    this.phase = Phase.PREP;
    this.resources = { oil: this.level.initialResources.oil };
    this.score = 0;
    this.waveIndex = 0;
    this.waveTime = 0;
    this.spawnTimers = new Map();
    this.spawnCounts = new Map();
    this.penaltyCooldown = 0;
    this.sweets = [];
    this.towers = [];
    this.blockers = [];
    this.ants = [];
    this.projectiles = [];
    this.statusMessage = `已載入：${this.level.name}。Space 開始進攻。`;
    this.loadLevelEntities();
  }

  resetLevelByNumber(levelNumber) {
    const index = this.levelNumbers.indexOf(levelNumber);
    if (index >= 0) this.resetLevel(index);
    else this.statusMessage = `尚未建立第 ${levelNumber} 關。`;
  }

  loadLevelEntities() {
    for (const item of this.level.sweets) {
      const def = this.data.sweets[item.type];
      this.sweets.push({
        id: item.id,
        kind: item.type,
        pos: new Vec(item.x, item.y),
        label: def.label,
        color: def.color,
        radius: def.radius,
        maxRadius: def.radius,
        mode: def.carryMode,
        units: item.units ?? def.units,
        maxUnits: item.units ?? def.units,
        chewTime: def.chewTime || 0,
        value: def.value,
      });
    }
    for (const item of this.level.blockers || []) {
      this.blockers.push(this.makeBlockerFromLevel(item));
    }
    if (this.level.randomBlockers) this.generateRandomBlockers(this.level.randomBlockers);
  }

  makeBlockerFromLevel(item) {
    const def = this.data.blockers[item.type];
    return {
      id: item.id,
      kind: item.type,
      pos: new Vec(item.x, item.y),
      label: def.label,
      color: def.color,
      radius: def.radius,
      hp: def.hp,
      maxHp: def.hp,
      cost: def.cost?.oil || 0,
      blocksMovement: def.blocksMovement,
      destructible: def.destructible ?? true,
    };
  }

  generateRandomBlockers(settings) {
    const random = seededRandom(settings.seed ?? this.levelIndex + 1);
    const def = this.data.blockers[settings.type];
    let placed = 0;
    let attempts = settings.count * 60;
    while (placed < settings.count && attempts > 0) {
      attempts -= 1;
      const area = settings.area;
      const pos = new Vec(randInt(random, area.xMin, area.xMax), randInt(random, area.yMin, area.yMax));
      if (this.overlapsExisting(pos, def.radius)) continue;
      this.blockers.push({
        id: `random_blocker_${placed}`,
        kind: settings.type,
        pos,
        label: def.label,
        color: def.color,
        radius: def.radius,
        hp: def.hp,
        maxHp: def.hp,
        cost: def.cost?.oil || 0,
        blocksMovement: def.blocksMovement,
        destructible: def.destructible ?? true,
      });
      placed += 1;
    }
  }

  update(dt) {
    dt *= this.cfg.gameplay.timeScale || 1;
    if (this.phase !== Phase.ATTACK) return;
    this.waveTime += dt;
    this.penaltyCooldown = Math.max(0, this.penaltyCooldown - dt);
    this.updateSpawning(dt);
    this.updateAnts(dt);
    this.updateTowers(dt);
    this.updateProjectiles(dt);
    this.cleanupEntities();
    this.checkWaveEnd();
    this.checkLoss();
  }

  updateSpawning(dt) {
    const wave = this.level.waves[this.waveIndex];
    wave.spawns.forEach((spawn, idx) => {
      this.spawnTimers.set(idx, (this.spawnTimers.get(idx) || 0) - dt);
      this.spawnCounts.set(idx, this.spawnCounts.get(idx) || 0);
      if (this.spawnCounts.get(idx) >= spawn.count) return;
      if (this.spawnTimers.get(idx) <= 0) {
        this.spawnAnt(spawn);
        this.spawnCounts.set(idx, this.spawnCounts.get(idx) + 1);
        this.spawnTimers.set(idx, spawn.interval);
      }
    });
  }

  spawnAnt(spawn) {
    const def = this.data.ants[spawn.antType];
    const entry = this.entryById(spawn.entryId);
    const sweet = this.pickTargetSweet(entry);
    if (!sweet) return;
    const state = def.role === "siege" ? AntState.SIEGE : AntState.TO_SWEET;
    const ant = this.makeAnt(spawn.antType, def, entry.copy(), state, entry.copy(), sweet, false, spawn.destroyCount ?? this.soldierDestroyGoal(spawn.antType));
    const path = this.pathfinder.findPath(ant.pos, sweet.pos, this.currentBlockers(), ant.radius);
    if (!path.found) {
      if (ant.role === "siege") this.ants.push(ant);
      else this.triggerSoldierPenalty(entry.copy(), sweet);
      return;
    }
    ant.setPath(path.points);
    this.ants.push(ant);
  }

  makeAnt(kind, def, pos, state, entry, sweet, penaltySoldier = false, destroyGoal = 0) {
    const ant = {
      id: `ant_${this.ants.length}_${Math.floor(Math.random() * 9000 + 1000)}`,
      kind,
      pos,
      label: def.label,
      color: def.color,
      radius: def.radius,
      hp: def.hp,
      maxHp: def.hp,
      speed: def.speed,
      oilReward: def.oilReward,
      scoreReward: def.scoreReward,
      role: def.role,
      canDamageBuildings: def.canDamageBuildings,
      buildingDamagePerSecond: def.buildingDamagePerSecond,
      state,
      entry,
      targetSweet: sweet,
      path: [],
      pathIndex: 0,
      harvestTimer: 0,
      carryingValue: 0,
      alive: true,
      finished: false,
      attackTarget: null,
      pathBiasSide: Math.random() < 0.5 ? -1 : 1,
      pathBiasStrength: def.role === "siege" ? randFloat(14, 30) : randFloat(3, 8),
      pathWobblePhase: Math.random() * Math.PI * 2,
      penaltySoldier,
      destroyedBuildings: 0,
      destroyGoal,
      facingAngle: -Math.PI / 2,
    };
    ant.takeDamage = (damage) => {
      ant.hp -= damage;
      if (ant.hp <= 0) ant.alive = false;
    };
    ant.setPath = (points) => {
      ant.path = this.withPathBias(ant, points);
      ant.pathIndex = points.length > 1 ? 1 : 0;
    };
    ant.moveToward = (target, maxDistance) => {
      const dx = target.x - ant.pos.x;
      const dy = target.y - ant.pos.y;
      if (dx || dy) ant.facingAngle = Math.atan2(dy, dx);
      return ant.pos.toward(target, maxDistance);
    };
    ant.followPath = (dt) => {
      if (ant.pathIndex >= ant.path.length) return true;
      if (ant.moveToward(ant.path[ant.pathIndex], ant.speed * dt)) ant.pathIndex += 1;
      return ant.pathIndex >= ant.path.length;
    };
    return ant;
  }

  withPathBias(ant, points) {
    if (points.length <= 1 || ant.pathBiasStrength <= 0) return points.map((point) => point.copy());
    const biased = [points[0].copy()];
    const isSoldier = ant.role === "siege";
    const minSegment = isSoldier ? 32 : 70;
    for (let i = 1; i < points.length; i += 1) {
      const start = points[i - 1];
      const end = points[i];
      const dx = end.x - start.x;
      const dy = end.y - start.y;
      const distance = Math.hypot(dx, dy);
      if (distance > minSegment) {
        const normalX = -dy / distance;
        const normalY = dx / distance;
        const wave = Math.sin(ant.pathWobblePhase + i * (isSoldier ? 2.37 : 1.73));
        const amount = ant.pathBiasSide * ant.pathBiasStrength * (0.65 + 0.35 * wave);
        const t = 0.45 + 0.08 * Math.sin(ant.pathWobblePhase + i * 2.11);
        biased.push(new Vec(start.x + dx * t + normalX * amount, start.y + dy * t + normalY * amount));
        if (isSoldier && distance > minSegment * 1.8) {
          const counterWave = Math.sin(ant.pathWobblePhase + i * 3.91);
          const counterAmount = -ant.pathBiasSide * ant.pathBiasStrength * (0.45 + 0.55 * counterWave);
          const t2 = 0.72 + 0.1 * Math.sin(ant.pathWobblePhase + i * 1.29);
          biased.push(new Vec(start.x + dx * t2 + normalX * counterAmount, start.y + dy * t2 + normalY * counterAmount));
        }
      }
      biased.push(end.copy());
    }
    return biased;
  }

  updateAnts(dt) {
    for (const ant of this.ants) {
      if (!ant.alive || ant.finished) continue;
      if (ant.state === AntState.TO_SWEET) this.updateWorkerToSweet(ant, dt);
      else if (ant.state === AntState.HARVESTING) this.updateHarvesting(ant, dt);
      else if (ant.state === AntState.LEAVING && ant.followPath(dt)) ant.finished = true;
      else if (ant.state === AntState.SIEGE) this.updateSoldier(ant, dt);
      else if (ant.state === AntState.SIEGE_LEAVING && ant.followPath(dt)) ant.finished = true;
    }
    this.resolveAntCollisions();
  }

  updateWorkerToSweet(ant, dt) {
    let sweet = ant.targetSweet;
    if (!sweet || sweet.units <= 0) {
      sweet = this.pickTargetSweet(ant.pos);
      ant.targetSweet = sweet;
      if (!sweet) {
        ant.finished = true;
        return;
      }
      const path = this.pathfinder.findPath(ant.pos, sweet.pos, this.currentBlockers(), ant.radius);
      if (!path.found) {
        this.triggerSoldierPenalty(ant.entry, sweet);
        ant.finished = true;
        return;
      }
      ant.setPath(path.points);
    }
    if (ant.followPath(dt)) {
      if (sweet.mode === "instant_pickup") {
        this.takeSweetUnit(ant, sweet);
        this.routeAntToExit(ant);
      } else {
        ant.state = AntState.HARVESTING;
        ant.harvestTimer = sweet.chewTime;
      }
    }
  }

  updateHarvesting(ant, dt) {
    const sweet = ant.targetSweet;
    if (!sweet || sweet.units <= 0) {
      this.routeAntToExit(ant);
      return;
    }
    ant.harvestTimer -= dt;
    if (ant.harvestTimer <= 0) {
      this.takeSweetUnit(ant, sweet);
      this.routeAntToExit(ant);
    }
  }

  takeSweetUnit(ant, sweet) {
    if (sweet.units > 0) {
      sweet.units -= 1;
      ant.carryingValue = sweet.value;
    }
  }

  routeAntToExit(ant) {
    ant.state = AntState.LEAVING;
    const exitPos = this.nearestExit(ant.pos);
    const path = this.pathfinder.findPath(ant.pos, exitPos, this.currentBlockers(), ant.radius);
    ant.setPath(path.found ? path.points : [ant.pos.copy(), exitPos]);
  }

  updateSoldier(ant, dt) {
    const sweet = ant.targetSweet;
    if (!sweet) {
      ant.finished = true;
      return;
    }
    if (ant.penaltySoldier && this.pathfinder.hasPath(ant.entry, sweet.pos, this.currentBlockers(), ant.radius)) {
      ant.state = AntState.SIEGE_LEAVING;
      const exitPos = this.nearestExit(ant.pos);
      const path = this.pathfinder.findPath(ant.pos, exitPos, this.currentBlockers(), ant.radius);
      ant.setPath(path.found ? path.points : [ant.pos.copy(), exitPos]);
      this.statusMessage = "巨大兵蟻已打通路徑，正在離開。";
      return;
    }
    const target = ant.attackTarget && this.isAlive(ant.attackTarget) ? ant.attackTarget : this.findSiegeTarget(ant, sweet);
    ant.attackTarget = target;
    if (!target) {
      this.routeAntToExit(ant);
      return;
    }
    if (ant.pos.distanceTo(target.pos) <= ant.radius + target.radius + 6) {
      target.hp -= ant.buildingDamagePerSecond * dt;
      if (target.hp <= 0) {
        ant.destroyedBuildings += 1;
        ant.attackTarget = null;
        if (!ant.penaltySoldier && ant.destroyGoal > 0 && ant.destroyedBuildings >= ant.destroyGoal) {
          this.statusMessage = `兵蟻已破壞 ${ant.destroyedBuildings} 個目標，正在離開。`;
          this.routeAntToExit(ant);
        }
      }
    } else {
      ant.moveToward(this.soldierApproachPoint(ant, target), ant.speed * dt);
    }
  }

  soldierApproachPoint(ant, target) {
    const dx = target.pos.x - ant.pos.x;
    const dy = target.pos.y - ant.pos.y;
    const distance = Math.hypot(dx, dy);
    if (!distance) return target.pos.copy();
    const normalX = -dy / distance;
    const normalY = dx / distance;
    const pulse = Math.sin(this.waveTime * 4.6 + ant.pathWobblePhase);
    const chop = Math.sin(this.waveTime * 9.5 + ant.pathWobblePhase * 0.7);
    let offset = ant.pathBiasSide * (10 + ant.pathBiasStrength * 0.55 * pulse + 7 * chop);
    offset *= Math.min(1, Math.max(0.15, distance / 140));
    return new Vec(target.pos.x + normalX * offset, target.pos.y + normalY * offset);
  }

  findSiegeTarget(ant, sweet) {
    const candidates = [...this.towers, ...this.blockers].filter((b) => this.isAlive(b) && b.destructible);
    candidates.sort((a, b) => (
      this.distancePointToSegment(a.pos, ant.entry, sweet.pos) + a.pos.distanceTo(ant.pos) * 0.25
      - (this.distancePointToSegment(b.pos, ant.entry, sweet.pos) + b.pos.distanceTo(ant.pos) * 0.25)
    ));
    return candidates[0] || null;
  }

  distancePointToSegment(p, a, b) {
    const lengthSq = Math.max(1, (b.x - a.x) ** 2 + (b.y - a.y) ** 2);
    const t = Math.max(0, Math.min(1, ((p.x - a.x) * (b.x - a.x) + (p.y - a.y) * (b.y - a.y)) / lengthSq));
    return p.distanceTo(new Vec(a.x + t * (b.x - a.x), a.y + t * (b.y - a.y)));
  }

  updateTowers(dt) {
    for (const tower of this.towers) {
      if (!this.isAlive(tower)) continue;
      tower.cooldown = Math.max(0, tower.cooldown - dt);
      if (tower.cooldown > 0) continue;
      const target = this.pickTowerTarget(tower);
      if (target) {
        this.projectiles.push({ pos: tower.pos.copy(), target, speed: this.data.towers[tower.kind].attack.projectileSpeed, damage: tower.damage, alive: true });
        tower.cooldown = 1 / tower.shotsPerSecond;
      }
    }
  }

  pickTowerTarget(tower) {
    const candidates = this.ants.filter((ant) => ant.alive && !ant.finished && ant.pos.distanceTo(tower.pos) <= tower.attackRange);
    candidates.sort((a, b) => a.pos.distanceTo(tower.pos) - b.pos.distanceTo(tower.pos));
    return candidates[0] || null;
  }

  updateProjectiles(dt) {
    for (const projectile of this.projectiles) {
      if (!projectile.target.alive) {
        projectile.alive = false;
        continue;
      }
      if (projectile.pos.toward(projectile.target.pos, projectile.speed * dt)) {
        projectile.target.takeDamage(projectile.damage);
        projectile.alive = false;
      }
    }
  }

  cleanupEntities() {
    for (const ant of this.ants) {
      if (!ant.alive && !ant.finished) {
        this.resources.oil += ant.oilReward;
        this.score += ant.scoreReward;
        ant.finished = true;
      }
    }
    this.projectiles = this.projectiles.filter((p) => p.alive);
    this.towers = this.towers.filter((t) => this.isAlive(t));
    this.blockers = this.blockers.filter((b) => this.isAlive(b));
    this.ants = this.ants.filter((a) => !((a.finished && !a.alive) || (a.finished && [AntState.LEAVING, AntState.SIEGE_LEAVING].includes(a.state))));
  }

  checkWaveEnd() {
    const wave = this.level.waves[this.waveIndex];
    const allSpawned = wave.spawns.every((spawn, i) => (this.spawnCounts.get(i) || 0) >= spawn.count);
    const activeAnts = this.ants.some((ant) => ant.alive && !ant.finished);
    if (this.waveTime >= wave.duration && allSpawned && !activeAnts) {
      this.waveIndex += 1;
      if (this.waveIndex >= this.level.waves.length) {
        this.phase = Phase.WIN;
        this.score += this.remainingSweetUnits() * this.level.scoring.sweetUnitBonus;
        this.statusMessage = "勝利！按 R 重新開始。";
      } else {
        this.phase = Phase.PREP;
        const bonus = this.level.wavePrepOilBonus;
        this.resources.oil += bonus;
        this.statusMessage = `波段結束，獲得 ${bonus} 油量。Space 開始下一波。`;
      }
    }
  }

  checkLoss() {
    if (this.remainingSweetUnits() <= 0) {
      this.phase = Phase.LOSE;
      this.statusMessage = "地圖上的甜食已全部被搬走，失敗。按 R 重新開始。";
    }
  }

  startWave() {
    if (this.waveIndex >= this.level.waves.length) {
      this.phase = Phase.WIN;
      return;
    }
    this.phase = Phase.ATTACK;
    this.waveTime = 0;
    this.spawnTimers.clear();
    this.spawnCounts.clear();
    this.statusMessage = `${this.level.waves[this.waveIndex].name} 開始`;
  }

  tryPlace(pos) {
    if (!this.inBuildableArea(pos)) {
      this.statusMessage = "邊緣禁放區或場外不能建造。";
      return;
    }
    const [def, isTower] = this.buildDefinition(this.selectedBuildable);
    const cost = def.cost.oil;
    if (this.resources.oil < cost) {
      this.statusMessage = "油量不足。";
      return;
    }
    if (this.overlapsExisting(pos, def.radius)) {
      this.statusMessage = "建造位置與其他物件重疊。";
      return;
    }
    let building;
    if (isTower) {
      building = {
        id: `tower_${this.towers.length}`,
        kind: this.selectedBuildable,
        pos,
        label: def.label,
        color: def.color,
        radius: def.radius,
        hp: def.hp,
        maxHp: def.hp,
        cost,
        blocksMovement: def.blocksMovement,
        destructible: def.destructible ?? true,
        attackRange: def.attack.range,
        damage: def.attack.damage,
        shotsPerSecond: def.attack.shotsPerSecond,
        cooldown: 0,
      };
      this.towers.push(building);
    } else {
      building = {
        id: `blocker_${this.blockers.length}`,
        kind: this.selectedBuildable,
        pos,
        label: def.label,
        color: def.color,
        radius: def.radius,
        hp: def.hp,
        maxHp: def.hp,
        cost,
        blocksMovement: def.blocksMovement,
        destructible: def.destructible ?? true,
      };
      this.blockers.push(building);
    }
    this.resources.oil -= cost;
    this.statusMessage = `建造完成，花費 ${cost} 油量。`;
    if (this.phase === Phase.ATTACK && building.blocksMovement) {
      const [rerouted, blocked] = this.rerouteAntsAfterNewBlocker();
      if (blocked) this.statusMessage = `建造完成，${rerouted} 隻螞蟻重新尋路；${blocked} 隻被封路。`;
      else if (rerouted) this.statusMessage = `建造完成，${rerouted} 隻螞蟻重新尋路。`;
    }
  }

  rerouteAntsAfterNewBlocker() {
    let rerouted = 0;
    let blocked = 0;
    for (const ant of this.ants) {
      if (!ant.alive || ant.finished) continue;
      if (ant.state === AntState.TO_SWEET && ant.targetSweet && ant.targetSweet.units > 0) {
        if (this.rerouteAnt(ant, ant.targetSweet.pos)) rerouted += 1;
        else {
          this.triggerSoldierPenalty(ant.entry, ant.targetSweet);
          ant.finished = true;
          blocked += 1;
        }
      } else if (ant.state === AntState.LEAVING) {
        if (this.rerouteAnt(ant, this.nearestExit(ant.pos))) rerouted += 1;
        else ant.setPath([ant.pos.copy(), this.nearestExit(ant.pos)]);
      } else if (ant.state === AntState.SIEGE_LEAVING && this.rerouteAnt(ant, this.nearestExit(ant.pos))) {
        rerouted += 1;
      }
    }
    return [rerouted, blocked];
  }

  rerouteAnt(ant, target) {
    const path = this.pathfinder.findPath(ant.pos, target, this.currentBlockers(), ant.radius);
    if (!path.found) return false;
    ant.setPath(path.points);
    return true;
  }

  triggerSoldierPenalty(entry, sweet) {
    const settings = this.cfg.blockedPathPenalty;
    const active = this.ants.filter((ant) => ant.role === "siege" && ant.alive).length;
    if (this.penaltyCooldown > 0 || active >= settings.maxActivePenaltyAnts) {
      this.statusMessage = "工蟻路徑被封死，兵蟻懲罰冷卻中。";
      return;
    }
    const def = this.data.ants[settings.triggerAntType];
    this.ants.push(this.makeAnt(settings.triggerAntType, def, entry.copy(), AntState.SIEGE, entry.copy(), sweet, true, 0));
    this.penaltyCooldown = settings.cooldownSeconds;
    this.statusMessage = "路徑被封死：巨大兵蟻出現，開始拆除阻礙。";
  }

  spawnManualSoldier() {
    if (this.levelNumbers[this.levelIndex] !== 9) return;
    const entry = this.level.entries[this.ants.filter((ant) => ant.role === "siege").length % this.level.entries.length];
    this.spawnAnt({ antType: "soldier_ant", entryId: entry.id, count: 1, interval: 0 });
    this.statusMessage = "手動召喚 1 隻兵蟻。";
  }

  resolveAntCollisions() {
    const active = this.ants.filter((ant) => ant.alive && !ant.finished);
    for (let pass = 0; pass < 2; pass += 1) {
      for (let i = 0; i < active.length; i += 1) {
        const first = active[i];
        const firstRadius = this.antCollisionRadius(first);
        for (let j = i + 1; j < active.length; j += 1) {
          const second = active[j];
          const dx = second.pos.x - first.pos.x;
          const dy = second.pos.y - first.pos.y;
          const distance = Math.hypot(dx, dy);
          const minDistance = firstRadius + this.antCollisionRadius(second);
          if (distance >= minDistance) continue;
          const angle = distance ? Math.atan2(dy, dx) : (i * 1.73 + j * 2.41) % (Math.PI * 2);
          const nx = Math.cos(angle);
          const ny = Math.sin(angle);
          const push = (minDistance - distance) * 0.5;
          first.pos.x -= nx * push;
          first.pos.y -= ny * push;
          second.pos.x += nx * push;
          second.pos.y += ny * push;
          this.keepAntInWorld(first);
          this.keepAntInWorld(second);
        }
      }
    }
  }

  antCollisionRadius(ant) {
    return Math.max(5, ant.radius * (ant.role === "siege" ? 0.75 : 0.65));
  }

  keepAntInWorld(ant) {
    ant.pos.x = clamp(ant.pos.x, 0, this.worldW);
    ant.pos.y = clamp(ant.pos.y, 0, this.worldH);
  }

  draw() {
    const ctx = this.ctx;
    ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
    ctx.fillStyle = "#eee8dc";
    ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);
    this.drawWorld();
    this.drawUi();
  }

  drawUi() {
    const ctx = this.ctx;
    ctx.fillStyle = "#262b32";
    ctx.fillRect(0, 0, this.canvas.width, this.uiH);
    const waveName = this.waveIndex >= this.level.waves.length ? "完成" : this.level.waves[this.waveIndex].name;
    this.drawText(`關卡:${this.levelNumbers[this.levelIndex]} ${this.level.name}  階段:${this.phase}  波:${waveName}  油量:${this.resources.oil}  分數:${this.score}  甜食:${this.remainingSweetUnits()}/${this.totalSweetUnitsStart()}`, 12, 24, "#fff", "18px");
    this.drawText(`左鍵建造  Space開始  R重開  F1-F3/F9切關  | ${this.statusMessage}`, 12, 54, "#dbe1e8", "14px");
    this.drawBuildButtons();
  }

  drawBuildButtons() {
    this.layoutBuildButtonRects();
    for (const [id, rect] of this.buildButtonRects) {
      const [def] = this.buildDefinition(id);
      const selected = id === this.selectedBuildable;
      this.roundRect(rect.x, rect.y, rect.w, rect.h, 6, selected ? "#4e71a8" : "#373e48", selected ? "#f1d874" : "#7c8592", 2);
      const image = this.images[id];
      if (image) this.drawImageCentered(image, rect.x + 22, rect.y + rect.h / 2, 34, 34);
      else this.drawCircleLabel(rect.x + 22, rect.y + rect.h / 2, 14, rgb(def.color), def.label, "#fff");
      this.drawText(this.buildableName(id), rect.x + 44, rect.y + 18, "#fff", "14px");
      this.drawText(`可放:${this.affordableCount(id)}  油:${this.buildableCost(id)}`, rect.x + 44, rect.y + 38, "#e1e8ee", "14px");
    }
  }

  drawWorld() {
    const ctx = this.ctx;
    const grass = this.images.grass_background;
    if (grass) ctx.drawImage(grass, 0, this.uiH, this.worldW, this.worldH);
    else {
      ctx.fillStyle = "#87b064";
      ctx.fillRect(0, this.uiH, this.worldW, this.worldH);
    }
    const margin = this.cfg.world.edgeBuildMargin;
    ctx.strokeStyle = "#4e7340";
    ctx.lineWidth = 2;
    ctx.strokeRect(margin, this.uiH + margin, this.worldW - margin * 2, this.worldH - margin * 2);

    for (const entry of this.level.entries) this.drawEntryExitMarker(new Vec(entry.x, entry.y), "入", "#232323", true);
    for (const exit of this.exits()) this.drawEntryExitMarker(exit, "出", "#4c6041", false);
    for (const sweet of this.sweets) if (sweet.units > 0) this.drawSweet(sweet);

    const mouseWorld = this.screenToWorld(this.mouse);
    const mouseInWorld = this.mouse.y >= this.uiH && this.mouse.y <= this.uiH + this.worldH;
    const [, selectedIsTower] = this.buildDefinition(this.selectedBuildable);
    const hoveredTower = mouseInWorld ? this.hoveredTower(mouseWorld) : null;
    const placingTower = [Phase.PREP, Phase.ATTACK].includes(this.phase) && mouseInWorld && selectedIsTower && !hoveredTower;

    for (const tower of this.towers) {
      if (placingTower || tower === hoveredTower) this.drawContrastCircle(tower.pos, tower.attackRange);
      this.drawBuilding(tower);
    }
    for (const blocker of this.blockers) this.drawBuilding(blocker);
    for (const projectile of this.projectiles) this.drawCircle(projectile.pos, 4, "#1e1e1e");
    for (const ant of this.ants) {
      if (!ant.alive || ant.finished) continue;
      this.drawAnt(ant);
      this.drawHpBar(ant.pos, ant.radius, ant.hp, ant.maxHp);
    }
    if ([Phase.PREP, Phase.ATTACK].includes(this.phase) && mouseInWorld) this.drawBuildPreview(mouseWorld);
    if ([Phase.WIN, Phase.LOSE].includes(this.phase)) this.drawCenterOverlay();
  }

  drawEntryExitMarker(pos, label, color, entering) {
    const screen = this.worldToScreen(pos);
    this.ctx.strokeStyle = color;
    this.ctx.lineWidth = 2;
    this.ctx.beginPath();
    this.ctx.arc(screen.x, screen.y, 16, 0, Math.PI * 2);
    this.ctx.stroke();
    this.drawCenteredText(label, pos, color, "18px");
    let direction = this.edgeDirection(pos);
    if (!entering) direction = new Vec(-direction.x, -direction.y);
    this.drawArrow(new Vec(pos.x - direction.x * 26, pos.y - direction.y * 26), new Vec(pos.x + direction.x * 26, pos.y + direction.y * 26), color);
  }

  drawArrow(start, end, color) {
    const ctx = this.ctx;
    const a = this.worldToScreen(start);
    const b = this.worldToScreen(end);
    ctx.strokeStyle = color;
    ctx.lineWidth = 3;
    ctx.beginPath();
    ctx.moveTo(a.x, a.y);
    ctx.lineTo(b.x, b.y);
    ctx.stroke();
    const angle = Math.atan2(end.y - start.y, end.x - start.x);
    ctx.fillStyle = color;
    ctx.beginPath();
    ctx.moveTo(b.x, b.y);
    ctx.lineTo(b.x - Math.cos(angle - 0.55) * 10, b.y - Math.sin(angle - 0.55) * 10);
    ctx.lineTo(b.x - Math.cos(angle + 0.55) * 10, b.y - Math.sin(angle + 0.55) * 10);
    ctx.closePath();
    ctx.fill();
  }

  drawSweet(sweet) {
    const ratio = Math.max(0.35, sweet.units / Math.max(1, sweet.maxUnits));
    const radius = Math.max(8, sweet.maxRadius * (0.55 + 0.45 * ratio));
    if (!this.drawSprite(sweet.kind, sweet.pos, radius * 2.6)) this.drawCircleLabelAtWorld(sweet.pos, radius, rgb(sweet.color), `${sweet.label}${sweet.units}`);
    this.drawCenteredText(String(sweet.units), new Vec(sweet.pos.x, sweet.pos.y + radius + 12), "#281f1a", "14px");
  }

  drawBuilding(entity) {
    if (!this.drawSprite(entity.kind, entity.pos, entity.radius * 3.1)) this.drawCircleLabelAtWorld(entity.pos, entity.radius, rgb(entity.color), entity.label);
    this.drawHpBar(entity.pos, entity.radius, entity.hp, entity.maxHp);
  }

  drawAnt(ant) {
    const frame = 1 + (Math.floor(performance.now() / 180) % 2);
    const size = ant.radius * (ant.role === "siege" ? 3.4 : 4.0);
    if (!this.drawSprite(`${ant.kind}_${frame}`, ant.pos, size, -ant.facingAngle - Math.PI / 2)) {
      this.drawCircleLabelAtWorld(ant.pos, ant.radius, rgb(ant.color), ant.label);
    }
  }

  drawSprite(id, pos, size, rotation = 0) {
    const image = this.images[id];
    if (!image) return false;
    const screen = this.worldToScreen(pos);
    const ctx = this.ctx;
    ctx.save();
    ctx.translate(screen.x, screen.y);
    ctx.rotate(rotation);
    ctx.drawImage(image, -size / 2, -size / 2, size, size);
    ctx.restore();
    return true;
  }

  drawBuildPreview(pos) {
    const [def, isTower] = this.buildDefinition(this.selectedBuildable);
    const valid = this.inBuildableArea(pos) && !this.overlapsExisting(pos, def.radius) && this.resources.oil >= def.cost.oil;
    const image = this.images[this.selectedBuildable];
    const screen = this.worldToScreen(pos);
    this.ctx.save();
    this.ctx.globalAlpha = valid ? 0.9 : 0.35;
    if (image) this.ctx.drawImage(image, screen.x - def.radius * 1.55, screen.y - def.radius * 1.55, def.radius * 3.1, def.radius * 3.1);
    else this.drawCircle(pos, def.radius, rgb(def.color));
    this.ctx.restore();
    if (isTower) this.drawContrastCircle(pos, def.attack.range);
  }

  drawHpBar(pos, radius, hp, maxHp) {
    if (hp >= maxHp) return;
    const screen = this.worldToScreen(pos);
    const width = radius * 2;
    this.ctx.fillStyle = "#5a1e1e";
    this.ctx.fillRect(screen.x - radius, screen.y - radius - 8, width, 4);
    this.ctx.fillStyle = "#44b45b";
    this.ctx.fillRect(screen.x - radius, screen.y - radius - 8, width * Math.max(0, hp / maxHp), 4);
  }

  drawCenterOverlay() {
    this.roundRect(280, this.uiH + 220, 450, 140, 0, "#1e2228", "#fff", 2);
    const title = this.phase === Phase.WIN ? "勝利" : "失敗";
    this.drawText(`${title}  分數：${this.score}`, 310, this.uiH + 276, "#fff", "22px");
    this.drawText("按 R 重新開始。", 310, this.uiH + 318, "#e6e6e6", "18px");
  }

  currentBlockers() {
    return [...this.towers, ...this.blockers]
      .filter((b) => this.isAlive(b) && b.blocksMovement)
      .map((b) => ({ x: b.pos.x, y: b.pos.y, radius: b.radius, entity: b, distanceTo: Vec.prototype.distanceTo }));
  }

  pickTargetSweet(source) {
    const active = this.sweets.filter((sweet) => sweet.units > 0);
    active.sort((a, b) => source.distanceTo(a.pos) - source.distanceTo(b.pos));
    return active[0] || null;
  }

  entryById(entryId) {
    const entry = this.level.entries.find((item) => item.id === entryId);
    return new Vec(entry.x, entry.y);
  }

  exits() {
    return this.level.exits.map((item) => new Vec(item.x, item.y));
  }

  nearestExit(pos) {
    return this.exits().sort((a, b) => pos.distanceTo(a) - pos.distanceTo(b))[0];
  }

  totalSweetUnitsStart() {
    return this.level.sweets.reduce((total, item) => total + (item.units ?? this.data.sweets[item.type].units), 0);
  }

  remainingSweetUnits() {
    return this.sweets.reduce((total, sweet) => total + sweet.units, 0);
  }

  soldierDestroyGoal(antType) {
    const def = this.data.ants[antType];
    if (def.role !== "siege") return 0;
    return Number(this.level.soldierDestructionGoals?.[antType] ?? this.levelIndex + 1);
  }

  buildDefinition(id) {
    if (this.data.towers[id]) return [this.data.towers[id], true];
    return [this.data.blockers[id], false];
  }

  buildableIds() {
    return [...Object.keys(this.data.towers), ...Object.keys(this.data.blockers)].filter((id) => this.buildableCost(id) > 0);
  }

  buildableName(id) {
    const [def] = this.buildDefinition(id);
    return def.name || def.label;
  }

  buildableCost(id) {
    const [def] = this.buildDefinition(id);
    return def.cost?.oil || 0;
  }

  affordableCount(id) {
    const cost = this.buildableCost(id);
    return cost > 0 ? Math.floor(this.resources.oil / cost) : 0;
  }

  layoutBuildButtonRects() {
    this.buildButtonRects.clear();
    const ids = this.buildableIds();
    const buttonW = 132;
    const buttonH = 48;
    const gap = 10;
    let x = this.canvas.width - ids.length * buttonW - (ids.length - 1) * gap - 12;
    for (const id of ids) {
      this.buildButtonRects.set(id, { x, y: 12, w: buttonW, h: buttonH });
      x += buttonW + gap;
    }
  }

  handleBuildButtonClick(pos) {
    this.layoutBuildButtonRects();
    for (const [id, rect] of this.buildButtonRects) {
      if (pos.x >= rect.x && pos.x <= rect.x + rect.w && pos.y >= rect.y && pos.y <= rect.y + rect.h) {
        this.selectedBuildable = id;
        this.statusMessage = `已選：${this.buildableName(id)}`;
        return true;
      }
    }
    return false;
  }

  inBuildableArea(pos) {
    const margin = this.cfg.world.edgeBuildMargin;
    return pos.x >= margin && pos.x <= this.worldW - margin && pos.y >= margin && pos.y <= this.worldH - margin;
  }

  overlapsExisting(pos, radius) {
    const padding = this.cfg.world.placementPadding || 0;
    for (const entity of [...this.towers, ...this.blockers, ...this.sweets, ...this.ants]) {
      if (entity.alive === false || entity.finished || entity.units === 0) continue;
      if (pos.distanceTo(entity.pos) < radius + entity.radius + padding) return true;
    }
    return false;
  }

  hoveredTower(pos) {
    return [...this.towers].reverse().find((tower) => tower.pos.distanceTo(pos) <= tower.radius) || null;
  }

  edgeDirection(pos) {
    if (pos.x <= 0) return new Vec(1, 0);
    if (pos.x >= this.worldW - 1) return new Vec(-1, 0);
    if (pos.y <= 0) return new Vec(0, 1);
    if (pos.y >= this.worldH - 1) return new Vec(0, -1);
    const center = new Vec(this.worldW / 2, this.worldH / 2);
    const dx = center.x - pos.x;
    const dy = center.y - pos.y;
    const dist = Math.hypot(dx, dy) || 1;
    return new Vec(dx / dist, dy / dist);
  }

  isAlive(entity) {
    return !entity.destructible || entity.hp > 0;
  }

  worldToScreen(pos) {
    return new Vec(pos.x, pos.y + this.uiH);
  }

  screenToWorld(pos) {
    return new Vec(pos.x, pos.y - this.uiH);
  }

  drawText(text, x, y, color, size) {
    this.ctx.fillStyle = color;
    this.ctx.font = `${size} system-ui, -apple-system, "Noto Sans TC", sans-serif`;
    this.ctx.textBaseline = "alphabetic";
    this.ctx.fillText(text, x, y);
  }

  drawCenteredText(text, pos, color, size) {
    const screen = this.worldToScreen(pos);
    this.ctx.fillStyle = color;
    this.ctx.font = `${size} system-ui, -apple-system, "Noto Sans TC", sans-serif`;
    this.ctx.textAlign = "center";
    this.ctx.textBaseline = "middle";
    this.ctx.fillText(text, screen.x, screen.y);
    this.ctx.textAlign = "start";
    this.ctx.textBaseline = "alphabetic";
  }

  drawCircle(pos, radius, color) {
    const screen = this.worldToScreen(pos);
    this.ctx.fillStyle = color;
    this.ctx.beginPath();
    this.ctx.arc(screen.x, screen.y, radius, 0, Math.PI * 2);
    this.ctx.fill();
  }

  drawCircleLabelAtWorld(pos, radius, color, label) {
    const screen = this.worldToScreen(pos);
    this.drawCircleLabel(screen.x, screen.y, radius, color, label, "#fff");
  }

  drawCircleLabel(x, y, radius, color, label, labelColor) {
    this.ctx.fillStyle = color;
    this.ctx.beginPath();
    this.ctx.arc(x, y, radius, 0, Math.PI * 2);
    this.ctx.fill();
    this.ctx.strokeStyle = "#202020";
    this.ctx.lineWidth = 2;
    this.ctx.stroke();
    this.ctx.fillStyle = labelColor;
    this.ctx.font = "14px system-ui, -apple-system, 'Noto Sans TC', sans-serif";
    this.ctx.textAlign = "center";
    this.ctx.textBaseline = "middle";
    this.ctx.fillText(label, x, y);
    this.ctx.textAlign = "start";
    this.ctx.textBaseline = "alphabetic";
  }

  drawContrastCircle(pos, radius) {
    const screen = this.worldToScreen(pos);
    this.ctx.strokeStyle = "#fff";
    this.ctx.lineWidth = 2;
    this.ctx.beginPath();
    this.ctx.arc(screen.x, screen.y, radius, 0, Math.PI * 2);
    this.ctx.stroke();
  }

  drawImageCentered(image, x, y, w, h) {
    this.ctx.drawImage(image, x - w / 2, y - h / 2, w, h);
  }

  roundRect(x, y, w, h, r, fill, stroke, lineWidth) {
    const ctx = this.ctx;
    ctx.beginPath();
    ctx.moveTo(x + r, y);
    ctx.arcTo(x + w, y, x + w, y + h, r);
    ctx.arcTo(x + w, y + h, x, y + h, r);
    ctx.arcTo(x, y + h, x, y, r);
    ctx.arcTo(x, y, x + w, y, r);
    ctx.closePath();
    ctx.fillStyle = fill;
    ctx.fill();
    if (stroke) {
      ctx.strokeStyle = stroke;
      ctx.lineWidth = lineWidth;
      ctx.stroke();
    }
  }
}

function rgb(color) {
  return `rgb(${color[0]}, ${color[1]}, ${color[2]})`;
}

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

function randFloat(min, max) {
  return min + Math.random() * (max - min);
}

function seededRandom(seed) {
  let state = seed >>> 0;
  return () => {
    state = (state * 1664525 + 1013904223) >>> 0;
    return state / 0x100000000;
  };
}

function randInt(random, min, max) {
  return Math.floor(random() * (max - min + 1)) + min;
}

async function loadJson(path) {
  const response = await fetch(path);
  if (!response.ok) throw new Error(`Cannot load ${path}`);
  return response.json();
}

async function loadImage(path) {
  return new Promise((resolve) => {
    const image = new Image();
    image.onload = () => resolve(image);
    image.onerror = () => resolve(null);
    image.src = path;
  });
}

async function boot() {
  const canvas = document.getElementById("game");
  const [config, ants, sweets, towers, blockers, levelPairs, imagePairs] = await Promise.all([
    loadJson(DATA_PATHS.config),
    loadJson(DATA_PATHS.ants),
    loadJson(DATA_PATHS.sweets),
    loadJson(DATA_PATHS.towers),
    loadJson(DATA_PATHS.blockers),
    Promise.all(DATA_PATHS.levels.map(async ([number, path]) => ({ number: Number(number), data: await loadJson(path) }))),
    Promise.all(Object.entries(IMAGE_PATHS).map(async ([id, path]) => [id, await loadImage(path)])),
  ]);
  const images = Object.fromEntries(imagePairs.filter(([, image]) => image));
  const game = new Game(canvas, { config, ants, sweets, towers, blockers, levels: levelPairs }, images);
  let last = performance.now();
  function frame(now) {
    const dt = Math.min(0.05, (now - last) / 1000);
    last = now;
    game.update(dt);
    game.draw();
    requestAnimationFrame(frame);
  }
  requestAnimationFrame(frame);
}

boot().catch((error) => {
  const canvas = document.getElementById("game");
  const ctx = canvas.getContext("2d");
  ctx.fillStyle = "#20242a";
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  ctx.fillStyle = "#fff";
  ctx.font = "20px system-ui, sans-serif";
  ctx.fillText("載入遊戲資料失敗，請用本機 HTTP server 開啟。", 230, 330);
  ctx.fillText(String(error.message || error), 230, 365);
});
