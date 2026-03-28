"""
TRON LIGHT CYCLE GAME  (Turtle Edition)
========================================
AI: A* Search + Flood-Fill survival fallback
Controls: Arrow Keys / WASD, R = restart, ESC/Q = quit
Mode select at startup: 1 = single opponent, 2 = four opponents
No external dependencies — pure Python standard library.
"""

import turtle
import heapq
import sys
import time
from dataclasses import dataclass, field
from typing import List, Tuple

# ──────────────────────────────────────────────────────────
# CONSTANTS
# ──────────────────────────────────────────────────────────
CELL   = 14
COLS   = 55
ROWS   = 38
WIDTH  = COLS * CELL
HEIGHT = ROWS * CELL
FPS    = 10

BG_COLOR     = "#020410"
PLAYER_COLOR = "#00C8FF"
PLAYER_TRAIL = "#004A70"

ENEMY_COLORS = [
    ("#FF3250", "#7A0F1E"),
    ("#FFB800", "#7A5500"),
    ("#A040FF", "#4A1880"),
    ("#00FF90", "#007040"),
]

TEXT_COLOR = "#C8F0FF"
WIN_COLOR  = "#00FF96"
LOSE_COLOR = "#FF3C3C"

UP    = ( 0,  1);  DOWN  = ( 0, -1)
LEFT  = (-1,  0);  RIGHT = ( 1,  0)
DIRS  = [UP, DOWN, LEFT, RIGHT]

def opposite(d): return (-d[0], -d[1])

def g2s(gx, gy):
    return gx * CELL - WIDTH//2 + CELL//2, gy * CELL - HEIGHT//2 + CELL//2

# ──────────────────────────────────────────────────────────
# A*
# ──────────────────────────────────────────────────────────
def heuristic(a, b): return abs(a[0]-b[0]) + abs(a[1]-b[1])

def astar(start, goal, blocked):
    h = [(heuristic(start, goal), 0, start, [start])]
    closed = set()
    while h:
        f, g, cur, path = heapq.heappop(h)
        if cur == goal: return path
        if cur in closed: continue
        closed.add(cur)
        for dx, dy in DIRS:
            nb = (cur[0]+dx, cur[1]+dy)
            if nb in closed or nb in blocked: continue
            if not (0 <= nb[0] < COLS and 0 <= nb[1] < ROWS): continue
            ng = g + 1
            heapq.heappush(h, (ng + heuristic(nb, goal), ng, nb, path+[nb]))
    return None

# ──────────────────────────────────────────────────────────
# FLOOD FILL
# ──────────────────────────────────────────────────────────
def flood_fill_count(start, blocked):
    visited, q = {start}, [start]
    while q:
        x, y = q.pop()
        for dx, dy in DIRS:
            nb = (x+dx, y+dy)
            if nb not in visited and nb not in blocked \
               and 0 <= nb[0] < COLS and 0 <= nb[1] < ROWS:
                visited.add(nb); q.append(nb)
    return len(visited)

# ──────────────────────────────────────────────────────────
# CYCLE
# ──────────────────────────────────────────────────────────
@dataclass
class Cycle:
    pos: Tuple[int,int]
    direction: Tuple[int,int]
    trail: set = field(default_factory=set)
    alive: bool = True

    def __post_init__(self): self.trail.add(self.pos)
    def next_pos(self, d=None):
        d = d or self.direction
        return (self.pos[0]+d[0], self.pos[1]+d[1])
    def move(self):
        self.pos = self.next_pos(); self.trail.add(self.pos)
    def in_bounds(self, p=None):
        x, y = p or self.pos
        return 0 <= x < COLS and 0 <= y < ROWS

# ──────────────────────────────────────────────────────────
# ENEMY AGENT
# ──────────────────────────────────────────────────────────
class EnemyAgent:
    REPLAN = 3
    def __init__(self, enemy, player, all_trails_fn):
        self.enemy = enemy; self.player = player
        self.all_trails_fn = all_trails_fn
        self.plan = []; self.ticks = 0

    def _predict(self, steps=5):
        px, py = self.player.pos; dx, dy = self.player.direction
        for _ in range(steps):
            nx, ny = px+dx, py+dy
            if 0 <= nx < COLS and 0 <= ny < ROWS: px, py = nx, ny
        return (px, py)

    def decide(self):
        self.ticks += 1
        blocked = self.all_trails_fn()
        if self.ticks % self.REPLAN == 0 or not self.plan:
            path = astar(self.enemy.pos, self._predict(), blocked) \
                or astar(self.enemy.pos, self.player.pos, blocked)
            self.plan = path[1:] if path and len(path) > 1 else []
        if self.plan:
            nxt = self.plan.pop(0)
            d = (nxt[0]-self.enemy.pos[0], nxt[1]-self.enemy.pos[1])
            np_ = self.enemy.next_pos(d)
            if np_ not in blocked and self.enemy.in_bounds(np_):
                return d
        return self._survival(blocked)

    def _survival(self, blocked):
        best_d, best_s = self.enemy.direction, -1
        opp = opposite(self.enemy.direction)
        for d in DIRS:
            if d == opp: continue
            nx, ny = self.enemy.next_pos(d)
            if not (0 <= nx < COLS and 0 <= ny < ROWS): continue
            if (nx, ny) in blocked: continue
            s = flood_fill_count((nx, ny), blocked | {(nx, ny)})
            if s > best_s: best_s, best_d = s, d
        return best_d

# ──────────────────────────────────────────────────────────
# STAMP-BASED CELL PAINTER  (delta — only draws NEW cells)
# ──────────────────────────────────────────────────────────
class CellPainter:
    """
    Turtle stamps are fast rectangles.
    We track every painted cell and only stamp NEW ones,
    so the rendering cost is O(new cells per tick) not O(total cells).
    """
    def __init__(self, screen):
        self.t = turtle.RawTurtle(screen)
        self.t.shape("square")
        self.t.shapesize(CELL / 20)   # built-in square shape is 20 px
        self.t.penup()
        self.t.speed(0)
        self.t.hideturtle()
        self._stamps: dict = {}        # (gx,gy) -> stamp_id

    def paint(self, gx, gy, color):
        """Paint a cell; if already painted, skip."""
        if (gx, gy) in self._stamps:
            return
        self._do_stamp(gx, gy, color)

    def repaint(self, gx, gy, color):
        """Erase existing stamp for cell then repaint (head → trail)."""
        sid = self._stamps.pop((gx, gy), None)
        if sid is not None:
            self.t.clearstamp(sid)
        self._do_stamp(gx, gy, color)

    def _do_stamp(self, gx, gy, color):
        x, y = g2s(gx, gy)
        self.t.goto(x, y)
        self.t.fillcolor(color)
        self.t.color(color)
        self.t.showturtle()
        sid = self.t.stamp()
        self.t.hideturtle()
        self._stamps[(gx, gy)] = sid

    def clear_all(self):
        self.t.clearstamps()
        self._stamps.clear()


# ──────────────────────────────────────────────────────────
# GAME
# ──────────────────────────────────────────────────────────
class TronGame:
    STARTS = [
        (8,        ROWS//2,   RIGHT),
        (COLS-8,   ROWS//2,   LEFT),
        (COLS//2,  4,         DOWN),
        (4,        4,         RIGHT),
        (COLS-4,   ROWS-4,    LEFT),
    ]

    def __init__(self):
        self.sc = turtle.Screen()
        self.sc.setup(WIDTH + 40, HEIGHT + 60)
        self.sc.title("TRON · Light Cycle · A* AI")
        self.sc.bgcolor(BG_COLOR)
        self.sc.tracer(0, 0)

        self.txt = turtle.RawTurtle(self.sc)
        self.txt.hideturtle(); self.txt.penup(); self.txt.speed(0)

        self.mode = self._choose_mode()
        self.num_enemies = 1 if self.mode == 1 else 4
        self.wins = 0
        self._running = True
        self._need_reset = False

        self.painter = CellPainter(self.sc)
        self._setup_keys()
        self.reset()
        self._loop()

    # ── mode select ──────────────────────────────────────
    def _choose_mode(self):
        self.sc.clear()
        self.sc.bgcolor(BG_COLOR)
        self.sc.tracer(0, 0)
        t = turtle.RawTurtle(self.sc)
        t.hideturtle(); t.penup(); t.speed(0)

        def wr(msg, y, color=TEXT_COLOR, sz=18):
            t.goto(0, y); t.color(color)
            t.write(msg, align="center", font=("Courier", sz, "bold"))

        wr("TRON  LIGHT  CYCLE",   90, "#00C8FF", 28)
        wr("Select Mode",           20, TEXT_COLOR, 16)
        wr("1  ─  Single Opponent", -20, "#FF3250", 18)
        wr("2  ─  Four  Opponents", -60, "#FFB800", 18)
        wr("ESC / Q  to quit",     -110, "#334455", 13)
        self.sc.update()

        chosen = [None]
        def k1(): chosen[0] = 1
        def k2(): chosen[0] = 2
        def kq(): turtle.bye(); sys.exit()

        self.sc.onkeypress(k1, "1"); self.sc.onkeypress(k2, "2")
        self.sc.onkeypress(kq, "Escape"); self.sc.onkeypress(kq, "q")
        self.sc.listen()
        while chosen[0] is None:
            self.sc.update(); time.sleep(0.05)

        t.clear()
        return chosen[0]

    # ── key bindings ─────────────────────────────────────
    def _setup_keys(self):
        sc = self.sc
        sc.onkeypress(lambda: self._turn(UP),    "Up")
        sc.onkeypress(lambda: self._turn(UP),    "w")
        sc.onkeypress(lambda: self._turn(DOWN),  "Down")
        sc.onkeypress(lambda: self._turn(DOWN),  "s")
        sc.onkeypress(lambda: self._turn(LEFT),  "Left")
        sc.onkeypress(lambda: self._turn(LEFT),  "a")
        sc.onkeypress(lambda: self._turn(RIGHT), "Right")
        sc.onkeypress(lambda: self._turn(RIGHT), "d")
        sc.onkeypress(self._restart, "r")
        sc.onkeypress(self._quit,    "Escape")
        sc.onkeypress(self._quit,    "q")
        sc.listen()

    def _turn(self, d):
        if self.state == "playing" and d != opposite(self.player.direction):
            self.player.direction = d

    def _restart(self):
        if self.state != "playing":
            self._need_reset = True

    def _quit(self):
        self._running = False
        try: turtle.bye()
        except: pass
        sys.exit()

    # ── reset ────────────────────────────────────────────
    def reset(self):
        self.painter.clear_all()
        self.txt.clear()

        col, row, d = self.STARTS[0]
        self.player = Cycle(pos=(col, row), direction=d)

        self.enemies: List[Cycle] = []
        self.agents:  List[EnemyAgent] = []

        def all_trails():
            b = set(self.player.trail)
            for e in self.enemies: b |= e.trail
            return b

        for i in range(self.num_enemies):
            c, r, di = self.STARTS[i+1]
            e = Cycle(pos=(c, r), direction=di)
            self.enemies.append(e)
            self.agents.append(EnemyAgent(e, self.player, all_trails))

        self.state = "playing"
        self.frame = 0

        # Paint starting heads
        self.painter.paint(*self.player.pos, PLAYER_COLOR)
        for i, e in enumerate(self.enemies):
            self.painter.paint(*e.pos, ENEMY_COLORS[i % 4][0])

        self._draw_hud()
        self.sc.update()

    # ── update ───────────────────────────────────────────
    def update(self):
        if self.state != "playing":
            return

        live = [(e, a, i) for i, (e, a) in
                enumerate(zip(self.enemies, self.agents)) if e.alive]

        for e, a, _ in live:
            e.direction = a.decide()

        blocked = set(self.player.trail)
        for e in self.enemies: blocked |= e.trail

        p_nxt = self.player.next_pos()
        if not self.player.in_bounds(p_nxt) or p_nxt in blocked:
            self.state = "lose"; return

        for e, a, i in live:
            e_nxt = e.next_pos()
            if not e.in_bounds(e_nxt) or e_nxt in blocked or e_nxt == p_nxt:
                e.alive = False
            else:
                self.painter.repaint(*e.pos, ENEMY_COLORS[i % 4][1])  # old head → trail
                e.move()
                self.painter.paint(*e.pos, ENEMY_COLORS[i % 4][0])    # new head

        alive_pos = {e.pos for e in self.enemies if e.alive}
        if p_nxt in alive_pos:
            self.state = "lose"; return

        self.painter.repaint(*self.player.pos, PLAYER_TRAIL)  # old head → trail
        self.player.move()
        self.painter.paint(*self.player.pos, PLAYER_COLOR)    # new head
        self.frame += 1

        if all(not e.alive for e in self.enemies):
            self.state = "win"; self.wins += 1

    # ── HUD ──────────────────────────────────────────────
    def _draw_hud(self):
        t = self.txt
        t.clear()

        t.goto(-WIDTH//2 + 6, HEIGHT//2 - 20)
        t.color("#00C8FF")
        t.write("TRON  LIGHT  CYCLE", font=("Courier", 13, "bold"))

        t.goto(WIDTH//2 - 5, HEIGHT//2 - 20)
        t.color(TEXT_COLOR)
        t.write(f"WINS:{self.wins}  F:{self.frame}",
                align="right", font=("Courier", 11, "normal"))

        mode_str = "1 opponent" if self.num_enemies == 1 else f"{self.num_enemies} opponents"
        t.goto(-WIDTH//2 + 6, -HEIGHT//2 + 4)
        t.color("#334455")
        t.write(f"{mode_str} | WASD/Arrows=move  R=restart  ESC=quit",
                font=("Courier", 9, "normal"))

        if self.state == "win":
            t.goto(0, 20); t.color(WIN_COLOR)
            t.write("  YOU  WIN!", align="center", font=("Courier", 34, "bold"))
            t.goto(0, -30); t.color(TEXT_COLOR)
            t.write("Press  R  to restart", align="center",
                    font=("Courier", 16, "normal"))
        elif self.state == "lose":
            t.goto(0, 20); t.color(LOSE_COLOR)
            t.write("  YOU  LOSE!", align="center", font=("Courier", 34, "bold"))
            t.goto(0, -30); t.color(TEXT_COLOR)
            t.write("Press  R  to restart", align="center",
                    font=("Courier", 16, "normal"))

    # ── main loop ────────────────────────────────────────
    def _loop(self):
        interval = 1.0 / FPS
        while self._running:
            t0 = time.perf_counter()

            if self._need_reset:
                self._need_reset = False
                self.reset()
            else:
                self.update()
                self._draw_hud()
                self.sc.update()
                self.sc.listen()

            time.sleep(max(0.0, interval - (time.perf_counter() - t0)))


if __name__ == "__main__":
    TronGame()
