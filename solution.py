from __future__ import annotations

import heapq
import math
import time
from typing import List, Tuple

from player import Player
from board import HexBoard


class SmartPlayer(Player):
    INF = 10**9
    WIN_SCORE = 10**6

    def __init__(self, player_id: int):
        super().__init__(player_id)
        self._deadline = 0.0

    def play(self, board: HexBoard) -> tuple:
        matrix = [row[:] for row in board.board]
        n = len(matrix)
        empties = self._empty_cells(matrix)

        if not empties:
            return (0, 0)

        if len(empties) == n * n:
            return self._best_opening_move(matrix)

        opponent = self._opponent(self.player_id)

        winning = self._find_immediate_win(matrix, self.player_id)
        if winning is not None:
            return winning

        block = self._find_immediate_win(matrix, opponent)
        if block is not None:
            return block

        self._deadline = time.perf_counter() + 4.2

        empty_count = len(empties)
        if empty_count <= 8:
            depth, max_candidates = 4, 8
        elif empty_count <= 16:
            depth, max_candidates = 3, 10
        else:
            depth, max_candidates = 2, 12

        candidates = self._generate_candidates(matrix, self.player_id, max_candidates=max_candidates)
        if not candidates:
            return self._closest_to_center(empties, n)

        best_move = candidates[0]
        best_value = -math.inf
        alpha = -math.inf
        beta = math.inf

        for r, c in candidates:
            if time.perf_counter() >= self._deadline:
                break
            matrix[r][c] = self.player_id
            value = self._alphabeta(
                matrix=matrix,
                depth=depth - 1,
                current_player=opponent,
                root_player=self.player_id,
                alpha=alpha,
                beta=beta,
            )
            matrix[r][c] = 0

            if value > best_value:
                best_value = value
                best_move = (r, c)
            if value > alpha:
                alpha = value

        return best_move

    def _alphabeta(
        self,
        matrix: List[List[int]],
        depth: int,
        current_player: int,
        root_player: int,
        alpha: float,
        beta: float,
    ) -> float:
        if time.perf_counter() >= self._deadline:
            return self._evaluate(matrix, root_player)

        if self._has_connection(matrix, root_player):
            return self.WIN_SCORE
        if self._has_connection(matrix, self._opponent(root_player)):
            return -self.WIN_SCORE

        empties = self._empty_cells(matrix)
        if depth == 0 or not empties:
            return self._evaluate(matrix, root_player)

        candidates = self._generate_candidates(
            matrix,
            current_player,
            max_candidates=8 if depth >= 2 else 10,
        )
        if not candidates:
            return self._evaluate(matrix, root_player)

        next_player = self._opponent(current_player)

        if current_player == root_player:
            value = -math.inf
            for r, c in candidates:
                matrix[r][c] = current_player
                child = self._alphabeta(
                    matrix=matrix,
                    depth=depth - 1,
                    current_player=next_player,
                    root_player=root_player,
                    alpha=alpha,
                    beta=beta,
                )
                matrix[r][c] = 0

                if child > value:
                    value = child
                if child > alpha:
                    alpha = child
                if alpha >= beta or time.perf_counter() >= self._deadline:
                    break
            return value

        value = math.inf
        for r, c in candidates:
            matrix[r][c] = current_player
            child = self._alphabeta(
                matrix=matrix,
                depth=depth - 1,
                current_player=next_player,
                root_player=root_player,
                alpha=alpha,
                beta=beta,
            )
            matrix[r][c] = 0

            if child < value:
                value = child
            if child < beta:
                beta = child
            if alpha >= beta or time.perf_counter() >= self._deadline:
                break
        return value

    def _evaluate(self, matrix: List[List[int]], root_player: int) -> float:
        opponent = self._opponent(root_player)

        if self._has_connection(matrix, root_player):
            return self.WIN_SCORE
        if self._has_connection(matrix, opponent):
            return -self.WIN_SCORE

        my_cost = self._connection_cost(matrix, root_player)
        opp_cost = self._connection_cost(matrix, opponent)

        score = 35.0 * (opp_cost - my_cost)
        score += 4.0 * self._friendly_adjacency_balance(matrix, root_player)
        score -= 4.0 * self._friendly_adjacency_balance(matrix, opponent)
        score += 2.0 * self._central_presence(matrix, root_player)
        score -= 2.0 * self._central_presence(matrix, opponent)

        return score

    def _generate_candidates(
        self,
        matrix: List[List[int]],
        player_id: int,
        max_candidates: int,
    ) -> List[Tuple[int, int]]:
        empties = self._empty_cells(matrix)
        if len(empties) <= max_candidates:
            return empties

        n = len(matrix)
        opponent = self._opponent(player_id)
        scored = []

        for r, c in empties:
            score = 0.0
            neighbor_values = []

            for nr, nc in self._neighbors(r, c, n):
                val = matrix[nr][nc]
                neighbor_values.append(val)
                if val == player_id:
                    score += 12.0
                elif val == opponent:
                    score += 8.0

            if player_id in neighbor_values and opponent in neighbor_values:
                score += 6.0

            score -= 1.5 * self._center_distance(r, c, n)
            score += self._cell_corridor_bonus(matrix, r, c, player_id)
            score += 0.8 * self._cell_corridor_bonus(matrix, r, c, opponent)

            scored.append((score, (r, c)))

        scored.sort(key=lambda item: item[0], reverse=True)
        return [move for _, move in scored[:max_candidates]]

    def _cell_corridor_bonus(self, matrix: List[List[int]], row: int, col: int, player_id: int) -> float:
        n = len(matrix)
        if player_id == 1:
            distance_to_goal_sides = min(col, (n - 1) - col)
        else:
            distance_to_goal_sides = min(row, (n - 1) - row)

        bonus = (n - 1) - distance_to_goal_sides

        friendly_neighbors = 0
        for nr, nc in self._neighbors(row, col, n):
            if matrix[nr][nc] == player_id:
                friendly_neighbors += 1

        return 1.5 * bonus + 2.0 * friendly_neighbors

    def _friendly_adjacency_balance(self, matrix: List[List[int]], player_id: int) -> int:
        n = len(matrix)
        total = 0
        for r in range(n):
            for c in range(n):
                if matrix[r][c] != player_id:
                    continue
                for nr, nc in self._neighbors(r, c, n):
                    if matrix[nr][nc] == player_id:
                        total += 1
        return total // 2

    def _central_presence(self, matrix: List[List[int]], player_id: int) -> float:
        n = len(matrix)
        total = 0.0
        for r in range(n):
            for c in range(n):
                if matrix[r][c] == player_id:
                    total -= self._center_distance(r, c, n)
        return total

    def _find_immediate_win(self, matrix: List[List[int]], player_id: int):
        for r, c in self._empty_cells(matrix):
            matrix[r][c] = player_id
            wins = self._has_connection(matrix, player_id)
            matrix[r][c] = 0
            if wins:
                return (r, c)
        return None

    def _has_connection(self, matrix: List[List[int]], player_id: int) -> bool:
        n = len(matrix)
        stack = []
        visited = set()

        if player_id == 1:
            for r in range(n):
                if matrix[r][0] == player_id:
                    stack.append((r, 0))
                    visited.add((r, 0))
            target = lambda rr, cc: cc == n - 1
        else:
            for c in range(n):
                if matrix[0][c] == player_id:
                    stack.append((0, c))
                    visited.add((0, c))
            target = lambda rr, cc: rr == n - 1

        while stack:
            r, c = stack.pop()
            if target(r, c):
                return True
            for nr, nc in self._neighbors(r, c, n):
                if matrix[nr][nc] == player_id and (nr, nc) not in visited:
                    visited.add((nr, nc))
                    stack.append((nr, nc))
        return False

    def _connection_cost(self, matrix: List[List[int]], player_id: int) -> int:
        n = len(matrix)
        dist = [[self.INF] * n for _ in range(n)]
        heap = []

        def cell_cost(value: int) -> int:
            if value == player_id:
                return 0
            if value == 0:
                return 1
            return self.INF

        if player_id == 1:
            for r in range(n):
                cost = cell_cost(matrix[r][0])
                if cost < self.INF:
                    dist[r][0] = cost
                    heapq.heappush(heap, (cost, r, 0))
            is_goal = lambda rr, cc: cc == n - 1
        else:
            for c in range(n):
                cost = cell_cost(matrix[0][c])
                if cost < self.INF:
                    dist[0][c] = cost
                    heapq.heappush(heap, (cost, 0, c))
            is_goal = lambda rr, cc: rr == n - 1

        best_goal = self.INF

        while heap:
            d, r, c = heapq.heappop(heap)
            if d != dist[r][c]:
                continue
            if d >= best_goal:
                continue
            if is_goal(r, c):
                best_goal = d
                continue

            for nr, nc in self._neighbors(r, c, n):
                step = cell_cost(matrix[nr][nc])
                if step >= self.INF:
                    continue
                nd = d + step
                if nd < dist[nr][nc]:
                    dist[nr][nc] = nd
                    heapq.heappush(heap, (nd, nr, nc))

        return best_goal if best_goal < self.INF else n * n

    def _neighbors(self, row: int, col: int, size: int):
        if row % 2 == 0:
            deltas = [(-1, -1), (-1, 0), (0, -1), (0, 1), (1, -1), (1, 0)]
        else:
            deltas = [(-1, 0), (-1, 1), (0, -1), (0, 1), (1, 0), (1, 1)]

        for dr, dc in deltas:
            nr, nc = row + dr, col + dc
            if 0 <= nr < size and 0 <= nc < size:
                yield nr, nc

    def _empty_cells(self, matrix: List[List[int]]) -> List[Tuple[int, int]]:
        empties = []
        n = len(matrix)
        for r in range(n):
            for c in range(n):
                if matrix[r][c] == 0:
                    empties.append((r, c))
        return empties

    def _center_distance(self, row: int, col: int, size: int) -> float:
        center = (size - 1) / 2.0
        return abs(row - center) + abs(col - center)

    def _best_opening_move(self, matrix: List[List[int]]) -> Tuple[int, int]:
        n = len(matrix)
        return self._closest_to_center(self._empty_cells(matrix), n)

    def _closest_to_center(self, cells: List[Tuple[int, int]], size: int) -> Tuple[int, int]:
        return min(
            cells,
            key=lambda pos: (self._center_distance(pos[0], pos[1], size), pos[0], pos[1]),
        )

    def _opponent(self, player_id: int) -> int:
        return 2 if player_id == 1 else 1
