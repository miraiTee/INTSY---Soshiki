import os
import random
import time
from typing import Optional, Tuple, Any, Type
from abc import ABC, abstractmethod
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame


class Cat(ABC):
    def __init__(self, grid_size: int, tile_size: int):
        self.grid_size = grid_size
        self.tile_size = tile_size
        self.pos = np.zeros(2, dtype=np.int32)
        self.visual_pos = np.zeros(2, dtype=float)
        
        self.player_pos = np.zeros(2, dtype=np.int32)
        self.prev_player_pos = np.zeros(2, dtype=np.int32)
        self.last_player_action = None
        
        self.current_distance = 0  
        self.prev_distance = 0     
        
        self._load_sprite()
    
    @abstractmethod
    def _get_sprite_path(self) -> str:
        pass
    
    def _load_sprite(self):
        img_path = self._get_sprite_path()
        if not os.path.exists(img_path):
            self.sprite = pygame.Surface((self.tile_size, self.tile_size))
            self.sprite.fill((200, 100, 100))
            return
        try:
            self.sprite = pygame.image.load(img_path)
            self.sprite = self.sprite.convert_alpha()
            self.sprite = pygame.transform.scale(self.sprite, (self.tile_size, self.tile_size))
        except Exception as e:
            self.sprite = pygame.Surface((self.tile_size, self.tile_size))
            self.sprite.fill((200, 100, 100))
    
    def update_player_info(self, player_pos: np.ndarray, player_action: int) -> None:
        self.prev_player_pos = self.player_pos.copy()
        self.player_pos = player_pos.copy()
        self.last_player_action = player_action

        self.prev_distance = abs(self.pos[0] - self.prev_player_pos[0]) + abs(self.pos[1] - self.prev_player_pos[1])
        self.current_distance = abs(self.pos[0] - self.player_pos[0]) + abs(self.pos[1] - self.player_pos[1])
    
    def player_moved_closer(self) -> bool:
        return self.current_distance < self.prev_distance
    
    @abstractmethod
    def move(self) -> None:
        pass
    
    def reset(self, pos: np.ndarray) -> None:
        self.pos = pos.copy()
        self.visual_pos = pos.astype(float)
    
    def update_visual_pos(self, dt: float, animation_speed: float) -> None:
        for i in range(2):
            diff = self.pos[i] - self.visual_pos[i]
            if abs(diff) > 0.01:
                self.visual_pos[i] += np.clip(diff * animation_speed * dt, -1, 1)

####################################
# CATT BEHAVIOR IMPLEMENTATIONS    #
####################################

### A predictable catt for testing manhattan distance as a bum
### should int at turn 10.
class TrainCatt(Cat):
    def _get_sprite_path(self) -> str:
        return "images/traincatt.jpg"

    def __init__(self, grid_size: int, tile_size: int):
        super().__init__(grid_size, tile_size)
        # Train route through the four corner stations (Clockwise)
        self.stations = [
            (0, 0),
            (0, self.grid_size - 1),
            (self.grid_size - 1, self.grid_size - 1),
            (self.grid_size - 1, 0),
        ]
        self.station_idx = 0
        self.move_timer = 0

    def reset(self, pos: np.ndarray) -> None:
        super().reset(pos)
        self.move_timer = 0

        #Goes to the next corner.
        curr_tile = (self.pos[0], self.pos[1])
        if curr_tile in self.stations:
            self.station_idx = self.stations.index(curr_tile)
        else:
            self.station_idx = 0
            self.pos[:] = self.stations[0]

    def move(self) -> None:
        self.move_timer += 1
        if self.move_timer < 5:
            return

        self.move_timer = 0

        self.station_idx = (self.station_idx + 1) % len(self.stations)
        self.pos[:] = self.stations[self.station_idx]


### another cyclic, this time with only 3 and random spots in the grid
###  unstable due to rng and random bag states could be the shortest or longest.
class PunishingBird(Cat):
    def _get_sprite_path(self) -> str:
        return "images/PunishingBird.png"

    def __init__(self, grid_size: int, tile_size: int):
        super().__init__(grid_size, tile_size)

        self.stations = []
        self.station_idx = 0

        self.step_cycle = [1, 2, 3]
        self.step_idx = 0

    def reset(self, pos: np.ndarray) -> None:
        super().reset(pos)
        positions = [
            (r, c)
            for r in range(self.grid_size)
            for c in range(self.grid_size)
        ]

        self.stations = random.sample(positions, 3)

        self.station_idx = 0
        self.step_idx = 0

        self.pos[:] = self.stations[0]
        self.station_idx = 1

    def move(self) -> None:
        station = self.stations[self.station_idx]
        steps = self.step_cycle[self.step_idx]

        for _ in range(steps):
            if tuple(self.pos) == station:
                break

            dr = station[0] - self.pos[0]
            dc = station[1] - self.pos[1]

            if abs(dr) >= abs(dc):
                self.pos[0] += 1 if dr > 0 else (-1 if dr < 0 else 0)
            else:
                self.pos[1] += 1 if dc > 0 else (-1 if dc < 0 else 0)

        if tuple(self.pos) == station:
            self.station_idx = (self.station_idx + 1) % 3

        # Next movement distance
        self.step_idx = (self.step_idx + 1) % 3

# Testing for random bag of moves.
# Unperceivable youkai.
# 0 = Copy, 1 = Avoid, 2 = Random
class Koishi(Cat):
    def _get_sprite_path(self) -> str:
        return "images/koishee.jpg"

    def __init__(self, grid_size: int, tile_size: int):
        super().__init__(grid_size, tile_size)
        self.state = 0
        self.turns = 0

    def reset(self, pos: np.ndarray) -> None:
        super().reset(pos)
        self.state = 0
        self.turns = 0

    def move(self) -> None:
        durations = [5, 3, 5]
        self.turns += 1
        if self.turns >= durations[self.state]:
            self.turns = 0

            bag = [0, 1, 2]
            bag.remove(self.state)
            self.state = random.choice(bag)

        if self.state == 0:
            if self.last_player_action == 0:  # Up
                self.pos[0] = max(0, self.pos[0] - 1)
            elif self.last_player_action == 1:  # Down
                self.pos[0] = min(self.grid_size - 1, self.pos[0] + 1)
            elif self.last_player_action == 2:  # Left
                self.pos[1] = max(0, self.pos[1] - 1)
            elif self.last_player_action == 3:  # Right
                self.pos[1] = min(self.grid_size - 1, self.pos[1] + 1)

        elif self.state == 1:
            moves = [
                (-1, 0),
                (1, 0),
                (0, -1),
                (0, 1),
                (0, 0),
            ]

            best_dist = -1
            best_moves = []

            for dr, dc in moves:
                nr = min(max(self.pos[0] + dr, 0), self.grid_size - 1)
                nc = min(max(self.pos[1] + dc, 0), self.grid_size - 1)

                dist = abs(nr - self.player_pos[0]) + abs(nc - self.player_pos[1])

                if dist > best_dist:
                    best_dist = dist
                    best_moves = [(nr, nc)]
                elif dist == best_dist:
                    best_moves.append((nr, nc))

            self.pos[:] = random.choice(best_moves)

        else:
            dr, dc = random.choice(
                [
                    (-1, 0),
                    (1, 0),
                    (0, -1),
                    (0, 1),
                ]
            )

            self.pos[0] = min(max(self.pos[0] + dr, 0), self.grid_size - 1)
            self.pos[1] = min(max(self.pos[1] + dc, 0), self.grid_size - 1)
            

# john Four
# Dances like a grenade
# Whose moves like a flower
# Who awits you to be seated for the grand finale

# dev notes: the bot doesn't even want to interact with its gimmick ToT
# so is this more of a 3 or a 4 kinda implementation.
# You go to (4,4)/(3,3) for 1/0 index starts.
class Jhin(Cat):
    def __init__(self, grid_size: int, tile_size: int):
        super().__init__(grid_size, tile_size)
        self.state = False

    def _get_sprite_path(self) -> str:
        return "images/jhin.jpg"

    # Whisper
    def reset(self, pos: np.ndarray) -> None:
        super().reset(pos)
        self.state = False

    # Curtain Call
    def move(self) -> None:

        # Death In 4 Acts
        player_just_left = (
            self.prev_player_pos[0] == 3
            and self.prev_player_pos[1] == 3
            and (self.player_pos[0] != 3 or self.player_pos[1] != 3)
        )

        # Captive Audience
        # Player steps onto (3,3)
        if self.player_pos[0] == 3 and self.player_pos[1] == 3:
            self.state = True
            return

        # Death In 4 Acts
        if self.state and player_just_left:
            self.pos[0] = 3
            self.pos[1] = 3
            self.state = False
            return

        # Deadly Flourish
        dirs = [
            (-1, 0),   # Up
            (1, 0),    # Down
            (0, -1),   # Left
            (0, 1),    # Right
            (-1, -1),  # Up-left
            (-1, 1),   # Up-right
            (1, -1),   # Down-left
            (1, 1)     # Down-right
        ]
        valid_moves = []

        for dr, dc in dirs:
            target_r = self.pos[0] + 3 * dr
            target_c = self.pos[1] + 3 * dc
            if (0 <= target_r < self.grid_size and 0 <= target_c < self.grid_size):
                valid_moves.append((target_r, target_c))

        # Dancing Grenade
        if valid_moves:
            new_move = random.choice(valid_moves)

            self.pos[0] = new_move[0]
            self.pos[1] = new_move[1]
        
# A mere star that will fade cannot hope to best the light of the rising sun. 
# Be prepared; for today, a Star of the City shall be gone. 

# Reverse Scale
# Player must go on the opposite corner where Xiao is, 
# afterwards she shrinks her move space and teleports to another corner not on player or prev where she is.
# Once she closes to the center. she no longer moves again.
class Xiao(Cat):
    def __init__(self, grid_size: int, tile_size: int):
        super().__init__(grid_size, tile_size)
        self.scale = 0
        self.turn_counter = 0
        self.finished = False

    def _get_sprite_path(self) -> str:
        return "images/xiao.png"

    def reset(self, pos: np.ndarray) -> None:
        super().reset(pos)
        self.scale = 0
        self.turn_counter = 0
        self.finished = False

    def _corner_is_safe(self, corner: tuple) -> bool:
        dist = abs(corner[0] - self.player_pos[0]) + abs(corner[1] - self.player_pos[1])
        return dist > 2

    def move(self) -> None:
        if self.finished:
            return

        self.turn_counter += 1

        low = self.scale
        high = self.grid_size - 1 - self.scale

        if low >= high:
            self.finished = True
            return
        
        player_is_close = self.current_distance <= 2
        if self.turn_counter < 6 and not player_is_close:
            return

        self.turn_counter = 0

        scale = (self.pos[0], self.pos[1])
        player_pos = (self.player_pos[0], self.player_pos[1])
        reverse_scale = (low + high - scale[0], low + high - scale[1])

        if player_pos == reverse_scale:
            self.scale += 1
            low = self.scale
            high = self.grid_size - 1 - self.scale

            if low >= high:
                self.finished = True
                return

        current_corners = [
            (low, low),
            (low, high),
            (high, low),
            (high, high),
        ]

        valid_corners = []
        for corner in current_corners:
            if corner != scale and self._corner_is_safe(corner):
                valid_corners.append(corner)

        if not valid_corners:
            for corner in current_corners:
                if corner != scale:
                    valid_corners.append(corner)

        if valid_corners:
            new_pos = random.choice(valid_corners)
            self.pos[0] = new_pos[0]
            self.pos[1] = new_pos[1]