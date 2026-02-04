import numpy as np
import random
import pygame

# Constants
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 400
LANE_HEIGHT = 100
SHUTTLE_WIDTH = 40
SHUTTLE_HEIGHT = 40
OBSTACLE_WIDTH = 40
OBSTACLE_HEIGHT = 40
FPS = 10

class WarehouseEnv:
    def __init__(self, render_mode=False, length=50):
        self.render_mode = render_mode
        self.length = length # Length of the corridor in steps
        self.n_lanes = 3
        self.action_space = [0, 1, 2] # 0: Up, 1: Down, 2: Stay

        # Colors
        self.WHITE = (255, 255, 255)
        self.BLACK = (0, 0, 0)
        self.RED = (255, 0, 0)
        self.GREEN = (0, 255, 0)
        self.BLUE = (0, 0, 255)
        self.GRAY = (200, 200, 200)

        if self.render_mode:
            pygame.init()
            self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
            pygame.display.set_caption("Warehouse Shuttle Environment")
            self.clock = pygame.time.Clock()

        self.reset()

    def reset(self):
        self.shuttle_x = 0
        self.shuttle_y = 1 # Start in middle lane
        self.steps = 0

        # Generate obstacles
        # Obstacles are dictionaries: {'x': int, 'y': int, 'dir': int}
        self.obstacles = []
        # Generate random obstacles along the path
        for x in range(5, self.length, 3): # Spaced out
            if random.random() < 0.6: # 60% chance of obstacle at this x
                y = random.randint(0, self.n_lanes - 1)
                direction = random.choice([-1, 0, 1])
                self.obstacles.append({'x': x, 'y': y, 'dir': direction})

        return self._get_state()

    def _get_state(self):
        # Find nearest obstacle ahead
        nearest_obs = None
        min_dist = float('inf')

        for obs in self.obstacles:
            dist = obs['x'] - self.shuttle_x
            if 0 <= dist < min_dist:
                min_dist = dist
                nearest_obs = obs

        # Discretize distance
        # Close: 0-2, Medium: 3-5, Far: >5 (or >6)
        if min_dist <= 2:
            dist_state = 0 # Close
        elif min_dist <= 5:
            dist_state = 1 # Medium
        else:
            dist_state = 2 # Far / No obstacle ahead (if min_dist is inf)

        # If no obstacle ahead, use default values
        obs_y = 0 if nearest_obs is None else nearest_obs['y']

        if nearest_obs is None:
             dist_state = 2 # Treat no obstacle as Far

        # State tuple: (shuttle_y, dist_state, obs_y)
        return (self.shuttle_y, dist_state, obs_y)

    def step(self, action):
        reward = -0.1 # Time step penalty
        done = False

        # 1. Move Shuttle (Y axis)
        if action == 0: # Up (y decreases)
            self.shuttle_y = max(0, self.shuttle_y - 1)
        elif action == 1: # Down (y increases)
            self.shuttle_y = min(self.n_lanes - 1, self.shuttle_y + 1)
        # action 2 is Stay

        # 2. Check collisions BEFORE moving forward (if obstacle is at same X)
        # Actually, let's assume continuous movement.
        # We check collision if shuttle and obstacle occupy same grid cell.

        if self._check_collision():
            reward = -100
            done = True
            return self._get_state(), reward, done

        # 3. Move Shuttle Forward (X axis)
        prev_x = self.shuttle_x
        self.shuttle_x += 1

        # Check "Evitement" (Passing an obstacle)
        # If we passed an obstacle without collision
        for obs in self.obstacles:
            if obs['x'] == prev_x:
                reward += 1 # Avoidance bonus

        # 4. Move Obstacles
        for obs in self.obstacles:
            # Obstacles move perpendicularly (Y axis)
            if random.random() < 0.3: # Randomly change direction sometimes
                obs['dir'] = random.choice([-1, 0, 1])

            new_y = obs['y'] + obs['dir']
            if 0 <= new_y < self.n_lanes:
                obs['y'] = new_y
            else:
                obs['dir'] *= -1 # Bounce

        # 5. Check collisions AFTER moving
        if self._check_collision():
            reward = -100
            done = True
            return self._get_state(), reward, done

        # 6. Check Goal
        if self.shuttle_x >= self.length:
            done = True
            reward += 10 # Goal reward (not specified but good practice)

        return self._get_state(), reward, done

    def _check_collision(self):
        for obs in self.obstacles:
            if obs['x'] == self.shuttle_x and obs['y'] == self.shuttle_y:
                return True
        return False

    def render(self):
        if not self.render_mode:
            return

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()

        self.screen.fill(self.WHITE)

        # Draw Lanes
        for i in range(self.n_lanes):
            y = (i + 1) * LANE_HEIGHT
            pygame.draw.line(self.screen, self.BLACK, (0, y), (SCREEN_WIDTH, y), 2)

        # Draw Shuttle
        # Map shuttle X to screen X.
        # Since the shuttle moves forward, we can keep the shuttle fixed on screen
        # and scroll obstacles, OR move shuttle on screen.
        # Moving shuttle on screen is easier for fixed length.

        # Scale X to fit screen width roughly
        scale_x = SCREEN_WIDTH / (self.length + 5)

        s_rect = pygame.Rect(
            self.shuttle_x * scale_x,
            self.shuttle_y * LANE_HEIGHT + (LANE_HEIGHT - SHUTTLE_HEIGHT)//2,
            SHUTTLE_WIDTH,
            SHUTTLE_HEIGHT
        )
        pygame.draw.rect(self.screen, self.BLUE, s_rect)

        # Draw Obstacles
        for obs in self.obstacles:
            if obs['x'] < self.shuttle_x - 5: continue # Don't draw far behind

            o_rect = pygame.Rect(
                obs['x'] * scale_x,
                obs['y'] * LANE_HEIGHT + (LANE_HEIGHT - OBSTACLE_HEIGHT)//2,
                OBSTACLE_WIDTH,
                OBSTACLE_HEIGHT
            )
            pygame.draw.rect(self.screen, self.RED, o_rect)

        pygame.display.flip()
        self.clock.tick(FPS)

    def close(self):
        if self.render_mode:
            pygame.quit()
