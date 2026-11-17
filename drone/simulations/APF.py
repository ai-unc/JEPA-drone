"""Artificial potential field over known obstacle-course geometry."""
import math
import numpy as np
from simulations.obstacle_course import ObstacleCourse


class APF:
    MAX_ATTRACTION = 2.0  # m/s
    MAX_REPULSION = 5.0   # m/s per obstacle
    MAX_OVERALL_SPEED = 4.0
    MIN_DISTANCE_FROM_GOAL = 1.0
    RANGE = 8.0

    class Force:
        def __init__(self, name, vector):
            self.name = name
            self.vector = np.asarray(vector, dtype=float)

        def clamp(self, max_magnitude):
            magnitude = np.linalg.norm(self.vector)
            if magnitude > max_magnitude:
                self.vector *= max_magnitude / magnitude
            return self

    def __init__(self):
        self.seen = set()
        self.velocity_vector = np.zeros(3)
        self.forces = []
        self.yaw = 0.0

    def run(self, position, obstacle_course: ObstacleCourse):
        position = np.asarray(position, dtype=float)
        self.seen.clear()
        self.forces.clear()
        self.calculate_goal_force(position, obstacle_course.goal)
        self.calculate_ceiling_and_floor_force(position, obstacle_course)
        grid = obstacle_course.occupancy_map
        if grid is None:
            obstacle_course.rebuild_occupancy_map()
            grid = obstacle_course.occupancy_map
        cell = np.asarray(obstacle_course.occupancy_cell_size)
        origin = obstacle_course.occupancy_origin
        low = np.floor((position[:2] - self.RANGE - origin) / cell).astype(int)
        high = np.floor((position[:2] + self.RANGE - origin) / cell).astype(int)
        for y in range(max(0, low[1]), min(len(grid)-1, high[1])+1):
            for x in range(max(0, low[0]), min(len(grid[0])-1, high[0])+1):
                self.calculate_cell_force(position, grid[y][x])
        return self.calculate_velocity_vector()

    def calculate_velocity_vector(self):
        # Attraction plus each local repulsion; clamp final commanded speed.
        velocity = sum((force.vector for force in self.forces), np.zeros(3))
        magnitude = np.linalg.norm(velocity)
        if magnitude > self.MAX_OVERALL_SPEED:
            velocity *= self.MAX_OVERALL_SPEED / magnitude
        self.velocity_vector = velocity
        if np.linalg.norm(velocity[:2]) > 0.05:
            self.yaw = math.atan2(velocity[1], velocity[0])
        return velocity

    def calculate_closest_point(self, position, obstacle: ObstacleCourse.Obstacle):
        return obstacle.get_closest_point(position)

    def calculate_goal_force(self, position, goal_position):
        delta = np.asarray(goal_position, dtype=float) - position
        distance = np.linalg.norm(delta)
        goal_yaw = math.atan2(delta[1], delta[0])
        speed = min(self.MAX_ATTRACTION, distance)
        force = np.zeros(3) if distance < self.MIN_DISTANCE_FROM_GOAL else delta / distance * speed
        self.forces.append(self.Force('goal', force))
        return force, goal_yaw

    def calculate_ceiling_and_floor_force(self, position, obstacle_course):
        ground = obstacle_course.ground_level
        ceiling = ground - obstacle_course.room_height
        margin = self.RANGE
        force = np.zeros(3)
        floor_gap = ground - position[2]
        ceiling_gap = position[2] - ceiling
        if floor_gap < margin:
            force[2] -= self.MAX_REPULSION * (margin - max(0., floor_gap)) / margin
        if ceiling_gap < margin:
            force[2] += self.MAX_REPULSION * (margin - max(0., ceiling_gap)) / margin
        self.forces.append(self.Force('vertical bounds', force))
        return force

    def calculate_cell_force(self, position, cell):
        # The same obstacle may occupy several neighboring cells.
        for obstacle in cell:
            key = id(obstacle)
            if key in self.seen:
                continue
            self.seen.add(key)
            closest = self.calculate_closest_point(position, obstacle)
            away = position - closest
            distance = np.linalg.norm(away)
            if distance >= self.RANGE:
                continue
            if obstacle.contains(position):
                away = -away
            if distance < 1e-8:
                away = position - obstacle.position
                distance = np.linalg.norm(away)
            if distance < 1e-8:
                away = np.array([1., 0., 0.])
                distance = 1.
            strength = self.MAX_REPULSION * (self.RANGE - min(distance, self.RANGE)) / self.RANGE
            self.forces.append(self.Force('obstacle', away / distance * strength))
