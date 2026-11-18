"""Artificial potential field over known obstacle-course geometry."""
import math
import numpy as np
from simulations.obstacle_course import ObstacleCourse


class APF:
    MAX_ATTRACTION = 2.0  # m/s
    MAX_REPULSION = 5.0   # m/s per obstacle
    MAX_OVERALL_SPEED = 4.0
    MIN_DISTANCE_FROM_GOAL = 1.0
    RANGE = 10

    BUG_VELOCITY_THRESHOLD = 0.5 #m/s
    BUG_STALL_STEPS = 50
    BUG_CLEARANCE = 3.0
    M_LINE_TOLERANCE = 1.5

    class Modes:
        APF = "APF"
        BUG = "BUG"



    class Force:
        def __init__(self, name, vector):
            self.name = name
            self.vector = np.asarray(vector, dtype=float)


    def __init__(self):
        self.seen = set()
        self.velocity_vector = np.zeros(3)
        self.forces = []
        self.yaw = 0.0
        self.mode = self.Modes.APF
        self.m_line_start = None
        self.bug_obstacle = None
        self.bug_hit = None
        self.bug_height = None
        self.bug_side = 1
        self.stall_steps = 0
        self.previous_position = None

    def run(self, position, obstacle_course: ObstacleCourse, actual_velocity=None):
        position = np.asarray(position, dtype=float)
        if self.m_line_start is None:
            self.m_line_start = position[:2].copy()
        if self.mode == self.Modes.BUG:
            result = self.calculate_bug_velocity(position, obstacle_course)
            self.previous_position = position.copy()
            return result
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
        result = self.calculate_apf_velocity_vector()
        commanded = np.linalg.norm(result[:2])
        measured = (np.linalg.norm(np.asarray(actual_velocity)[:2])
                    if actual_velocity is not None else np.inf)
        progress = (np.linalg.norm(position[:2] - self.previous_position[:2])
                    if self.previous_position is not None else np.inf)
        stalled = (commanded < self.BUG_VELOCITY_THRESHOLD or
                   (commanded > self.BUG_VELOCITY_THRESHOLD and measured < self.BUG_VELOCITY_THRESHOLD and progress < .02))
        self.stall_steps = self.stall_steps + 1 if stalled else 0
        if self.stall_steps >= self.BUG_STALL_STEPS and np.linalg.norm(obstacle_course.goal[:2]-position[:2]) > self.RANGE:
            candidates = self.nearby_section_obstacles(position, obstacle_course, self.RANGE)
            if candidates:
                self.bug_obstacle = min(candidates, key=lambda o: np.linalg.norm(
                    self.closest_section_point(position[:2],o.horizontal_section(position[2]))[0]-position[:2]))
                self.bug_hit = position[:2].copy()
                self.bug_height = float(position[2])
                self.bug_side = self.choose_bug_side(position, self.bug_obstacle, obstacle_course.goal)
                self.mode = self.Modes.BUG
                result = self.calculate_bug_velocity(position, obstacle_course)
        self.previous_position = position.copy()
        return result

    def calculate_apf_velocity_vector(self):
        # Add goal attraction and each unique obstacle and boundary repulsion.
        velocity = sum((force.vector for force in self.forces), np.zeros(3))
        magnitude = np.linalg.norm(velocity)
        if magnitude > self.MAX_OVERALL_SPEED:
            velocity *= self.MAX_OVERALL_SPEED / magnitude
        self.velocity_vector = velocity
        if np.linalg.norm(velocity[:2]) > 0.05:
            self.yaw = math.atan2(velocity[1], velocity[0])
        return velocity

    def calculate_velocity_vector(self):
        return self.calculate_apf_velocity_vector()

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

    @staticmethod
    def closest_section_point(point, polygon):
        """Closest XY boundary point and whether XY lies inside its section."""
        point = np.asarray(point, dtype=float)
        edges = np.roll(polygon,-1,axis=0)-polygon
        offsets = point-polygon
        t = np.clip(np.sum(offsets*edges,axis=1)/np.sum(edges*edges,axis=1),0,1)
        candidates = polygon + t[:,None]*edges
        nearest = candidates[np.argmin(np.sum((candidates-point)**2,axis=1))]
        cross = edges[:,0]*offsets[:,1]-edges[:,1]*offsets[:,0]
        inside = bool(np.all(cross>=-1e-7) or np.all(cross<=1e-7))
        return nearest, inside

    def nearby_section_obstacles(self, position, course, radius):
        """Deduplicate obstacles indexed by several nearby occupancy cells."""
        if course.occupancy_map is None:
            course.rebuild_occupancy_map()
        cell = np.asarray(course.occupancy_cell_size)
        low = np.floor((position[:2]-radius-course.occupancy_origin)/cell).astype(int)
        high = np.floor((position[:2]+radius-course.occupancy_origin)/cell).astype(int)
        found = {}
        grid = course.occupancy_map
        for y in range(max(0,low[1]),min(len(grid)-1,high[1])+1):
            for x in range(max(0,low[0]),min(len(grid[0])-1,high[0])+1):
                for obstacle in grid[y][x]:
                    found[id(obstacle)] = obstacle
        return [o for o in found.values() if o.horizontal_section(position[2]) is not None]

    def choose_bug_side(self, position, obstacle, goal):
        polygon = obstacle.horizontal_section(position[2])
        boundary, _ = self.closest_section_point(position[:2],polygon)
        outward = position[:2]-boundary
        if np.linalg.norm(outward)<1e-8:
            outward = position[:2]-obstacle.position[:2]
        outward /= max(np.linalg.norm(outward),1e-8)
        tangent = np.array([-outward[1],outward[0]])
        return 1 if np.dot(tangent,goal[:2]-position[:2])>=0 else -1

    def bug_m_line_restored(self, position, course):
        """Bug2 exit: rejoin the m-line closer to the goal with a clear route."""
        goal = course.goal[:2]
        line = goal-self.m_line_start
        length = np.linalg.norm(line)
        if length < 1e-8 or self.bug_hit is None:
            return False
        heading = line/length
        offset = position[:2]-self.m_line_start
        lateral = abs(heading[0]*offset[1]-heading[1]*offset[0])
        if lateral > self.M_LINE_TOLERANCE or np.linalg.norm(position[:2]-self.bug_hit)<2*self.BUG_CLEARANCE:
            return False
        if np.linalg.norm(goal-position[:2]) >= np.linalg.norm(goal-self.bug_hit)-self.M_LINE_TOLERANCE:
            return False
        return self.bug_path_clear(position, heading, np.linalg.norm(goal-position[:2]), course)

    def bug_path_clear(self, position, heading, goal_distance, course):
        """Check the next RANGE metres at the height held during Bug mode."""
        horizon = min(self.RANGE, goal_distance)
        sections = [obstacle.horizontal_section(self.bug_height)
                    for obstacle in self.nearby_section_obstacles(
                        np.r_[position[:2], self.bug_height], course,
                        horizon + self.BUG_CLEARANCE)]
        for distance in np.linspace(0., horizon, max(2, math.ceil(horizon / 0.5) + 1)):
            probe = position[:2] + heading * distance
            for section in sections:
                boundary, inside = self.closest_section_point(probe,section)
                if inside or np.linalg.norm(probe-boundary)<self.BUG_CLEARANCE:
                    return False
        return True

    def bug_goal_route_clear(self, position, course):
        """Leave the boundary early when a shorter route toward the goal opens."""
        if self.bug_hit is None:
            return False
        goal_delta = course.goal[:2] - position[:2]
        goal_distance = np.linalg.norm(goal_delta)
        if goal_distance < 1e-8 or np.linalg.norm(position[:2] - self.bug_hit) < 2 * self.BUG_CLEARANCE:
            return False
        if goal_distance >= np.linalg.norm(course.goal[:2] - self.bug_hit) - self.M_LINE_TOLERANCE:
            return False
        return self.bug_path_clear(position, goal_delta / goal_distance, goal_distance, course)

    def calculate_bug_velocity(self, position, course):
        if self.bug_m_line_restored(position,course) or self.bug_goal_route_clear(position,course):
            self.mode = self.Modes.APF
            self.bug_obstacle = self.bug_hit = self.bug_height = None
            self.stall_steps = 0
            return self.run(position,course)
        section = self.bug_obstacle.horizontal_section(self.bug_height)
        if section is None:
            self.mode = self.Modes.APF
            self.stall_steps = 0
            return self.run(position,course)
        boundary, inside = self.closest_section_point(position[:2],section)
        outward = position[:2]-boundary
        distance = np.linalg.norm(outward)
        if inside:
            outward = -outward
        if distance<1e-8:
            outward = position[:2]-self.bug_obstacle.position[:2]
            distance = np.linalg.norm(outward)
        outward /= max(distance,1e-8)
        tangent = self.bug_side*np.array([-outward[1],outward[0]])
        correction = np.clip(self.BUG_CLEARANCE-distance,-self.BUG_CLEARANCE,self.BUG_CLEARANCE)
        direction = tangent + outward*correction/self.BUG_CLEARANCE
        direction /= max(np.linalg.norm(direction),1e-8)
        velocity = np.r_[direction*min(self.MAX_OVERALL_SPEED,self.MAX_ATTRACTION),
                         np.clip(2*(self.bug_height-position[2]),-1.,1.)]
        self.velocity_vector = velocity
        if np.linalg.norm(direction)>0.05:
            self.yaw = math.atan2(direction[1],direction[0])
        return velocity
