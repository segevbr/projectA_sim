import pygame
import numpy as np
import math
import random

# --- CONFIGURATION ---
MAP_WIDTH = 800
MAP_HEIGHT = 600
SIDEBAR_WIDTH = 250
WINDOW_SIZE = (MAP_WIDTH + SIDEBAR_WIDTH, MAP_HEIGHT)

GRID_SIZE = 40  # pixels
MAP_COLS = MAP_WIDTH // GRID_SIZE
MAP_ROWS = MAP_HEIGHT // GRID_SIZE

# Physics / Sensor Config
CELL_SIZE_MM = 180.0
PIXELS_PER_MM = GRID_SIZE / CELL_SIZE_MM
# Sensor Ranges
MIN_RANGE_MM = 1.0
MAX_RANGE_MM = 1300.0
SENSOR_ACCURACY_MM = 20.0 # Standard Deviation for noise
# Convert limits to pixels
MIN_RANGE_PX = MIN_RANGE_MM * PIXELS_PER_MM
MAX_RANGE_PX = MAX_RANGE_MM * PIXELS_PER_MM

# Angles
# 0 is east, angle grows clockwise 
SENSOR_ANGLES = [-90, -40.33, 0, 37.22, 90] 
NUM_DISCRETE_ANGLES = 8
ROTATION_STEP = 360 / NUM_DISCRETE_ANGLES 

# Map Wall Indices
WALL_UP = 0
WALL_DOWN = 1
WALL_LEFT = 2
WALL_RIGHT = 3

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
BLUE = (0, 100, 255)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
GRAY = (50, 50, 50)
CYAN = (0, 255, 255)
DARK_RED = (100, 0, 0)
YELLOW = (255, 255, 0)
SIDEBAR_BG = (30, 30, 30)

class Robot:
    def __init__(self, x, y, theta):
        self.x = x # True float position
        self.y = y
        self.theta = theta # True float angle (degrees)
        
        # Active state for sensors (Default all ON)
        self.active_sensors = [True] * len(SENSOR_ANGLES)
        
    def move(self, speed, rot_speed, wall_map):
        """
        Continuous movement.
        speed: pixels per frame
        rot_speed: degrees per frame
        """
        # Update Angle
        self.theta = (self.theta + rot_speed) % 360
        
        # Calculate proposed new position
        rad = math.radians(self.theta)
        dx = math.cos(rad) * speed
        dy = math.sin(rad) * speed
        
        new_x = self.x + dx
        new_y = self.y + dy
        
        # Simple Collision Check (Stop at walls)
        if self._is_valid(new_x, new_y, wall_map):
            self.x = new_x
            self.y = new_y
            return dx, dy # Return actual movement for odometry
        return 0, 0

    def _is_valid(self, x, y, wall_map):
        c, r = int(x // GRID_SIZE), int(y // GRID_SIZE)
        if not (0 <= c < MAP_COLS and 0 <= r < MAP_ROWS): return False
        
        # Basic check: are we inside a wall line?
        # Ideally we check crossing lines, but checking current cell bounds is safer
        # Logic: If we are very close to a wall, check the wall_map
        local_x = x % GRID_SIZE
        local_y = y % GRID_SIZE
        buffer = 5 # radius of robot collision
        
        if local_x < buffer and wall_map[r, c, WALL_LEFT] == 0: return False
        if local_x > GRID_SIZE - buffer and wall_map[r, c, WALL_RIGHT] == 0: return False
        if local_y < buffer and wall_map[r, c, WALL_UP] == 0: return False
        if local_y > GRID_SIZE - buffer and wall_map[r, c, WALL_DOWN] == 0: return False
        
        return True

    def toggle_sensor(self, idx):
        if 0 <= idx < len(self.active_sensors):
            self.active_sensors[idx] = not self.active_sensors[idx]

    def measure(self, wall_map, add_noise=False):
        """
        Raycasts in the CONTINUOUS map using precise float math.
        """
        measurements = []
        hit_points = []
        
        robot_angle = self.theta

        for i, sensor_angle in enumerate(SENSOR_ANGLES):
            if not self.active_sensors[i]:
                measurements.append(None)
                continue

            total_angle = (robot_angle + sensor_angle) % 360
            rad = math.radians(total_angle)
            
            dx = math.cos(rad)
            dy = math.sin(rad) 
            
            curr_x, curr_y = self.x, self.y
            dist_px = 0
            hit = False
            step_size = 2.0 
            
            while dist_px < MAX_RANGE_PX:
                next_x = curr_x + dx * step_size
                next_y = curr_y + dy * step_size
                
                curr_c, curr_r = int(curr_x // GRID_SIZE), int(curr_y // GRID_SIZE)
                next_c, next_r = int(next_x // GRID_SIZE), int(next_y // GRID_SIZE)
                
                # Check Bounds
                if not (0 <= next_c < MAP_COLS and 0 <= next_r < MAP_ROWS):
                    hit = True
                    break
                
                # Check Walls (Crossing from curr cell to next cell)
                if next_c != curr_c:
                    if next_c > curr_c: # Right
                        if wall_map[curr_r, curr_c, WALL_RIGHT] == 0: hit = True; break
                    else: # Left
                        if wall_map[curr_r, curr_c, WALL_LEFT] == 0: hit = True; break     
                if next_r != curr_r:
                    if next_r > curr_r: # Down
                        if wall_map[curr_r, curr_c, WALL_DOWN] == 0: hit = True; break
                    else: # Up
                        if wall_map[curr_r, curr_c, WALL_UP] == 0: hit = True; break
                
                curr_x, curr_y = next_x, next_y
                dist_px += step_size
                
            dist_mm = dist_px / PIXELS_PER_MM
            
            # Limits
            dist_mm = max(MIN_RANGE_MM, min(dist_mm, MAX_RANGE_MM))
            
            # Noise
            if add_noise and hit:
                dist_mm = random.gauss(dist_mm, SENSOR_ACCURACY_MM)
                dist_mm = max(MIN_RANGE_MM, min(dist_mm, MAX_RANGE_MM))

            measurements.append(dist_mm)
            hit_points.append((curr_x, curr_y))
            
        return measurements, hit_points

# --- MAP HELPERS ---

def generate_wall_map():
    maze = np.ones((MAP_ROWS, MAP_COLS, 4), dtype=int)
    def set_v_wall(r, c):
        maze[r, c, WALL_RIGHT] = 0
        if c + 1 < MAP_COLS: maze[r, c+1, WALL_LEFT] = 0
    def set_h_wall(r, c):
        maze[r, c, WALL_DOWN] = 0
        if r + 1 < MAP_ROWS: maze[r+1, c, WALL_UP] = 0
        
    maze[0, :, WALL_UP] = 0
    maze[-1, :, WALL_DOWN] = 0
    maze[:, 0, WALL_LEFT] = 0
    maze[:, -1, WALL_RIGHT] = 0
    
    # Alon House map
    for i in range(0,9): set_h_wall(2, i)
    for i in range(0,3): set_v_wall(i, 8)
    
    return maze

def precompute_all_orientations(wall_map):
    """
    Pre-computes expected sensor readings for grid centers.
    Shape: (Rows, Cols, 8, 5)
    """
    print("Pre-computing for 8 discrete orientations... (Wait for it)")
    rows, cols, _ = wall_map.shape
    num_angles = NUM_DISCRETE_ANGLES
    
    data = np.zeros((rows, cols, num_angles, len(SENSOR_ANGLES)))
    
    # Use a virtual robot for precomputation
    v_bot = Robot(0, 0, 0)
    
    for r in range(rows):
        for c in range(cols):
            # Place virtual bot at center of cell
            v_bot.x = (c * GRID_SIZE) + (GRID_SIZE / 2)
            v_bot.y = (r * GRID_SIZE) + (GRID_SIZE / 2)
            
            for ang_idx in range(num_angles):
                v_bot.theta = ang_idx * ROTATION_STEP
                dists, _ = v_bot.measure(wall_map, add_noise=False)
                data[r, c, ang_idx] = dists
                
    print("Pre-computation Complete.")
    return data

def toggle_wall_click(wall_map, mx, my):
    """
    Toggles the wall nearest to the mouse click (mx, my).
    Syncs the wall change between adjacent cells.
    """
    if not (0 <= mx < MAP_WIDTH and 0 <= my < MAP_HEIGHT):
        return

    c, r = mx // GRID_SIZE, my // GRID_SIZE
    
    # Calculate offset within the cell
    lx = mx % GRID_SIZE
    ly = my % GRID_SIZE
    
    # Distances to each edge
    d_left = lx
    d_right = GRID_SIZE - lx
    d_up = ly
    d_down = GRID_SIZE - ly
    
    min_dist = min(d_left, d_right, d_up, d_down)
    
    # Threshold in pixels to click a wall
    CLICK_THRESHOLD = 15
    if min_dist > CLICK_THRESHOLD:
        return
        
    # Toggle the specific wall (0->1 or 1->0)
    # And ensure the neighbor's corresponding wall is updated
    if min_dist == d_up:
        val = 1 - wall_map[r, c, WALL_UP]
        wall_map[r, c, WALL_UP] = val
        if r > 0: 
            wall_map[r-1, c, WALL_DOWN] = val
            
    elif min_dist == d_down:
        val = 1 - wall_map[r, c, WALL_DOWN]
        wall_map[r, c, WALL_DOWN] = val
        if r < MAP_ROWS - 1: 
            wall_map[r+1, c, WALL_UP] = val
            
    elif min_dist == d_left:
        val = 1 - wall_map[r, c, WALL_LEFT]
        wall_map[r, c, WALL_LEFT] = val
        if c > 0: 
            wall_map[r, c-1, WALL_RIGHT] = val
            
    elif min_dist == d_right:
        val = 1 - wall_map[r, c, WALL_RIGHT]
        wall_map[r, c, WALL_RIGHT] = val
        if c < MAP_COLS - 1: 
            wall_map[r, c+1, WALL_LEFT] = val

def dump_selected_cells(data, selected_cells):
    """
    Dumps distance data for selected cells in the format requested.
    """
    print("\n=== SELECTED CELLS DUMP (PYTHON) ===")
    
    # Sort selected cells by row then col for cleaner output
    # selected_cells is set of (py_r, py_c)
    sorted_cells = sorted(list(selected_cells), key=lambda x: (x[0], x[1]))

    for (py_r, c) in sorted_cells:
        # Calculate C++ Row equivalent for display if needed, or just standard row
        # In the original file provided: cpp_r = MAP_ROWS - 1 - py_r
        # We will keep py_r for clarity unless specific C++ conversion is needed, 
        # but the original code had this line, so we keep the logic structure.
        
        print(f"Cell ({py_r}, {c}):")
        for a in range(8):
            d = data[py_r, c, a]
            vals = ", ".join([f"{x:6.2f}" for x in d/1000]) # Convert to meters for display
            print(f"  Angle {a} ({int(a*ROTATION_STEP):3d}°): {{ {vals} }},")
    print("=== END DUMP ===\n")

# --- GRID LOCALIZATION ALGORITHMS ---

def update_probability(prob_matrix, measured_dists, expected_data, current_discrete_angle_idx):
    """
    Bayes Update. 
    Compare 'measured_dists' (Real Robot) vs 'expected_data' (Grid Precalc).
    We use the robot's current approximate discrete angle to slice the expected data.
    """
    rows, cols = prob_matrix.shape
    new_prob = np.zeros_like(prob_matrix)
    
    # Sigma needs to cover both sensor noise AND discretization error
    # (The robot might be 15 pixels off-center in the cell, changing readings)
    sigma = 40.0 # mm
    
    # Vectorized calculation for speed
    # Get the slice of expectations for the current discrete angle: (Rows, Cols, Sensors)
    expected_slice = expected_data[:, :, current_discrete_angle_idx, :]
    
    # Create mask for valid sensors (not None)
    valid_indices = [i for i, m in enumerate(measured_dists) if m is not None]
    
    if not valid_indices:
        return prob_matrix # No data, no update

    # Calculate Likelihoods
    likelihood_grid = np.ones((rows, cols))
    
    for i in valid_indices:
        z = measured_dists[i]
        mu_grid = expected_slice[:, :, i]
        
        # Avoid zero division or weirdness with walls
        mu_grid = np.maximum(mu_grid, MIN_RANGE_MM)
        
        # Gaussian: exp( - (z - mu)^2 / (2sigma^2) )
        diff = z - mu_grid
        likelihood_grid *= np.exp(-(diff**2) / (2 * sigma**2))
    
    new_prob = prob_matrix * likelihood_grid
            
    total = np.sum(new_prob)
    if total > 0:
        new_prob /= total
    else:
        new_prob.fill(1.0 / np.count_nonzero(prob_matrix))
        
    return new_prob

def predict_motion(prob_matrix, dr, dc, wall_map):
    """
    Shifts the probability grid by (dr, dc) cells.
    Also adds blur (convolution) to represent motion uncertainty.
    """
    rows, cols = prob_matrix.shape
    shifted = np.zeros_like(prob_matrix)
    
    # 1. Shift
    for r in range(rows):
        for c in range(cols):
            if prob_matrix[r, c] < 1e-9: continue
            
            # Check walls
            blocked = False
            if dr == -1 and (wall_map[r,c,WALL_UP]==0 or r-1<0): blocked=True
            elif dr == 1 and (wall_map[r,c,WALL_DOWN]==0 or r+1>=rows): blocked=True
            elif dc == -1 and (wall_map[r,c,WALL_LEFT]==0 or c-1<0): blocked=True
            elif dc == 1 and (wall_map[r,c,WALL_RIGHT]==0 or c+1>=cols): blocked=True
            
            if not blocked:
                shifted[r+dr, c+dc] += prob_matrix[r, c]
            else:
                shifted[r, c] += prob_matrix[r, c] # Stay put if hit wall

    # 2. Blur (Uncertainty)
    blurred = np.zeros_like(shifted)
    
    # Kernel: Center=60%, Neighbors=10%
    # This represents that even if we moved 1 cell, we might have stayed or moved 2
    for r in range(rows):
        for c in range(cols):
            val = shifted[r, c]
            if val < 1e-9: continue
            
            # Simple 3x3 blur
            blurred[r, c] += val * 0.6
            for nr, nc in [(r+1,c), (r-1,c), (r,c+1), (r,c-1)]:
                if 0 <= nr < rows and 0 <= nc < cols:
                    blurred[nr, nc] += val * 0.1
            
    total = np.sum(blurred)
    if total > 0: blurred /= total
    
    return blurred

def main():
    pygame.init()
    screen = pygame.display.set_mode(WINDOW_SIZE)
    pygame.display.set_caption("Continuous Robot vs Discrete Grid")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("Arial", 18)
    small_font = pygame.font.SysFont("Arial", 14)

    # Setup
    wall_map = generate_wall_map()
    expected_data = precompute_all_orientations(wall_map)
    
    # Init Uniform Probability
    prob_matrix = np.ones((MAP_ROWS, MAP_COLS))
    prob_matrix /= (MAP_ROWS * MAP_COLS)
    
    # Robot spawns at floating point coordinate (e.g., middle of cell 3,3)
    robot = Robot(3.5 * GRID_SIZE, 3.5 * GRID_SIZE, 90.0)
    
    # ODOMETRY ACCUMULATORS
    odom_x = 0.0
    odom_y = 0.0
    
    # MODES
    building_mode = False
    analysis_mode = False
    selected_cells = set()
    
    running = True
    while running:
        speed = 0
        rot_speed = 0
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            
            # --- MOUSE INTERACTIONS ---
            elif event.type == pygame.MOUSEBUTTONDOWN:
                mx, my = pygame.mouse.get_pos()
                
                # Check if click is inside map area
                if mx < MAP_WIDTH:
                    # Priority 1: Analysis Mode
                    if analysis_mode:
                        c, r = mx // GRID_SIZE, my // GRID_SIZE
                        if 0 <= c < MAP_COLS and 0 <= r < MAP_ROWS:
                            coord = (r, c)
                            if coord in selected_cells:
                                selected_cells.remove(coord)
                            else:
                                selected_cells.add(coord)
                    
                    # Priority 2: Builder Mode
                    elif building_mode:
                        toggle_wall_click(wall_map, mx, my)
                        
                    # Priority 3: Teleport (Simulation Mode)
                    else:
                        # Teleport robot to click location
                        # Check bounds first
                        if 0 <= mx < MAP_WIDTH and 0 <= my < MAP_HEIGHT:
                            robot.x = mx
                            robot.y = my
                            # Reset probabilities since we teleported
                            prob_matrix.fill(1.0 / (MAP_ROWS*MAP_COLS))
                            odom_x, odom_y = 0, 0
            
            # --- KEYBOARD INTERACTIONS ---
            elif event.type == pygame.KEYDOWN:
                # 1. Builder Mode Toggle
                if event.key == pygame.K_b:
                    building_mode = not building_mode
                    if not building_mode:
                        # Exit Builder: Recompute everything
                        screen.fill(BLACK)
                        txt = font.render("Recomputing Map Data... Please Wait...", True, WHITE)
                        screen.blit(txt, (MAP_WIDTH//2 - 150, MAP_HEIGHT//2))
                        pygame.display.flip()
                        
                        expected_data = precompute_all_orientations(wall_map)
                        prob_matrix.fill(1.0 / (MAP_ROWS*MAP_COLS))
                        odom_x, odom_y = 0, 0
                
                # 2. Analysis Mode Toggle
                elif event.key == pygame.K_p and not building_mode:
                    if analysis_mode:
                        # Exiting: Dump and Clear
                        if selected_cells:
                            dump_selected_cells(expected_data, selected_cells)
                        else:
                            print("\nNo cells selected for dump.\n")
                        selected_cells.clear()
                        analysis_mode = False
                    else:
                        # Entering
                        analysis_mode = True
                        print("\n--- ANALYSIS MODE STARTED ---")
                        print("1. Click cells on the map to mark them (Cyan).")
                        print("2. Press 'P' again to dump their distance data to console.")

                # 3. Manual sensor toggle (1-5)
                elif not building_mode and not analysis_mode:
                    if event.key == pygame.K_1: robot.toggle_sensor(0)
                    elif event.key == pygame.K_2: robot.toggle_sensor(1)
                    elif event.key == pygame.K_3: robot.toggle_sensor(2)
                    elif event.key == pygame.K_4: robot.toggle_sensor(3)
                    elif event.key == pygame.K_5: robot.toggle_sensor(4)
                    
                    # Kidnap / Reset
                    elif event.key == pygame.K_r:
                        prob_matrix.fill(1.0 / (MAP_ROWS*MAP_COLS))
                        print("Global Localization Reset")

        # --- MOVEMENT & LOGIC (Only in Sim Mode) ---
        if not building_mode and not analysis_mode:
            # Continuous Input
            keys = pygame.key.get_pressed()
            if keys[pygame.K_w]: speed = 3
            if keys[pygame.K_s]: speed = -3
            if keys[pygame.K_a]: rot_speed = -3 
            if keys[pygame.K_d]: rot_speed = 3
            
            # 1. MOVE ROBOT (Continuous)
            dx_actual, dy_actual = robot.move(speed, rot_speed, wall_map)
            
            # 2. ACCUMULATE ODOMETRY
            odom_x += dx_actual
            odom_y += dy_actual
            
            # 3. CHECK FOR GRID STEP
            grid_shift_c = 0
            grid_shift_r = 0
            
            if abs(odom_x) >= GRID_SIZE * 0.75:
                direction = 1 if odom_x > 0 else -1
                grid_shift_c = direction
                odom_x -= direction * GRID_SIZE 
                
            if abs(odom_y) >= GRID_SIZE * 0.75:
                direction = 1 if odom_y > 0 else -1
                grid_shift_r = direction
                odom_y -= direction * GRID_SIZE

            # 4. PREDICT STEP
            if grid_shift_r != 0 or grid_shift_c != 0:
                prob_matrix = predict_motion(prob_matrix, grid_shift_r, grid_shift_c, wall_map)

            # 5. UPDATE STEP
            dists, hit_points = robot.measure(wall_map, add_noise=True)
            
            normalized_angle = robot.theta % 360
            discrete_angle_idx = int((normalized_angle + (ROTATION_STEP/2)) // ROTATION_STEP) % 8
            
            prob_matrix = update_probability(prob_matrix, dists, expected_data, discrete_angle_idx)
        
        else:
            # In Builder/Analysis mode, we still want to see the robot's rays if it's there
            dists, hit_points = robot.measure(wall_map, add_noise=False)


        # --- DRAWING ---
        screen.fill(BLACK)
        
        # Draw Map & Probability
        for r in range(MAP_ROWS):
            for c in range(MAP_COLS):
                rect = (c*GRID_SIZE, r*GRID_SIZE, GRID_SIZE, GRID_SIZE)
                
                # Probability (Red Haze) - Only in Sim Mode
                if not building_mode and not analysis_mode:
                    p = prob_matrix[r, c]
                    val = min(int(p * 2000), 255) 
                    if val > 10:
                        s = pygame.Surface((GRID_SIZE, GRID_SIZE))
                        s.set_alpha(150)
                        s.fill((val, 0, 0))
                        screen.blit(s, rect[:2])
                
                # Highlight Selected Cells (Analysis Mode)
                if analysis_mode and (r, c) in selected_cells:
                    pygame.draw.rect(screen, CYAN, rect)

                # Grid Lines
                grid_col = (60, 60, 0) if building_mode else (30, 30, 30)
                pygame.draw.rect(screen, grid_col, rect, 1)
                
                # Walls
                w_thick = 4 if building_mode else 2
                if wall_map[r, c, WALL_UP] == 0:
                    pygame.draw.line(screen, RED, (c*GRID_SIZE, r*GRID_SIZE), ((c+1)*GRID_SIZE, r*GRID_SIZE), w_thick)
                if wall_map[r, c, WALL_DOWN] == 0:
                    pygame.draw.line(screen, RED, (c*GRID_SIZE, (r+1)*GRID_SIZE), ((c+1)*GRID_SIZE, (r+1)*GRID_SIZE), w_thick)
                if wall_map[r, c, WALL_LEFT] == 0:
                    pygame.draw.line(screen, RED, (c*GRID_SIZE, r*GRID_SIZE), (c*GRID_SIZE, (r+1)*GRID_SIZE), w_thick)
                if wall_map[r, c, WALL_RIGHT] == 0:
                    pygame.draw.line(screen, RED, ((c+1)*GRID_SIZE, r*GRID_SIZE), ((c+1)*GRID_SIZE, (r+1)*GRID_SIZE), w_thick)

        # Draw Analysis Border
        if analysis_mode:
            pygame.draw.rect(screen, BLUE, (0, 0, MAP_WIDTH, MAP_HEIGHT), 5)

        # Draw Robot (Continuous Position)
        pygame.draw.circle(screen, BLUE, (int(robot.x), int(robot.y)), 10)
        
        head_rad = math.radians(robot.theta)
        head_x = robot.x + math.cos(head_rad) * 15
        head_y = robot.y + math.sin(head_rad) * 15
        pygame.draw.line(screen, WHITE, (robot.x, robot.y), (head_x, head_y), 2)
        
        # Lidar Rays
        for p in hit_points:
            pygame.draw.line(screen, GREEN, (robot.x, robot.y), p, 1)
            pygame.draw.circle(screen, GREEN, (int(p[0]), int(p[1])), 2)

        # --- SIDEBAR ---
        sidebar_rect = pygame.Rect(MAP_WIDTH, 0, SIDEBAR_WIDTH, MAP_HEIGHT)
        pygame.draw.rect(screen, SIDEBAR_BG, sidebar_rect)
        pygame.draw.line(screen, WHITE, (MAP_WIDTH, 0), (MAP_WIDTH, MAP_HEIGHT), 2)
        
        x_start = MAP_WIDTH + 10
        y_off = 10
        
        # Mode Status
        if analysis_mode:
            mode_txt = "ANALYSIS MODE"
            mode_col = CYAN
        elif building_mode:
            mode_txt = "BUILDER MODE"
            mode_col = YELLOW
        else:
            mode_txt = "SIMULATION MODE"
            mode_col = GREEN
            
        screen.blit(font.render(mode_txt, True, mode_col), (x_start, y_off)); y_off += 30
        
        # Controls Text
        screen.blit(font.render("CONTROLS", True, GREEN), (x_start, y_off)); y_off += 25
        screen.blit(small_font.render("B: Toggle Builder", True, WHITE), (x_start, y_off)); y_off += 20
        screen.blit(small_font.render("P: Toggle Analysis", True, WHITE), (x_start, y_off)); y_off += 20
        
        if building_mode:
             screen.blit(small_font.render("Click Walls to Toggle", True, YELLOW), (x_start, y_off)); y_off += 20
        elif analysis_mode:
             screen.blit(small_font.render("Click Cells to Select", True, CYAN), (x_start, y_off)); y_off += 20
             screen.blit(small_font.render("Press P to Dump Data", True, CYAN), (x_start, y_off)); y_off += 20
             screen.blit(small_font.render(f"Selected: {len(selected_cells)}", True, CYAN), (x_start, y_off)); y_off += 20
        else:
            screen.blit(small_font.render("WASD: Move/Turn", True, WHITE), (x_start, y_off)); y_off += 20
            screen.blit(small_font.render("Click: Teleport", True, WHITE), (x_start, y_off)); y_off += 20

        y_off += 20

        # Robot Stats
        screen.blit(font.render("ROBOT STATE", True, GREEN), (x_start, y_off)); y_off += 25
        screen.blit(small_font.render(f"X: {robot.x:.1f}  Y: {robot.y:.1f}", True, WHITE), (x_start, y_off)); y_off += 20
        screen.blit(small_font.render(f"Θ: {robot.theta:.1f}°", True, WHITE), (x_start, y_off)); y_off += 30
        
        # Top Estimates
        screen.blit(font.render("Top Probabilities:", True, GREEN), (x_start, y_off)); y_off += 25
        
        if not building_mode and not analysis_mode:
            r_disc = int(robot.y // GRID_SIZE)
            c_disc = int(robot.x // GRID_SIZE)
            
            flat_indices = np.argsort(prob_matrix.ravel())[::-1]
            count = 0
            for idx in flat_indices:
                r_idx, c_idx = divmod(idx, MAP_COLS)
                prob = prob_matrix[r_idx, c_idx]
                if prob < 0.005 or count >= 6: break
                
                is_correct = (r_idx == r_disc and c_idx == c_disc)
                color = GREEN if is_correct else WHITE
                txt = f"({r_idx}, {c_idx}): {prob:.3f}"
                if is_correct: txt += " <--"
                
                screen.blit(small_font.render(txt, True, color), (x_start, y_off)); y_off += 20
                count += 1
        else:
             screen.blit(small_font.render("Paused...", True, GRAY), (x_start, y_off))

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()

if __name__ == "__main__":
    main()