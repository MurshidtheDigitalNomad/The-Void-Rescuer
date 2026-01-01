# The Void Rescuer - Contributor Guide

## Project Overview

**The Void Rescuer** is a 3D space rescue game built with Python and OpenGL where players navigate near a black hole to rescue stranded astronauts while managing gravity physics, energy, and obstacles.

---

## Quick Start

### Prerequisites

- Python 3.7+
- PyOpenGL library

### Installation

```bash
pip install PyOpenGL
```

### Running the Game

```bash
python main.py
```

### Controls (Current Implementation)

- **W / Up Arrow**: Forward thrust
- **S / Down Arrow**: Backward thrust
- **A / Left Arrow**: Rotate left
- **D / Right Arrow**: Rotate right
- **SPACE**: Tether/untether nearest astronaut
- **ESC**: Quit game

---

## Architecture Overview

### Core Components

#### 1. **Abstract Base Classes** (Lines 1-150)

The project uses abstract interfaces to separate concerns:

- `Vector3`: 3D math helper with add, subtract, multiply, magnitude, and normalize
- `Updatable`: Objects that advance simulation each frame
- `Renderable`: Objects that can draw themselves in OpenGL
- `PhysicsBody`: Objects affected by forces (ships, astronauts, asteroids)
- `GravitySource`: Objects that apply gravitational forces
- `Tetherable`: Objects that can be linked via tether beam
- `Ship`: Player-controlled vessel
- `Astronaut`: Rescue targets
- `Asteroid`: Obstacles
- `CameraRig`: Camera control system
- `HudLayer`: UI overlay
- `Scene`: Collection of game objects
- `GameApplication`: Main game loop

#### 2. **Physics System** (Lines 150-200)

**BlackHole Class**

- Implements inverse-square gravity law: `F = G * M / r²`
- Has event horizon for game-over detection
- Located at origin (0, 0, 0)

**Key Methods:**

```python
force_at(position: Vector3) -> Vector3
is_inside_event_horizon(position: Vector3) -> bool
```

#### 3. **Game Objects** (Lines 200-440)

**SpaceShip**

- Player-controlled with thrust and rotation
- Affected by gravity and drag
- Renders as cylinder body + cube wings + sphere cockpit

**SpaceAstronaut**

- Can be tethered to ship
- Floating animation (sin wave bobbing)
- Renders as sphere helmet + cyan visor + cube backpack
- Auto-follows ship when tethered using unit vector math

**SpaceAsteroid**

- Orbiting obstacles (rendering to be implemented by Member 2)
- Affected by black hole gravity

#### 4. **Rendering System** (Lines 440-500)

**Tether Beam**

- Pulsing GL_LINES effect using sin wave
- Brightness and line width oscillate
- Color fades from ship to astronaut

#### 5. **Game Loop** (Lines 500-773)

**VoidRescuerGame Class**

- GLUT-based game loop running at ~60 FPS
- Handles initialization, update, and rendering
- Camera follows ship from behind and above

---

## Work Distribution

### ✅ Member 1: Physics & Movement (COMPLETED)

**Implemented:**

- ✅ Gravity logic with inverse-square law
- ✅ Tether connection system with smooth following
- ✅ Ship controls (WASD + Arrow keys)
- ✅ Ship rendering (cylinder + cube wings)
- ✅ Astronaut rendering (helmet + visor + backpack)
- ✅ Tether beam rendering (pulsing lines)
- ✅ Floating animation for astronauts

### 🔲 Member 2: Visuals & Camera (TODO)

**Your Tasks:**

1. **Split-Screen HUD**

   - Implement dual viewport rendering
   - Main view: 3D perspective (already working)
   - Mini-map: Top-down orthographic view
   - Location: Add after line 640 in `VoidRescuerGame.render()`

2. **Black Hole Visual Design**

   - Replace simple sphere with layered effect:
     - Dark core sphere
     - Spinning particle ring (use GL_POINTS)
     - Accretion disk with rotation animation
   - Add glow/emission effect
   - Location: Update lines 620-625

3. **Speed Visual Effects (FOV Warp)**
   - Calculate ship speed from velocity magnitude
   - Modify FOV in real-time: `gluPerspective(base_fov + speed_factor, ...)`
   - Add motion blur or star-streak effect
   - Location: Add in `VoidRescuerGame.render()` before `gluLookAt`

**Code Example - Split Screen:**

```python
def render(self):
    # Main viewport (left 75% of screen)
    glViewport(0, 0, int(self.window_width * 0.75), self.window_height)
    self.render_main_view()

    # Mini-map (right 25% of screen)
    glViewport(int(self.window_width * 0.75), 0,
               int(self.window_width * 0.25), int(self.window_height * 0.3))
    self.render_minimap()
```

**Code Example - FOV Warp:**

```python
speed = self.ship.velocity.magnitude()
fov = 45 + min(speed * 0.1, 20)  # Max +20 degrees
gluPerspective(fov, aspect_ratio, 1, 3000)
```

### 🔲 Member 3: Gameplay & Rules (TODO)

**Your Tasks:**

1. **Spawning System**

   - Implement `spawn_astronauts(count: int)` with danger-zone check
   - Implement `spawn_asteroids(count: int)` with orbital velocity
   - Use distance formula to ensure safe spawn locations (>300 units from center)
   - Add to `VoidRescuerGame.initialize()`

2. **Rescue Gate**

   - Create `RescueGate` class (extends `Renderable`)
   - Place at safe location (e.g., 500 units from center)
   - Check collision with tethered astronauts
   - Award points and mark astronaut as rescued
   - Location: New class after line 440

3. **Information Display**
   - Implement HUD text rendering using GLUT bitmap fonts
   - Display: Speed, Energy, Distance from black hole, Astronauts rescued
   - Use `glutBitmapString()` for text
   - Location: Add `render_hud()` method to `VoidRescuerGame`

**Code Example - Spawning:**

```python
def spawn_astronauts(self, count: int):
    for _ in range(count):
        while True:
            angle = random.uniform(0, 360)
            distance = random.uniform(300, 600)  # Safe zone
            x = math.cos(math.radians(angle)) * distance
            y = math.sin(math.radians(angle)) * distance

            # Verify distance from black hole
            if distance >= 300:
                astronaut = SpaceAstronaut(Vector3(x, y, 0))
                astronaut.gravity_source = self.black_hole
                self.astronauts.append(astronaut)
                break
```

**Code Example - HUD Text:**

```python
def render_hud(self):
    glMatrixMode(GL_PROJECTION)
    glPushMatrix()
    glLoadIdentity()
    gluOrtho2D(0, self.window_width, 0, self.window_height)
    glMatrixMode(GL_MODELVIEW)
    glPushMatrix()
    glLoadIdentity()

    glColor3f(1, 1, 1)
    glRasterPos2f(10, self.window_height - 20)

    speed = self.ship.velocity.magnitude()
    text = f"Speed: {speed:.1f}"
    for char in text:
        glutBitmapCharacter(GLUT_BITMAP_9_BY_15, ord(char))

    glPopMatrix()
    glMatrixMode(GL_PROJECTION)
    glPopMatrix()
    glMatrixMode(GL_MODELVIEW)
```

---

## Adding New Features

### Step 1: Define Abstract Interface (if needed)

If your feature is a new type of game object:

```python
class MyFeature(Renderable, Updatable, ABC):
    @abstractmethod
    def special_behavior(self) -> None:
        """Describe what this does."""
```

### Step 2: Create Concrete Implementation

```python
class ConcreteFeature(MyFeature):
    def __init__(self, position: Vector3):
        self.position = position

    def update(self, dt: float) -> None:
        # Game logic here
        pass

    def render(self) -> None:
        glPushMatrix()
        glTranslatef(self.position.x, self.position.y, self.position.z)
        # OpenGL rendering here
        glPopMatrix()

    def special_behavior(self) -> None:
        # Feature-specific logic
        pass
```

### Step 3: Integrate into Game Loop

Add to `VoidRescuerGame`:

```python
def initialize(self):
    # ... existing code ...
    self.my_features: List[ConcreteFeature] = []
    self.my_features.append(ConcreteFeature(Vector3(100, 100, 0)))

def step(self, dt: float):
    # ... existing code ...
    for feature in self.my_features:
        feature.update(dt)

def render(self):
    # ... existing code ...
    for feature in self.my_features:
        feature.render()
```

---

## Rendering Guidelines

### OpenGL Best Practices

1. **Always use glPushMatrix/glPopMatrix pairs**

   ```python
   glPushMatrix()
   # transformations and rendering
   glPopMatrix()
   ```

2. **Set color before drawing**

   ```python
   glColor3f(r, g, b)  # Values 0.0 to 1.0
   glutSolidSphere(radius, slices, stacks)
   ```

3. **Common primitives**

   - `glutSolidSphere(radius, slices, stacks)` - Sphere
   - `glutSolidCube(size)` - Cube
   - `gluCylinder(quadric, base, top, height, slices, stacks)` - Cylinder
   - `GL_LINES`, `GL_TRIANGLES`, `GL_QUADS` - Custom shapes

4. **Transformations**
   - `glTranslatef(x, y, z)` - Move
   - `glRotatef(angle, x, y, z)` - Rotate around axis
   - `glScalef(x, y, z)` - Scale

### Animation Techniques

**Floating/Bobbing:**

```python
wave = math.sin(time * speed) * amplitude
glTranslatef(x, y, z + wave)
```

**Rotation:**

```python
angle = (time * rotation_speed) % 360
glRotatef(angle, 0, 0, 1)  # Rotate around Z-axis
```

**Pulsing:**

```python
scale = 1.0 + math.sin(time * frequency) * 0.2
glScalef(scale, scale, scale)
```

---

## Physics Guidelines

### Adding Forces

All `PhysicsBody` objects integrate forces in their `integrate()` method:

```python
def integrate(self, dt: float) -> None:
    # 1. Calculate forces
    total_force = Vector3()

    if self.gravity_source:
        total_force = total_force + self.gravity_source.force_at(self.position)

    # Add custom forces
    total_force = total_force + self.custom_force

    # 2. F = ma, so a = F/m
    acceleration = total_force * (1.0 / self.mass)

    # 3. Update velocity
    self.velocity = self.velocity + acceleration * dt

    # 4. Update position
    self.position = self.position + self.velocity * dt
```

### Collision Detection

Basic sphere-sphere collision:

```python
def check_collision(obj1, obj2, combined_radius: float) -> bool:
    delta = obj2.position - obj1.position
    distance = delta.magnitude()
    return distance < combined_radius
```

---

## Testing Your Changes

### 1. Test Rendering

Add temporary objects in `initialize()`:

```python
test_obj = MyNewObject(Vector3(200, 0, 0))
```

### 2. Test Physics

Print debug info in `step()`:

```python
print(f"Position: {obj.position}, Velocity: {obj.velocity}")
```

### 3. Test Input

Add keyboard handlers:

```python
def keyboard_callback(key: bytes, x: int, y: int):
    if key == b't':  # Test key
        print("Test triggered!")
        # Your test code
```

---

## Common Patterns

### Pattern 1: Orbital Motion

```python
# Give initial perpendicular velocity for orbit
angle = math.atan2(position.y, position.x)
perpendicular_angle = angle + math.pi / 2
orbital_speed = math.sqrt(gravity_strength / distance)
velocity = Vector3(
    math.cos(perpendicular_angle) * orbital_speed,
    math.sin(perpendicular_angle) * orbital_speed,
    0
)
```

### Pattern 2: Smooth Following

```python
# Move toward target at a percentage of the distance
direction = target.position - self.position
self.position = self.position + direction * smoothing_factor * dt
```

### Pattern 3: Look-At Rotation

```python
# Point object toward target
dx = target.position.x - self.position.x
dy = target.position.y - self.position.y
angle_degrees = math.degrees(math.atan2(dy, dx))
glRotatef(angle_degrees, 0, 0, 1)
```

---

## Troubleshooting

### Objects Not Visible

- Check camera position with `gluLookAt`
- Verify object is within far clipping plane (3000 units)
- Ensure lighting is enabled if using solid primitives
- Check color isn't black or same as background

### Physics Issues

- Verify `gravity_source` is set on objects
- Check `mass` is not zero
- Ensure `integrate()` is called in `update()`
- Print position/velocity values for debugging

### Performance Issues

- Reduce polygon count (slices/stacks) on spheres/cylinders
- Limit number of objects
- Use display lists for static geometry (advanced)

---

## File Structure

```
The-Void-Rescuer/
├── main.py                 # Main game code
├── CONTRIBUTOR_GUIDE.md    # This file
└── ProjectOutline.pdf      # Original project specification
```

---

## Coding Standards

1. **Use type hints** for all function parameters and returns
2. **Document with docstrings** for all classes and public methods
3. **Follow existing naming conventions**:
   - Classes: `PascalCase`
   - Functions/methods: `snake_case`
   - Constants: `UPPER_SNAKE_CASE`
4. **Keep the abstract/concrete separation**:
   - Define interfaces first
   - Implement concrete classes later

---

## Getting Help

1. **Read the code**: The existing implementations show patterns you can follow
2. **Check ProjectOutline.pdf**: Contains detailed math and specifications
3. **Test incrementally**: Add small pieces and verify they work
4. **Use print statements**: Debug by printing values to console

---

## Next Steps

### For Member 2 (Visuals & Camera):

1. Start with split-screen implementation
2. Then enhance black hole rendering
3. Finally add FOV warp effect

### For Member 3 (Gameplay & Rules):

1. Start with spawning system
2. Add rescue gate
3. Implement HUD display

---

**Good luck and happy coding! 🚀**
