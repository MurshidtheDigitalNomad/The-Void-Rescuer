# The Void Rescuer

A 3D space rescue game built with Python and OpenGL where players pilot a ship to save astronauts near a black hole.

## Installation

```bash
pip install PyOpenGL
```

## Running the Game

```bash
python main.py
```

## Controls

- **W / Up Arrow**: Forward thrust
- **S / Down Arrow**: Backward thrust
- **A / Left Arrow**: Rotate left
- **D / Right Arrow**: Rotate right
- **SPACE**: Tether/untether nearest astronaut
- **ESC**: Quit

## Current Status

✅ **Member 1: Physics & Movement** - COMPLETE

- Inverse-square gravity physics
- Ship controls with thrust and rotation
- Tether system with smooth following
- Astronaut floating animation
- Pulsing tether beam rendering

🔲 **Member 2: Visuals & Camera** - TODO

- Split-screen HUD (main view + mini-map)
- Enhanced black hole visual effects
- FOV warp for speed effects

🔲 **Member 3: Gameplay & Rules** - TODO

- Spawning system for astronauts/asteroids
- Rescue gate and scoring
- HUD information display

## For Contributors

See [CONTRIBUTOR_GUIDE.md](CONTRIBUTOR_GUIDE.md) for detailed implementation instructions.

## Project Structure

- `main.py` - Complete game implementation
- Abstract base classes define the architecture
- Concrete implementations for physics and rendering
- GLUT-based game loop running at 60 FPS
