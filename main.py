"""
Project scaffolding for The Void Rescuer.

This file sets up abstract interfaces and lightweight data structures so we can
iterate on gameplay (gravity, tethering, HUD) without committing to OpenGL
details yet.
"""

from __future__ import annotations

import math
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Iterable, Optional, List

try:
	from OpenGL.GL import *
	from OpenGL.GLU import *
	from OpenGL.GLUT import *
except ImportError:
	raise ImportError(
		"PyOpenGL not found. Install with: pip install PyOpenGL"
	)


@dataclass
class Vector3:
	x: float = 0.0
	y: float = 0.0
	z: float = 0.0

	def __add__(self, other: "Vector3") -> "Vector3":
		return Vector3(self.x + other.x, self.y + other.y, self.z + other.z)

	def __sub__(self, other: "Vector3") -> "Vector3":
		return Vector3(self.x - other.x, self.y - other.y, self.z - other.z)

	def __mul__(self, scalar: float) -> "Vector3":
		return Vector3(self.x * scalar, self.y * scalar, self.z * scalar)

	__rmul__ = __mul__

	def magnitude(self) -> float:
		return (self.x * self.x + self.y * self.y + self.z * self.z) ** 0.5

	def normalized(self) -> "Vector3":
		mag = self.magnitude()
		if mag == 0:
			return Vector3()
		return Vector3(self.x / mag, self.y / mag, self.z / mag)


class Updatable(ABC):
	@abstractmethod
	def update(self, dt: float) -> None:
		"""Advance simulation by dt seconds."""


class Renderable(ABC):
	@abstractmethod
	def render(self) -> None:
		"""Draw the object (OpenGL implementation to be supplied later)."""


class PhysicsBody(Updatable, ABC):
	mass: float
	position: Vector3
	velocity: Vector3

	@abstractmethod
	def integrate(self, dt: float) -> None:
		"""Integrate motion; subclasses decide which forces apply."""

	def update(self, dt: float) -> None:
		self.integrate(dt)


class GravitySource(ABC):
	@abstractmethod
	def force_at(self, position: Vector3) -> Vector3:
		"""Return gravitational force applied at a world position."""


class Tetherable(ABC):
	tethered_to: Optional["Ship"]

	@abstractmethod
	def attach_tether(self, ship: "Ship") -> None:
		"""Link this object to a ship."""

	@abstractmethod
	def detach_tether(self) -> None:
		"""Remove any existing tether link."""

	@abstractmethod
	def update_tether(self, dt: float) -> None:
		"""Move toward the tether anchor each frame."""


class Ship(PhysicsBody, Renderable, ABC):
	heading_degrees: float

	@abstractmethod
	def apply_thrust(self, amount: float) -> None:
		"""Accelerate along the current heading."""

	@abstractmethod
	def rotate(self, delta_degrees: float) -> None:
		"""Adjust heading based on input."""


class Astronaut(PhysicsBody, Renderable, Tetherable, ABC):
	is_rescued: bool


class Asteroid(PhysicsBody, Renderable, ABC):
	radius: float


class CameraRig(Updatable, Renderable, ABC):
	@abstractmethod
	def follow(self, target: Ship) -> None:
		"""Adjust camera parameters based on a target ship."""


class HudLayer(Renderable, Updatable, ABC):
	@abstractmethod
	def set_status(self, speed: float, energy: float, distance: float) -> None:
		"""Update HUD metrics displayed to the player."""


class Scene(ABC):
	@abstractmethod
	def objects(self) -> Iterable[Renderable]:
		"""Return all renderable objects in the scene."""

	@abstractmethod
	def update(self, dt: float) -> None:
		"""Advance simulation for all objects in the scene."""


class GameApplication(ABC):
	@abstractmethod
	def initialize(self) -> None:
		"""Set up OpenGL state, load assets, and seed the world."""

	@abstractmethod
	def step(self, dt: float) -> None:
		"""Single frame update including input, simulation, and rendering."""

	@abstractmethod
	def shutdown(self) -> None:
		"""Release resources and exit cleanly."""


# =========================================================================
# CONCRETE IMPLEMENTATIONS - Member 1: Physics & Movement
# =========================================================================


class BlackHole(GravitySource):
	"""
	Singularity at (0,0,0) that pulls objects using inverse-square law.
	"""

	def __init__(self, strength: float = 50000.0, event_horizon: float = 150.0):
		self.position = Vector3(0, 0, 0)
		self.strength = strength
		self.event_horizon = event_horizon

	def force_at(self, position: Vector3) -> Vector3:
		"""
		Calculate gravitational force using inverse-square law.
		F = G * M / r^2, directed toward center.
		"""
		delta = self.position - position
		distance = delta.magnitude()

		if distance < 1.0:  # Prevent division by zero
			distance = 1.0

		# Inverse square law: force magnitude
		force_magnitude = self.strength / (distance * distance)

		# Direction toward black hole (normalized)
		direction = delta.normalized()

		return direction * force_magnitude

	def is_inside_event_horizon(self, position: Vector3) -> bool:
		"""Check if position has crossed the event horizon (game over)."""
		distance = (position - self.position).magnitude()
		return distance < self.event_horizon


class SpaceShip(Ship):
	"""
	Player-controlled ship with thrust, rotation, and gravity effects.
	"""

	def __init__(
		self,
		position: Vector3 = Vector3(400, 0, 0),
		heading_degrees: float = 180.0,
		mass: float = 10.0,
	):
		self.position = position
		self.velocity = Vector3()
		self.heading_degrees = heading_degrees
		self.mass = mass
		self.thrust_power = 50.0
		self.drag_coefficient = 0.98  # Natural slowdown
		self.gravity_source: Optional[GravitySource] = None

	def apply_thrust(self, amount: float) -> None:
		"""
		Accelerate along current heading.
		Using bullet movement math: speed * cos(theta), speed * sin(theta)
		"""
		angle_rad = math.radians(self.heading_degrees)
		thrust_x = amount * self.thrust_power * math.cos(angle_rad)
		thrust_y = amount * self.thrust_power * math.sin(angle_rad)

		acceleration = Vector3(thrust_x, thrust_y, 0) * (1.0 / self.mass)
		self.velocity = self.velocity + acceleration

	def rotate(self, delta_degrees: float) -> None:
		"""Adjust heading based on input."""
		self.heading_degrees += delta_degrees
		# Keep angle in [0, 360)
		self.heading_degrees %= 360

	def integrate(self, dt: float) -> None:
		"""
		Update position based on velocity and apply gravity + drag.
		"""
		# Apply gravity force if source exists
		if self.gravity_source:
			gravity_force = self.gravity_source.force_at(self.position)
			acceleration = gravity_force * (1.0 / self.mass)
			self.velocity = self.velocity + acceleration * dt

		# Apply drag (natural slowdown)
		self.velocity = self.velocity * self.drag_coefficient

		# Update position
		self.position = self.position + self.velocity * dt

	def render(self) -> None:
		"""Render the spaceship using OpenGL."""
		glPushMatrix()
		
		# Move to ship position
		glTranslatef(self.position.x, self.position.y, self.position.z)
		
		# Rotate to heading
		glRotatef(self.heading_degrees, 0, 0, 1)
		
		# Ship body (cylinder)
		glColor3f(0.7, 0.7, 0.9)  # Light blue
		glPushMatrix()
		glRotatef(-90, 1, 0, 0)  # Point cylinder forward
		quadric = gluNewQuadric()
		gluCylinder(quadric, 8, 5, 30, 16, 16)  # Base radius, top radius, height
		gluDeleteQuadric(quadric)
		glPopMatrix()
		
		# Left wing (cube)
		glColor3f(0.5, 0.5, 0.8)
		glPushMatrix()
		glTranslatef(-15, 0, -10)
		glScalef(10, 2, 15)
		glutSolidCube(1)
		glPopMatrix()
		
		# Right wing (cube)
		glPushMatrix()
		glTranslatef(15, 0, -10)
		glScalef(10, 2, 15)
		glutSolidCube(1)
		glPopMatrix()
		
		# Cockpit (sphere)
		glColor3f(0.3, 0.6, 0.9)
		glPushMatrix()
		glTranslatef(0, 0, 15)
		glutSolidSphere(6, 16, 16)
		glPopMatrix()
		
		glPopMatrix()


class SpaceAstronaut(Astronaut):
	"""
	Stranded astronaut that can be tethered and rescued.
	"""

	def __init__(
		self,
		position: Vector3,
		mass: float = 5.0,
	):
		self.position = position
		self.velocity = Vector3()
		self.mass = mass
		self.is_rescued = False
		self.tethered_to: Optional[Ship] = None
		self.gravity_source: Optional[GravitySource] = None
		self.tether_max_distance = 500.0
		self.tether_pull_speed = 0.3
		self.float_offset = 0.0  # For floating animation

	def attach_tether(self, ship: Ship) -> None:
		"""Link this astronaut to a ship."""
		self.tethered_to = ship

	def detach_tether(self) -> None:
		"""Remove tether link."""
		self.tethered_to = None

	def update_tether(self, dt: float) -> None:
		"""
		Move astronaut toward tethered ship.
		Uses unit vector and ship's speed for smooth following.
		"""
		if not self.tethered_to:
			return

		# Calculate direction from astronaut to ship
		direction = self.tethered_to.position - self.position
		distance = direction.magnitude()

		# Check if tether should snap
		if distance > self.tether_max_distance:
			self.detach_tether()
			return

		# Normalize direction to get unit vector
		if distance > 0.1:
			unit_direction = direction.normalized()

			# Pull astronaut toward ship
			# Speed is proportional to distance and ship's velocity
			pull_speed = distance * self.tether_pull_speed
			if hasattr(self.tethered_to, 'velocity'):
				ship_speed = self.tethered_to.velocity.magnitude()
				pull_speed += ship_speed * 0.5

			self.velocity = unit_direction * pull_speed

	def integrate(self, dt: float) -> None:
		"""
		Update position based on velocity and apply gravity.
		"""
		# Apply gravity force if source exists
		if self.gravity_source:
			gravity_force = self.gravity_source.force_at(self.position)
			acceleration = gravity_force * (1.0 / self.mass)
			self.velocity = self.velocity + acceleration * dt

		# If tethered, update tether movement
		if self.tethered_to:
			self.update_tether(dt)

		# Update position
		self.position = self.position + self.velocity * dt
		
		# Update floating animation
		self.float_offset += dt * 2.0  # Animation speed

	def render(self) -> None:
		"""Render the astronaut with floating animation."""
		glPushMatrix()
		
		# Move to astronaut position
		float_wave = math.sin(self.float_offset) * 5.0  # Bobbing motion
		glTranslatef(self.position.x, self.position.y, self.position.z + float_wave)
		
		# Helmet (sphere)
		glColor3f(0.9, 0.9, 1.0)  # White-ish helmet
		glPushMatrix()
		glutSolidSphere(8, 16, 16)
		glPopMatrix()
		
		# Visor (cyan sphere, slightly smaller and forward)
		glColor3f(0.0, 1.0, 1.0)  # Cyan
		glPushMatrix()
		glTranslatef(0, 5, 0)
		glutSolidSphere(4, 12, 12)
		glPopMatrix()
		
		# Life Support Pack (cube behind)
		glColor3f(0.6, 0.6, 0.6)  # Gray backpack
		glPushMatrix()
		glTranslatef(0, -6, 0)
		glScalef(1.2, 1.5, 0.5)  # Flat backpack shape
		glutSolidCube(8)
		glPopMatrix()
		
		glPopMatrix()


class SpaceAsteroid(Asteroid):
	"""
	Orbiting asteroid obstacle.
	"""

	def __init__(
		self,
		position: Vector3,
		radius: float = 20.0,
		mass: float = 15.0,
	):
		self.position = position
		self.velocity = Vector3()
		self.radius = radius
		self.mass = mass
		self.gravity_source: Optional[GravitySource] = None

	def integrate(self, dt: float) -> None:
		"""
		Update position based on velocity and apply gravity.
		"""
		# Apply gravity force if source exists
		if self.gravity_source:
			gravity_force = self.gravity_source.force_at(self.position)
			acceleration = gravity_force * (1.0 / self.mass)
			self.velocity = self.velocity + acceleration * dt

		# Update position
		self.position = self.position + self.velocity * dt

	def render(self) -> None:
		"""Placeholder for OpenGL rendering."""
		pass


def render_tether_beam(ship: SpaceShip, astronaut: SpaceAstronaut, time_value: float) -> None:
	"""
	Render pulsing tether beam between ship and astronaut.
	"""
	if not astronaut.tethered_to:
		return
	
	# Pulsing effect using sine wave
	pulse = (math.sin(time_value * 5.0) + 1.0) * 0.5  # Oscillates 0 to 1
	brightness = 0.5 + pulse * 0.5  # 0.5 to 1.0
	
	# Line width pulsing
	line_width = 2.0 + pulse * 2.0  # 2 to 4
	glLineWidth(line_width)
	
	# Draw beam
	glBegin(GL_LINES)
	glColor3f(0.0, brightness, brightness)  # Cyan beam
	glVertex3f(ship.position.x, ship.position.y, ship.position.z)
	glColor3f(0.0, brightness * 0.5, brightness * 0.5)  # Fade at astronaut
	glVertex3f(astronaut.position.x, astronaut.position.y, astronaut.position.z)
	glEnd()
	
	# Reset line width
	glLineWidth(1.0)


class InputController:
	"""
	Handles keyboard input for ship control.
	"""

	def __init__(self, ship: SpaceShip):
		self.ship = ship
		self.keys_pressed = set()

	def key_down(self, key: str) -> None:
		"""Register key press."""
		self.keys_pressed.add(key.lower())

	def key_up(self, key: str) -> None:
		"""Register key release."""
		self.keys_pressed.discard(key.lower())

	def update(self, dt: float) -> None:
		"""
		Process input and update ship accordingly.
		WASD for movement, Arrow keys for rotation.
		"""
		# Rotation (Arrow keys or A/D)
		if 'left' in self.keys_pressed or 'a' in self.keys_pressed:
			self.ship.rotate(5.0)

		if 'right' in self.keys_pressed or 'd' in self.keys_pressed:
			self.ship.rotate(-5.0)

		# Thrust (WASD or Arrow keys)
		if 'w' in self.keys_pressed or 'up' in self.keys_pressed:
			self.ship.apply_thrust(1.0)  # Forward

		if 's' in self.keys_pressed or 'down' in self.keys_pressed:
			self.ship.apply_thrust(-0.5)  # Backward (slower)


# =========================================================================
# GAME APPLICATION & GLUT SETUP
# =========================================================================


class VoidRescuerGame(GameApplication):
	"""
	Main game application with OpenGL/GLUT rendering.
	"""

	def __init__(self):
		self.window_width = 1200
		self.window_height = 800
		self.game_over = False
		self.last_time = time.time()
		self.current_time = 0.0
		
		# Game objects
		self.black_hole: Optional[BlackHole] = None
		self.ship: Optional[SpaceShip] = None
		self.astronauts: List[SpaceAstronaut] = []
		self.asteroids: List[SpaceAsteroid] = []
		self.input_controller: Optional[InputController] = None

	def initialize(self) -> None:
		"""Set up OpenGL state and initialize game objects."""
		# OpenGL setup
		glClearColor(0.0, 0.0, 0.05, 1.0)  # Dark blue space background
		glEnable(GL_DEPTH_TEST)
		glEnable(GL_LIGHTING)
		glEnable(GL_LIGHT0)
		glEnable(GL_COLOR_MATERIAL)
		glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)
		
		# Lighting setup
		glLightfv(GL_LIGHT0, GL_POSITION, [500, 500, 500, 1])
		glLightfv(GL_LIGHT0, GL_AMBIENT, [0.3, 0.3, 0.3, 1])
		glLightfv(GL_LIGHT0, GL_DIFFUSE, [0.8, 0.8, 0.8, 1])
		
		# Projection setup
		glMatrixMode(GL_PROJECTION)
		glLoadIdentity()
		gluPerspective(45, self.window_width / self.window_height, 1, 3000)
		glMatrixMode(GL_MODELVIEW)
		
		# Create game objects
		self.black_hole = BlackHole(strength=50000.0, event_horizon=150.0)
		self.ship = SpaceShip(position=Vector3(400, 0, 0))
		self.ship.gravity_source = self.black_hole
		
		# Create test astronauts
		for angle in [45, 135, 225, 315]:
			rad = math.radians(angle)
			pos = Vector3(math.cos(rad) * 300, math.sin(rad) * 300, 0)
			astronaut = SpaceAstronaut(position=pos)
			astronaut.gravity_source = self.black_hole
			self.astronauts.append(astronaut)
		
		# Create test asteroids
		for angle in [0, 90, 180, 270]:
			rad = math.radians(angle)
			pos = Vector3(math.cos(rad) * 250, math.sin(rad) * 250, 0)
			asteroid = SpaceAsteroid(position=pos)
			asteroid.gravity_source = self.black_hole
			self.asteroids.append(asteroid)
		
		self.input_controller = InputController(self.ship)

	def step(self, dt: float) -> None:
		"""Single frame update including input, simulation, and rendering."""
		if self.game_over:
			return
		
		# Update current time for animations
		self.current_time += dt
		
		# Process input
		if self.input_controller:
			self.input_controller.update(dt)
		
		# Update physics
		self.ship.update(dt)
		for astronaut in self.astronauts:
			astronaut.update(dt)
		for asteroid in self.asteroids:
			asteroid.update(dt)
		
		# Check event horizon
		if self.black_hole.is_inside_event_horizon(self.ship.position):
			self.game_over = True
			print("GAME OVER! Ship crossed the event horizon!")
		
		# Render
		self.render()

	def render(self) -> None:
		"""Render the scene."""
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
		glLoadIdentity()
		
		# Camera setup - follow ship from behind and above
		if self.ship:
			angle_rad = math.radians(self.ship.heading_degrees)
			cam_distance = 200
			cam_x = self.ship.position.x - cam_distance * math.cos(angle_rad)
			cam_y = self.ship.position.y - cam_distance * math.sin(angle_rad)
			cam_z = self.ship.position.z + 100
			
			gluLookAt(
				cam_x, cam_y, cam_z,  # Camera position
				self.ship.position.x, self.ship.position.y, self.ship.position.z,  # Look at ship
				0, 0, 1  # Up vector
			)
		
		# Draw black hole (simple marker for now)
		glPushMatrix()
		glColor3f(0.2, 0.0, 0.2)  # Dark purple
		glutSolidSphere(self.black_hole.event_horizon, 32, 32)
		glPopMatrix()
		
		# Draw ship
		if self.ship:
			self.ship.render()
		
		# Draw astronauts and tethers
		for astronaut in self.astronauts:
			if astronaut.tethered_to:
				render_tether_beam(self.ship, astronaut, self.current_time)
			astronaut.render()
		
		# Draw asteroids
		for asteroid in self.asteroids:
			asteroid.render()
		
		glutSwapBuffers()

	def shutdown(self) -> None:
		"""Release resources and exit cleanly."""
		print("Shutting down...")


# Global game instance for GLUT callbacks
game: Optional[VoidRescuerGame] = None


def display_callback() -> None:
	"""GLUT display callback."""
	if game:
		game.render()


def timer_callback(value: int) -> None:
	"""GLUT timer callback for game loop."""
	if game and not game.game_over:
		current = time.time()
		dt = current - game.last_time
		game.last_time = current
		
		game.step(dt)
		glutPostRedisplay()
		glutTimerFunc(16, timer_callback, 0)  # ~60 FPS


def keyboard_callback(key: bytes, x: int, y: int) -> None:
	"""GLUT keyboard down callback."""
	if game and game.input_controller:
		key_str = key.decode('utf-8').lower()
		game.input_controller.key_down(key_str)
		
		# Escape to quit
		if key == b'\x1b':  # ESC
			sys.exit(0)
		
		# Space to tether nearest astronaut
		if key == b' ':
			nearest = None
			min_dist = float('inf')
			for astronaut in game.astronauts:
				if not astronaut.is_rescued:
					dist = (astronaut.position - game.ship.position).magnitude()
					if dist < min_dist and dist < 200:  # Within 200 units
						min_dist = dist
						nearest = astronaut
			
			if nearest:
				if nearest.tethered_to:
					nearest.detach_tether()
					print("Tether released")
				else:
					nearest.attach_tether(game.ship)
					print(f"Tethered! Distance: {min_dist:.1f}")


def keyboard_up_callback(key: bytes, x: int, y: int) -> None:
	"""GLUT keyboard up callback."""
	if game and game.input_controller:
		key_str = key.decode('utf-8').lower()
		game.input_controller.key_up(key_str)


def special_callback(key: int, x: int, y: int) -> None:
	"""GLUT special key down callback (arrow keys)."""
	if game and game.input_controller:
		if key == GLUT_KEY_LEFT:
			game.input_controller.key_down('left')
		elif key == GLUT_KEY_RIGHT:
			game.input_controller.key_down('right')
		elif key == GLUT_KEY_UP:
			game.input_controller.key_down('up')
		elif key == GLUT_KEY_DOWN:
			game.input_controller.key_down('down')


def special_up_callback(key: int, x: int, y: int) -> None:
	"""GLUT special key up callback (arrow keys)."""
	if game and game.input_controller:
		if key == GLUT_KEY_LEFT:
			game.input_controller.key_up('left')
		elif key == GLUT_KEY_RIGHT:
			game.input_controller.key_up('right')
		elif key == GLUT_KEY_UP:
			game.input_controller.key_up('up')
		elif key == GLUT_KEY_DOWN:
			game.input_controller.key_up('down')


def main() -> None:
	"""Entry point - initialize GLUT and start game loop."""
	global game
	
	print("="*60)
	print("THE VOID RESCUER - Member 1: Physics & Movement Demo")
	print("="*60)
	print("\nControls:")
	print("  W/Up Arrow    - Forward thrust")
	print("  S/Down Arrow  - Backward thrust")
	print("  A/Left Arrow  - Rotate left")
	print("  D/Right Arrow - Rotate right")
	print("  SPACE         - Tether/untether nearest astronaut")
	print("  ESC           - Quit")
	print("\nObjective: Test physics and tethering!")
	print("="*60)
	print()
	
	# Initialize GLUT
	glutInit()
	glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH)
	glutInitWindowSize(1200, 800)
	glutInitWindowPosition(100, 100)
	glutCreateWindow(b"The Void Rescuer - Physics Demo")
	
	# Create and initialize game
	game = VoidRescuerGame()
	game.initialize()
	
	# Register callbacks
	glutDisplayFunc(display_callback)
	glutKeyboardFunc(keyboard_callback)
	glutKeyboardUpFunc(keyboard_up_callback)
	glutSpecialFunc(special_callback)
	glutSpecialUpFunc(special_up_callback)
	glutTimerFunc(16, timer_callback, 0)
	
	# Start game loop
	glutMainLoop()


if __name__ == "__main__":
	import sys
	main()
