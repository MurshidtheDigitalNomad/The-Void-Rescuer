"""
Project scaffolding for The Void Rescuer.

This file sets up abstract interfaces and lightweight data structures so we can
iterate on gameplay (gravity, tethering, HUD) without committing to OpenGL
details yet.
"""

from __future__ import annotations

import math
import time
import random
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
		Incorporates the mass of the object for realistic attraction.
		"""
		delta = self.position - position
		distance = delta.magnitude()

		if distance < 1.0:  # Prevent division by zero
			distance = 1.0

		# Inverse square law with stronger pull very close to singularity
		# Force increases dramatically as distance decreases
		force_magnitude = self.strength / (distance * distance)
		
		# Add exponential factor for extreme pull near event horizon
		if distance < 200:
			force_magnitude *= (200.0 / distance) ** 0.5  # Extra pull near singularity

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
		mass: float = 50.0,  # Increased mass for stronger gravity effects
	):
		self.position = position
		self.velocity = Vector3()
		self.heading_degrees = heading_degrees
		self.mass = mass
		self.thrust_power = 100.0  # Base thrust power
		self.thrust_power_min = 20.0  # Minimum thrust
		self.thrust_power_max = 300.0  # Maximum thrust
		self.drag_coefficient = 0.98  # Natural slowdown
		self.gravity_source: Optional[GravitySource] = None
		# Fuel system
		self.fuel = 1000.0  # Starting fuel
		self.fuel_max = 1000.0  # Maximum fuel capacity
		self.fuel_consumption_rate = 0.05  # Base consumption per unit power per second

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

	def increase_power(self) -> None:
		"""Increase thrust power (P key)."""
		self.thrust_power = min(self.thrust_power + 10.0, self.thrust_power_max)
		print(f"Thrust power: {self.thrust_power:.1f}")

	def decrease_power(self) -> None:
		"""Decrease thrust power (O key)."""
		self.thrust_power = max(self.thrust_power - 10.0, self.thrust_power_min)
		print(f"Thrust power: {self.thrust_power:.1f}")

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

		# Consume fuel based on thrust power
		# Higher power = higher consumption
		fuel_consumed = self.thrust_power * self.fuel_consumption_rate * dt
		self.fuel = max(0, self.fuel - fuel_consumed)

		# Update position
		self.position = self.position + self.velocity * dt

	def render(self) -> None:
		"""Render the spaceship using OpenGL - Star Wars X-wing style."""
		glPushMatrix()
		
		# Move to ship position
		glTranslatef(self.position.x, self.position.y, self.position.z)
		
		# Rotate to heading
		glRotatef(self.heading_degrees, 0, 0, 1)
		
		# Main fuselage (elongated body)
		glColor3f(0.85, 0.85, 0.90)  # Light gray/white
		glPushMatrix()
		glRotatef(-90, 1, 0, 0)
		quadric = gluNewQuadric()
		gluCylinder(quadric, 6, 4, 35, 16, 16)  # Long tapered body
		gluDeleteQuadric(quadric)
		glPopMatrix()
		
		# Cockpit canopy (front sphere)
		glColor3f(0.2, 0.3, 0.4)  # Dark blue-gray glass
		glPushMatrix()
		glTranslatef(0, 0, 18)
		glutSolidSphere(5, 12, 12)
		glPopMatrix()
		
		# Nose cone
		glColor3f(0.9, 0.3, 0.2)  # Red/orange nose
		glPushMatrix()
		glTranslatef(0, 0, 22)
		glRotatef(-90, 1, 0, 0)
		quadric = gluNewQuadric()
		gluCylinder(quadric, 3, 0, 8, 12, 12)  # Cone shape
		gluDeleteQuadric(quadric)
		glPopMatrix()
		
		# Top-left wing (X-wing configuration)
		glColor3f(0.75, 0.75, 0.80)
		glPushMatrix()
		glTranslatef(-12, 0, 0)
		glRotatef(15, 0, 1, 0)  # Slight angle
		glScalef(18, 1.5, 12)
		glutSolidCube(1)
		glPopMatrix()
		
		# Top-right wing
		glPushMatrix()
		glTranslatef(12, 0, 0)
		glRotatef(-15, 0, 1, 0)
		glScalef(18, 1.5, 12)
		glutSolidCube(1)
		glPopMatrix()
		
		# Bottom-left wing
		glPushMatrix()
		glTranslatef(-12, 0, -5)
		glRotatef(15, 0, 1, 0)
		glScalef(18, 1.5, 12)
		glutSolidCube(1)
		glPopMatrix()
		
		# Bottom-right wing
		glPushMatrix()
		glTranslatef(12, 0, -5)
		glRotatef(-15, 0, 1, 0)
		glScalef(18, 1.5, 12)
		glutSolidCube(1)
		glPopMatrix()
		
		# Engine glow (left)
		glColor3f(0.3, 0.6, 1.0)  # Blue engine glow
		glPushMatrix()
		glTranslatef(-12, 0, -15)
		glutSolidSphere(3, 8, 8)
		glPopMatrix()
		
		# Engine glow (right)
		glPushMatrix()
		glTranslatef(12, 0, -15)
		glutSolidSphere(3, 8, 8)
		glPopMatrix()
		
		# Rear stabilizer
		glColor3f(0.8, 0.8, 0.85)
		glPushMatrix()
		glTranslatef(0, 0, -18)
		glScalef(8, 12, 2)
		glutSolidCube(1)
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

		# Check if astronaut reached the ship (rescued)
		if distance < 20.0:
			self.is_rescued = True
			self.detach_tether()
			print(f"✓ Astronaut Rescued! Total rescued: {self.is_rescued}")
			return

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
		"""Render the astronaut with floating animation and rescue status."""
		glPushMatrix()
		
		# Move to astronaut position
		float_wave = math.sin(self.float_offset) * 5.0  # Bobbing motion
		glTranslatef(self.position.x, self.position.y, self.position.z + float_wave)
		
		# Helmet (sphere) - color changes if rescued
		if self.is_rescued:
			glColor3f(0.0, 1.0, 0.0)  # Green - rescued!
		else:
			glColor3f(0.9, 0.9, 1.0)  # White-ish helmet
		glPushMatrix()
		glutSolidSphere(8, 16, 16)
		glPopMatrix()
		
		# Visor (cyan sphere, slightly smaller and forward)
		if self.is_rescued:
			glColor3f(0.0, 1.0, 0.5)  # Light green
		else:
			glColor3f(0.0, 1.0, 1.0)  # Cyan
		glPushMatrix()
		glTranslatef(0, 5, 0)
		glutSolidSphere(4, 12, 12)
		glPopMatrix()
		
		# Life Support Pack (cube behind)
		if self.is_rescued:
			glColor3f(0.2, 0.8, 0.2)  # Darker green
		else:
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


class RescueGate(Renderable):
	"""
	Safe zone where astronauts must be brought to be rescued.
	Located far from the black hole.
	"""

	def __init__(self, position: Vector3 = Vector3(-600, 0, 0), radius: float = 50.0):
		self.position = position
		self.radius = radius
		self.score = 0

	def is_inside_gate(self, position: Vector3) -> bool:
		"""Check if a position is inside the gate."""
		distance = (position - self.position).magnitude()
		return distance < self.radius

	def add_score(self, points: int = 100) -> None:
		"""Add points when astronaut is rescued."""
		self.score += points

	def render(self) -> None:
		"""Render the rescue gate as a glowing sphere with a ring."""
		glPushMatrix()
		
		# Move to gate position
		glTranslatef(self.position.x, self.position.y, self.position.z)
		
		# Outer glowing sphere (semi-transparent effect via color)
		glColor4f(0.0, 1.0, 0.5, 0.3)  # Semi-transparent green
		glutSolidSphere(self.radius + 10, 32, 32)
		
		# Inner solid sphere (brighter green)
		glColor3f(0.0, 1.0, 0.5)  # Bright green
		glutSolidSphere(self.radius, 24, 24)
		
		# Ring marker (yellow)
		glColor3f(1.0, 1.0, 0.0)
		glBegin(GL_LINE_LOOP)
		for i in range(32):
			angle = (i / 32.0) * 2 * math.pi
			x = self.radius * math.cos(angle)
			y = self.radius * math.sin(angle)
			glVertex3f(x, y, 0)
		glEnd()
		
		glPopMatrix()


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


def calculate_required_power(black_hole: BlackHole, astronaut_position: Vector3, astronaut_mass: float = 5.0) -> float:
	"""
	Calculate the minimum thrust power needed to pull an astronaut against gravity.
	Returns the required power value in the same units as ship.thrust_power.
	"""
	gravity_force = black_hole.force_at(astronaut_position)
	force_magnitude = gravity_force.magnitude()
	
	# Convert gravity force to required power
	# Power needed = force_magnitude * scaling_factor to match thrust power units
	# Scale so it aligns with thrust_power range (20-300)
	required_power = force_magnitude * 10.0  # Direct scaling of force to power units
	return required_power


def render_hud_text(window_width: int, window_height: int, ship: SpaceShip, 
	                   astronauts: List[SpaceAstronaut], black_hole: BlackHole, gate: Optional["RescueGate"] = None, game_over: bool = False) -> None:
	"""
	Render HUD text in top-left corner showing power, fuel, speed, energy, distance, and rescue information.
	"""
	# Find nearest astronaut
	nearest_astronaut = None
	min_distance = float('inf')
	
	for astronaut in astronauts:
		if not astronaut.is_rescued:
			dist = (astronaut.position - ship.position).magnitude()
			if dist < min_distance:
				min_distance = dist
				nearest_astronaut = astronaut
	
	# Count rescued astronauts
	rescued_count = sum(1 for a in astronauts if a.is_rescued)
	total_astronauts = len(astronauts)
	
	# Calculate ship speed
	ship_speed = ship.velocity.magnitude()
	
	# Calculate distance to rescue gate
	distance_to_gate = float('inf')
	if gate:
		distance_to_gate = (ship.position - gate.position).magnitude()
	
	# Switch to 2D projection for HUD
	glMatrixMode(GL_PROJECTION)
	glPushMatrix()
	glLoadIdentity()
	glOrtho(0, window_width, window_height, 0, -1, 1)
	glMatrixMode(GL_MODELVIEW)
	glPushMatrix()
	glLoadIdentity()
	
	# Disable depth testing for HUD
	glDisable(GL_DEPTH_TEST)
	
	# Display ship speed
	glColor3f(0.2, 1.0, 0.2)  # Bright green
	glRasterPos2f(15, 25)
	text = f"Speed: {ship_speed:.1f} u/s"
	for char in text:
		glutBitmapCharacter(GLUT_BITMAP_HELVETICA_18, ord(char))
	
	# Always display current power
	glRasterPos2f(15, 50)
	text = f"Current Power: {ship.thrust_power:.1f}"
	for char in text:
		glutBitmapCharacter(GLUT_BITMAP_HELVETICA_18, ord(char))
	
	# Display fuel level (energy) with color coding
	fuel_percent = (ship.fuel / ship.fuel_max) * 100 if ship.fuel_max > 0 else 0
	if fuel_percent > 50:
		glColor3f(0.2, 1.0, 0.2)  # Green - plenty of fuel
	elif fuel_percent > 20:
		glColor3f(1.0, 1.0, 0.0)  # Yellow - low fuel warning
	else:
		glColor3f(1.0, 0.0, 0.0)  # Red - critical fuel
	
	glRasterPos2f(15, 75)
	text = f"Energy: {ship.fuel:.0f}/{ship.fuel_max:.0f} ({fuel_percent:.1f}%)"
	for char in text:
		glutBitmapCharacter(GLUT_BITMAP_HELVETICA_18, ord(char))
	
	# Display rescue count
	glColor3f(0.2, 1.0, 0.2)  # Green for rescue status
	glRasterPos2f(15, 100)
	text = f"Rescued: {rescued_count}/{total_astronauts}"
	for char in text:
		glutBitmapCharacter(GLUT_BITMAP_HELVETICA_18, ord(char))
	
	# Display distance to rescue gate
	if gate:
		glColor3f(1.0, 1.0, 0.0)  # Yellow for gate info
		glRasterPos2f(15, 125)
		text = f"Distance to Gate: {distance_to_gate:.1f}"
		for char in text:
			glutBitmapCharacter(GLUT_BITMAP_HELVETICA_18, ord(char))
		
		glColor3f(0.0, 1.0, 1.0)  # Cyan for score
		glRasterPos2f(15, 150)
		text = f"Gate Score: {gate.score}"
		for char in text:
			glutBitmapCharacter(GLUT_BITMAP_HELVETICA_18, ord(char))

	
	# Display required power if astronaut is nearby
	if nearest_astronaut and min_distance < 500:
		required_power = calculate_required_power(black_hole, nearest_astronaut.position)
		current_power = ship.thrust_power
		
		# Color based on whether power is sufficient
		if current_power >= required_power:
			glColor3f(0.0, 1.0, 0.0)  # Green - sufficient power
		else:
			glColor3f(1.0, 0.0, 0.0)  # Red - insufficient power
		
		glRasterPos2f(15, 175)
		text = f"Required Power: {required_power:.1f}"
		for char in text:
			glutBitmapCharacter(GLUT_BITMAP_HELVETICA_18, ord(char))
		
		glRasterPos2f(15, 200)
		text = f"Distance to Astronaut: {min_distance:.1f}"
		for char in text:
			glutBitmapCharacter(GLUT_BITMAP_HELVETICA_18, ord(char))
	
	# Display game over message if game is over
	if game_over:
		glColor3f(1.0, 0.0, 0.0)  # Red for game over
		glRasterPos2f(window_width // 2 - 200, window_height // 2 - 50)
		text = "GAME OVER - OUT OF FUEL!"
		for char in text:
			glutBitmapCharacter(GLUT_BITMAP_HELVETICA_18, ord(char))
		
		glColor3f(1.0, 1.0, 0.0)  # Yellow for instructions
		glRasterPos2f(window_width // 2 - 100, window_height // 2)
		text = "Press R to Reload"
		for char in text:
			glutBitmapCharacter(GLUT_BITMAP_HELVETICA_18, ord(char))
	
	# Re-enable depth testing
	glEnable(GL_DEPTH_TEST)
	
	# Restore projection matrices
	glPopMatrix()
	glMatrixMode(GL_PROJECTION)
	glPopMatrix()
	glMatrixMode(GL_MODELVIEW)


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
		P to increase power, O to decrease power.
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

		# Power adjustment
		if 'p' in self.keys_pressed:
			self.ship.increase_power()
			self.keys_pressed.discard('p')  # One-time press

		if 'o' in self.keys_pressed:
			self.ship.decrease_power()
			self.keys_pressed.discard('o')  # One-time press


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
		self.ship_destroyed = False  # Track if ship was destroyed
		
		# Game objects
		self.black_hole: Optional[BlackHole] = None
		self.ship: Optional[SpaceShip] = None
		self.astronauts: List[SpaceAstronaut] = []
		self.asteroids: List[SpaceAsteroid] = []
		self.input_controller: Optional[InputController] = None
		self.rescue_gate: Optional[RescueGate] = None
		
		# Spawning system
		self.spawn_timer = 0.0
		self.spawn_interval = 5.0  # Spawn new astronaut every 5 seconds
		self.max_astronauts = 8  # Maximum number of astronauts in orbit

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
		
		# Create test astronauts with random positions in orbit
		for _ in range(4):
			angle = random.uniform(0, 360)
			rad = math.radians(angle)
			distance = random.uniform(280, 320)
			pos = Vector3(math.cos(rad) * distance, math.sin(rad) * distance, 0)
			astronaut = SpaceAstronaut(position=pos)
			astronaut.gravity_source = self.black_hole
			self.astronauts.append(astronaut)
		
		# Create test asteroids with random positions in orbit
		for _ in range(4):
			angle = random.uniform(0, 360)
			rad = math.radians(angle)
			distance = random.uniform(230, 270)
			pos = Vector3(math.cos(rad) * distance, math.sin(rad) * distance, 0)
			asteroid = SpaceAsteroid(position=pos)
			asteroid.gravity_source = self.black_hole
			self.asteroids.append(asteroid)
		
		# Create rescue gate
		self.rescue_gate = RescueGate(position=Vector3(-600, 0, 0), radius=50.0)
		
		self.input_controller = InputController(self.ship)

	def reload_game(self) -> None:
		"""Reset the game to initial state."""
		self.game_over = False
		self.ship_destroyed = False
		self.current_time = 0.0
		self.last_time = time.time()
		self.spawn_timer = 0.0
		
		# Reset ship
		self.ship = SpaceShip(position=Vector3(400, 0, 0))
		self.ship.gravity_source = self.black_hole
		
		# Reset astronauts with random positions
		self.astronauts = []
		for _ in range(4):
			angle = random.uniform(0, 360)
			rad = math.radians(angle)
			distance = random.uniform(280, 320)
			pos = Vector3(math.cos(rad) * distance, math.sin(rad) * distance, 0)
			astronaut = SpaceAstronaut(position=pos)
			astronaut.gravity_source = self.black_hole
			self.astronauts.append(astronaut)
		
		# Reset asteroids with random positions
		self.asteroids = []
		for _ in range(4):
			angle = random.uniform(0, 360)
			rad = math.radians(angle)
			distance = random.uniform(230, 270)
			pos = Vector3(math.cos(rad) * distance, math.sin(rad) * distance, 0)
			asteroid = SpaceAsteroid(position=pos)
			asteroid.gravity_source = self.black_hole
			self.asteroids.append(asteroid)
		
		# Reset rescue gate score
		self.rescue_gate.score = 0
		
		self.input_controller = InputController(self.ship)
		print("\nGame reloaded!\n")

	def spawn_new_astronaut(self) -> None:
		"""Spawn a new astronaut at a random orbital position."""
		if len(self.astronauts) >= self.max_astronauts:
			return
		
		angle = random.uniform(0, 360)
		rad = math.radians(angle)
		distance = random.uniform(280, 320)
		pos = Vector3(math.cos(rad) * distance, math.sin(rad) * distance, 0)
		astronaut = SpaceAstronaut(position=pos)
		astronaut.gravity_source = self.black_hole
		self.astronauts.append(astronaut)

	def step(self, dt: float) -> None:
		"""Single frame update including input, simulation, and rendering."""
		if self.game_over:
			return
		
		# Update current time for animations
		self.current_time += dt
		
		# Update spawning timer
		self.spawn_timer += dt
		if self.spawn_timer >= self.spawn_interval:
			self.spawn_new_astronaut()
			self.spawn_timer = 0.0
		
		# Process input
		if self.input_controller:
			self.input_controller.update(dt)
		
		# Update physics
		self.ship.update(dt)
		for astronaut in self.astronauts:
			astronaut.update(dt)
		for asteroid in self.asteroids:
			asteroid.update(dt)
		
		# Check if astronauts reach the rescue gate
		for astronaut in self.astronauts:
			if not astronaut.is_rescued and self.rescue_gate.is_inside_gate(astronaut.position):
				astronaut.is_rescued = True
				astronaut.detach_tether()
				self.rescue_gate.add_score(100)
				print(f"✓ Astronaut Rescued at Gate! Gate Score: {self.rescue_gate.score}")
		
		# Mark astronauts as saved if they're 500+ units away from black hole
		for astronaut in self.astronauts:
			if not astronaut.is_rescued:
				distance_from_black_hole = (astronaut.position - self.black_hole.position).magnitude()
				if distance_from_black_hole >= 500.0:
					astronaut.is_rescued = True
					astronaut.detach_tether()
					print(f"✓ Astronaut Escaped! Distance from black hole: {distance_from_black_hole:.1f}")
		
		# Calculate distance from black hole
		ship_distance = (self.ship.position - self.black_hole.position).magnitude()
		
		# Check if ship is inside event horizon
		if self.black_hole.is_inside_event_horizon(self.ship.position):
			self.game_over = True
			self.ship_destroyed = True
			print("\n" + "="*60)
			print("█" * 60)
			print("███  GAME OVER! SHIP DESTROYED BY BLACK HOLE!  ███")
			print("█" * 60)
			print(f"Final distance from singularity: {ship_distance:.1f} units")
			print(f"Ship's thrust power was: {self.ship.thrust_power:.1f}")
			print("="*60 + "\n")
			# Hide ship by moving it far away
			self.ship.position = Vector3(10000, 10000, 10000)
			return
		
		# Check if fuel is depleted with unsaved astronauts
		rescued_count = sum(1 for a in self.astronauts if a.is_rescued)
		total_astronauts = len(self.astronauts)
		
		if self.ship.fuel <= 0 and rescued_count < total_astronauts:
			self.game_over = True
			print("\n" + "="*60)
			print("█" * 60)
			print("███  GAME OVER! OUT OF FUEL!  ███")
			print("█" * 60)
			print(f"Rescued: {rescued_count}/{total_astronauts}")
			print("="*60 + "\n")
			# Hide ship by moving it far away
			self.ship.position = Vector3(10000, 10000, 10000)
			return
		
		# Warning when getting close
		if ship_distance < 300 and not self.ship_destroyed:
			if int(self.current_time * 2) % 2 == 0:  # Flash warning
				print(f"\r⚠️  WARNING! Distance: {ship_distance:.1f} units - INCREASE POWER!", end="")
		
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
		
		# Draw rescue gate
		if self.rescue_gate:
			self.rescue_gate.render()
		
		# Render HUD text (required power info and game over message)
		render_hud_text(self.window_width, self.window_height, self.ship, 
		               self.astronauts, self.black_hole, self.rescue_gate, self.game_over)
		
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
		
		# R to reload game when game over
		if key == b'r' and game.game_over:
			game.reload_game()
			return
		
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
	
	print("="*70)
	print("THE VOID RESCUER - Member 1: Physics & Movement Demo")
	print("="*70)
	print("\nControls:")
	print("  W/Up Arrow    - Forward thrust")
	print("  S/Down Arrow  - Backward thrust")
	print("  A/Left Arrow  - Rotate left")
	print("  D/Right Arrow - Rotate right")
	print("  P             - Increase ship thrust power")
	print("  O             - Decrease ship thrust power")
	print("  SPACE         - Tether/untether nearest astronaut")
	print("  ESC           - Quit")
	print("\nObjective: Escape the black hole's gravity!")
	print("Strategy: Get closer to rescue astronauts, but increase power to escape!")
	print("="*70)
	print()
	
	# Initialize GLUT
	glutInit()
	glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH)
	glutInitWindowSize(800, 600)
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
