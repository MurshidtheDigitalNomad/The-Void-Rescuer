
from __future__ import annotations

import math
import time
import sys
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Iterable, Optional, List

from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *

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


class StarField:
	"""
	Background star field with random small white dots.
	"""

	def __init__(self, num_stars: int = 200, spread: float = 2000.0):
		import random
		self.stars = []
		for _ in range(num_stars):
			# Random position in 3D space
			x = random.uniform(-spread, spread)
			y = random.uniform(-spread, spread)
			z = random.uniform(-spread, spread)
			# Random size (small dots)
			size = random.uniform(0.5, 2.5)
			# Random brightness
			brightness = random.uniform(0.6, 1.0)
			self.stars.append({
				'position': Vector3(x, y, z),
				'size': size,
				'brightness': brightness
			})

	def render(self) -> None:
		"""Render all stars as small white dots."""
		glDisable(GL_LIGHTING)  # Disable lighting for stars
		for star in self.stars:
			glPushMatrix()
			glTranslatef(star['position'].x, star['position'].y, star['position'].z)
			# White color with varying brightness
			b = star['brightness']
			glColor3f(b, b, b)
			gluSphere(gluNewQuadric(), star['size'], 4, 4)  # Small sphere
			glPopMatrix()
		glEnable(GL_LIGHTING)  # Re-enable lighting for other objects


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
		# Base pull strength (no multiplier)
		force_magnitude = (self.strength / (distance * distance)) * 1.0
		
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
		position: Vector3 = Vector3(600, 0, 0),
		heading_degrees: float = 180.0,
		mass: float = 50.0,  # Increased mass for stronger gravity effects
	):
		self.position = position
		self.velocity = Vector3()
		self.heading_degrees = heading_degrees
		self.mass = mass
		self.thrust_power = 50.0  # Base thrust power
		self.thrust_power_min = 20.0  # Minimum thrust
		self.thrust_power_max = 200.0  # Maximum thrust
		self.drag_coefficient = 0.98  # Natural slowdown
		self.gravity_source: Optional[GravitySource] = None
		# Collision
		self.collision_radius = 25.0  # Ship collision sphere
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
		self.collision_radius = 10.0  # Astronaut collision sphere
		self.tether_max_distance = 300.0  # Increased from 500
		self.tether_pull_strength = 200.0  # Strong pull force
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

			# Apply strong pull force toward ship
			# Force increases with distance (like a spring)
			pull_force = unit_direction * self.tether_pull_strength * (distance / 100.0)
			acceleration = pull_force * (1.0 / self.mass)
			
			# Add to existing velocity (don't replace it)
			self.velocity = self.velocity + acceleration * 0.016

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
		scale: tuple[float, float, float] = (1.0, 1.0, 1.0),
		rotation_axis: Vector3 = Vector3(0, 0, 1),
		rotation_speed: float = 0.0,  # degrees per second
	):
		self.position = position
		self.velocity = Vector3()
		self.radius = radius
		self.scale = scale
		self.rotation_axis = rotation_axis
		self.rotation_speed = rotation_speed
		self.rotation_angle = 0.0
		self.mass = mass
		self.gravity_source: Optional[GravitySource] = None
		# Collision radius based on largest scale to keep detection fair
		self.collision_radius = radius * max(scale)

	def integrate(self, dt: float) -> None:
		"""
		Update position based on velocity, apply gravity, and spin.
		"""
		# Apply gravity force if source exists
		if self.gravity_source:
			gravity_force = self.gravity_source.force_at(self.position)
			acceleration = gravity_force * (1.0 / self.mass)
			self.velocity = self.velocity + acceleration * dt

		# Update position
		self.position = self.position + self.velocity * dt

		# Spin the asteroid
		self.rotation_angle = (self.rotation_angle + self.rotation_speed * dt) % 360.0

	def render(self) -> None:
		"""Render asteroid as an irregular gray rock."""
		glPushMatrix()
		glTranslatef(self.position.x, self.position.y, self.position.z)
		glRotatef(self.rotation_angle, self.rotation_axis.x, self.rotation_axis.y, self.rotation_axis.z)
		glScalef(self.scale[0], self.scale[1], self.scale[2])
		glColor3f(0.55, 0.55, 0.55)  # Medium gray
		glutSolidSphere(self.radius, 14, 14)
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
	
	# Convert gravity force to required power and add a 20-point safety margin
	# Power needed = gravity pull converted to thrust units + 20 to overcome pull
	required_power = force_magnitude * 10.0 + 20.0
	return required_power


def render_hud_text(window_width: int, window_height: int, ship: SpaceShip, 
	                   astronauts: List[SpaceAstronaut], black_hole: BlackHole, game_over: bool = False, game_won: bool = False, paused: bool = False, difficulty: str = "medium") -> None:
	"""
	Render HUD text in top-left corner showing power, fuel, and rescue information.
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
	
	# Always display current power (will be recolored below if comparing)
	glColor3f(0.2, 1.0, 0.2)  # Bright green
	glRasterPos2f(15, 25)
	text = f"Current Power: {ship.thrust_power:.1f}"
	for char in text:
		glutBitmapCharacter(GLUT_BITMAP_HELVETICA_18, ord(char)) # type: ignore
	
	# Display fuel level with color coding
	fuel_percent = (ship.fuel / ship.fuel_max) * 100 if ship.fuel_max > 0 else 0
	if fuel_percent > 50:
		glColor3f(0.2, 1.0, 0.2)  # Green - plenty of fuel
	elif fuel_percent > 20:
		glColor3f(1.0, 1.0, 0.0)  # Yellow - low fuel warning
	else:
		glColor3f(1.0, 0.0, 0.0)  # Red - critical fuel
	
	glRasterPos2f(15, 50)
	text = f"Fuel: {ship.fuel:.0f}/{ship.fuel_max:.0f} ({fuel_percent:.1f}%)"
	for char in text:
		glutBitmapCharacter(GLUT_BITMAP_HELVETICA_18, ord(char)) # type: ignore
	
	# Display rescue count
	glColor3f(0.2, 1.0, 0.2)  # Green for rescue status
	glRasterPos2f(15, 75)
	text = f"Rescued: {rescued_count}/{total_astronauts}"
	for char in text:
		glutBitmapCharacter(GLUT_BITMAP_HELVETICA_18, ord(char)) # type: ignore
	
	# Display required power if astronaut is nearby
	if nearest_astronaut and min_distance < 500:
		required_power = calculate_required_power(black_hole, nearest_astronaut.position)
		current_power = ship.thrust_power
		ratio = current_power / required_power if required_power > 0 else 0
		
		# Color based on whether power is sufficient (needs +20 margin baked into required)
		if current_power >= required_power:
			glColor3f(0.0, 1.0, 0.0)  # Green - sufficient power
		else:
			glColor3f(1.0, 0.0, 0.0)  # Red - insufficient power
		
		glRasterPos2f(15, 100)
		text = f"Required Power: {required_power:.1f} (gravity+20)"
		for char in text:
			glutBitmapCharacter(GLUT_BITMAP_HELVETICA_18, ord(char)) # type: ignore
		
		glRasterPos2f(15, 125)
		text = f"Current Power: {current_power:.1f} ({ratio:.2f}x req)"
		for char in text:
			glutBitmapCharacter(GLUT_BITMAP_HELVETICA_18, ord(char)) # type: ignore
		
		glRasterPos2f(15, 150)
		text = f"Distance to Astronaut: {min_distance:.1f}"
		for char in text:
			glutBitmapCharacter(GLUT_BITMAP_HELVETICA_18, ord(char)) # type: ignore
	
	# Display game over message if game is over
	if game_over:
		glColor3f(1.0, 0.0, 0.0)  # Red for game over
		glRasterPos2f(window_width // 2 - 200, window_height // 2 - 50)
		text = "===== GAME OVER ====="
		for char in text:
			glutBitmapCharacter(GLUT_BITMAP_HELVETICA_18, ord(char)) # type: ignore
		
		glColor3f(1.0, 1.0, 0.0)  # Yellow for instructions
		glRasterPos2f(window_width // 2 - 100, window_height // 2)
		text = "Press R to Restart"
		for char in text:
			glutBitmapCharacter(GLUT_BITMAP_HELVETICA_18, ord(char)) # type: ignore
	
	# Display victory message if won
	if not game_over and game_won:
		glColor3f(0.0, 1.0, 0.0)  # Green for victory
		glRasterPos2f(window_width // 2 - 200, window_height // 2 - 80)
		text = "====== VICTORY! ======"
		for char in text:
			glutBitmapCharacter(GLUT_BITMAP_HELVETICA_18, ord(char)) # type: ignore
		
		glColor3f(0.0, 1.0, 1.0)  # Cyan
		glRasterPos2f(window_width // 2 - 180, window_height // 2 - 20)
		text = "ALL ASTRONAUTS RESCUED!"
		for char in text:
			glutBitmapCharacter(GLUT_BITMAP_HELVETICA_18, ord(char)) # type: ignore
		
		glColor3f(1.0, 1.0, 0.0)  # Yellow for instructions
		glRasterPos2f(window_width // 2 - 120, window_height // 2 + 40)
		text = "Press R to Play Again"
		for char in text:
			glutBitmapCharacter(GLUT_BITMAP_HELVETICA_18, ord(char)) # type: ignore
	
	# Display pause message if paused
	if paused and not game_over and not game_won:
		glColor3f(1.0, 1.0, 1.0)  # White for pause
		glRasterPos2f(window_width // 2 - 120, window_height - 100)
		text = "|| PAUSED"
		for char in text:
			glutBitmapCharacter(GLUT_BITMAP_HELVETICA_18, ord(char)) # type: ignore
		
		glColor3f(1.0, 1.0, 0.0)  # Yellow for instructions
		glRasterPos2f(window_width // 2 - 150, window_height - 130)
		text = "Press TAB to Resume"
			# Display current level on right side
			glColor3f(1.0, 1.0, 0.0)  # Yellow
			level_text = f"LEVEL: {difficulty.upper()}"
			# Calculate position for right-aligned text
			text_width = len(level_text) * 10  # Approximate width
			glRasterPos2f(window_width - text_width - 15, 25)
			for char in level_text:
				glutBitmapCharacter(GLUT_BITMAP_HELVETICA_18, ord(char))  # type: ignore
	
		for char in text:
			glutBitmapCharacter(GLUT_BITMAP_HELVETICA_18, ord(char)) # type: ignore
	
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

	def __init__(self, difficulty: str = "medium"):
		self.window_width = 1200
		self.window_height = 800
		self.game_over = False
		self.game_over_reason = ""  # Reason for game over
		self.game_won = False  # Track victory
		self.victory_announced = False  # Track if victory was announced
		self.paused = False  # Track pause state
		self.last_time = time.time()
		self.current_time = 0.0
		self.ship_destroyed = False  # Track if ship was destroyed
		self.difficulty = difficulty  # Game difficulty level
		self.station_position = Vector3(800, 0, 0)  # Rescue station location
		self.station_half_size = 40.0  # Square half-size (80x80 pad)
		self.station_rescue_radius = 100.0  # Rescue radius around station center
		self.difficulty_order = ["easy", "medium", "hard"]
		self.level_index = max(0, self.difficulty_order.index(difficulty)) if difficulty in self.difficulty_order else 1
		
		# Game objects
		self.black_hole: Optional[BlackHole] = None
		self.ship: Optional[SpaceShip] = None
		self.astronauts: List[SpaceAstronaut] = []
		self.asteroids: List[SpaceAsteroid] = []
		self.input_controller: Optional[InputController] = None
		self.star_field: Optional[StarField] = None  # Background stars
		
		# Difficulty settings
		self.difficulty_settings = {
			"easy": {
				"num_astronauts": 4,
				"num_asteroids": 5,
				"fuel_consumption": 0.02,
				"thrust_power": 80.0,
				"initial_fuel": 2000.0
			},
			"medium": {
				"num_astronauts": 6,
				"num_asteroids": 9,
				"fuel_consumption": 0.03,
				"thrust_power": 60.0,
				"initial_fuel": 1500.0
			},
			"hard": {
				"num_astronauts": 10,
				"num_asteroids": 15,
				"fuel_consumption": 0.05,
				"thrust_power": 50.0,
				"initial_fuel": 1200.0
			}
		}
		
		self.current_settings = self.difficulty_settings.get(difficulty, self.difficulty_settings["medium"])

	def _set_difficulty(self, difficulty: str) -> None:
		"""Update difficulty and current settings safely."""
		self.difficulty = difficulty if difficulty in self.difficulty_settings else "medium"
		self.current_settings = self.difficulty_settings.get(self.difficulty, self.difficulty_settings["medium"])
		if self.difficulty in self.difficulty_order:
			self.level_index = self.difficulty_order.index(self.difficulty)

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
		self.star_field = StarField(num_stars=200, spread=2000.0)
		self.black_hole = BlackHole(strength=50000.0, event_horizon=150.0)
		self.ship = SpaceShip(position=Vector3(400, 0, 0))
		self.ship.gravity_source = self.black_hole
		
		# Apply difficulty settings to ship
		self.ship.fuel = self.current_settings["initial_fuel"]
		self.ship.fuel_max = self.current_settings["initial_fuel"]
		self.ship.fuel_consumption_rate = self.current_settings["fuel_consumption"]
		self.ship.thrust_power = self.current_settings["thrust_power"]
		
		# Create astronauts based on difficulty
		num_astronauts = self.current_settings["num_astronauts"]
		angle_step = 360.0 / num_astronauts
		spawn_radius = 420.0  # Keep astronauts ≥400 units from the black hole
		for i in range(num_astronauts):
			angle = i * angle_step
			rad = math.radians(angle)
			pos = Vector3(math.cos(rad) * spawn_radius, math.sin(rad) * spawn_radius, 0)
			astronaut = SpaceAstronaut(position=pos)
			astronaut.gravity_source = self.black_hole
			self.astronauts.append(astronaut)
		
		# Create asteroids based on difficulty
		num_asteroids = self.current_settings["num_asteroids"]
		for _ in range(num_asteroids):
			asteroid = self._create_random_asteroid()
			self.asteroids.append(asteroid)
		
		self.input_controller = InputController(self.ship)
		
		# Print difficulty info
		print(f"\n[DIFFICULTY: {self.difficulty.upper()}]")
		print(f"  Astronauts: {num_astronauts}")
		print(f"  Asteroids: {num_asteroids}")
		print(f"  Initial Fuel: {self.current_settings['initial_fuel']}")
		print(f"  Fuel Consumption: {self.current_settings['fuel_consumption']}")
		print(f"  Thrust Power: {self.current_settings['thrust_power']}")
		print()

	def _create_random_asteroid(self) -> SpaceAsteroid:
		"""Generate a random asteroid position, size, and shape."""
		distance = random.uniform(180.0, 380.0)
		angle_deg = random.uniform(0.0, 360.0)
		angle_rad = math.radians(angle_deg)
		pos = Vector3(
			math.cos(angle_rad) * distance,
			math.sin(angle_rad) * distance,
			random.uniform(-30.0, 30.0),  # Slight Z variation
		)
		base_radius = random.uniform(12.0, 28.0)
		scale = (
			random.uniform(0.8, 1.4),
			random.uniform(0.8, 1.4),
			random.uniform(0.7, 1.3),
		)
		axis = Vector3(
			random.uniform(-1.0, 1.0),
			random.uniform(-1.0, 1.0),
			random.uniform(-1.0, 1.0),
		)
		if axis.magnitude() < 0.1:
			axis = Vector3(0, 0, 1)
		rotation_speed = random.uniform(-35.0, 35.0)
		asteroid = SpaceAsteroid(
			position=pos,
			radius=base_radius,
			mass=15.0,
			scale=scale,
			rotation_axis=axis.normalized(),
			rotation_speed=rotation_speed,
		)
		asteroid.gravity_source = self.black_hole
		return asteroid

	def reload_game(self) -> None:
		"""Reset the game to initial state with current difficulty."""
		self.game_over = False
		self.game_over_reason = ""
		self.game_won = False
		self.victory_announced = False
		self.paused = False
		self.ship_destroyed = False
		self.current_time = 0.0
		self.last_time = time.time()
		self.station_position = Vector3(500, 0, 0)
		
		# Reset ship
		self.ship = SpaceShip(position=Vector3(400, 0, 0))
		self.ship.gravity_source = self.black_hole
		
		# Apply difficulty settings to ship
		self.ship.fuel = self.current_settings["initial_fuel"]
		self.ship.fuel_max = self.current_settings["initial_fuel"]
		self.ship.fuel_consumption_rate = self.current_settings["fuel_consumption"]
		self.ship.thrust_power = self.current_settings["thrust_power"]
		
		# Reset astronauts based on difficulty
		self.astronauts = []
		num_astronauts = self.current_settings["num_astronauts"]
		angle_step = 360.0 / num_astronauts
		spawn_radius = 420.0  # Keep astronauts ≥400 units from the black hole
		for i in range(num_astronauts):
			angle = i * angle_step
			rad = math.radians(angle)
			pos = Vector3(math.cos(rad) * spawn_radius, math.sin(rad) * spawn_radius, 0)
			astronaut = SpaceAstronaut(position=pos)
			astronaut.gravity_source = self.black_hole
			self.astronauts.append(astronaut)
		
		# Reset asteroids based on difficulty
		self.asteroids = []
		num_asteroids = self.current_settings["num_asteroids"]
		for _ in range(num_asteroids):
			asteroid = self._create_random_asteroid()
			self.asteroids.append(asteroid)
		
		self.input_controller = InputController(self.ship)
		
		# Print current difficulty info
		print(f"\n[DIFFICULTY: {self.difficulty.upper()}]")
		print(f"  Astronauts: {num_astronauts}")
		print(f"  Asteroids: {num_asteroids}")
		print(f"  Initial Fuel: {self.current_settings['initial_fuel']}")
		print(f"  Fuel Consumption: {self.current_settings['fuel_consumption']}")
		print(f"  Thrust Power: {self.current_settings['thrust_power']}")
		print("\nGame reloaded!\n")

	def check_collisions(self) -> None:
		"""Check for collisions between game objects."""
		# Ship-Asteroid collisions - GAME OVER
		for asteroid in self.asteroids:
			distance = (self.ship.position - asteroid.position).magnitude() # type: ignore 
			if distance < (self.ship.collision_radius + asteroid.collision_radius): # type: ignore
				# Collision detected - GAME OVER
				self.game_over = True
				self.game_over_reason = "SHIP HIT ASTEROID!"
				self.ship_destroyed = True
				print("\n" + "="*60)
				print("#" * 60)
				print("###  GAME OVER! " + self.game_over_reason + "  ###")
				print("#" * 60)
				print("="*60 + "\n")
				# Hide ship by moving it far away
				self.ship.position = Vector3(10000, 10000, 10000) # type: ignore
				return
		
		# Astronaut-Asteroid collisions
		for astronaut in self.astronauts:
			if astronaut.is_rescued:
				continue
			for asteroid in self.asteroids:
				distance = (astronaut.position - asteroid.position).magnitude()
				if distance < (astronaut.collision_radius + asteroid.collision_radius):
					# Astronaut hit by asteroid - GAME OVER
					self.game_over = True
					self.game_over_reason = "ASTRONAUT HIT BY ASTEROID!"
					print("\n" + "="*60)
					print("#" * 60)
					print("###  GAME OVER! " + self.game_over_reason + "  ###")
					print("#" * 60)
					print("Mission Failed: Could not save astronaut")
					print("="*60 + "\n")
					# Mark astronaut as lost
					astronaut.position = Vector3(10000, 10000, 10000)
					return

	def step(self, dt: float) -> None:
		"""Single frame update including input, simulation, and rendering."""
		# Process input
		if self.input_controller:
			self.input_controller.update(dt)
		
		# Don't process physics if game is over or paused, but DO render
		if self.game_over or self.paused:
			self.render()
			return
		
		# Update current time for animations
		self.current_time += dt
		
		# Update physics
		self.ship.update(dt) # type: ignore
		for astronaut in self.astronauts:
			astronaut.update(dt)
		for asteroid in self.asteroids:
			asteroid.update(dt)
		
		# Check collisions
		self.check_collisions()
		
		# If game over from collision, stop here
		if self.game_over:
			self.render()
			return
		
		# Check station rescue condition: within 20 units of station center
		for astronaut in self.astronauts:
			if astronaut.is_rescued:
				continue
			dist_station = (astronaut.position - self.station_position).magnitude()
			if dist_station <= self.station_rescue_radius:
				astronaut.is_rescued = True
				astronaut.detach_tether()
				print(f"[SUCCESS] Astronaut docked at station! Total: {sum(1 for a in self.astronauts if a.is_rescued)}/{len(self.astronauts)}")
		
		# Calculate distance from black hole
		ship_distance = (self.ship.position - self.black_hole.position).magnitude() # type: ignore
		
		# Check if ship is inside event horizon
		if self.black_hole.is_inside_event_horizon(self.ship.position): # type: ignore
			self.game_over = True
			self.game_over_reason = "SHIP DESTROYED BY BLACK HOLE!"
			ship_distance = (self.ship.position - self.black_hole.position).magnitude() # type: ignore
			self.ship_destroyed = True
			print("\n" + "="*60)
			print("#" * 60)
			print("###  GAME OVER! " + self.game_over_reason + "  ###")
			print("#" * 60)
			print(f"Final distance from singularity: {ship_distance:.1f} units")
			print(f"Ship's thrust power was: {self.ship.thrust_power:.1f}") # type: ignore
			print("="*60 + "\n")
			# Hide ship by moving it far away
			self.ship.position = Vector3(10000, 10000, 10000) # type: ignore
			return
		
		# Check for astronaut lost to black hole
		for astronaut in self.astronauts:
			if not astronaut.is_rescued and self.black_hole.is_inside_event_horizon(astronaut.position): # type: ignore
				self.game_over = True
				self.game_over_reason = "ASTRONAUT LOST TO BLACK HOLE!"
				print("\n" + "="*60)
				print("#" * 60)
				print("###  GAME OVER! " + self.game_over_reason + "  ###")
				print("#" * 60)
				print("Mission Failed: Could not save all astronauts")
				print("="*60 + "\n")
				return
		
		# Check if fuel is depleted with unsaved astronauts
		rescued_count = sum(1 for a in self.astronauts if a.is_rescued)
		total_astronauts = len(self.astronauts)
		
		# Check for victory - all astronauts rescued!
		if rescued_count == total_astronauts and rescued_count > 0:
			self.game_won = True
			# Only print victory message once
			if not self.victory_announced:
				self.victory_announced = True
				print("\n" + "="*60)
				print("#" * 60)
				print("###  VICTORY! ALL ASTRONAUTS RESCUED!  ###")
				print("#" * 60)
				print(f"Mission Complete: {rescued_count}/{total_astronauts} saved")
				print("="*60 + "\n")
				# Auto-advance to next level if available
				next_index = self.level_index + 1
				if next_index < len(self.difficulty_order):
					next_diff = self.difficulty_order[next_index]
					print(f"\n[LEVEL COMPLETE] Advancing to {next_diff.upper()}...")
					self._set_difficulty(next_diff)
					self.reload_game()
					return
				else:
					print("\n[CAMPAIGN COMPLETE] All levels finished!\n")
		
		if self.ship.fuel <= 0 and rescued_count < total_astronauts: # type: ignore
			self.game_over = True
			self.game_over_reason = "OUT OF FUEL!"
			print("\n" + "="*60)
			print("#" * 60)
			print("###  GAME OVER! " + self.game_over_reason + "  ###")
			print("#" * 60)
			print(f"Rescued: {rescued_count}/{total_astronauts}")
			print("="*60 + "\n")
			# Hide ship by moving it far away
			self.ship.position = Vector3(10000, 10000, 10000) # type: ignore
			return
		
		# Warning when getting close
		if ship_distance < 300 and not self.ship_destroyed:
			if int(self.current_time * 2) % 2 == 0:  # Flash warning
				print(f"\r[WARNING] Distance: {ship_distance:.1f} units - INCREASE POWER!", end="")
		
		# Render
		self.render()

	def render(self) -> None:
		"""Render the scene."""
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT) # type: ignore
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
		
		# Draw background stars
		if self.star_field:
			self.star_field.render()

		# Draw rescue station (square pad 1000 units from black hole)
		glPushMatrix()
		glTranslatef(self.station_position.x, self.station_position.y, self.station_position.z)
		glColor3f(0.7, 0.7, 0.7)  # Light gray pad
		glBegin(GL_QUADS)
		glVertex3f(-self.station_half_size, -self.station_half_size, 0)
		glVertex3f(self.station_half_size, -self.station_half_size, 0)
		glVertex3f(self.station_half_size, self.station_half_size, 0)
		glVertex3f(-self.station_half_size, self.station_half_size, 0)
		glEnd()
		# Border
		glColor3f(0.2, 0.8, 0.2)  # Green border
		glLineWidth(3.0)
		glBegin(GL_LINE_LOOP)
		glVertex3f(-self.station_half_size, -self.station_half_size, 0.2)
		glVertex3f(self.station_half_size, -self.station_half_size, 0.2)
		glVertex3f(self.station_half_size, self.station_half_size, 0.2)
		glVertex3f(-self.station_half_size, self.station_half_size, 0.2)
		glEnd()
		glLineWidth(1.0)
		glPopMatrix()
		
		# Draw black hole (simple marker for now)
		glPushMatrix()
		glColor3f(0.2, 0.0, 0.2)  # Dark purple
		glutSolidSphere(self.black_hole.event_horizon, 32, 32) # type: ignore
		glPopMatrix()
		
		# Draw ship
		if self.ship:
			self.ship.render()
		
		# Draw astronauts and tethers
		for astronaut in self.astronauts:
			if astronaut.tethered_to:
				render_tether_beam(self.ship, astronaut, self.current_time) # type: ignore
			astronaut.render()
		
		# Draw asteroids
		for asteroid in self.asteroids:
			asteroid.render()
		
		# Render HUD text (required power info and game over message)
		render_hud_text(self.window_width, self.window_height, self.ship, self.astronauts, self.black_hole, self.game_over, self.game_won, self.paused) # type: ignore
		
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
	if game:
		try:
			current = time.time()
			dt = current - game.last_time
			game.last_time = current
			
			game.step(dt)
			glutPostRedisplay()
			glutTimerFunc(16, timer_callback, 0)  # ~60 FPS
		except Exception as e:
			print(f"\nERROR in game loop: {e}")
			import traceback
			traceback.print_exc()
			glutLeaveMainLoop()


def keyboard_callback(key: bytes, x: int, y: int) -> None: # type: ignore
	"""GLUT keyboard down callback."""
	if not game:
		return
	
	# Escape to quit - HIGHEST PRIORITY
	if key == b'\x1b':  # ESC
		print("Exiting game via ESC...")
		glutLeaveMainLoop()
		sys.exit(0)
		return
	
	# TAB to pause/unpause
	if key == b'\t':
		game.paused = not game.paused
		if game.paused:
			print("Game PAUSED")
		else:
			print("Game RESUMED")
		return
	
	# R to reload game anytime
	if key == b'r' or key == b'R':
		print("Restarting game...")
		game.reload_game()
		return
	
	# Process other input
	if game.input_controller:
		key_str = key.decode('utf-8').lower()
		game.input_controller.key_down(key_str)
		
		# Space to tether nearest astronaut
		if key == b' ':
			nearest = None
			min_dist = float('inf')
			for astronaut in game.astronauts:
				if not astronaut.is_rescued:
					dist = (astronaut.position - game.ship.position).magnitude() # type: ignore
					if dist < min_dist and dist < 300:  # Within 300 units
						min_dist = dist
						nearest = astronaut
			
			if nearest:
				if nearest.tethered_to:
					nearest.detach_tether()
					print("Tether released")
				else:
					nearest.attach_tether(game.ship) # type: ignore
					print(f"Tethered! Distance: {min_dist:.1f}")


def keyboard_up_callback(key: bytes, x: int, y: int) -> None: # type: ignore
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
	print("THE VOID RESCUER - Space Rescue Mission")
	print("="*70)
	print("\n*** SELECT DIFFICULTY ***\n")
	print("1. EASY   - 4 astronauts, 5 asteroids  (More fuel, more power)")
	print("2. MEDIUM - 6 astronauts, 9 asteroids  (Balanced)")
	print("3. HARD   - 10 astronauts, 15 asteroids (Less fuel, less power)")
	print()
	
	difficulty = "medium"
	while True:
		choice = input("Enter your choice (1/2/3): ").strip()
		if choice == "1":
			difficulty = "easy"
			break
		elif choice == "2":
			difficulty = "medium"
			break
		elif choice == "3":
			difficulty = "hard"
			break
		else:
			print("Invalid choice. Please enter 1, 2, or 3.")
	
	print("\n" + "="*70)
	print(f"DIFFICULTY: {difficulty.upper()}")
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
	print("\nObjective: Rescue all astronauts and escape the black hole!")
	print("WARNING: Avoid asteroids! Collision = GAME OVER!")
	print("="*70)
	print()
	
	# Initialize GLUT
	glutInit()
	glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH) # type: ignore
	glutInitWindowSize(800, 600)
	glutInitWindowPosition(100, 100)
	glutCreateWindow(b"The Void Rescuer")
	
	# Create and initialize game with selected difficulty
	game = VoidRescuerGame(difficulty=difficulty)
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
	main()
