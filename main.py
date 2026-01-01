"""
Project scaffolding for The Void Rescuer.

This file sets up abstract interfaces and lightweight data structures so we can
iterate on gameplay (gravity, tethering, HUD) without committing to OpenGL
details yet.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Iterable, Optional


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


def main() -> None:
	"""Entry point placeholder; wire to GLUT/pygame later."""
	raise SystemExit("Game loop not initialized yet.")


if __name__ == "__main__":
	main()
