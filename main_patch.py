# Patch for showing current level on right side of screen

# 1. Update function signature (line ~618)
# OLD:
# def render_hud_text(window_width: int, window_height: int, ship: SpaceShip, 
#                    astronauts: List[SpaceAstronaut], black_hole: BlackHole, game_over: bool = False, game_won: bool = False, paused: bool = False) -> None:

# NEW:
def render_hud_text(window_width: int, window_height: int, ship: SpaceShip, 
                   astronauts: List[SpaceAstronaut], black_hole: BlackHole, game_over: bool = False, game_won: bool = False, paused: bool = False, difficulty: str = "medium") -> None:
    pass

# 2. Add level display before re-enabling depth testing (after pause message, before glEnable(GL_DEPTH_TEST))
# INSERT:
    # Display current level on right side
    glColor3f(1.0, 1.0, 0.0)  # Yellow
    level_text = f"LEVEL: {difficulty.upper()}"
    # Calculate position for right-aligned text
    text_width = len(level_text) * 10  # Approximate width
    glRasterPos2f(window_width - text_width - 15, 25)
    for char in level_text:
        glutBitmapCharacter(GLUT_BITMAP_HELVETICA_18, ord(char))

# 3. Update function call (line ~1249)
# OLD:
# render_hud_text(self.window_width, self.window_height, self.ship, self.astronauts, self.black_hole, self.game_over, self.game_won, self.paused)

# NEW:
render_hud_text(self.window_width, self.window_height, self.ship, self.astronauts, self.black_hole, self.game_over, self.game_won, self.paused, self.difficulty)
