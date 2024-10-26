import pygame


class PygameRenderer:
    def __init__(self, window_size=300):
        """Initialize pygame and set up the display."""
        pygame.init()
        self.window_size = window_size
        self.cell_size = window_size // 3
        self.screen = pygame.display.set_mode((window_size, window_size))
        pygame.display.set_caption("Tic-Tac-Toe")

        # Colors
        self.bg_color = (255, 255, 255)
        self.line_color = (0, 0, 0)
        self.x_color = (242, 85, 96)   # Red for X
        self.o_color = (28, 170, 156)  # Green for O

        # Font
        self.font = pygame.font.SysFont(None, 100)

    def draw_board(self, state):
        """Draw the Tic-Tac-Toe board and symbols based on the state."""
        self.screen.fill(self.bg_color)

        # Draw grid lines
        for i in range(1, 3):
            pygame.draw.line(self.screen, self.line_color, (0, i * self.cell_size), (self.window_size, i * self.cell_size), 5)
            pygame.draw.line(self.screen, self.line_color, (i * self.cell_size, 0), (i * self.cell_size, self.window_size), 5)

        # Draw X and O symbols
        for row in range(3):
            for col in range(3):
                symbol = state[row * 3 + col]
                center_x = col * self.cell_size + self.cell_size // 2
                center_y = row * self.cell_size + self.cell_size // 2
                
                if symbol == 1:
                    # Draw X
                    text_surface = self.font.render("X", True, self.x_color)
                    text_rect = text_surface.get_rect(center=(center_x, center_y))
                    self.screen.blit(text_surface, text_rect)
                elif symbol == -1:
                    # Draw O
                    text_surface = self.font.render("O", True, self.o_color)
                    text_rect = text_surface.get_rect(center=(center_x, center_y))
                    self.screen.blit(text_surface, text_rect)

        # Update display
        pygame.display.flip()

    def close(self):
        """Properly close the pygame display."""
        pygame.quit()
