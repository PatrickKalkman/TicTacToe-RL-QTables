from dataclasses import dataclass
from typing import Tuple
import pygame


@dataclass
class RenderConfig:
    window_size: int = 300
    line_thickness: int = 5
    font_size: int = 100
    background_color: Tuple[int, int, int] = (255, 255, 255)
    line_color: Tuple[int, int, int] = (0, 0, 0)
    x_symbol_color: Tuple[int, int, int] = (242, 85, 96)
    o_symbol_color: Tuple[int, int, int] = (28, 170, 156)


class PygameRenderer:
    def __init__(self, config: RenderConfig = RenderConfig()):
        pygame.init()
        self.config = config
        self.cell_size = self.config.window_size // 3

        self.screen = pygame.display.set_mode(
            (self.config.window_size, self.config.window_size)
        )
        pygame.display.set_caption("Tic-Tac-Toe")

        self.font = pygame.font.SysFont(None, self.config.font_size)

    def render_game_state(self, board_state: str) -> None:
        self._clear_screen()
        self._draw_grid()
        self._render_symbols(board_state)
        self._update_display()

    def _clear_screen(self) -> None:
        self.screen.fill(self.config.background_color)

    def _draw_grid(self) -> None:
        for i in range(1, 3):
            position = i * self.cell_size

            pygame.draw.line(
                self.screen,
                self.config.line_color,
                (0, position),
                (self.config.window_size, position),
                self.config.line_thickness,
            )

            pygame.draw.line(
                self.screen,
                self.config.line_color,
                (position, 0),
                (position, self.config.window_size),
                self.config.line_thickness,
            )

    def _render_symbols(self, board_state: str) -> None:
        for position, symbol in enumerate(board_state):
            if symbol not in ("X", "O"):
                continue

            row, col = divmod(position, 3)
            symbol_position = self._calculate_symbol_position(row, col)
            self._draw_symbol(symbol, symbol_position)

    def _calculate_symbol_position(self, row: int, col: int) -> Tuple[int, int]:
        return (
            col * self.cell_size + self.cell_size // 2,
            row * self.cell_size + self.cell_size // 2,
        )

    def _draw_symbol(self, symbol: str, position: Tuple[int, int]) -> None:
        symbol_color = (
            self.config.x_symbol_color if symbol == "X" else self.config.o_symbol_color
        )

        symbol_surface = self.font.render(symbol, True, symbol_color)
        symbol_rect = symbol_surface.get_rect(center=position)
        self.screen.blit(symbol_surface, symbol_rect)

    def _update_display(self) -> None:
        pygame.display.flip()

    def close(self) -> None:
        pygame.quit()
