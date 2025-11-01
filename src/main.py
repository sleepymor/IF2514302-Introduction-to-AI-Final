from environment.enivronment import TacticalEnvironment
import pygame

def main():
    pygame.init()
    env = TacticalEnvironment(width=20, height=12, seed=42)
    screen = pygame.display.set_mode((env.width * 40, env.height * 40))
    clock = pygame.time.Clock()

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN and env.turn == 'player':
                mx, my = pygame.mouse.get_pos()
                gx, gy = mx // 40, my // 40
                env.step((gx, gy))

        screen.fill((20, 20, 30))
        env.draw(screen)
        pygame.display.flip()
        clock.tick(30)

    pygame.quit()

if __name__ == "__main__":
    main()