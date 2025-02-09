#!/usr/bin/env python

import sys
from typing import Sequence
import gymnasium as gym
from gymnasium.utils.play import PlayableGame, PlayPlot, play, display_arr
from gymnasium.envs.box2d import lunar_lander

import numpy as np
import pygame
import pygame.freetype


def playplot_callback(obs_t, obs_tp1, action, rew, terminated, truncated, info):
    if terminated:
        if rew >= 100:
            print('Landed')
        else:
            print('Crashed')
    return [rew,]

def run_game(env: gym.Env,
             controls: dict[tuple[int, ...], int],
             noop: int=0,
             fps: int=30,
             autopilot: bool=False,
             transpose: bool | None = True,
             seed: int | None = None):
    # see https://gymnasium.farama.org/_modules/gymnasium/utils/play/#play

    obs, info = env.reset(seed=seed)
    game = PlayableGame(env, controls, zoom=None)

    clock = pygame.time.Clock()

    total: float = 0
    landed = False
    done = False
    # play at most for two minutes
    for i in range(fps * 120):
        if done or not game.running:
            break
        if autopilot:
            action = lunar_lander.heuristic(env=env, s=obs)
        else:
            action = controls.get(tuple(sorted(game.pressed_keys)), noop)
        obs, rew, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total += rew
        if terminated and rew >= 100:
            landed = True

        if obs is not None:
            rendered = env.render()
            if isinstance(rendered, Sequence):
                rendered = rendered[-1]
            display_arr(
                game.screen, rendered, transpose=transpose, video_size=game.video_size
            )

        for event in pygame.event.get():
            game.process_event(event)

        pygame.display.flip()
        clock.tick(fps)

    pygame.freetype.init()
    font = pygame.freetype.SysFont(pygame.freetype.get_default_font(), size=50)
    if landed:
        text = font.render('Landed', fgcolor=pygame.Color('green'))
    else:
        text = font.render('Crashed', fgcolor=pygame.Color('red'))
    game.screen.blit(text[0], text[1])
    pygame.display.flip()

    print("{}, with a total reward of {}".format('Landed' if landed else 'Crashed', total))

    while True:
        event = pygame.event.wait()
        if event.type == pygame.QUIT:
            break

    env.close()
    pygame.quit()



def main(*args: str) -> int:
    autopilot = 'autopilot' in args
    env = gym.make("LunarLander-v3", render_mode='rgb_array',
                   continuous=False, gravity=-10.0,
                   enable_wind=False, wind_power=15.0, turbulence_power=1.5)
    game_control = {
        (pygame.K_LEFT,): 3, # left engine
        (pygame.K_SPACE,): 2, # main engine
        (pygame.K_RIGHT,): 1, # right engine
    }

    run_game(env, controls=game_control, fps=20, autopilot=autopilot)
    return 0

if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:]))