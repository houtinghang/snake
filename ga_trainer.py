# ga_trainer.py

import os
import random
import numpy as np
from pygame.math import Vector2
from snake import MAIN, cell_number as GRID_SIZE

# ---------- 若要訓練時不跳視窗，取消下列註解 ----------
# os.environ["SDL_VIDEODRIVER"] = "dummy"
# import pygame
# pygame.display.init()
# --------------------------------------------------

# ===== GA 超參數 =====
POP_SIZE   = 30      # 族群大小
GENS       = 100     # 世代數
MUT_RATE   = 0.5     # 突變機率
ELITE_RATE = 0.1     # 精英保留比例

# NN 結構
INPUT_SIZE  = 4
HIDDEN_SIZE = 8
OUTPUT_SIZE = 4
CHROMO_LEN  = INPUT_SIZE*HIDDEN_SIZE + HIDDEN_SIZE*OUTPUT_SIZE + HIDDEN_SIZE + OUTPUT_SIZE

DIRECTIONS = [
    Vector2(1,0), Vector2(-1,0),
    Vector2(0,1), Vector2(0,-1)
]

class Individual:
    def __init__(self, genome=None):
        self.genome = genome if genome is not None else np.random.uniform(-1,1,CHROMO_LEN)
        self.fitness = 0

    def decode(self):
        g = self.genome
        w1_end = INPUT_SIZE * HIDDEN_SIZE
        w2_end = w1_end + HIDDEN_SIZE * OUTPUT_SIZE
        b1_end = w2_end + HIDDEN_SIZE
        w1 = g[:w1_end].reshape(INPUT_SIZE, HIDDEN_SIZE)
        w2 = g[w1_end:w2_end].reshape(HIDDEN_SIZE, OUTPUT_SIZE)
        b1 = g[w2_end:b1_end]
        b2 = g[b1_end:]
        return w1, w2, b1, b2

    def decide(self, head, fruit_pos, last_dir):
        state = np.array([
            (fruit_pos.x - head.x) / GRID_SIZE,
            (fruit_pos.y - head.y) / GRID_SIZE,
            last_dir.x,
            last_dir.y
        ])
        w1, w2, b1, b2 = self.decode()
        h   = np.tanh(state.dot(w1) + b1)
        out = h.dot(w2) + b2
        return DIRECTIONS[np.argmax(out)]

class SnakeGame:
    """只取用 MAIN 的邏輯，不跳視窗或 sys.exit()。並禁用障礙物與加速道具。"""
    def __init__(self, individual):
        self.game = MAIN(individual)
        # 攔截結束呼叫
        self.game.is_over = False
        self.game.game_over = lambda: setattr(self.game, 'is_over', True)
        self.game.quit      = lambda: setattr(self.game, 'is_over', True)
        # 禁用障礙物生成與檢查
        self.game.blocks = []
        self.game.check_collision = lambda: None
        # 禁用加速道具
        if hasattr(self.game, 'speedup'):
            self.game.speedup.update = lambda: None

    def run(self, max_steps=1000):
        steps = 0
        while not self.game.is_over and steps < max_steps:
            head      = self.game.snake.body[0].copy()
            fruit_pos = self.game.fruit.pos
            last_dir  = self.game.snake.direction
            self.game.snake.direction = self.game.ai.decide(head, fruit_pos, last_dir)
            self.game.update()
            steps += 1
        ate = len(self.game.snake.body) - 3
        return ate * 1000 + steps


def tournament_selection(pop, k=3):
    return max(random.sample(pop, k), key=lambda ind: ind.fitness)


def one_point_crossover(g1, g2):
    pt = random.randrange(1, CHROMO_LEN)
    c1 = np.concatenate([g1[:pt], g2[pt:]])
    c2 = np.concatenate([g2[:pt], g1[pt:]])
    return c1, c2


def mutate(genome, rate):
    for i in range(len(genome)):
        if random.random() < rate:
            genome[i] += np.random.randn() * 0.1
    return genome

class GATrainer:
    def __init__(self):
        self.population = [Individual() for _ in range(POP_SIZE)]

    def evolve(self):
        for gen in range(GENS):
            # 評估
            for ind in self.population:
                ind.fitness = SnakeGame(ind).run()

            # 排序並印出當代最佳
            self.population.sort(key=lambda x: x.fitness, reverse=True)
            best = self.population[0]
            print(f"Gen {gen:03d}  Best: {best.fitness}", flush=True)

            # 構造下一代
            next_pop = self.population[:int(ELITE_RATE * POP_SIZE)]
            while len(next_pop) < POP_SIZE:
                p1 = tournament_selection(self.population)
                p2 = tournament_selection(self.population)
                c1, c2 = one_point_crossover(p1.genome, p2.genome)
                next_pop.append(Individual(mutate(c1.copy(), MUT_RATE)))
                if len(next_pop) < POP_SIZE:
                    next_pop.append(Individual(mutate(c2.copy(), MUT_RATE)))
            self.population = next_pop

        return self.population[0]

if __name__ == "__main__":
    trainer = GATrainer()
    best = trainer.evolve()
    np.save("best_genome.npy", best.genome)
    print("訓練完成，最佳基因已儲存於 best_genome.npy")