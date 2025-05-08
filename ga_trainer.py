# ga_trainer.py

import random
import numpy as np
from pygame.math import Vector2

# --------------------------------------------------
# ※ 這裡要從 snake.py 匯入 MAIN 類別和格子數量
from snake import MAIN, cell_number as GRID_SIZE
# --------------------------------------------------

# ===== GA 超參數 =====
POP_SIZE   = 30      # 族群大小
GENS       = 200     # 演化世代數
MUT_RATE   = 0.2    # 突變機率
ELITE_RATE = 0.1     # 精英保留比例

# ===== 神經網路結構 =====
INPUT_SIZE   = 4     # [相對 X, 相對 Y, 方向 X, 方向 Y]
HIDDEN_SIZE  = 8
OUTPUT_SIZE  = 4     # 四個方向
# 總基因長度 = W1 + W2 + b1 + b2
CHROMO_LEN   = INPUT_SIZE*HIDDEN_SIZE + HIDDEN_SIZE*OUTPUT_SIZE + HIDDEN_SIZE + OUTPUT_SIZE

# 四個候選移動向量
DIRECTIONS = [
    Vector2(1,0), Vector2(-1,0),
    Vector2(0,1), Vector2(0,-1)
]

class Individual:
    """染色體：封裝前饋網路的所有權重與偏差。"""
    def __init__(self, genome=None):
        if genome is None:
            # 隨機初始化
            self.genome = np.random.uniform(-1, 1, CHROMO_LEN)
        else:
            self.genome = genome
        self.fitness = 0

    def decode(self):
        """拆分 genome 成 W1, W2, b1, b2"""
        g = self.genome
        w1_end = INPUT_SIZE * HIDDEN_SIZE
        w2_end = w1_end + HIDDEN_SIZE * OUTPUT_SIZE
        b1_end = w2_end + HIDDEN_SIZE
        w1 = g[:w1_end].reshape((INPUT_SIZE, HIDDEN_SIZE))
        w2 = g[w1_end:w2_end].reshape((HIDDEN_SIZE, OUTPUT_SIZE))
        b1 = g[w2_end:b1_end]
        b2 = g[b1_end:]
        return w1, w2, b1, b2

    def decide(self, head, fruit_pos, last_dir):
        """
        根據 (蛇頭座標, 果實座標, 上一步方向) 決定下一步方向。
        回傳 DIRECTIONS 之一。
        """
        # 標準化輸入狀態
        state = np.array([
            (fruit_pos.x - head.x) / GRID_SIZE,
            (fruit_pos.y - head.y) / GRID_SIZE,
            last_dir.x,
            last_dir.y
        ])
        w1, w2, b1, b2 = self.decode()
        hidden = np.tanh(state.dot(w1) + b1)   # 隱藏層
        output = hidden.dot(w2) + b2           # 輸出層
        idx = np.argmax(output)                # 選最大分對應的方向
        return DIRECTIONS[idx]

class SnakeGame:
    """
    用完整版 MAIN 來模擬遊戲流程，並計算 fitness。
    ※ 不顯示視窗，只呼叫 update()。
    """
    def __init__(self, individual):
        # 1. 建立有 AI 個體的遊戲實例
        self.game = MAIN(individual)
        # 2. 攔截原本的 game_over(), 只設定 flag 不結束程式
        self.game.is_over = False
        def _override_game_over():
            self.game.is_over = True
        self.game.game_over = _override_game_over

    def run(self, max_steps=5000):
        """
        執行到死亡或走滿 max_steps，回傳：
        fitness = 吃到果實數*1000 + 存活步數
        """
        steps = 0
        while not self.game.is_over and steps < max_steps:
            head      = self.game.snake.body[0].copy()
            fruit_pos = self.game.fruit.pos
            last_dir  = self.game.snake.direction
            # AI 決策下一步
            self.game.snake.direction = self.game.ai.decide(head, fruit_pos, last_dir)
            # 執行一次完整遊戲邏輯（含移動、障礙、加速、碰撞）
            self.game.update()
            steps += 1

        # 計算吃到的果實顆數
        ate = len(self.game.snake.body) - 3
        return ate * 1000 + steps

class GATrainer:
    """遺傳演算法主流程：評估 → 排序 → 精英保留 → 交配 → 突變 → 重複。"""
    def __init__(self):
        # 初始族群
        self.population = [Individual() for _ in range(POP_SIZE)]

    def evolve(self):
        best = None
        for gen in range(GENS):
            # 1) 計算每隻蛇的 fitness
            for ind in self.population:
                ind.fitness = SnakeGame(ind).run()

            # 2) 排序由高到低，並輸出當代最佳
            self.population.sort(key=lambda x: x.fitness, reverse=True)
            best = self.population[0]
            print(f"Generation {gen}: Best Fitness = {best.fitness}")

            # 3) 精英保留
            num_elite = max(1, int(ELITE_RATE * POP_SIZE))
            elites    = self.population[:num_elite]

            # 4) 交配 + 突變，產生下一代
            next_pop = elites.copy()
            while len(next_pop) < POP_SIZE:
                p1, p2 = random.sample(elites, 2)
                # 單點交配
                pt = random.randint(1, CHROMO_LEN - 1)
                g1 = np.concatenate([p1.genome[:pt], p2.genome[pt:]])
                g2 = np.concatenate([p2.genome[:pt], p1.genome[pt:]])
                # 突變函式
                def mutate(g):
                    for i in range(len(g)):
                        if random.random() < MUT_RATE:
                            g[i] += random.uniform(-0.5, 0.5)
                    return g
                next_pop.append(Individual(mutate(g1)))
                if len(next_pop) < POP_SIZE:
                    next_pop.append(Individual(mutate(g2)))

            # 5) 限制族群大小
            self.population = next_pop[:POP_SIZE]

        return best

if __name__ == "__main__":
    # 直接執行時開始訓練並儲存最佳基因
    trainer = GATrainer()
    best_ind = trainer.evolve()
    np.save("best_genome.npy", best_ind.genome)
    print("訓練完成！最佳基因已儲存至 best_genome.npy")
