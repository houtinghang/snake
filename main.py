# 負責切換模式：train / play / ai
import sys
from snake import MAIN
from ga_trainer import Individual
from dqn_trainer import DQNPlayer, train_dqn
import numpy as np
import os


def main():
    if len(sys.argv) > 1:
        mode = sys.argv[1]
    else:
        mode = 'dqn'  # 預設使用DQN模式

    if mode == 'train_ga':
        print("開始GA訓練...")
        from ga_trainer import GATrainer

        trainer = GATrainer()
        best = trainer.evolve()
        np.save("best_genome.npy", best.genome)
        print("GA訓練完成")

    elif mode == 'train_dqn':
        print("開始DQN訓練...")
        train_dqn(episodes=2000)

    elif mode == 'play':
        print("手動遊戲模式")
        game = MAIN()
        game.run()

    elif mode == 'ga':
        print("GA AI 自動遊戲")
        if os.path.exists("best_genome.npy"):
            genome = np.load("best_genome.npy")
            ai = Individual(genome)
            game = MAIN(ai)
            game.run()
        else:
            print("找不到 best_genome.npy，請先執行訓練")

    elif mode == 'dqn':
        print("DQN AI 自動遊戲")
        if os.path.exists("best_dqn_model.pth"):
            ai = DQNPlayer("best_dqn_model.pth")
            game = MAIN(ai)
            game.run()
        else:
            print("找不到 best_dqn_model.pth，開始訓練...")
            train_dqn(episodes=1000)
            ai = DQNPlayer("best_dqn_model.pth")
            game = MAIN(ai)
            game.run()
    else:
        print("使用方式:")
        print("python main.py play       # 手動遊戲")
        print("python main.py ga         # GA AI 自動遊戲")
        print("python main.py dqn        # DQN AI 自動遊戲")
        print("python main.py train_ga   # 訓練 GA")
        print("python main.py train_dqn  # 訓練 DQN")


if __name__ == "__main__":
    main()