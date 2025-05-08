
# 負責切換模式：train / play / ai
import sys
import numpy as np
from snake import MAIN
from ga_trainer import GATrainer, Individual


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else 'play'
    if mode == 'train':
        best = GATrainer().evolve()
        np.save('best_genome.npy', best.genome)
        print('訓練完成，基因儲存於 best_genome.npy')
        return

    ai = None
    if mode == 'ai':
        genome = np.load('best_genome.npy')
        ai = Individual(genome)

    game = MAIN(ai)
    game.run()

if __name__ == '__main__':
    main()