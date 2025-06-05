"""
快速啟動DQN訓練和遊戲的腳本
"""
from dqn_trainer import train_dqn, DQNPlayer
from snake import MAIN
import os

def main():
    print("=== DQN 貪食蛇 AI ===")
    print("1. 訓練新的DQN模型")
    print("2. 使用現有模型遊戲")
    print("3. 快速訓練並遊戲")
    
    choice = input("請選擇 (1/2/3): ").strip()
    
    if choice == '1':
        episodes = int(input("請輸入訓練回合數 (建議1000-5000): ") or "2000")
        print(f"開始訓練 {episodes} 回合...")
        train_dqn(episodes=episodes)
        
    elif choice == '2':
        if os.path.exists("best_dqn_model.pth"):
            print("載入模型並開始遊戲...")
            ai = DQNPlayer("best_dqn_model.pth")
            game = MAIN(ai)
            game.run()
        else:
            print("找不到訓練好的模型，請先訓練！")
            
    elif choice == '3':
        print("快速訓練 500 回合...")
        train_dqn(episodes=500)
        print("訓練完成，開始遊戲...")
        ai = DQNPlayer("best_dqn_model.pth")
        game = MAIN(ai)
        game.run()
        
    else:
        print("無效選擇")

if __name__ == "__main__":
    main()