import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from collections import deque
import matplotlib.pyplot as plt
import os
from pygame.math import Vector2
from snake import MAIN, cell_number as GRID_SIZE

# 優化的DQN超參數
LEARNING_RATE = 0.0005        # 降低學習率提高穩定性
GAMMA = 0.95                  # 提高折扣因子
EPSILON_START = 1.0
EPSILON_END = 0.01            # 提高最終探索率
EPSILON_DECAY = 0.995         # 調整衰減率
MEMORY_SIZE = 50000           # 減少記憶體使用
BATCH_SIZE = 64               # 減小批次大小
UPDATE_TARGET_EVERY = 50      # 更頻繁更新目標網路

class DQN(nn.Module):  # 修正類名
    def __init__(self, input_size=11, hidden_size=256, output_size=4):
        super(DQN, self).__init__()  # 修正super調用
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)  # 增加一層
        self.fc4 = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x
    
    def save(self, filepath):
        torch.save(self.state_dict(), filepath)
    
    def load(self, filepath):
        self.load_state_dict(torch.load(filepath, map_location='cpu'))

class DQNAgent:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"使用設備: {self.device}")
        
        self.epsilon = EPSILON_START
        self.memory = deque(maxlen=MEMORY_SIZE)
        self.steps_done = 0
        
        # 主網路和目標網路
        self.q_network = DQN().to(self.device)
        self.target_network = DQN().to(self.device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=LEARNING_RATE, weight_decay=1e-7)
        
        # 學習率調度器
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=1000, gamma=0.9)
        
        # 更新目標網路
        self.update_target_network()
        
        # 動作映射 (上, 下, 左, 右)
        self.actions = [
            Vector2(0, -1),  # UP
            Vector2(0, 1),   # DOWN
            Vector2(-1, 0),  # LEFT
            Vector2(1, 0)    # RIGHT
        ]
    
    def update_target_network(self):
        """軟更新目標網路參數"""
        tau = 0.005
        for target_param, local_param in zip(self.target_network.parameters(), self.q_network.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)
    
    def get_state(self, game):
        """獲取遊戲狀態特徵"""
        head = game.snake.body[0]
        fruit_pos = game.fruit.pos
        current_dir = game.snake.direction
        
        # 檢查各方向是否有危險
        danger_straight = self.is_collision(game, head + current_dir)
        danger_right = self.is_collision(game, head + self.get_right_direction(current_dir))
        danger_left = self.is_collision(game, head + self.get_left_direction(current_dir))
        
        # 當前移動方向
        dir_up = current_dir == Vector2(0, -1)
        dir_down = current_dir == Vector2(0, 1)
        dir_left = current_dir == Vector2(-1, 0)
        dir_right = current_dir == Vector2(1, 0)
        
        # 食物相對位置
        food_left = fruit_pos.x < head.x
        food_right = fruit_pos.x > head.x
        food_up = fruit_pos.y < head.y
        food_down = fruit_pos.y > head.y
        
        state = [
            int(danger_straight),
            int(danger_right),
            int(danger_left),
            int(dir_up),
            int(dir_down),
            int(dir_left),
            int(dir_right),
            int(food_left),
            int(food_right),
            int(food_up),
            int(food_down)
        ]
        
        return np.array(state, dtype=np.float32)  # 改為float32
    
    def is_collision(self, game, point):
        """檢查指定點是否會碰撞"""
        # 撞牆
        if not (0 <= point.x < GRID_SIZE and 0 <= point.y < GRID_SIZE):
            return True
        # 撞蛇身
        if point in game.snake.body:
            return True
        # 撞障礙物
        for block in game.blocks:
            if point == block.pos:
                return True
        return False
    
    def get_right_direction(self, current_dir):
        """獲取右轉方向"""
        if current_dir == Vector2(0, -1):  # UP
            return Vector2(1, 0)  # RIGHT
        elif current_dir == Vector2(1, 0):  # RIGHT
            return Vector2(0, 1)  # DOWN
        elif current_dir == Vector2(0, 1):  # DOWN
            return Vector2(-1, 0)  # LEFT
        else:  # LEFT
            return Vector2(0, -1)  # UP
    
    def get_left_direction(self, current_dir):
        """獲取左轉方向"""
        if current_dir == Vector2(0, -1):  # UP
            return Vector2(-1, 0)  # LEFT
        elif current_dir == Vector2(-1, 0):  # LEFT
            return Vector2(0, 1)  # DOWN
        elif current_dir == Vector2(0, 1):  # DOWN
            return Vector2(1, 0)  # RIGHT
        else:  # RIGHT
            return Vector2(0, -1)  # UP
    
    def get_action(self, state):
        """改進的動作選擇策略"""
        self.steps_done += 1
        
        # 動態epsilon衰減
        self.epsilon = EPSILON_END + (EPSILON_START - EPSILON_END) * \
                      np.exp(-1. * self.steps_done / 2000)
        
        if random.random() < self.epsilon:
            return random.randint(0, 3)
        else:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.q_network(state_tensor)
            return q_values.argmax().item()
    
    def remember(self, state, action, reward, next_state, done):
        """儲存經驗"""
        self.memory.append((state, action, reward, next_state, done))
    
    def replay(self):
        """改進的經驗重播訓練"""
        if len(self.memory) < BATCH_SIZE:
            return
        
        batch = random.sample(self.memory, BATCH_SIZE)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.BoolTensor(dones).to(self.device)
        
        # Double DQN
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # 使用主網路選擇動作，目標網路評估價值
        next_actions = self.q_network(next_states).argmax(1, keepdim=True)
        next_q_values = self.target_network(next_states).gather(1, next_actions).squeeze()
        target_q_values = rewards + (GAMMA * next_q_values * ~dones)
        
        # Huber Loss (更穩定)
        loss = F.smooth_l1_loss(current_q_values.squeeze(), target_q_values.detach())
        
        self.optimizer.zero_grad()
        loss.backward()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        
        self.optimizer.step()
        self.scheduler.step()

class DQNSnakeGame:
    """適配DQN的貪食蛇遊戲"""
    def __init__(self):
        self.game = MAIN()
        self.game.is_over = False
        self.game.quit = lambda: setattr(self.game, 'is_over', True)
        self.steps = 0
        self.max_steps = 1500  # 增加最大步數
        
    def reset(self):
        """重置遊戲"""
        self.game = MAIN()
        self.game.is_over = False
        self.game.quit = lambda: setattr(self.game, 'is_over', True)
        self.steps = 0
        return self.game
    
    def step(self, action_idx, agent):
        """執行一步遊戲"""
        # 將動作索引轉換為方向向量
        new_direction = agent.actions[action_idx]
        
        # 檢查是否為相反方向（不允許）
        if new_direction == -self.game.snake.direction:
            new_direction = self.game.snake.direction
        
        self.game.snake.direction = new_direction
        
        # 記錄吃果實前的分數和頭部位置
        score_before = len(self.game.snake.body)
        head_before = self.game.snake.body[0].copy()
        fruit_pos = self.game.fruit.pos.copy()
        
        # 執行遊戲邏輯
        self.game.update()
        self.steps += 1
        
        # 計算獎勵
        reward = 0
        done = False
        
        # 檢查遊戲是否結束
        if self.game.is_over:
            reward = -100  # 死亡重懲罰
            done = True
        elif self.steps >= self.max_steps:
            reward = -50   # 超時懲罰
            done = True
        else:
            # 吃到果實獎勵
            score_after = len(self.game.snake.body)
            if score_after > score_before:
                reward = 100 + score_after * 10  # 越長獎勵越多
                self.steps = 0  # 重置步數計數器
            else:
                # 距離獎勵：越靠近食物獎勵越多
                distance_before = abs(head_before.x - fruit_pos.x) + abs(head_before.y - fruit_pos.y)
                distance_after = abs(self.game.snake.body[0].x - fruit_pos.x) + abs(self.game.snake.body[0].y - fruit_pos.y)
                
                if distance_after < distance_before:
                    reward = 10  # 接近食物
                elif distance_after > distance_before:
                    reward = -5  # 遠離食物
                else:
                    reward = 1   # 生存獎勵
        
        return reward, done, len(self.game.snake.body) - 3

def train_dqn(episodes=5000):
    """DQN訓練主函數"""
    agent = DQNAgent()
    game = DQNSnakeGame()
    
    scores = []
    mean_scores = []
    total_score = 0
    record = 0
    episode_rewards = []
    
    print("開始DQN訓練...")
    print(f"目標回合數: {episodes}")
    print(f"使用設備: {agent.device}")
    print(f"學習率: {LEARNING_RATE}")
    print(f"折扣因子: {GAMMA}")
    
    for episode in range(episodes):
        game.reset()
        state = agent.get_state(game.game)
        total_reward = 0
        
        while True:
            action = agent.get_action(state)
            reward, done, score = game.step(action, agent)
            next_state = agent.get_state(game.game) if not done else np.zeros(11)
            
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            
            if done:
                break
        
        # 訓練網路
        if episode > 100:  # 累積一定經驗後開始訓練
            agent.replay()
        
        # 更新目標網路
        if episode % UPDATE_TARGET_EVERY == 0:
            agent.update_target_network()
        
        # 記錄分數
        scores.append(score)
        episode_rewards.append(total_reward)
        total_score += score
        mean_score = total_score / (episode + 1)
        mean_scores.append(mean_score)
        
        if score > record:
            record = score
            agent.q_network.save('best_dqn_model.pth')
            print(f'新記錄！Episode {episode + 1}, Score: {score}')
        
        # 定期保存檢查點
        if episode % 1000 == 0 and episode > 0:
            agent.q_network.save(f'dqn_checkpoint_{episode}.pth')
        
        # 輸出進度
        if episode % 100 == 0:
            avg_reward = np.mean(episode_rewards[-100:]) if len(episode_rewards) >= 100 else np.mean(episode_rewards)
            avg_score = np.mean(scores[-100:]) if len(scores) >= 100 else np.mean(scores)
            print(f'Episode {episode:4d} | Score: {score:2d} | Avg Score: {avg_score:5.2f} | '
                  f'Record: {record:2d} | Reward: {total_reward:6.1f} | Avg Reward: {avg_reward:6.1f} | '
                  f'Epsilon: {agent.epsilon:.3f}')
        
        # 每500個episode繪製進度圖  
        #if(episode + 1) % 500 == 0:
           #plot_training_progress(scores, mean_scores, episode_rewards)
    
    # 儲存最終模型
    agent.q_network.save('final_dqn_model.pth')
    print("DQN訓練完成！")
    print(f"最佳分數: {record}")
    print(f"平均分數: {total_score / episodes:.2f}")
    return agent

# def plot_training_progress(scores, mean_scores, rewards):
#     """繪製訓練進度"""
#     try:
#         fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
#         # 分數圖
#         ax1.clear()
#         ax1.set_title('DQN Training Progress - Scores')
#         ax1.set_xlabel('Episode')
#         ax1.set_ylabel('Score')
#         ax1.plot(scores, alpha=0.6, label='Score')
#         ax1.plot(mean_scores, label='Mean Score', linewidth=2)
#         ax1.legend()
#         ax1.grid(True, alpha=0.3)
        
#         # 獎勵圖
#         ax2.clear()
#         ax2.set_title('DQN Training Progress - Rewards')
#         ax2.set_xlabel('Episode')
#         ax2.set_ylabel('Total Reward')
        
#         # 計算移動平均
#         if len(rewards) > 100:
#             moving_avg = np.convolve(rewards, np.ones(100)/100, mode='valid')
#             ax2.plot(range(99, len(rewards)), moving_avg, label='Moving Average (100)', linewidth=2)
        
#         ax2.plot(rewards, alpha=0.3, label='Reward')
#         ax2.legend()
#         ax2.grid(True, alpha=0.3)
        
#         plt.tight_layout()
#         plt.show(block=False)
#         plt.pause(0.1)
#     except Exception as e:
#         print(f"繪圖錯誤: {e}")

class DQNPlayer:
    """使用訓練好的DQN模型來玩遊戲"""
    def __init__(self, model_path='best_dqn_model.pth'):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = DQN().to(self.device)
        
        if os.path.exists(model_path):
            self.model.load(model_path)
            print(f"載入模型: {model_path}")
        else:
            print(f"找不到模型檔案: {model_path}")
        
        self.model.eval()
        
        # 動作映射
        self.actions = [
            Vector2(0, -1),  # UP
            Vector2(0, 1),   # DOWN
            Vector2(-1, 0),  # LEFT
            Vector2(1, 0)    # RIGHT
        ]
    
    def get_state(self, game):
        """獲取遊戲狀態 (與DQNAgent相同)"""
        agent = DQNAgent()
        return agent.get_state(game)
    
    def decide(self, head, fruit_pos, current_dir):
        """決定下一步動作"""
        # 創建一個臨時遊戲物件來獲取狀態
        temp_game = type('TempGame', (), {
            'snake': type('Snake', (), {'body': [head], 'direction': current_dir})(),
            'fruit': type('Fruit', (), {'pos': fruit_pos})(),
            'blocks': []
        })()
        
        state = self.get_state(temp_game)
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            q_values = self.model(state_tensor)
            action_idx = q_values.argmax().item()
        
        new_direction = self.actions[action_idx]
        
        # 檢查是否為相反方向（不允許）
        if new_direction == -current_dir:
            return current_dir
        
        return new_direction

if __name__ == "__main__":
    print("開始DQN訓練...")
    print("使用 Ctrl+C 可以中斷訓練")
    
    try:
        trained_agent = train_dqn(episodes=5000)
        print("訓練完成！")
    except KeyboardInterrupt:
        print("\n訓練被中斷")