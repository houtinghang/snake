# snake.py
import pygame, sys, random
from pygame.math import Vector2

# 全域設定（可依需求調整）
cell_size = 40       # 每格像素大小
cell_number = 20     # 格子數量

class SNAKE:
    """處理蛇身繪製、移動與增長邏輯"""
    def __init__(self):
        # 初始身體：三格
        self.body = [Vector2(5,10), Vector2(4,10), Vector2(3,10)]
        self.direction = Vector2(1,0)
        self.new_block = False
        # 載入頭尾與身體圖檔（請放在 Graphics/ 下）
        self.head_up       = pygame.image.load('Graphics/head_up.png').convert_alpha()
        self.head_down     = pygame.image.load('Graphics/head_down.png').convert_alpha()
        self.head_right    = pygame.image.load('Graphics/head_right.png').convert_alpha()
        self.head_left     = pygame.image.load('Graphics/head_left.png').convert_alpha()
        self.tail_up       = pygame.image.load('Graphics/tail_up.png').convert_alpha()
        self.tail_down     = pygame.image.load('Graphics/tail_down.png').convert_alpha()
        self.tail_right    = pygame.image.load('Graphics/tail_right.png').convert_alpha()
        self.tail_left     = pygame.image.load('Graphics/tail_left.png').convert_alpha()
        self.body_vertical   = pygame.image.load('Graphics/body_vertical.png').convert_alpha()
        self.body_horizontal = pygame.image.load('Graphics/body_horizontal.png').convert_alpha()
        self.body_tr = pygame.image.load('Graphics/body_tr.png').convert_alpha()
        self.body_tl = pygame.image.load('Graphics/body_tl.png').convert_alpha()
        self.body_br = pygame.image.load('Graphics/body_br.png').convert_alpha()
        self.body_bl = pygame.image.load('Graphics/body_bl.png').convert_alpha()
        # 加速道具效果速度
        self.base_speed  = 12
        self.boost_speed = 36
        self.boost_end   = 0

    def draw(self, screen):
        """根據身體節點繪製蛇，每格一圖"""
        self.update_head_graphics()
        self.update_tail_graphics()
        for i, block in enumerate(self.body):
            rect = pygame.Rect(block.x*cell_size, block.y*cell_size, cell_size, cell_size)
            if i == 0:
                screen.blit(self.head, rect)
            elif i == len(self.body)-1:
                screen.blit(self.tail, rect)
            else:
                prev_rel = self.body[i+1] - block
                next_rel = self.body[i-1] - block
                if prev_rel.x == next_rel.x:
                    screen.blit(self.body_vertical, rect)
                elif prev_rel.y == next_rel.y:
                    screen.blit(self.body_horizontal, rect)
                else:
                    if (prev_rel.x==-1 and next_rel.y==-1) or (prev_rel.y==-1 and next_rel.x==-1):
                        screen.blit(self.body_tl, rect)
                    elif (prev_rel.x==1 and next_rel.y==-1) or (prev_rel.y==-1 and next_rel.x==1):
                        screen.blit(self.body_tr, rect)
                    elif (prev_rel.x==-1 and next_rel.y==1) or (prev_rel.y==1 and next_rel.x==-1):
                        screen.blit(self.body_bl, rect)
                    else:
                        screen.blit(self.body_br, rect)

    def update_head_graphics(self):
        """依照第二節位置決定頭方向圖"""
        d = self.body[1] - self.body[0]
        if d == Vector2(1,0):   self.head = self.head_left
        elif d == Vector2(-1,0):self.head = self.head_right
        elif d == Vector2(0,1): self.head = self.head_up
        elif d == Vector2(0,-1):self.head = self.head_down

    def update_tail_graphics(self):
        """依照倒數第二節決定尾巴方向圖"""
        d = self.body[-2] - self.body[-1]
        if d == Vector2(1,0):   self.tail = self.tail_left
        elif d == Vector2(-1,0):self.tail = self.tail_right
        elif d == Vector2(0,1): self.tail = self.tail_up
        elif d == Vector2(0,-1):self.tail = self.tail_down

    def move(self):
        """格子移動，若 new_block 為 True 則身長+1"""
        if self.new_block:
            b = self.body[:]
            b.insert(0, b[0] + self.direction)
            self.body = b
            self.new_block = False
        else:
            b = self.body[:-1]
            b.insert(0, b[0] + self.direction)
            self.body = b

    def add_block(self):
        """下一次 move() 多加一節"""
        self.new_block = True

    def activate_boost(self, duration):
        """啟動加速效果，持續 duration 毫秒"""
        self.boost_end = pygame.time.get_ticks() + duration

    def current_speed(self):
        """回傳當前速率 (加速 / 正常)"""
        return self.boost_speed if pygame.time.get_ticks() < self.boost_end else self.base_speed

class FRUIT:
    """果實位置管理"""
    def __init__(self, snake_body):
        self.randomize(snake_body)
    def randomize(self, snake_body):
        """隨機放在蛇身以外的位置"""
        while True:
            p = Vector2(random.randrange(cell_number), random.randrange(cell_number))
            if p not in snake_body:
                self.pos = p
                break
    def draw(self, screen):
        rect = pygame.Rect(self.pos.x*cell_size, self.pos.y*cell_size, cell_size, cell_size)
        screen.blit(apple, rect)

class BLOCK:
    """障礙物管理"""
    def __init__(self, snake_body, forbidden, existing):
        self.randomize(snake_body, forbidden, existing)
    def randomize(self, snake_body, forbidden, existing):
        while True:
            p = Vector2(random.randrange(cell_number), random.randrange(cell_number))
            if p not in snake_body and p not in forbidden and all(p != b.pos for b in existing):
                self.pos = p
                break
    def draw(self, screen):
        rect = pygame.Rect(self.pos.x*cell_size, self.pos.y*cell_size, cell_size, cell_size)
        pygame.draw.rect(screen, (100,100,100), rect)

class SPEEDUP:
    """加速道具管理"""
    def __init__(self, snake, interval=10000, duration=5000):
        self.snake = snake
        self.interval = interval
        self.duration = duration
        self.next_spawn = pygame.time.get_ticks() + interval
        self.pos = None
    def update(self):
        """定時產生 & 被吃後重置"""
        t = pygame.time.get_ticks()
        if self.pos is None and t >= self.next_spawn:
            self.randomize()
        if self.pos and self.snake.body[0] == self.pos:
            self.snake.activate_boost(self.duration)
            self.pos = None
            self.next_spawn = t + self.interval
    def randomize(self):
        """隨機放置在蛇身外"""
        while True:
            p = Vector2(random.randrange(cell_number), random.randrange(cell_number))
            if p not in self.snake.body:
                self.pos = p
                break
    def draw(self, screen):
        if self.pos:
            rect = pygame.Rect(self.pos.x*cell_size, self.pos.y*cell_size, cell_size, cell_size)
            pygame.draw.rect(screen, (255,0,0), rect)

class MAIN:
    """遊戲主類別：負責初始化、主迴圈與繪製"""
    def __init__(self, ai=None):
        pygame.init()
        self.screen = pygame.display.set_mode((cell_size*cell_number, cell_size*cell_number))
        self.clock = pygame.time.Clock()
        global apple, game_font
        apple = pygame.image.load('Graphics/apple.png').convert_alpha()
        game_font = pygame.font.Font('Snake/Font/PoetsenOne-Regular.ttf', 25)
        self.snake = SNAKE()
        self.fruit = FRUIT(self.snake.body)
        self.blocks = []
        self.speedup = SPEEDUP(self.snake)
        self.ai = ai  # 傳入 Individual 實例，若 None 則手動
        self.is_over = False  # GA 模擬時攔截結束

    def update(self):
        """※ 新增：對應 ga_trainer.py 的逐步模擬呼叫"""
        self.snake.move()
        self.speedup.update()
        self.check_collision()
        self.check_fail()

    def run(self):
        """主迴圈：事件、邏輯更新、繪製"""
        while True:
            self.clock.tick(self.snake.current_speed())
            for e in pygame.event.get():
                if e.type == pygame.QUIT:
                    self.quit()
                if self.ai is None and e.type == pygame.KEYDOWN:
                    if e.key == pygame.K_UP and self.snake.direction.y != 1:
                        self.snake.direction = Vector2(0,-1)
                    if e.key == pygame.K_DOWN and self.snake.direction.y != -1:
                        self.snake.direction = Vector2(0,1)
                    if e.key == pygame.K_LEFT and self.snake.direction.x != 1:
                        self.snake.direction = Vector2(-1,0)
                    if e.key == pygame.K_RIGHT and self.snake.direction.x != -1:
                        self.snake.direction = Vector2(1,0)
            if self.ai:
                head = self.snake.body[0].copy()
                self.snake.direction = self.ai.decide(head, self.fruit.pos, self.snake.direction)
            # 遊戲邏輯更新 (含障礙、加速、碰撞檢查)
            self.update()
            # 畫面繪製
            self.screen.fill((175,215,70))
            self.draw_elements()
            pygame.display.flip()

    def check_collision(self):
        """吃到果實後處理：長身 & 重置果實 & 重產障礙"""
        if self.fruit.pos == self.snake.body[0]:
            self.fruit.randomize(self.snake.body)
            self.snake.add_block()
            num = random.randint(1,10)
            self.blocks.clear()
            for _ in range(num):
                b = BLOCK(self.snake.body, [self.fruit.pos], self.blocks)
                self.blocks.append(b)

    def check_fail(self):
        """撞牆或撞身或撞障礙則結束"""
        head = self.snake.body[0]
        if not (0 <= head.x < cell_number and 0 <= head.y < cell_number):
            self.quit()
        for seg in self.snake.body[1:]:
            if seg == head:
                self.quit()
        for b in self.blocks:
            if b.pos == head:
                self.quit()

    def draw_elements(self):
        """繪製草地、果實、障礙、道具、蛇、分數"""
        color = (167,209,61)
        for r in range(cell_number):
            for c in range(cell_number):
                if (r+c) % 2 == 0:
                    pygame.draw.rect(self.screen, color,
                                     (c*cell_size, r*cell_size, cell_size, cell_size))
        self.fruit.draw(self.screen)
        for b in self.blocks:
            b.draw(self.screen)
        self.speedup.draw(self.screen)
        self.snake.draw(self.screen)
        txt = str(len(self.snake.body) - 3)
        surf = game_font.render(txt, True, (56,74,12))
        x, y = cell_size*cell_number-60, cell_size*cell_number-40
        rect_score = surf.get_rect(center=(x,y))
        rect_apple = apple.get_rect(midright=(rect_score.left, rect_score.centery))
        bg = pygame.Rect(rect_apple.left, rect_apple.top,
                         rect_apple.width + rect_score.width + 6, rect_apple.height)
        pygame.draw.rect(self.screen, (167,209,61), bg)
        self.screen.blit(surf, rect_score)
        self.screen.blit(apple, rect_apple)
        pygame.draw.rect(self.screen, (56,74,12), bg, 2)

    def quit(self):
        """結束遊戲並關閉程式"""
        pygame.quit()
        sys.exit()
