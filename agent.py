import tensorflow as tf
import random
import numpy as np
from collections import deque
from snake import SnakeGame as Game
from snake import Direction, Point, BLOCK_SIZE, SPEED
from helper import plot

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001

class Agent:
    def __init__(self): 
        self.n_games = 0
        self.epsilon = 0
        self.gamma = 0.9
        self.replay_memory = deque(maxlen=MAX_MEMORY)
        self.model = self.create_model()
        self.target_model = self.create_model()
        self.target_update_counter = 0
        
    def create_model(self):
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Dense(24, input_shape=(11,), activation='relu'))
        model.add(tf.keras.layers.Dense(24, activation='relu'))
        model.add(tf.keras.layers.Dense(3, activation='softmax'))
        model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(lr=LR), metrics=['accuracy'])
        return model

    def update_replay_memory(self, transition):
        self.replay_memory.append(transition)

    def get_state(self, game):
        head = game.snake[0]
        point_l = Point(head.x - BLOCK_SIZE, head.y)
        point_r = Point(head.x + BLOCK_SIZE, head.y)
        point_u = Point(head.x, head.y - BLOCK_SIZE)
        point_d = Point(head.x, head.y + BLOCK_SIZE)

        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        state = [
            # Danger straight
            (dir_r and game.is_collision(point_r)) or
            (dir_l and game.is_collision(point_l)) or
            (dir_u and game.is_collision(point_u)) or
            (dir_d and game.is_collision(point_d)),
            
            # Danger right
            (dir_u and game.is_collision(point_r)) or 
            (dir_d and game.is_collision(point_l)) or
            (dir_l and game.is_collision(point_u)) or
            (dir_r and game.is_collision(point_d)),

            # Danger left
            (dir_d and game.is_collision(point_r)) or 
            (dir_u and game.is_collision(point_l)) or 
            (dir_r and game.is_collision(point_u)) or 
            (dir_l and game.is_collision(point_d)),

            # Move direction
            dir_l,
            dir_r, 
            dir_u,
            dir_d,

            # Food location
            game.food.x < game.head.x,  # food left
            game.food.x > game.head.x,  # food right
            game.food.y < game.head.y,  # food up
            game.food.y > game.head.y  # food down
        ]
        return np.array(state, dtype=int)    
            
    def remember(self, state, action, reward, next_state, done):
        self.replay_memory.append((state, action, reward, next_state, done))

    def train_long_memory(self):
        if len(self.replay_memory) >= BATCH_SIZE:
            batch = random.sample(self.replay_memory, BATCH_SIZE)
        else:
            batch = self.replay_memory
        
        states, actions, rewards, next_states, dones = zip(*batch)

        current_qs_list = self.model.predict(np.array(states))
        future_qs_list = self.target_model.predict(np.array(next_states))

        X = []
        y = []

        for index, (state, action, reward, next_state, done) in enumerate(batch):
            if not done:
                max_future_q = np.max(future_qs_list[index])
                new_q = reward + self.gamma * max_future_q
            else:
                new_q = reward

            current_qs = current_qs_list[index]
            current_qs[action] = new_q

            X.append(state)
            y.append(current_qs)

        self.model.fit(np.array(X), np.array(y), batch_size=BATCH_SIZE, verbose=0, shuffle=False)

        self.target_update_counter += 1
        if self.target_update_counter > 10:
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter = 0

    def train_short_memory(self, state, action, reward, next_state, done):
        target = reward
        if not done:
            target = reward + self.gamma * np.max(self.target_model.predict(np.array([next_state]))[0])

        target_f = self.model.predict(np.array([state]))
        target_f[0][action] = target

        self.model.fit(np.array([state]), target_f, epochs=1, verbose=0)

    def get_action(self, state):
        self.epsilon = 80 - self.n_games
        final_move = [0, 0, 0]

        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
            print("random")
        else:
            move = np.argmax(self.model.predict(np.array([state]))[0])
            final_move[move] = 1
            print("predict")
        return final_move
    
def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = Agent()
    game = Game()
    
    while True:
        # get old state
        old_state = agent.get_state(game)

        # get move
        final_move = agent.get_action(old_state)

        # perform move and get new state
        reward, done, score = game.play_step(final_move)
        state_new = agent.get_state(game)

        # train short memory
        agent.train_short_memory(old_state, final_move, reward, state_new, done)

        # remember
        agent.remember(old_state, final_move, reward, state_new, done)

        if done:
            # train long memory, plot result
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()

            if score > record:
                record = score
                agent.model.save()

            print('Game', agent.n_games, 'Score', score, 'Record:', record)

            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores) 

if __name__ == '__main__':
    train()