# Adapted from: https://github.com/TheAILearner/Snake-Game-using-OpenCV-Python/blob/master/snake_game_using_opencv.ipynb
# Get from Sentdex: https://www.youtube.com/watch?v=uKnjGn8fF70&list=PLQVvvaa0QuDf0O2DWwLZBfJeYY-JOeZB1&index=3
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import cv2
import random
import time
from collections import deque

SNAKE_LEN_GOAL = 100
SNAKE_INITIAL_LEN = 3  # (1-5)

APPLE_REWARD = 10
MAX_STEPS_LIMIT = 100


class SnakeEnv(gym.Env):
    def __init__(self):

        super(SnakeEnv, self).__init__()

        # Estados y acciones
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(low=-500, high=500,
                                            shape=(5 + SNAKE_LEN_GOAL,), dtype=np.float64)
        # Acciones anteriores
        self.prev_actions = deque([-1 for i in range(SNAKE_LEN_GOAL)], maxlen=SNAKE_LEN_GOAL)

        # Visualizar juego
        self.render = False

        # Número de steps hasta comer la manzana
        self.num_steps = 0

        # Si se ha llegado a un estado terminal
        self.truncated = False
        self.done = False

        # Reward y Score
        self.actual_reward, self.score = 0, 3

        # Posición de la serpiente y la manzana
        self.snake_head = [250, 250]
        self.snake_position = [[250, 250], [240, 250], [230, 250], [220, 250], [210, 250]][:SNAKE_INITIAL_LEN]
        self.apple_position = [random.randrange(1, 50) * 10, random.randrange(1, 50) * 10]

        # Distancia inicial hasta la manzana
        self.initial_dist_to_apple = np.sum(np.abs(np.array(self.snake_head) - np.array(self.apple_position))) / 10

        # Distancias opuestas (idx <-> num)
        self.actions = [1, 0, 3, 2]

        # Tablero
        self.img = np.zeros((500, 500, 3), dtype='uint8')

    def step(self, action):

        self.prev_actions.append(action)
        self.num_steps += 1
        info = {}

        #################### Visualization ####################

        if self.render:

            cv2.imshow('Snake', self.img)
            cv2.waitKey(1)
            self.img = np.zeros((500, 500, 3), dtype='uint8')

            # Display Snake
            red = 0
            for position in self.snake_position[1:]:
                cv2.rectangle(self.img, (position[0], position[1]), (position[0] + 10, position[1] + 10),
                              (0, 150, red), 3)
                if red <= 255 - red: red += 15

            # Display Snake head
            cv2.rectangle(self.img, (self.snake_head[0], self.snake_head[1]),
                          (self.snake_head[0] + 10, self.snake_head[1] + 10), (0, 255, 0), 3)

            # Display Apple
            cv2.rectangle(self.img, (self.apple_position[0], self.apple_position[1]),
                          (self.apple_position[0] + 10, self.apple_position[1] + 10), (0, 0, 255), 3)

            # Print info
            cv2.putText(self.img, f"Score: {self.score}", (5, 450),
                        cv2.FONT_HERSHEY_PLAIN, 1, (255, 50, 50), 1, cv2.LINE_AA)
            cv2.putText(self.img, f"Steps: {self.num_steps}", (5, 470),
                        cv2.FONT_HERSHEY_PLAIN, 1, (255, 50, 50), 1, cv2.LINE_AA)
            cv2.putText(self.img, f"Reward: {round(self.actual_reward, 4)}", (5, 490),
                        cv2.FONT_HERSHEY_PLAIN, 1, (255, 50, 50), 1, cv2.LINE_AA)

            # Takes step after fixed time
            t_end = time.time() + 0.05
            k = -1
            while time.time() < t_end:
                if k == -1:
                    k = cv2.waitKey(1)
                else:
                    continue

        ####################### Move snake #######################

        if action == 1:
            self.snake_head[0] += 10  # RIGHT
        elif action == 0:
            self.snake_head[0] -= 10  # LEFT
        elif action == 2:
            self.snake_head[1] += 10  # DOWN
        elif action == 3:
            self.snake_head[1] -= 10  # UP

        self.snake_position.insert(0, list(self.snake_head))

        ########### Calculate if collision with apple ############

        if self.snake_head == self.apple_position:
            self.apple_position = [random.randrange(1, 50) * 10, random.randrange(1, 50) * 10]
            self.score += 1
            self.num_steps = 0
            apple_reward = APPLE_REWARD

        else:

            self.snake_position.pop()
            apple_reward = 0

        ################ Calculate if game over #################

        # Si ha colisionado con los bordes
        if (0 > self.snake_head[0] or self.snake_head[0] >= 500 or
                0 > self.snake_head[1] or self.snake_head[1] >= 500):
            info['cause_of_death'] = 'Collision with bounders'
            self.truncated = True
            self.done = True

        # Si ha colisionado consigo misma
        elif self.snake_head in self.snake_position[1:]:
            info['cause_of_death'] = 'Collision with self'
            self.truncated = True
            self.done = True

        # Si ha superado el número límite de steps
        elif self.num_steps > MAX_STEPS_LIMIT:
            info['cause_of_death'] = 'Step limit exceeded'
            self.truncated = True
            self.done = True

        if self.render and self.truncated:
            self.img = np.zeros((500, 500, 3), dtype='uint8')
            cv2.putText(self.img, 'You Lose! Your Score is {}'.format(self.score), (25, 250), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 0, 255), 2, cv2.LINE_AA)
            cv2.imshow('Snake', self.img)

        ################## Calculate if it wins ##################

        # if len(self.snake_position) >= SNAKE_LEN_GOAL:
        #
        #     if self.render:
        #         font = cv2.FONT_HERSHEY_SIMPLEX
        #         self.img = np.zeros((500, 500, 3), dtype='uint8')
        #         cv2.putText(self.img, 'You Win! Your Score is {}'.format(self.score), (25, 250), font, 1,
        #                     (0, 255, 0),
        #                     2,
        #                     cv2.LINE_AA)
        #         cv2.imshow('Snake', self.img)
        #
        #     self.done = True

        #################### Calculate reward ####################

        # Obtener distancias
        manhattan_dist_to_apple = np.sum(np.abs(np.array(self.snake_head) - np.array(self.apple_position))) / 10

        manhattan_dist_to_nearst_border = min([self.snake_head[0], self.snake_head[1],
                                               499 - self.snake_head[0], 499 - self.snake_head[1]])

        manhattan_dist_to_nearst_snake_part = min([(np.sum(np.abs(np.array(self.snake_head) - np.array(elem))))
                                                   for elem in self.snake_position[1:]])

        # Premiar comer manzana
        self.actual_reward = apple_reward

        # Premiar si se acerca a la manzana por la ruta más corta
        if self.num_steps <= self.initial_dist_to_apple and apple_reward == 0:
            self.actual_reward += 1 / manhattan_dist_to_apple

        # Penalizar muerte
        if self.truncated:
            self.actual_reward -= 10

        # Penalizar doble si vuelve hacia atrás
        if self.actions[self.prev_actions[0]] == self.prev_actions[1]:
            self.actual_reward -= 10

        # Penalizar si está más cerca de un borde que de la manzana
        if manhattan_dist_to_nearst_border < manhattan_dist_to_apple:
            self.actual_reward -= 1

        # Penalizar si está a 1 de distancia de sí misma
        if manhattan_dist_to_nearst_snake_part == 1:
            self.actual_reward -= 1

        ######################### Return #########################

        head_x = self.snake_head[0]
        head_y = self.snake_head[1]

        snake_length = len(self.snake_position)
        apple_delta_x = self.apple_position[0] - head_x
        apple_delta_y = self.apple_position[1] - head_y

        # create observation
        observation = [head_x, head_y, apple_delta_x, apple_delta_y, snake_length] + list(self.prev_actions)
        observation = np.array(observation)

        info["score"] = self.score

        return observation, self.actual_reward, self.done, self.truncated, info

    def reset(self, seed=None, options=None):

        super().reset(seed=seed, options=options)

        # Acciones anteriores
        self.prev_actions = deque([-1 for i in range(SNAKE_LEN_GOAL)], maxlen=SNAKE_LEN_GOAL)

        # Número de steps hasta comer la manzana
        self.num_steps = 0

        # Si se ha llegado a un estado terminal
        self.truncated = False
        self.done = False

        # Reward y Score
        self.actual_reward, self.score = 0, 3

        # Posición de la serpiente y la manzana
        self.snake_head = [250, 250]
        self.snake_position = [[250, 250], [240, 250], [230, 250], [220, 250], [210, 250]][:SNAKE_INITIAL_LEN]
        self.apple_position = [random.randrange(1, 50) * 10, random.randrange(1, 50) * 10]

        # Distancia inicial hasta la manzana
        self.initial_dist_to_apple = np.sum(np.abs(np.array(self.snake_head) - np.array(self.apple_position))) / 10

        # Tablero
        self.img = np.zeros((500, 500, 3), dtype='uint8')

        ######################### Return #########################

        head_x = self.snake_head[0]
        head_y = self.snake_head[1]

        snake_length = len(self.snake_position)
        apple_delta_x = self.apple_position[0] - head_x
        apple_delta_y = self.apple_position[1] - head_y

        # create observation:
        observation = [head_x, head_y, apple_delta_x, apple_delta_y, snake_length] + list(self.prev_actions)
        observation = np.array(observation)

        info = {}

        return observation, info

    def activate_render(self):
        self.render = True
