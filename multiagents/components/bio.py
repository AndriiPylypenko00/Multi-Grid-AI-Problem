from typing import Optional, Tuple
import numpy as np
from multiagents.components.lib import *


class Cookie:
    def __init__(self, x: int, y: int) -> None:
        self.x = x
        self.y = y


class NeuralNetwork:
    def __init__(self, num_layers, num_inputs, num_outputs, num_of_neurons):
        self.num_inputs, self.num_outputs, self.num_layers = (num_inputs, num_outputs, num_layers)
        self.b = [np.random.rand() for _ in range(num_layers + 1)]
        self.weights = [
            np.random.randint(-1, 2, (num_inputs, num_of_neurons)),
            *[np.random.randint(-1, 2, (num_of_neurons, num_of_neurons)) for _ in range(num_layers)],
            np.random.randint(-1, 2, (num_of_neurons, num_outputs)),
        ]

    def relu(self, x):
        return np.maximum(0, x)

    def feed_forward(self, input_vector):
        x = self.relu(np.dot(input_vector, self.weights[0]) + self.b[0])
        for idx in range(self.num_layers):
            x = self.relu(np.dot(x, self.weights[idx + 1]) + self.b[idx])
        return self.relu(np.dot(x, self.weights[-1]) + self.b[-1])


class Agent(NeuralNetwork):
    def __init__(self, num_layers, num_inputs, num_outputs, num_of_neurons, max_score=100):
        super().__init__(num_layers, num_inputs, num_outputs, num_of_neurons)
        self.x = None
        self.y = None
        self.score = 0
        self.isdead = 0
        self.is_completed = 0
        self.max_score = max_score
        color = np.random.randint(3, 255)
        self.color = color if color > 100 else color + 20
        self.episode = 30
        self.episode_score = self.episode
        self.score_calculated = False

    def is_dead(self):
        if self.episode_score < 10:
            self.isdead = 1
        if self.episode_score > self.max_score:
            self.is_completed = 1
            self.score = 100000000
            self.episode_score = 10000000
            print('COMPLETE')
    
    def calculate_score(self):
        self.score += self.episode_score
        self.episode_score = self.episode
        self.score_calculated = True

class Map:
    def __init__(
        self,
        size_x: int,
        size_y: int,
        num_of_cookies: int,
        cookie_prize: int = 5,
        move_punishment: int = 1,
        border_punishment: int = 10,
    ) -> None:
        self.size_x = size_x
        self.size_y = size_y
        self.num_of_cookies = num_of_cookies
        self.cookie_prize = cookie_prize
        self.move_punishment = move_punishment
        self.border_punishment = border_punishment
        self.reset_world()

    def __generate_new_map(self) -> None:
        self.map = np.zeros((self.size_x, self.size_y))
        self.map[0:, 0] = -1
        self.map[0, :] = -1
        self.map[-1, :] = -1
        self.map[:, -1] = -1

    def __add_cookie_to_map(self, cookie_x: int, cookie_y: int) -> None:
        self.map[cookie_x, cookie_y] = 1

    def __add_agent_to_map(self, x: int, y: int, agent_color: int) -> None:
        self.map[x, y] = agent_color

    def __generate_new_cookie(self, cookie: Optional[Cookie]) -> Cookie:
        cookie_x, cookie_y = (np.random.randint(self.size_x), np.random.randint(self.size_y))
        while self.map[cookie_x, cookie_y] != 0:
            cookie_x, cookie_y = (np.random.randint(self.size_x), np.random.randint(self.size_y))
        if cookie is not None:
            self.map[cookie.x, cookie.y] = 0
        self.__add_cookie_to_map(cookie_x, cookie_y)
        return Cookie(cookie_x, cookie_y)

    def __move_agent(
        self, move_x: int, move_y: int, agent: Agent, cookies: Tuple[Cookie, ...]
    ) -> Tuple[Cookie, ...]:
        move_x = agent.x + move_x
        move_y = agent.y + move_y
        new_cookies = list(cookies)
        if self.map[move_x, move_y] != 1 and self.map[move_x, move_y] != 0:
            self.map[agent.x, agent.y] = 0
            agent.isdead = 1
            agent.score -= self.border_punishment
            return tuple(new_cookies)
        if self.map[move_x, move_y] == 1:
            agent.score += self.cookie_prize
            for cookie in range(len(new_cookies)):
                if move_x == new_cookies[cookie].x and move_y == new_cookies[cookie].y:
                    new_cookies[cookie] = self.__generate_new_cookie(new_cookies[cookie])
                    break
        buf_agent = self.map[agent.x, agent.y]
        self.map[agent.x, agent.y] = 0
        agent.x = move_x
        agent.y = move_y
        self.map[agent.x, agent.y] = buf_agent
        agent.score -= self.move_punishment
        return tuple(new_cookies)

    def reset_world(self) -> Tuple[Cookie, ...]:
        self.__generate_new_map()
        new_cookies = [self.__generate_new_cookie(None) for _ in range(self.num_of_cookies)]
        return tuple(new_cookies)

    def print_world(self) -> None:
        print(self.map)

    def move(self, way: int, agent: Agent, cookies: Tuple[Cookie, ...]) -> Tuple[Cookie, ...]:
        if way == 0:
            return self.__move_agent(0, -1, agent, cookies)
        elif way == 1:
            return self.__move_agent(-1, -1, agent, cookies)
        elif way == 2:
            return self.__move_agent(-1, 0, agent, cookies)
        elif way == 3:
            return self.__move_agent(-1, 1, agent, cookies)
        elif way == 4:
            return self.__move_agent(0, 1, agent, cookies)
        elif way == 5:
            return self.__move_agent(1, 1, agent, cookies)
        elif way == 6:
            return self.__move_agent(1, 0, agent, cookies)
        elif way == 7:
            return self.__move_agent(1, -1, agent, cookies)
        raise ValueError("Way is missing or not found")

    def generate_agent_pos(self, agent: Agent, agent_color: int) -> None:
        x, y = (np.random.randint(self.size_x), np.random.randint(self.size_y))
        while self.map[x, y] != 0:
            x, y = (np.random.randint(self.size_x), np.random.randint(self.size_y))
        if agent.x is not None and agent.y is not None:
            self.map[agent.x, agent.y] = 0
        agent.x = x
        agent.y = y
        self.__add_agent_to_map(x, y, agent_color)


class Evolutionary:
    def __init__(
        self,
        num_layers: int,
        num_inputs: int,
        num_outputs: int,
        num_of_agents: int,
        mutation_chance: float,
        num_of_neurons: int,
        percentage_of_parents: float = 0.2,
    ):
        self.num_of_agents = num_of_agents
        self.num_layers = num_layers
        self.num_inputs, self.num_outputs = (num_inputs, num_outputs)
        self.mutation_chance = mutation_chance
        self.num_of_neurons = num_of_neurons
        self.num_inputs, self.num_outputs = (num_inputs, num_outputs)
        self.reset_agents()
        self.last_best: Agent = self.agents[0]
        self.percentage_of_parents = percentage_of_parents

    def reset_agents(self) -> None:
        self.agents = tuple(
            [
                Agent(
                    num_layers=self.num_layers,
                    num_inputs=self.num_inputs,
                    num_outputs=self.num_outputs,
                    num_of_neurons=self.num_of_neurons,
                )
                for _ in range(self.num_of_agents)
            ]
        )

    def selection(self) -> Tuple[Agent, ...]:
        agents = list(self.agents)
        agents.sort(key=lambda x: x.score, reverse=True)
        s = 0
        props = []
        parents = [agents[0], agents[1]]
        for agent in agents:
            s += agent.score
        for agent in agents:
            props.append(agent.score/s)
        
        while len(parents) < self.num_of_agents * 0.5:
            for idx, prop in enumerate(props):
                if prop > np.random.rand():
                    parents.append(agents[idx])
        return tuple(parents)

    def crossover(self) -> None:
        new_population = []
        a_l = list(self.agents)
        a_l.sort(key=lambda x: x.score, reverse=True)
        self.last_best = a_l[0]
        parents = self.selection()
        for _ in range(len(self.agents)):
            new_genome = [
                np.zeros((self.num_inputs, self.num_of_neurons)),
                *[np.zeros((self.num_of_neurons, self.num_of_neurons)) for _ in range(self.num_layers)],
                np.zeros((self.num_of_neurons, self.num_outputs)),
            ]
            for genome_id in range(len(new_genome)):
                for x in range(new_genome[genome_id].shape[0]):
                    for y in range(new_genome[genome_id].shape[1]):
                        new_genome[genome_id][x, y] = self.mutation(
                            parents[np.random.randint(len(parents))].weights[genome_id][x, y]
                        )
            new_agent = Agent(self.num_layers, self.num_inputs, self.num_outputs, self.num_of_neurons)
            new_agent.weights = new_genome
            new_population.append(new_agent)
        self.agents = tuple(new_population)

    def mutation(self, gen: int) -> int:
        if np.random.rand() > self.mutation_chance:
            return -1 if np.random.rand() > 0.5 else (1 if np.random.rand() > 0.5 else 0)
        return gen


if __name__ == "__main__":
    num_of_cookies = 10
    world = Map(
        size_x=10,
        size_y=10,
        num_of_cookies=num_of_cookies,
        cookie_prize=20,
        move_punishment=1,
        border_punishment=50,
    )
    num_of_neurons = 50
    num_layers = 2
    num_of_agents = 5

    evolution = Evolutionary(
        num_layers=num_layers,
        num_inputs=24,
        num_outputs=8,
        num_of_agents=num_of_agents,
        mutation_chance=0.9,
        num_of_neurons=num_of_neurons,
    )
    for agent_idx in range(len(evolution.agents)):
        world.generate_agent_pos(evolution.agents[agent_idx], evolution.agents[agent_idx].color)
    epochs = 100
    num_moves = 5
    for epoch in range(epochs):
        cookies = world.reset_world()
        for _ in range(num_moves):
            for id, agent in enumerate(evolution.agents):
                if not agent.isdead and not agent.is_completed:
                    agent_decision = agent.feed_forward(
                        [
                            *calculate_agents(agent.x, agent.y, world.map),
                            *calculate_cookies(agent.x, agent.y, world.map),
                            *calculate_walls(agent.x, agent.y, world.map),
                        ]
                    )
                    cookies = world.move(int(np.argmax(agent_decision)), evolution.agents[id], cookies)
                    agent.is_dead()

        evolution.crossover()
        for agent_idx in range(len(evolution.agents)):
            world.generate_agent_pos(evolution.agents[agent_idx], evolution.agents[agent_idx].color)
        print(
            f"EPOCH: {epoch}, BEST AGENT SCORE: {evolution.last_best.score}, NUMBER OF AGENTS: {len(evolution.agents)}"
        )

    cookies = world.reset_world()
    for agent_idx in range(len(evolution.agents)):
        world.generate_agent_pos(evolution.agents[agent_idx], evolution.agents[agent_idx].color)
    for turn in range(num_moves):
        for agent in evolution.agents:
            if not agent.isdead and not agent.is_completed:
                agent_decision = agent.feed_forward(
                    [
                        *calculate_agents(agent.x, agent.y, world.map),
                        *calculate_cookies(agent.x, agent.y, world.map),
                        *calculate_walls(agent.x, agent.y, world.map),
                    ]
                )
                cookies = world.move(int(np.argmax(agent_decision)), agent, cookies)
                agent.is_dead()
        print(f"\nTurn {turn}, best agent score: {evolution.last_best.score}")
        world.print_world()
        print()
