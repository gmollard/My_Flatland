import numpy as np
from matplotlib.patches import Rectangle, Polygon, Circle



# Map of agent actions indices to vectors
ACTIONS = {
            0: np.array([-1, 0]),  # North
            1: np.array([0, 1]),  # East
            2: np.array([1, 0]),  # South
            3: np.array([0, -1]),  # West
            4: np.array([0, 0])  # wait
}


class Environment:
    
    def __init__(self, n, m, N_agents, goal_pos=None):
        assert N_agents < n*m, f'Not enough space available on the grid to place all agents:' +\
            f' {N_agents} agents for {n*m - 1} cells'
        
        self.n = n
        self.m = m
        self.cells = np.empty((n, m), dtype=Cell)
        
        # We instantiate every cells.
        for i in range(n):
            for j in range(m):
                self.cells[i, j] = Cell((i, j), [1])
                
        # We create empty array of Agents.
        self.agents = np.empty((N_agents), dtype=Agent)
        
        # For the moment, we instantiate only one Goal.
        if goal_pos is None:
            self.goals = [Goal((np.random.randint(n), np.random.randint(m)))]
        else:
            self.goals = [Goal(goal_pos)]
        
        # We instantiate every Agent at a random position, checking that this position
        # is not already occupied (by an other agent or by a Goal).
        for i in range(N_agents):
            rand_pos = (np.random.randint(n), np.random.randint(m))
            while rand_pos in [g.pos for g in self.goals] or\
            rand_pos in [a.pos for a in self.agents[np.where(self.agents != None)[0]]]:
                rand_pos = (np.random.randint(n), np.random.randint(m))
            self.agents[i] = Agent(rand_pos, self)
        
    def random_evolution(self):
        for i in range(len(self.agents)):
            agent = self.agents[i]
            if agent.state == 'active':
                self.agent_random_action(i)
            if tuple(agent.pos) in [goal.pos for goal in self.goals]:
                agent.state = 'success'
                
                
    def calculate_reward(self, agent_index, action_index):
        done = False
        reward = -0.1
        
        # Punish invalid movement -> Incentive for agent to walk within maze
        if self.agents[agent_index].invalid_move:
            reward -= 1
            
#         if action_index == 4:
#             reward -= 1
        
        if self.agents[agent_index].pos == self.goals[0].pos and action_index == 4:
            done = True
        
#         if done:
#             reward = 100
        
#         reward -= np.sqrt((agent.pos[0] - self.goals[0].pos[0])**2 + (agent.pos[1] - self.goals[0].pos[1])**2)
        return reward, done
    
    def in_limits(self, pos):
        if pos[0] >= self.n or pos[0] < 0 or pos[1] >= self.m or pos[1] < 0:
            return False
        return True
    
    
    def agent_action(self, action_index, agent_index):
        agent = self.agents[agent_index]
        direction = ACTIONS[action_index]
        if action_index != 4:
            if self.allowed_movement(agent.pos + direction):
                agent.invalid_move = False
                agent.pos += direction
                agent.pos = tuple(agent.pos)
            else:
                agent.invalid_move = True
        
        reward, done = self.calculate_reward(agent_index, action_index)
        if done:
            self.agents[agent_index].state = 'success'
        return self.state(agent_index), reward, done
    
    def agent_random_action(self, agent_index):
        r = np.random.randint(len(ACTIONS))
        self.agent_action(r, agent_index)
    
    def allowed_movement(self, pos):
        active_agents = [a.state == 'active' for a in self.agents]
        active_agents = self.agents[np.where(active_agents)[0]]
        if self.in_limits(pos) and (tuple(pos) not in [a.pos for a in active_agents]):
            return True
        return False
    
    def state(self, agent_index):
        obs = np.zeros((3, self.n, self.m))# + CELL_STATE_TO_VALUE['empty']
#         obs_goal = np.zeros((1, self.n, self.m))# + CELL_STATE_TO_VALUE['empty']
        for i, agent in enumerate(self.agents):
            if i == agent_index:
                obs[0, agent.pos[0], agent.pos[1]] = 1
            elif agent.state == 'active':
                obs[1, agent.pos[0], agent.pos[1]] = 1#CELL_STATE_TO_VALUE['agent']
        for goal in self.goals:
            obs[2, goal.pos[0], goal.pos[1]] = 1#CELL_STATE_TO_VALUE['goal']
            
#         print(np.concatenate(obs_ag, obs_goal))
#         return np.concatenate((obs_ag, obs_goal))
        return obs
#         return np.concatenate((obs_ag.flatten(), obs_goal.flatten()))
    
    def render(self, fig, ax):
        ax.set_xticks(range(self.n + 1))
        ax.set_yticks(range(self.m + 1))
        img = np.zeros((self.m + 1, self.n + 1, 3), dtype=int) + 255
        for agent in self.agents:
            agent.render(ax)
        for goal in self.goals:
            goal.render(ax)
        ax.imshow(img)
        ax.grid()

        
        
class Agent:
    
    def __init__(self, pos, env):
        self.pos = pos
        self.env = env
        self.state = 'active'
        self.invalid_move = False
        
    def render(self, r):
        pos = self.pos
        r.add_patch(Circle((self.pos[0] + 0.5, self.pos[1] + 0.5), radius=0.3, color='orange'))
        
        
class Goal:
    
    def __init__(self, pos):
        self.pos = pos
        
    def render(self, r):
        pos = self.pos
        r.add_patch(Rectangle((pos[0], pos[1]),
                              1, 1, color='green'))
        
        
class Cell:
    
    def __init__(self, pos, cap):
        self.pos = pos
        self.cap = cap

        
