import random
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator
from collections import namedtuple
import pprint

class QLearningAgent(Agent):
    """An agent that learns to drive in the smartcab world using Q learning"""

    def __init__(self, env):
        """
        this function is used to initialize the agent variables
        that will be used for the process.
        Even Q learning parameters like alpha, beta (epsilon) and gamma are initialized
        these are used in updating the Q Table.
        """
        super(QLearningAgent, self).__init__(env)                               # sets self.env = env, state = None, next_waypoint = None, and a default color
        self.color = 'black'                                                    # color
        self.planner = RoutePlanner(self.env, self)                             # route planner to get next_waypoint
        self.q_table = dict()                                                    #initialize q table here
        self.alpha    = 0.70                                                    #initializes learning rate
        self.beta = 0.0                                                         #initialize beta
        self.gamma    = 0.50
        self.discount = self.gamma                                              #initialize gamma
        self.prev_state = None
        self.state = None
        self.prev_action = None
        self.deadline = self.env.get_deadline(self)
        self.prev_reward = None
        self.sum_rewards = 0


    def rnd_value_gen(self, param ):
        """
        generates a random value that depends on the paramter passed to it
        """
        rnd = random.random()
        return rnd < param

    def reset(self, destination=None):
        """
        resets the agent variables to restart the process
        destinationis given as input
        """
        self.planner.route_to(destination)
        self.prev_state = None
        self.state = None
        self.prev_action = None
        self.beta = 0.0
        self.sum_rewards = 0

    def proper_actions(self, state):
        """
        returns the allowed proper actions for the agent
        takes current states as input
        """
        return ['forward', 'left', 'right', None]


    def q_value(self, state, action):
        """
        current state and action are inputs and the q value is given after refering the dictionary
        if value not present in the dictionary, it returns 15
        """
        return self.q_table.get((state, action), 15.0)

    def value(self, state):
        """

        takes state as input and gives the best q value based on all the legal actions
        that can be taken.

        """
        proper_actions = self.proper_actions(state)
        best_q = - 9999

        for action in proper_actions:
            #for each action check if the q value for the action is greater than minus infinity
            if(self.q_value(state, action) > best_q):
                best_q = self.q_value(state, action)

        return best_q

    def policy(self, state):
        """
        determines the action to be taken based on the state
        takes any one action in the event of a tie.
        """
        proper_actions = self.proper_actions(state)
        best_action = None
        best_q = - 9999
        for action in proper_actions:
            if(self.q_value(state, action) > best_q):
                best_q = self.q_value(state, action)
                best_action = action
            if(self.q_value(state, action) == best_q):
                if(self.rnd_value_gen(.5)):
                    best_q = self.q_value(state, action)
                    best_action = action
        return best_action

    def make_state(self, state):
        """

        useful to create the Q dictionary.
        Creates a namedtuple with the states.

        """

        State = namedtuple("State", ["light","next_waypoint","oncoming"])
        return State(light = state['light'],
                        next_waypoint = self.planner.next_waypoint(),
                        oncoming = state['oncoming'])


    def update(self, t):
        """

        updates the state of the agent and sets the reward based on action taken.

        """

        self.next_waypoint = self.planner.next_waypoint()
        self.state = self.make_state(self.env.sense(self))                      #current state
        action = self.action(self.state)                                        #current best action
        reward = self.env.act(self, action)                                     #reward for action determined
        if self.prev_reward!= None:
            #update q table based on change in previous reward
            self.change_qtable(self.prev_state,self.prev_action,self.state,self.prev_reward)

        #update states, action and rewards
        self.prev_action = action
        self.prev_state = self.state
        self.prev_reward = reward
        self.sum_rewards += reward
        #pprint.pprint(self.q_table,width= 1)
    def action(self, state):
        """

        determine action based on all probable actions
        choose random choice beta times and rest of the time choose
        policy choice

        """

        proper_actions = self.proper_actions(state)
        action = None
        if (self.rnd_value_gen(self.beta)):
            action = random.choice(proper_actions)
        else:
            action = self.policy(state)
        return action

    def change_qtable(self, state, action, nxt_state, reward):
        """

        Q-table update done by calling this function.

        """

        if((state, action) in self.q_table):
            #previous state += alpha*(reward + gamma*value of next state - old q value)
            self.q_table[(state, action)] += self.alpha*(reward + self.discount*self.value(nxt_state) - self.q_table[(state, action)])
        else:

            self.q_table[(state, action)] = 15.0
