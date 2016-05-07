import random
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator
from QLearningAgent import QLearningAgent



class LearningAgent(Agent):
    """An agent that learns to drive in the smartcab world."""

    def __init__(self, env):
        
        super(LearningAgent, self).__init__(env)
        self.color = 'magenta'
        self.planner = RoutePlanner(self.env, self)  # simple route planner to get next_waypoint
        self.prev_action = None
        self.prev_reward = 0


    def reset(self, destination=None):
        self.planner.route_to(destination)
        # TODO: Prepare for a new trip; reset any variables here, if required
        self.state = None
        self.prev_action = None
        self.prev_reward = 0



    def update(self, t):
        """
        updates the agent action.
        """
        # Gather inputs
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        inputs = self.env.sense(self)
        deadline = self.env.get_deadline(self)

        # TODO: Update state
        #two states : random and init
        if(self.state == None):
            self.state = 'Random'
        #print 'environment state:'
        # {'light': 'green', 'oncoming': None, 'right': None, 'left': None}
        current_environment_state = self.env.sense(self)
        action = None

        probable_actions = []
        if(current_environment_state['light'] == 'red'):
            if(current_environment_state['oncoming'] != 'left'):
                probable_actions = ['right', None]
        else:
            # traffic light is green and now check for oncoming
            #if no oncoming
            if(current_environment_state['oncoming'] == 'forward'):
                probable_actions = [ 'forward','right']
            else:
                probable_actions = ['right','forward', 'left']

        # TODO: Select action according to your policy
        if probable_actions != [] and self.state == 'Random':
            action_int_value =  random.randint(0,len(probable_actions)-1)
            action = probable_actions[action_int_value]
        elif probable_actions != [] and self.state == 'Init':
            action = self.prev_action



        # Execute action and get reward
        reward = self.env.act(self, action)

        # TODO: Learn policy based on state, action, reward

        if(action != None):
            if(reward > self.prev_reward):
                self.state = 'Init'
                self.prev_action = action
                self.prev_reward = reward
            else:
                self.state = 'Random'
                self.prev_action = action
                self.prev_reward = reward




        #print "LearningAgent.update(): deadline = {}, inputs = {}, action = {}, reward = {}".format(deadline, inputs, action, reward)  # [debug]

def run():
    """Run the agent for a finite number of trials."""

    e = Environment()  # create environment (also adds some dummy traffic)
    a = e.create_agent(QLearningAgent)  # create agent
    e.set_primary_agent(a, enforce_deadline=True)  # set agent to track
        # Now simulate it
    sim = Simulator(e, update_delay=0.0000001)  # reduce update_delay to speed up simulation
    sim.run(n_trials=100)  # press Esc or close pygame window to quit
    print "No. of fails = {}".format(e.count_fail)



if __name__ == '__main__':
    run()
