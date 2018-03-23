import random
import math
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator

class LearningAgent(Agent):
    """ An agent that learns to drive in the Smartcab world.
        This is the object you will be modifying. """ 

    def __init__(self, env, learning=False, epsilon=1.0, alpha=0.5):
        super(LearningAgent, self).__init__(env)     # Set the agent in the evironment 
        self.planner = RoutePlanner(self.env, self)  # Create a route planner
        self.valid_actions = self.env.valid_actions  # The set of valid actions

        # Set parameters of the learning agent
        self.learning = learning # Whether the agent is expected to learn
        self.Q = dict()          # Create a Q-table which will be a dictionary of tuples
        self.epsilon = epsilon   # Random exploration factor
        self.alpha = alpha       # Learning factor

        ###########
        ## TO DO ##
        ###########
        # Set any additional class parameters as needed
        # for debugging/tuning
        self.trial_stats = []
        self.qtable_stats = dict()

        # best action, create@trial, none_random_count, none_maxq_count, ..., best policy, all actions visited
        for k1 in ['red', 'green']:
            for k2 in ['forward', 'left', 'right']:
                for k3 in [None, 'forward', 'left', 'right']:
                    for k4 in [None, 'forward', 'left', 'right']:
                            if k1 == 'red':
                                if k2 == 'forward' or k2 == 'left':
                                    self.qtable_stats[(k1, k2, k3, k4)] = [None, 0, 0,0,0,0,0,0,0,0, False, False]
                                elif k4 == 'forward':
                                    self.qtable_stats[(k1, k2, k3, k4)] = [None, 0, 0,0,0,0,0,0,0,0, False, False]
                                else:
                                    self.qtable_stats[(k1, k2, k3, k4)] = [k2, 0, 0,0,0,0,0,0,0,0, False, False]
                            else:
                                if k2 == 'forward' or k2 == 'right':
                                    self.qtable_stats[(k1, k2, k3, k4)] = [k2, 0, 0,0,0,0,0,0,0,0, False, False]
                                elif k3 == 'forward' or k3 == 'right':
                                    self.qtable_stats[(k1, k2, k3, k4)] = [None, 0, 0,0,0,0,0,0,0,0, False, False]
                                else:
                                    self.qtable_stats[(k1, k2, k3, k4)] = [k2, 0, 0,0,0,0,0,0,0,0, False, False]


    def update_stats_newstate(self, state):
        if self.qtable_stats[state][1] == 0:
            # to be updated with the trial number
            self.qtable_stats[state][1] = -1

    def update_stats_newaction(self, state, action, is_maxq):
        action_index = 2 + 2 * self.env.valid_actions.index(action) + (1 if is_maxq else 0)
        if self.env.verbose:
            print 'update_stats_newaction: %s @ %d' % (action, action_index)
        self.qtable_stats[state][action_index] += 1

    def is_best_policy(self, q_values, best_action):
        maxq = None
        maxq_actions = []

        for a_q in q_values.itervalues():
            if a_q > maxq:
                maxq = a_q

        for a, a_q in q_values.items():
            if a_q == maxq:
                maxq_actions.append(a)

        return (len(maxq_actions) == 1) and (best_action in maxq_actions)

    def all_actions_visited(self, stat):
        return (stat[2] + stat[3] > 0) and (stat[4] + stat[5] > 0) and (stat[6] + stat[7] > 0) and (stat[8] + stat[9] > 0)

    def update_learnings(self, trial):
        new_states = 0
        total_states = 0
        total_states_all_actions = 0
        total_maxq_best = 0;

        for k1 in ['red', 'green']:
            for k2 in ['forward', 'left', 'right']:
                for k3 in [None, 'forward', 'left', 'right']:
                    for k4 in [None, 'forward', 'left', 'right']:
                            state = (k1, k2, k3, k4)
                            if self.qtable_stats[state][1] == -1:
                                self.qtable_stats[state][1] = trial
                                new_states += 1

                            if self.qtable_stats[state][1] > 0:
                                total_states += 1
                                self.qtable_stats[state][10] = self.is_best_policy(self.Q[state], \
                                    self.qtable_stats[state][0]
                                )

                                if self.qtable_stats[state][10] == True:
                                    total_maxq_best += 1

                                if self.qtable_stats[state][11] == False:
                                    self.qtable_stats[state][11] = self.all_actions_visited(self.qtable_stats[state])

                                if self.qtable_stats[state][11] == True:
                                    total_states_all_actions += 1

        assert len(self.Q) == total_states, 'inconsistent states'

        # add to
        self.trial_stats.append([trial, new_states, total_states, total_states_all_actions, total_maxq_best])

        if self.env.verbose:
            self.print_stats(print_allstates=False, print_qvalues=True)

    def print_stats(self, print_allstates=True, print_qvalues=False):
        print "[best action, create@trial, none_random_count, none_maxq_count, ..., best policy, all actions visited]"
        print "========== ========== ========== ========== =========="
        for k1 in ['red', 'green']:
            for k2 in ['forward', 'left', 'right']:
                for k3 in [None, 'forward', 'left', 'right']:
                    for k4 in [None, 'forward', 'left', 'right']:
                            state = (k1, k2, k3, k4)
                            stat = self.qtable_stats[state]
                            if stat[1] == 0:
                                if print_allstates: print '  (%s, %s - %s, %s): ' % (k1, k2, k3, k4), stat[0]
                            else:
                                print ' %s(%s, %s - %s, %s): %s  (%d, %d | %d, %d | %d, %d | %d, %d)%s' % \
                                      ('*' if stat[10] else ' ', k1, k2, k3, k4, stat[0],
                                       stat[2],stat[3],stat[4],stat[5],stat[6],stat[7],stat[8],stat[9],
                                       '*' if stat[11] else '')
                                if print_qvalues: print "    ", self.Q.get(state)
                    print "~~~~~~~~~~ ~~~~~~~~~~ ~~~~~~~~~~ ~~~~~~~~~~"

        print "total states: 96"
        print "[trial, new_states, total_states, total_states_all_actions, total_maxq_best]"
        for i in self.trial_stats: print i


    def reset(self, destination=None, testing=False):
        """ The reset function is called at the beginning of each trial.
            'testing' is set to True if testing trials are being used
            once training trials have completed. """

        # Select the destination as the new location to route to
        self.planner.route_to(destination)
        
        ########### 
        ## TO DO ##
        ###########
        # Update epsilon using a decay function of your choice
        # Update additional class parameters as needed
        # If 'testing' is True, set epsilon and alpha to 0
        if testing:
            self.epsilon = 0
            self.alpha = 0
        else:
            self.epsilon -= 0.002

        return None

    def build_state(self):
        """ The build_state function is called when the agent requests data from the 
            environment. The next waypoint, the intersection inputs, and the deadline 
            are all features available to the agent. """

        # Collect data about the environment
        waypoint = self.planner.next_waypoint() # The next waypoint 
        inputs = self.env.sense(self)           # Visual input - intersection light and traffic
        deadline = self.env.get_deadline(self)  # Remaining deadline

        ########### 
        ## TO DO ##
        ###########
        
        # NOTE : you are not allowed to engineer features outside of the inputs available.
        # Because the aim of this project is to teach Reinforcement Learning, we have placed 
        # constraints in order for you to learn how to adjust epsilon and alpha, and thus learn about the balance between exploration and exploitation.
        # With the hand-engineered features, this learning process gets entirely negated.
        
        # Set 'state' as a tuple of relevant data for the agent        
        #state = None
        # waypoint: the direction the Smartcab should drive leading to the destination, relative to the Smartcab's heading.
        # light: the color of the light.
        # left: the intended direction of travel for a vehicle to the Smartcab's left. Returns None if no vehicle is present.
        # right: the intended direction of travel for a vehicle to the Smartcab's right. Returns None if no vehicle is present.
        # oncoming: the intended direction of travel for a vehicle across the intersection from the Smartcab. Returns None if no vehicle is present.
        # try this first: 2*4*4*4 = 96
        # will try adding waypoint later
        state = (inputs['light'], waypoint, inputs['oncoming'], inputs['left'])

        return state


    def get_maxQ(self, state):
        """ The get_maxQ function is called when the agent is asked to find the
            maximum Q-value of all actions based on the 'state' the smartcab is in. """

        ########### 
        ## TO DO ##
        ###########
        # Calculate the maximum Q-value of all actions for a given state

        maxQ = None
        for a_q in self.Q[state].itervalues():
            if a_q > maxQ:
                maxQ = a_q

        return maxQ 

    def createQ(self, state):
        """ The createQ function is called when a state is generated by the agent. """

        ########### 
        ## TO DO ##
        ###########
        # When learning, check if the 'state' is not in the Q-table
        # If it is not, create a new dictionary for that state
        #   Then, for each action available, set the initial Q-value to 0.0
        if not state in self.Q:
            self.Q[state] = {None:0.0, 'forward':0.0, 'left':0.0, 'right':0.0}
            self.update_stats_newstate(state)

        return

    def choose_action(self, state):
        """ The choose_action function is called when the agent is asked to choose
            which action to take, based on the 'state' the smartcab is in. """

        # Set the agent state and default action
        self.state = state
        self.next_waypoint = self.planner.next_waypoint()
        action = None

        ########### 
        ## TO DO ##
        ###########
        # When not learning, choose a random action
        # When learning, choose a random action with 'epsilon' probability
        # Otherwise, choose an action with the highest Q-value for the current state
        # Be sure that when choosing an action with highest Q-value that you randomly select between actions that "tie".
        random.seed()

        if self.learning:
            if self.env.verbose == True:
                print "q table: ", self.Q[state]

            if random.random() < self.epsilon:
                # choose a random action
                action = random.choice(self.valid_actions)
                self.update_stats_newaction(state, action, False)
                if self.env.verbose == True:
                    print "random action: ", action
            else:
                # choose an action with the highest Q-value
                maxq = None
                maxq_actions = []

                for a_q in self.Q[state].itervalues():
                    if a_q > maxq:
                        maxq = a_q

                for a, a_q in self.Q[state].items():
                    if a_q == maxq:
                        maxq_actions.append(a)

                l = len(maxq_actions)
                if l == 1:
                    action = maxq_actions[0]
                    if self.env.verbose == True:
                        print "maxq action: ", action
                else:
                    # randomly select among ties
                    action = maxq_actions[random.randint(0, l-1)]
                    if self.env.verbose == True:
                        print "maxq action (random): ", action

                self.update_stats_newaction(state, action, True)
        else:
            action = random.choice(self.valid_actions)
            self.update_stats_newaction(state, action, False)

        return action
        # before any code is added
        #return None
        # test best policy returns A for safety and A+ for reliability
        #return self.qtable_stats[state][0];

    def learn(self, state, action, reward):
        """ The learn function is called after the agent completes an action and
            receives a reward. This function does not consider future rewards 
            when conducting learning. """

        ########### 
        ## TO DO ##
        ###########
        # When learning, implement the value iteration update rule
        #   Use only the learning rate 'alpha' (do not use the discount factor 'gamma')
        if self.env.verbose == True:
            print "LearningAgent.learn():"
            print "-state: ", state
            print "-action:", action
            print "-reward: ", reward

        q_hat = self.Q[state].get(action)

        self.Q[state][action] = ( 1 - self.alpha) * q_hat + self.alpha * reward
        return


    def update(self):
        """ The update function is called when a time step is completed in the 
            environment for a given trial. This function will build the agent
            state, choose an action, receive a reward, and learn if enabled. """

        state = self.build_state()          # Get current state
        self.createQ(state)                 # Create 'state' in Q-table
        action = self.choose_action(state)  # Choose an action
        reward = self.env.act(self, action) # Receive a reward
        self.learn(state, action, reward)   # Q-learn

        return
        

def run():
    """ Driving function for running the simulation. 
        Press ESC to close the simulation, or [SPACE] to pause the simulation. """

    ##############
    # Create the environment
    # Flags:
    #   verbose     - set to True to display additional output from the simulation
    #   num_dummies - discrete number of dummy agents in the environment, default is 100
    #   grid_size   - discrete number of intersections (columns, rows), default is (8, 6)
    env = Environment(verbose=False)


    ##############
    # Create the driving agent
    # Flags:
    #   learning   - set to True to force the driving agent to use Q-learning
    #    * epsilon - continuous value for the exploration factor, default is 1
    #    * alpha   - continuous value for the learning rate, default is 0.5
    agent = env.create_agent(LearningAgent, learning=True, alpha=0.5)


    ##############
    # Follow the driving agent
    # Flags:
    #   enforce_deadline - set to True to enforce a deadline metric
    env.set_primary_agent(agent, enforce_deadline=True)

    ##############
    # Create the simulation
    # Flags:
    #   update_delay - continuous time (in seconds) between actions, default is 2.0 seconds
    #   display      - set to False to disable the GUI if PyGame is enabled
    #   log_metrics  - set to True to log trial and simulation results to /logs
    #   optimized    - set to True to change the default log file name
    sim = Simulator(env, update_delay=0.01, log_metrics=True, optimized=True)
    
    ##############
    # Run the simulator
    # Flags:
    #   tolerance  - epsilon tolerance before beginning testing, default is 0.05 
    #   n_test     - discrete number of testing trials to perform, default is 0
    sim.run(n_test=10)

    agent.print_stats()


if __name__ == '__main__':
    run()
