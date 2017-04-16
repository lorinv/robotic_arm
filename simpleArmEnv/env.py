from state import ArmState
from actions import ArmActions

class Arm_Env:

    def __init__(self):
        self.state_manager = ArmState()
        self.action_manager = ArmActions()
        self.current_state = 0
        self.total_reward = 0
        self.num_states = self.state_manager.get_num_states()
        self.num_actions = self.action_manager.get_num_actions()

    def reset(self):
        self.action_manager.reset()
        return self.step(0)

    def get_reward(self):
        if self.current_state == 12:         
            return 1
        else:
            return -1

    def print_state(self, state, action, reward, done):
        print "------------------------------"
        print "State: %d" % state
        print "Action: %d" % action
        print "Reward: %d" % reward
        print "Total Reward: %d" % self.total_reward
        print "Done: %s" % str(done)
        print "-------------------------------"

    def step(self, action_id):      
        self.action_manager.take_action(action_id, self.current_state)
        self.current_state = self.state_manager.get_state()        
        reward = self.get_reward()        
        self.total_reward += reward
        done = self.total_reward >= 10
        if done:
            self.total_reward = 0
            self.reset()
        self.print_state(self.current_state, action_id, reward, done)
        return self.current_state, reward, done, None

    def render(self):
        print "Look at the robot..."    
        return 1

