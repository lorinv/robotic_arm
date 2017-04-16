import gym
#For Robotic Arm
#from rl_env import *
from env import Arm_Env
from TDLambdaAgent import *

def main():

    env = Arm_Env()

    perfLog = open("data/performance.txt","a+")
    perfLog.write("\n") #start convergence values on a fresh line
    
    #decent default values, just based on observation
    alpha =0.2
    gamma = 0.9
    randomActionRate = 0.2
    randomActionDecayRate = 0.9
    tdLambda = 0.5
    algorithm = "sarsa" # "sarsa" or "watkins" for q-learning
    traceMethod = "normal" # "normal" for normal (per Barto), or "replacing" for replacing traces (see Barto)
    resetQVals = False
    resetCache = True
    
    #get any cmd line params
    for arg in sys.argv:
        if "alpha=" in arg:
            alpha = float(arg.split("=")[1])
        elif "gamma=" in arg:
            gamma = float(arg.split("=")[1])
        elif "randomRate=" in arg:
            randomActionRate = float(arg.split("=")[1])
        elif "randomDecay=" in arg:
            randomActionDecayRate = float(arg.split("=")[1])
        elif "lambda=" in arg:
            tdLambda = float(arg.split("=")[1])
        elif "algorithm=" in arg:
            algorithm = arg.split("=")[1]
        elif "traceMethod=" in arg:
            traceMethod = arg.split("=")[1]
        elif "maxEpisodes=" in arg:
            maxEpisodes = int(arg.split("=")[1])
        elif "--resetq" in arg:
            resetQVals = True
        elif "--resetCache" in arg:
            resetCache = True

    """
    
    """
    learner = TDLambdaAgent(env.num_states, env.num_actions, alpha, gamma,
                                                randomActionRate, randomActionDecayRate, tdLambda,
                                                200, traceMethod, algorithm, True, "qValues.txt", resetQVals, resetCache)

    done = False
    convergence = False
    while not convergence:
        done = False
        #reset the learner's eligibility traces for this episode
        learner.ResetEligibilities()
        observation, reward, done, info = env.reset()
        action = learner.InitState(observation)
        while not done:
            observation, reward, done, info = env.step(action)
            action = learner.move(observation, reward)
            print ">>Action: "+str(action)
            #raw_input("")
        
        #re-run training offline, eg Dyna-Q
        #learner.Dream()
        
    perfLog.close()
    #env.monitor.close()

if __name__ == "__main__":
    random.seed(0)
    main()