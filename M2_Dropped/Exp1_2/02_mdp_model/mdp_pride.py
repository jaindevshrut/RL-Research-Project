#%%
import random
import csv
# import numpy as np
import agent
#%%


class pride_mdp():
    def __init__(self):
        self.permitted_states = ['S','S1','S2','G','G_plus']
        self.actions = ["frwd","a1","a2"]
        self.make_transition()
        self.model_changed = False
        self.story_m = False
        self.chosen_state = None
        self.chosen_action = None
        self.reset() 

    def reset(self):
        self.state = 'S'
        self.action = None

        self.reward = 0
        self.terminal = False
        self.tde = []

        self.previous_state = None
        self.previous_action = None


    def make_transition(self, story_mode = False, model_changed = False):
        # trainning mode
        # Populate transition matrix with 0s.
        self.t = {}
        for s in self.permitted_states:
            self.t[s] = {}
            for a in self.actions:
                self.t[s][a] = {}
                for s2 in self.permitted_states:
                    self.t[s][a][s2] = 0

        self.t["S"]["frwd"]["S1"] = 1
        self.t["S1"]["a1"]["G"] = 1
        self.t["S1"]["a2"]["S2"] = 1

        self.t["S2"]["frwd"]["G_plus"] = 0.5
        self.t["S2"]["frwd"]["G"] = 0.5

        self.t["G"]["frwd"]["G"] = 1
        self.t["G_plus"]["frwd"]["G_plus"] = 1

        for s in self.permitted_states:
            for a in list(self.t[s].keys()):
                if sum(self.t[s][a].values()) == 0:       
                    del self.t[s][a]  
        # This is for preventing the state with an action that doesn't exist for that state.

        if story_mode:
            self.story_m = True
            self.chosen_state = "S1"
            self.chosen_action = "a2"
            self.t["S2"]["frwd"]["G_plus"] = 1
            self.t["S2"]["frwd"]["G"] = 0

        # if story mode, the agent should use the path S1->a2->G_plus,
        # before that, this path shouldn't exist, or only exist very limited times. 


    def calculate_reward(self, state = None):
        if state == None:
            state = self.state
        if state == 'G':
            self.reward = 5
        elif state == "S2":
            self.reward = -5
        elif state == "G_plus":
            self.reward = 10
        else:
            self.reward = -1
        return self.reward


    def transition(self):
        state_action_p = self.t[self.state][self.action]
        self.previous_state = self.state 

        if self.state in ["G_plus", "G"]: 
            self.state = random.choices(list(state_action_p.keys()), weights=state_action_p.values(), k=1)[0]
            self.terminal = True
            return self.state, self.terminal
    
        self.state = random.choices(list(state_action_p.keys()), weights=state_action_p.values(), k=1)[0]
        return self.state, self.terminal



    #%%
    # Agent is always the same, different scenarios would be presented with different mdps.

print("Here is the Pride scenario")
a = agent.agent(pride_mdp())
a.train(i_max=20000)
a.simulate_episode(terminate = "G_plus")


with open('data/model_result.csv','a',newline='') as new_file:
    writer_object = csv.writer(new_file)
    writer_object.writerow(['Pride',a.sud_app,a.goal_app,a.cdc_app,
        a.power_app])
    new_file.close() 
