#%%
import random
import csv
import agent
#%%
for x in range (1):
    class happy_mdp():
        def __init__(self):
            self.permitted_states = ['S','S1','G','E']
            self.actions = ["frwd","a1","a2"]
            self.make_transition()
            self.reset() 
            self.model_changed = None
            self.story_m = None
            self.chosen_state = None
            self.chosen_action = None
            
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

            self.t = {}
            for s in self.permitted_states:
                self.t[s] = {}
                for a in self.actions:
                    self.t[s][a] = {}
                    for s2 in self.permitted_states:
                        self.t[s][a][s2] = 0

            self.t["S"]["frwd"]["S1"] = 1
            self.t["S1"]["a1"]["G"] = 1
            self.t["S1"]["a2"]["E"] = 1

            self.t["G"]["frwd"]["G"] = 1
            self.t["E"]["frwd"]["E"] = 1
    
            for s in self.permitted_states:
                for a in list(self.t[s].keys()):
                    if sum(self.t[s][a].values()) == 0:  
                        del self.t[s][a]

            if model_changed:
                self.model_changed = True
            if story_mode:
                self.story_m = True
                self.chosen_state = "S1"
                self.chosen_action = "a1"

        def calculate_reward(self, state = None):
            if state == None:
                state = self.state
            if state == 'G':
                self.reward = 7
                if self.model_changed or self.story_m:
                    self.reward = 10
            elif state == "E":
                self.reward = -10
            else:
                self.reward = -3
                if state == "S1" and (self.model_changed or self.story_m):
                # if (self.model_changed or self.story_m):
                    self.reward = 0
            return self.reward

        def transition(self):
            state_action_p = self.t[self.state][self.action]
            self.previous_state = self.state 

            if self.state in ["E", "G"]: 
                self.state = random.choices(list(state_action_p.keys()), weights=state_action_p.values(), k=1)[0]
                self.terminal = True
                return self.state, self.terminal
        
            self.state = random.choices(list(state_action_p.keys()), weights=state_action_p.values(), k=1)[0]
            return self.state, self.terminal


    #%%
    print("Here is the happiness scenario")

    a = agent.agent(happy_mdp())
    a.train(i_max=20000,i_change = 5)
    a.simulate_episode(terminate = "S1")

    with open('data/model_result.csv','a',newline='') as new_file:
        writer_object = csv.writer(new_file)
        writer_object.writerow(['Happiness',a.sud_app,a.goal_app,a.cdc_app,
            a.power_app])
        new_file.close() 

# %%
