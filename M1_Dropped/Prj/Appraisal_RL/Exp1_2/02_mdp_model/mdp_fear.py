#%%
import random
import csv
import agent
#%%
for x in range (1):
    class fear_mdp():
        def __init__(self):
            self.permitted_states = ['S','S1','P','E','G']
            self.actions = ["frwd"]
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
            self.t = {}
            for s in self.permitted_states:
                self.t[s] = {}
                for a in self.actions:
                    self.t[s][a] = {}
                    for s2 in self.permitted_states:
                        self.t[s][a][s2] = 0

            self.t["S"]["frwd"]["S1"] = 1
            self.t["S1"]["frwd"]["P"] = 0.2
            self.t["S1"]["frwd"]["G"] = 0.8

            self.t["P"]["frwd"]["E"] = 1
            # self.t["P"]["a2"]["E"] = 1   

            self.t["E"]["frwd"]["E"]=1
            self.t["G"]["frwd"]["G"]=1

            for s in self.permitted_states:
                for a in list(self.t[s].keys()):
                    if sum(self.t[s][a].values()) == 0:       
                        del self.t[s][a]  
            
            if story_mode:
                self.story_m = True
                self.t["S1"]["frwd"]["P"] = 1
                self.t["S1"]["frwd"]["G"] = 0
                self.chosen_state = "P"
                self.chosen_action = "frwd"
            

        def calculate_reward(self, state = None):
            if state == None:
                state = self.state
            if state == 'E':
                self.reward = -10
            elif state == 'G':
                self.reward = 10
            else:
                self.reward = -1
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
    # Agent is always the same, different scenarios would be presented with different mdps.

    print("Here is the fear scenario")

    a = agent.agent(fear_mdp())
    a.train(i_max=20000)
    a.simulate_episode(terminate = "P")

    with open('data/model_result.csv','a',newline='') as new_file:
        writer_object = csv.writer(new_file)
        writer_object.writerow(['Fear',a.sud_app,a.goal_app,a.cdc_app,
        a.power_app])
        new_file.close()   


# %%
