#%%
import random
import csv
import agent
#%%

class bored_mdp():
    def __init__(self):
        self.permitted_states = ['S','P','E','G']
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

        self.t["S"]["frwd"]["P"] = 1

        self.t["P"]["a1"]["E"]=1
        self.t["P"]["a2"]["G"]=1
        
        self.t["E"]["frwd"]["E"]=1
        self.t["G"]["frwd"]["G"]=1

        # Add "circular actions" where the state does not change
        # The action it takes will lead it back to itself
        for s in self.permitted_states:
            for a in list(self.t[s].keys()):
                if sum(self.t[s][a].values()) == 0:  
                    del self.t[s][a]          

        # story mode
        if story_mode:
            self.story_m = True
            self.chosen_state = "P"
            self.chosen_action = "a1"


    def calculate_reward(self, state = None):
        if state == None:
            state = self.state
        if state == 'E':
            self.reward = -1
        elif state == 'G':
            self.reward = 5
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

print("Here is the boredom scenario")

a = agent.agent(bored_mdp())
a.train(i_max=20000)
a.simulate_episode(terminate = "P")

# terminate_else means another possibility for the terminate. 

with open('data/model_result.csv','a',newline='') as new_file:
    writer_object = csv.writer(new_file)
    writer_object.writerow\
        (['Boredom',a.sud_app,a.goal_app,a.cdc_app,a.power_app,])
    new_file.close()  


# %%
