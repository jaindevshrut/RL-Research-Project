import random
from operator import itemgetter
import csv

class agent():
    def __init__(self,mdp):
        self.epsilon = 0.3
        self.gamma = 0.9
        # discount factor not so important
        self.alpha = 0.3
        # make different plots for different alpha 0.3, 0.5
        self.mdp = mdp
        self.q={}
        self.td_error = 0
        self.old_q = 0
        self.t_hat = {}
        self.max_q_table = 0
        #Q table is for every State Action pair.
        
        for s in self.mdp.t.keys():
            self.q[s]={}
            self.t_hat[s]={}
            for a in self.mdp.t[s]:
                self.q[s][a]=0
                self.t_hat[s][a]={}
                for s2 in self.mdp.t.keys():
                    self.t_hat[s][a][s2]=0
    

    def update_q_learning(self):
        if  self.mdp.action != None and self.mdp.action not in self.q[self.mdp.previous_state]:
            self.q[self.mdp.previous_state][self.mdp.action] = 0
            self.t_hat[self.mdp.previous_state][self.mdp.action]={}
            self.t_hat[self.mdp.previous_state][self.mdp.action][self.mdp.state]=0

        if self.mdp.previous_state != None:
            previous_q = self.q[self.mdp.previous_state][self.mdp.action]
            self.old_q = previous_q
            next_q = max(self.q[self.mdp.state].items(), key = itemgetter(1))[1]
            self.td_error = self.alpha * (self.mdp.reward + self.gamma * next_q - previous_q)
            new_q = previous_q + self.td_error
            self.q[self.mdp.previous_state][self.mdp.action] = new_q
            self.t_hat[self.mdp.previous_state][self.mdp.action][self.mdp.state] += 1
            # Here, next_q is calculated in terms of the max_q, so it doesn't know about 
            # the action it is going to take. It is still expecting the best thing to happen
            # Until it actually takes the action.
        

    def get_td_error(self):
        if self.mdp.previous_state != None:
            tde = self.td_error
            # self.mdp.tde_sum[self.mdp.state] += tde
            self.mdp.tde.append(tde)

    def update_q_td(self):
        previous_q = self.q[self.mdp.previous_state][self.mdp.action]

        self.q[self.mdp.previous_state][self.mdp.action] = \
            previous_q + self.alpha * (self.mdp.reward - previous_q)
        
        self.t_hat[self.mdp.previous_state][self.mdp.action][self.mdp.state] += 1

    def get_max_q_table(self):
        max_q_table = max({key: max(val.values()) for key, val in self.q.items()}.values())
        if max_q_table == 0:
            max_q_table = max_q_table + 1
        self.max_q_table = max_q_table
        return max_q_table   

    def choose_action_epsilon_greedy(self): 
        self.mdp.previous_action = self.mdp.action
        # if random.random() < self.epsilon and self.mdp.state == "P":


        if(self.mdp.story_m or self.mdp.model_changed) and self.mdp.state == self.mdp.chosen_state:
            self.mdp.action = self.mdp.chosen_action
        elif random.random() < self.epsilon:
            actions = []
            for key, value in self.mdp.t[self.mdp.state].items():
                actions.append(key)
            self.mdp.action = random.choice(actions)
        else:
            self.mdp.action = max(self.q[self.mdp.state].items(), key = itemgetter(1))[0]

    def do_step(self):
        
        self.update_q_learning()
        self.get_td_error()
        self.choose_action_epsilon_greedy()
        self.mdp.transition()
        self.mdp.calculate_reward()

        if self.mdp.terminal:
            self.update_q_td()

    def train(self,i_max,i_change=0):
        i=0
        while i < i_max:
            if i == i_max-i_change and not self.mdp.model_changed:
                self.mdp.make_transition(story_mode = False, model_changed = True)
                # self.mdp.model_changed = True
            self.do_step()

            if self.mdp.terminal:
                i += 1
                self.mdp.reset()

    def simulate_episode(self, terminate = None):
        # if not self.internal_active and app_acc:
        #     # when calculating instd, the behavior doesn't need to apprised 
        #     # only the sccores are trying to be calculated
        #     # instd needs to be active inside generate_instandard()
        #     # self.instandard=self.generate_instandard()
        #     sm=True

        # if self.mdp.repr_is_state:
        # self.terminate_else = terminate_else
        self.mdp.make_transition(story_mode = True) 
        self.mdp.reset()

        while True:
            self.do_step()
            if self.mdp.terminal:
                return

            if terminate == self.mdp.state:

                self.update_q_learning()
                self.get_td_error()
                # self.choose_action_epsilon_greedy()
                self.get_max_q_table()
                rounded_tde = [round(num,3)for num in self.mdp.tde]
                print("Q value:\t",self.q)
                print("TDE list:\t", rounded_tde)
                print("Manual terminate")
                print("Suddenness:\t", round(self.appraise_suddenness(),4))
                print("Goal relevance:\t", round(self.appraise_goal_relevance(),4))
                print("Conduciveness:\t", round(self.appraise_conduciveness(),4))
                print("Power:\t\t", round(self.appraise_power(),4))
                # print("In standard:\t", round(self.appraise_instandard(),4))
                return

    def appraise_power(self):
        # If two q are very high, the power is very low
        # reward having too much influence
        # state = self.mdp.state
        state = self.mdp.chosen_state
        avg_q = sum(self.q[state].values())/len(self.q[state].values())
        min_q = min(self.q[state].values())
        max_q = max(self.q[state].values())
        if abs(min_q)<max_q:
            self.power_app = abs((max_q-avg_q)/max_q)
        else:
            self.power_app = abs((min_q-avg_q)/min_q)
        return self.power_app

    def appraise_goal_relevance(self):
        self.goal_app = min(1,abs(self.td_error))
        return self.goal_app

    def appraise_suddenness(self):
        # It calculates p(s'|at-1)
        s = sum(self.t_hat[self.mdp.previous_state][self.mdp.previous_action].values())
        if s > 0:
            self.sud_app = 1-self.t_hat[self.mdp.previous_state][self.mdp.previous_action][self.mdp.state]/s
            # suddennes = 1- (frequency)
        else:
            self.sud_app = 0
        return self.sud_app
        
    def appraise_conduciveness(self):
        self.cdc_app = max(-1,min(1, self.td_error))/2+0.5
        return self.cdc_app