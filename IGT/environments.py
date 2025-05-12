import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns


class IGTEnv():
    def __init__(self,
                 mean_reward: np.ndarray, 
                 std_reward: np.ndarray, 
                 mean_loss: np.ndarray, 
                 std_loss: np.ndarray):
        self.num_arms = 4

        self.mean_reward = mean_reward
        self.std_reward = std_reward

        self.mean_loss = mean_loss
        self.std_loss = std_loss
        self.loss_tstamps = {0: np.array([0,1,0,1,0,1,0,1,0,1]),
                             1: np.array([0,0,0,0,1,0,0,0,0,0]),
                             2: np.array([0,1,0,1,0,1,0,1,0,1]),
                             3: np.array([0,0,0,0,1,0,0,0,0,0])}
        # shuffle each of the key in loss_tstamps
        for key in self.loss_tstamps:
            np.random.shuffle(self.loss_tstamps[key])

        # pick counter for each arm to give loss
        self.counts = np.zeros((self.num_arms))

        # asserts if the shapes donot match
        assert self.num_arms == self.mean_reward.shape[0] == self.std_reward.shape[0] == self.mean_loss.shape[0]== self.std_loss.shape[0], 'Invalid shape of mean_reward or std array'
        self.arms = dict(enumerate(zip(self.mean_reward, self.std_reward, self.mean_loss, self.std_loss)))

    def step(self, chosen_arm):
        arm_mean_rew, arm_dev_rew, arm_mean_loss, arm_dev_loss = self.arms[chosen_arm]

        # reward associated with the arm
        rew = np.random.normal(arm_mean_rew, arm_dev_rew)
        # loss associated with the arm
        loss = np.random.normal(arm_mean_loss, arm_dev_loss)
        loss = loss * self.loss_tstamps[chosen_arm][int(self.counts[chosen_arm])]
        rew = rew + loss

        self.counts[chosen_arm] +=1

        if self.counts[chosen_arm] == 10:
            # Once the arm is picked 10 times reset the counts and shuffle the array to change the loss times of the arm
            self.counts[chosen_arm] = 0
            np.random.shuffle(self.loss_tstamps[chosen_arm])
        return rew
    
    def reset(self):
        self.loss_tstamps = {0: np.array([0,1,0,1,0,1,0,1,0,1]),
                             1: np.array([0,0,0,0,1,0,0,0,0,0]),
                             2: np.array([0,1,0,1,0,1,0,1,0,1]),
                             3: np.array([0,0,0,0,1,0,0,0,0,0])}
        # shuffle each of the key in loss_tstamps
        for key in self.loss_tstamps:
            np.random.shuffle(self.loss_tstamps[key])

        # pick counter for each arm to give loss
        self.counts = np.zeros((self.num_arms))

        # asserts if the shapes donot match
        assert self.num_arms == self.mean_reward.shape[0] == self.std_reward.shape[0] == self.mean_loss.shape[0]== self.std_loss.shape[0], 'Invalid shape of mean_reward or std array'
        self.arms = dict(enumerate(zip(self.mean_reward, self.std_reward, self.mean_loss, self.std_loss)))


class NonStochasticBanditEnv():
    ''' Stationary environment where the mean reward is fixed and rewards are sampled from ~N(mean, std)'''
    def __init__(self,num_arms: int,mean_reward: np.ndarray, std: np.ndarray):
        self.num_arms = num_arms
        self.mean_reward = mean_reward
        self.std = std
        assert self.num_arms == self.mean_reward.shape[0] == self.std.shape[0], 'Invalid shape of mean_reward or std array'
        self.arms = dict(enumerate(zip(self.mean_reward, self.std)))

    def step(self, chosen_arm):
        arm_mean, arm_dev = self.arms[chosen_arm]
        return np.random.normal(arm_mean, arm_dev)
    
    def plot_arms(self):
        arm_data = np.zeros((500,self.num_arms))
        for i, data in enumerate(zip(self.mean_reward, self.std)):
            mu,sigma = data
            arm_data[:,i] = np.random.normal(mu, sigma,size = (500))
        
        plt.title('Reward distribution for arms', fontsize = 9)
        plt.ylabel('Rewards distribution', fontsize = 8)
        plt.xlabel('Arms', fontsize = 8)
        plt.xticks(range(1,self.num_arms+1))
        plt.violinplot(arm_data, positions=range(1,self.num_arms+1))
        plt.show()
        plt.tight_layout()

    def plot_arms_reward_distribution(self, num_samples=1000):
        """
        This function is only used to visualize the arm's distrbution.
        """
        fig, ax = plt.subplots(1, 1, sharex=False, sharey=False, figsize=(9, 5))
        colors = sns.color_palette("hls", self.num_arms)
        for i, arm_id in enumerate(self.arms):
            reward_samples = [self.step(arm_id) for _ in range(num_samples)]
            sns.histplot(reward_samples, ax=ax, stat="density", kde=True, bins=100, color=colors[i], label=f'arm_{arm_id}')
        ax.legend()
        plt.show()

    def reset(self):
        pass

class NonStationaryEnv():
    ''' Nonstationary environment where the reward probabilities or values change after certain trials.
    Rewards are sampled from ~N(mean, std) with specified probabilities.'''
    def __init__(self, num_arms, mean_reward, std, 
                 probabilities,
                 mean_rew_change=None, std_change=None, 
                 probabilities_change=None,
                 stationary=False, time_stamp_change=None,
                 change_type='both'):
        self.num_arms = num_arms
        self.mean_reward = mean_reward
        self.std = std
        self.probabilities = probabilities
        self.stationary = stationary
        
        # Parameters for change
        self.mean_rew_change = mean_reward if mean_rew_change is None else mean_rew_change
        self.std_change = std if std_change is None else std_change
        self.probabilities_change = probabilities if probabilities_change is None else probabilities_change
        self.time_stamp_change = time_stamp_change
        
        # What type of change to apply: 'reward', 'probability', or 'both'
        self.change_type = change_type

        # Check that arrays have correct shapes
        assert self.num_arms == self.mean_reward.shape[0] == self.std.shape[0], 'Invalid shape of mean_reward or std array'
        assert self.num_arms == self.probabilities.shape[0], 'Invalid shape of probabilities array'
        
        if not stationary and time_stamp_change is not None:
            if self.change_type in ['reward', 'both']:
                assert self.num_arms == self.mean_rew_change.shape[0] == self.std_change.shape[0], 'Invalid shape of mean_rew_change or std_change array'
            if self.change_type in ['probability', 'both']:
                assert self.num_arms == self.probabilities_change.shape[0], 'Invalid shape of probabilities_change array'
        
        # Generate reward probability timestamps
        self.rew_timestamps = self.create_rew_timepoints()
        
        # Shuffle each key in rew_timestamps
        for key in self.rew_timestamps:
            np.random.shuffle(self.rew_timestamps[key])
        
        # Pick counter for each arm
        self.counts = np.zeros((self.num_arms))
        
        # Create arms dictionary for rewards
        self.arms = dict(enumerate(zip(self.mean_reward, self.std)))
        self.step_counter = 0

    def create_rew_timepoints(self, probs=None):
        """
        Create binary arrays representing when rewards are given based on probabilities
        """
        # Use current probabilities if none provided
        if probs is None:
            probs = self.probabilities
            
        # Number of columns
        num_columns = 10

        # Create binary array
        binary_array = np.zeros((len(probs), num_columns), dtype=int)

        for i, prob in enumerate(probs):
            # Calculate the number of 1s based on the probability
            num_ones = int(prob * num_columns)
            
            # Randomly choose `num_ones` indices to set as 1
            one_indices = np.random.choice(num_columns, num_ones, replace=False)
            binary_array[i, one_indices] = 1
        
        binary_dict = {i: row.tolist() for i, row in enumerate(binary_array)}
        
        return binary_dict

    def step(self, chosen_arm):
        if not self.stationary and self.step_counter >= self.time_stamp_change:
            # Apply changes based on change_type
            if self.change_type in ['reward', 'both']:
                self.arms = dict(enumerate(zip(self.mean_rew_change, self.std_change)))
            
            if self.change_type in ['probability', 'both']:
                self.rew_timestamps = self.create_rew_timepoints(self.probabilities_change)
                # Shuffle the new timestamps
                for key in self.rew_timestamps:
                    np.random.shuffle(self.rew_timestamps[key])
                # Reset counts when probabilities change
                self.counts = np.zeros((self.num_arms))
        
        # Get mean and std for the chosen arm
        arm_mean, arm_dev = self.arms[chosen_arm]
        
        # Check if reward should be given based on probability
        if self.rew_timestamps[chosen_arm][int(self.counts[chosen_arm])] == 1:
            # Reward associated with the arm
            rew = np.random.normal(arm_mean, arm_dev)
        else:
            rew = 0
            
        # Update counter for this arm
        self.counts[chosen_arm] += 1
        
        # If we've used all timestamps for this arm, reset and reshuffle
        if self.counts[chosen_arm] == 10:
            self.counts[chosen_arm] = 0
            np.random.shuffle(self.rew_timestamps[chosen_arm])
            
        self.step_counter += 1
        return rew
    
    def reset(self):
        # Reset arms to initial values
        self.arms = dict(enumerate(zip(self.mean_reward, self.std)))
        
        # Reset reward timestamps based on initial probabilities
        self.rew_timestamps = self.create_rew_timepoints()
        for key in self.rew_timestamps:
            np.random.shuffle(self.rew_timestamps[key])
            
        # Reset counters
        self.counts = np.zeros((self.num_arms))
        self.step_counter = 0


class StochasticBanditEnv():
    ''' Stochastic bandits where arms give rewards with certain probabilities'''
    def __init__(self,
                 num_arms: int,
                 mean_reward: np.ndarray, 
                 std_reward: np.ndarray, 
                 probabilities: np.ndarray):
        self.num_arms = num_arms

        self.mean_reward = mean_reward
        self.std_reward = std_reward
        self.probabilities = probabilities
        self.mean_loss = np.zeros(num_arms)
        self.std_loss = np.zeros(num_arms)
        assert self.num_arms == self.mean_reward.shape[0] == self.std_reward.shape[0] == self.probabilities.shape[0], 'Invalid shape of mean_reward or std_reward or probabilities array'
        

        self.rew_timestams = self.create_rew_timepoints()
        
        # shuffle each of the key in rew_timestams
        for key in self.rew_timestams:
            np.random.shuffle(self.rew_timestams[key])

        # pick counter for each arm to give loss
        self.counts = np.zeros((self.num_arms))

        # asserts if the shapes donot match
       
        self.arms = dict(enumerate(zip(self.mean_reward, self.std_reward)))

    def create_rew_timepoints(self):
        # Number of columns
        num_columns = 10

        # Create binary array of shape (4, 10)
        binary_array = np.zeros((len(self.probabilities), num_columns), dtype=int)

        for i, prob in enumerate(self.probabilities):
            # Calculate the number of 1s based on the probability
            num_ones = int(prob * num_columns)
            
            # Randomly choose `num_ones` indices to set as 1
            one_indices = np.random.choice(num_columns, num_ones, replace=False)
            binary_array[i, one_indices] = 1
        
        binary_dict = {i: row.tolist() for i, row in enumerate(binary_array)}
        
        return binary_dict

    def step(self, chosen_arm):
        arm_mean_rew, arm_dev_rew = self.arms[chosen_arm]

        if self.rew_timestams[chosen_arm][int(self.counts[chosen_arm])] == 1:
            # reward associated with the arm
            rew = np.random.normal(arm_mean_rew, arm_dev_rew)
        else:
            rew = 0
        self.counts[chosen_arm] +=1

        if self.counts[chosen_arm] == 10:
            # Once the arm is picked 10 times reset the counts and shuffle the array to change the loss times of the arm
            self.counts[chosen_arm] = 0
            np.random.shuffle(self.rew_timestams[chosen_arm])
        return rew
    
    def reset(self):
        self.rew_timestams = self.create_rew_timepoints()
        
        # shuffle each of the key in rew_timestams
        for key in self.rew_timestams:
            np.random.shuffle(self.rew_timestams[key])

        # pick counter for each arm to give loss
        self.counts = np.zeros((self.num_arms))

        # asserts if the shapes donot match
        assert self.num_arms == self.mean_reward.shape[0] == self.std_reward.shape[0] == self.mean_loss.shape[0]== self.std_loss.shape[0], 'Invalid shape of mean_reward or std array'
        self.arms = dict(enumerate(zip(self.mean_reward, self.std_reward)))


class RestlessBanditsEnv():
    ''' Restless bandit environment where the mean of the reward keeps drifting away irresoective of whether the arm is chosen or not'''
    def __init__(self, num_arms, mean_reward, std, 
                 lambda_decay = 0.9836/100, theta = 50/100, 
                 diffusive_noise_std = 2.8/100, diffusive_noise_mean = 0):
        self.num_arms = num_arms
        self.mean_reward_initial = mean_reward
        self.std_initial = std
        self.mean_reward = mean_reward
        self.std = std
        assert self.num_arms == self.mean_reward.shape[0] == self.std.shape[0], 'Invalid shape of mean_reward or std array'
        
        self.lambda_decay = lambda_decay
        self.theta = theta
        self.diffusive_noise_std = diffusive_noise_std
        self.diffusive_noise_mean = diffusive_noise_mean

        self.arms = dict(enumerate(zip(self.mean_reward, self.std)))
        self.step_counter = 0

    def update_arm_mean(self):
        diffusive_noise = np.random.normal(self.diffusive_noise_mean, self.diffusive_noise_std,4)
        self.mean_reward = self.lambda_decay * self.mean_reward + (1 - self.lambda_decay) * self.theta + diffusive_noise
        self.arms = dict(enumerate(zip(self.mean_reward, self.std)))

    def step(self, chosen_arm):
        arm_mean, arm_dev = self.arms[chosen_arm]
        self.step_counter += 1
        reward = np.random.normal(arm_mean, arm_dev)
        self.update_arm_mean()
        return reward
    
    def reset(self):
        assert self.num_arms == self.mean_reward_initial.shape[0] == self.std_initial.shape[0], 'Invalid shape of mean_reward or std array'
        self.arms = dict(enumerate(zip(self.mean_reward_initial, self.std_initial)))
        self.step_counter = 0
