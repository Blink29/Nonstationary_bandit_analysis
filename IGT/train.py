from IGT.BGNetwork import BGNetwork
import torch
import numpy as np
# from tqdm.autonotebook import tqdm



def train(env, trails, epochs, bins, lr , d1_amp = 1, 
          d2_amp = 5, gpi_threshold = 3, max_gpi_iters = 250,
          STN_spike_output = None, STN_neurons = 256,
          stn_mean = 0, stn_std = 0.5, del_lim = None, 
          train_IP = True, del_med = None, printing = False, 
          gpi_var = 1e-4, gpi_mean = 0.5, track_arms = False):
    
    if STN_spike_output is None:
        print('Using Random noise')
    else: 
        # print('Using Spike output') # time , num_osc
        assert STN_spike_output.shape[0] >= 950, "Time steps of STN is less than 1000 (10 sec)" 


    arm_tracker_full = []
    picks_per_bin = int(trails//bins)
    arm_chosen_monitor = torch.zeros(epochs,trails)
    num_arms = env.num_arms

    reward_monitor = torch.zeros(epochs,trails)
    avg_counts = {i: torch.zeros(epochs,bins,1) for i in np.arange(num_arms)}
    ip_monitor = {i: torch.zeros(epochs,trails,1) for i in np.arange(num_arms)}
    dp_monitor = {i: torch.zeros(epochs,trails,1) for i in np.arange(num_arms)}

    for epoch in range(epochs):
        if track_arms:
            arm_tracker = []
            arm_tracker.append(env.arms)
        env.reset()
        
        bg_network = BGNetwork(STN_neurons = STN_neurons, 
                               max_gpi_iters = max_gpi_iters, 
                               d1_amp = d1_amp, 
                               d2_amp = d2_amp, 
                               gpi_threshold = gpi_threshold,
                               seed = epoch,
                               num_arms=env.num_arms,
                               gpi_mean=gpi_mean,
                               gpi_var=gpi_var)
        
        optimizer = torch.optim.Adam(params = bg_network.parameters(), lr = lr)
        for trail in range(trails):
            # print('*******TRAIL*********')
            bin_num = int(trail//picks_per_bin)
            if STN_spike_output is None:
                stn_output = torch.normal(stn_mean, stn_std, size=(1, max_gpi_iters,STN_neurons)) #torch.randn((1,25,256))
            else:
                t_sample = np.random.uniform(low = 0, high = 750)
                stn_output = torch.tensor(STN_spike_output[int(t_sample): int(t_sample + max_gpi_iters),:], dtype=torch.float32).unsqueeze(dim=0)
            gpi_out, gpi_iters, dp_output, ip_output, value = bg_network(stn_output)

            arm_chosen = torch.argmax(gpi_out)

            avg_counts[arm_chosen.item()][epoch,bin_num] = avg_counts[arm_chosen.item()][epoch,bin_num] + 1
            
            for arm in range(num_arms):
                ip_monitor[arm][epoch,trail] = ip_output[0,arm]
                dp_monitor[arm][epoch,trail] = dp_output[0,arm]

            reward = env.step(arm_chosen.item())
            if track_arms:
                arm_tracker.append(env.arms)

            TD_error =  reward - dp_output[:, arm_chosen] #gpi_out[:,arm_chosen]

            if train_IP == False:
                # print('Not training IP')
                for param in bg_network.D2_pathway.parameters():
                    param.requires_grad = False  
            
            if del_lim is not None:             
                TD_error = torch.clamp(TD_error, max=del_lim)
            
            if del_med is not None:
                TD_error = TD_error + del_med

            if printing:
                print(dp_output, ip_output, gpi_out, gpi_iters,  arm_chosen, reward, TD_error)

            loss = TD_error**2
            
            # setting gradients to zero
            optimizer.zero_grad()
            
            # Computing gradient
            loss.backward()
            
            # Updating weights
            optimizer.step()

            #network weights to clamped to only positive
            with torch.no_grad():
                for param in bg_network.parameters():
                    param.clamp_(min=0)  

            arm_chosen_monitor[epoch, trail] = arm_chosen.item()
            reward_monitor[epoch, trail] = reward
        if track_arms:
            arm_tracker_full.append(arm_tracker)
    return reward_monitor, arm_chosen_monitor,avg_counts,ip_monitor, dp_monitor, arm_tracker_full
