import numpy as np
from matplotlib import pyplot as plt

class Analysis:

    def __init__(self, spike_array):
        self.spike_array = np.array(spike_array)

    def spike_rate(self,binsize):
        ''' 
        Function to convert spike to rate of change

        Args:
            spike_array (np.ndarray): Spike train with shape (time, num_neurons)
            binsize (int): binsize
        '''
        num_neurons = self.spike_array.shape[1] 
        grid_size = int(np.sqrt(num_neurons)) # 16self.
        time = self.spike_array.shape[0]
        rate_coded = np.zeros((num_neurons, time))
        for neuron in range(num_neurons):
            spikes = self.spike_array[:,neuron]
            for i in range(time):
                rate = np.sum(spikes[i:i+binsize])
                rate_coded[neuron][i] = rate
        rate_coded = rate_coded.reshape(grid_size,grid_size,time)
        mean_rate_a = np.mean(rate_coded[0:grid_size//2,0:grid_size//2,:].reshape(grid_size//2*grid_size//2, -1), axis = 0)
        mean_rate_b = np.mean(rate_coded[grid_size//2:grid_size, 0:grid_size//2,:].reshape(grid_size//2*grid_size//2, -1), axis = 0)
        mean_rate_c = np.mean(rate_coded[0:grid_size//2, grid_size//2:grid_size, :].reshape(grid_size//2*grid_size//2, -1), axis = 0)
        mean_rate_d = np.mean(rate_coded[grid_size//2:grid_size, grid_size//2:grid_size, :].reshape(grid_size//2*grid_size//2, -1), axis = 0)
        rate_data = {'all_rate_data': rate_coded,
                     '1': mean_rate_a,
                     '2': mean_rate_b,
                     '3': mean_rate_c,
                     '4': mean_rate_d}

        return rate_data
    
    def compute_spike_rate_newfunc(self, bin_size, spike_array_ = None):
        """
        Compute the spike rate for each neuron by binning the spike data.

        Parameters:
        spike_array (numpy.ndarray): 2D array of shape (n_neurons, n_timepoints) containing spike data (0 or 1).
        bin_size (int): Size of the time window to bin the spikes.

        Returns:
        numpy.ndarray: 2D array of spike rates of shape (n_neurons, n_bins).
        """

        if spike_array_ is None:
            spike_array_ = self.spike_array.T
        n_neurons, n_timepoints = spike_array_.shape
        n_bins = n_timepoints // bin_size  # Number of bins

        # Reshape spike array by grouping time points into bins
        binned_spikes = spike_array_[:, :n_bins * bin_size].reshape(n_neurons, n_bins, bin_size)

        # Sum the spikes within each bin and normalize by bin size to get the rate
        spike_rates = binned_spikes.sum(axis=2)

        return spike_rates

    def raster_plot(self, h = 0.1, title = 'Raster Plot'):
        '''
        Function to plot spike raster

        Args:
        title (str): Name of the title of the plot
        h (float): step size, Default = 0.1
        spike_array(ndarray) : 2D-NumPy array (time_steps, num_neurons)
        spike array with 0s and 1s, where 1-denoting spike

        Returns: None

        '''
        num_neurons = self.spike_array.shape[1]
        time_steps = self.spike_array.shape[0]
        iter = time_steps 
        t = np.linspace(0,iter*h/1000,iter)
        
        fig = plt.figure(figsize = (15,2))
        for n in range(num_neurons):
            plt.scatter(t, (n+1)*self.spike_array[:,n],color='black', s=0.5)
        #range(time_steps)
        plt.ylim(0.5,256)
        plt.xlabel("t (s)")
        plt.ylabel("# neuron")
        plt.title(title)
        plt.show()

    def frequency(self, dt = 0.1):
        '''
        Function to compute frequency
        Args:
            spike_array : 2D-NumPy array (time_steps, num_neurons)
            dt (float) : dt value used for euler
            spike array with 0s and 1s, where 1-denoting spike

        Returns:
            frequency (float): Mean frequency
            frequency_max (float): Max frequency
            frequency_min (float): Min frequency

        '''
        num_neurons = self.spike_array.shape[1]
        time_steps = self.spike_array.shape[0]

        frequency_avg = 1000* np.sum(self.spike_array)/(num_neurons * time_steps * dt)
        frequency_max = np.max(1000* np.sum(self.spike_array, axis = 0)/(time_steps * dt))
        frequency_min = np.min(1000* np.sum(self.spike_array, axis = 0)/(time_steps * dt))


        return frequency_avg, frequency_max, frequency_min

    def synchrony(self):
        '''
        Function to compute synchrony

        Args:
            spike_array: 2D-NumPy array (time_steps, num_neurons)
            spike array with 0s and 1s, where 1-denoting spike

        Returns:
            Ravg, Rvalue, phi
        '''

        spike_array = np.transpose(self.spike_array)

        (num_neurons, time_steps)= spike_array.shape # rm: num_neurons
        Rvalue=[]
        phi=[]
        phi=3000*np.ones((num_neurons,time_steps))

        for neur in range(num_neurons):
                temp=spike_array[neur,:]
                temptime=np.where(temp==1)[0]
                j=1
                while j<temptime.shape[0]-1:
                    for i in range(temptime[j],temptime[j+1]-1):
                        phi[neur,i]=(2*np.pi*(i-temptime[j]))/(temptime[j+1]-temptime[j])
                    j=j+1

        tempM=np.mean(phi, axis =0)
        a=np.sqrt(-1+0j)
        M=np.exp(a*tempM)
        Rvalue = ((np.sum(np.exp(a*phi),axis =0 )/num_neurons))/M
        Ravg = np.mean(abs(Rvalue))
        return Rvalue, Ravg
