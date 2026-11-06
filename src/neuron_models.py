import torch
import torch.nn as nn
from tqdm import trange


def spike_fn(x):
    out = torch.zeros_like(x)
    out[x > 0] = 1.0
    return out

def kappa(delta_t):
    K = torch.where((delta_t>=0.5) & (delta_t<=2.5), 1, 0)
    return K

def f(x, E_d, D_d):
    return 1/(1+torch.exp(-(x-E_d)/D_d))

class Bursting_Model(nn.Module):
    def __init__(self, num_units, time_step, neuron_params, noise_params):
        super().__init__()
        
        self.num_units, self.time_step =  num_units, time_step
        
        # Unpack neuron parameters
        self.E_L = neuron_params['E_L'] #mV
        self.tau_s = neuron_params['tau_s'] #ms
        self.g_s = neuron_params['g_s'] #pA
        self.C_s =  neuron_params['C_s'] #pF
        self.tau_ws = neuron_params['tau_ws'] #ms
        self.b_ws = neuron_params['b_ws'] #pA
        self.tau_d = neuron_params['tau_d'] #ms
        self.g_d =  neuron_params['g_d'] #pA
        self.c_d = neuron_params['c_d'] #pA
        self.C_d = neuron_params['C_d'] #pF
        self.a_wd = neuron_params['a_wd'] #nS
        self.tau_wd = neuron_params['tau_wd'] #ms
        self.E_d = neuron_params['E_d'] #mV
        self.D_d = neuron_params['D_d'] #mV
        self.V_T = neuron_params['V_T'] #mV
        self.abs_ref = neuron_params['abs_ref'] #ms

        # Unpack noise parameters
        self.mu_s = noise_params['mu_s'] # pA
        self.sigma_s = noise_params['sigma_s'] # pA
        self.mu_d = noise_params['mu_d']  # pA
        self.sigma_d = noise_params['sigma_d'] # pA
        self.tau_noise = noise_params['tau_noise'] # ms
        
        # Define integration constants
        self.alpha_s = torch.exp(-self.time_step/self.tau_s)
        self.beta_s = torch.exp(-self.time_step/self.tau_ws).float()
        self.alpha_d = torch.exp(-self.time_step/self.tau_d)
        self.beta_d = torch.exp(-self.time_step/self.tau_wd).float()
        self.alpha_noise = torch.exp(-self.time_step/self.tau_noise)

        # Define state variables
        self.register_buffer('mem_s', torch.ones(self.num_units) * self.E_L)
        self.register_buffer('mem_d', torch.ones( self.num_units) * self.E_L)
        self.register_buffer('w_s', torch.zeros(self.num_units))
        self.register_buffer('w_d', torch.zeros(self.num_units))
        self.register_buffer('I_s_noise', torch.zeros( self.num_units))
        self.register_buffer('I_d_noise', torch.zeros(self.num_units))
        self.register_buffer('delta_t', torch.ones(self.num_units) * 1000) # long initial spike time

    def forward(self, I_s_factors, I_d_factors, W_s, W_d, I_s_baseline, I_d_baseline, record_potential=False):
        # Somatic and dendritic input currents. Will have dimensions (num_units, nb_steps)
        # inputs should be in units of pA

        _, nb_steps = I_s_factors.size()

        # Initialize states
        mem_s, mem_d = self.mem_s, self.mem_d
        w_s, w_d = self.w_s, self.w_d
        I_s_noise, I_d_noise = self.I_s_noise, self.I_d_noise
        delta_t = self.delta_t

        # Allocate tensors for stored somatic varaibles
        spk_rec = torch.zeros((self.num_units, nb_steps), device=self.mem_s.device)
        
        if record_potential==True:
            mem_d_rec = torch.zeros((self.num_units, nb_steps), device=self.mem_s.device)
            mem_s_rec = torch.zeros((self.num_units, nb_steps), device=self.mem_s.device)

        # Compute neuron layer activity
        for t in range(nb_steps):

            mthr = mem_s - self.V_T
            out = spike_fn(mthr)
            rst = out.detach() # We do not want to backprop through the reset
            
            delta_t[rst==1]=0
            delta_t[rst!=1]+=self.time_step

            # Compute the input currents using the rank-1 factors for this timestep
            I_s_tmp = W_s @ I_s_factors[:, t]+ I_s_baseline
            I_d_tmp = W_d @ I_d_factors[:, t]+ I_d_baseline
  
            new_mem_s_tmp = self.alpha_s*mem_s + self.tau_s*(1-self.alpha_s)*((self.g_s*f(mem_d, self.E_d, self.D_d) + I_s_tmp + I_s_noise + w_s)/self.C_s + self.E_L/self.tau_s) - rst*(self.V_T-self.E_L)
            new_mem_s = torch.where((delta_t<=self.abs_ref) & (delta_t>0), mem_s, new_mem_s_tmp.float())
            
            new_w_s = self.beta_s*w_s + (1-self.beta_s)*self.tau_ws*self.b_ws*rst
            
            new_mem_d  = self.alpha_d*mem_d + (1-self.alpha_d)*self.tau_d*((self.g_d*f(mem_d, self.E_d, self.D_d) + self.c_d*kappa(delta_t)+ I_d_tmp + I_d_noise + w_d)/self.C_d +self.E_L/self.tau_d)
            new_w_d = self.beta_d*w_d + (1-self.beta_d)*self.tau_wd*(self.a_wd*(mem_d - self.E_L)/self.tau_wd)

            new_I_s_noise = self.alpha_noise*I_s_noise + (1-self.alpha_noise)*self.mu_s + self.sigma_s*torch.sqrt((self.tau_noise*(1-self.alpha_noise**2))/2)*torch.randn(I_s_noise.size(), device=I_s_noise.device)
            new_I_d_noise = self.alpha_noise*I_d_noise + (1-self.alpha_noise)*self.mu_d + self.sigma_d*torch.sqrt((self.tau_noise*(1-self.alpha_noise**2))/2)*torch.randn(I_d_noise.size(), device=I_d_noise.device)

            if record_potential==True:
                mem_d_rec[:, t] = mem_d
                mem_s_rec[:, t] = mem_s
                
            spk_rec[:, t] = out

            mem_s = new_mem_s.detach()
            mem_d = new_mem_d.detach()
            w_s = new_w_s.detach()
            w_d = new_w_d.detach()
            I_s_noise = new_I_s_noise.detach()
            I_d_noise = new_I_d_noise.detach()

        if record_potential==True:
            return mem_s_rec, mem_d_rec, spk_rec
        else:
            return spk_rec  