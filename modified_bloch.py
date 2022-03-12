import math as m 
import numpy as np 
import matplotlib.pyplot as plt
import csv 
import random
import timeit
from numba import njit, types, typed


time_step = 1e-5 #параметры модели
display_step = 1000
num_cycles = 4
#density_matrix: eg -> 1,2

delta = 0
Rabi_freq = 2 * np.pi

rho_0 = np.array([1.0+0.0j, 0.0+0.0j,0.0+0.0j,0.0+0.0j])
print('Вот здесь все хорошо: ', rho_0)

#@jit(nopython=False)
@njit('float64[:](complex128[:], float64, float64)', cache=True, nogil=False, fastmath=True, parallel=False)
def model_evolution(initial_state, time_step, Rabi_freq):
    #print('initial state:', initial_state)
    rho = initial_state #np.array([1+0j, 0+0j,0+0j,0+0j])
    time = 0 
    d_rho = np.array([0+0j,0+0j,0+0j,0+0j])
    num_points_in_model = int(num_cycles*2*np.pi/((Rabi_freq**2+delta**2)**0.5 * time_step))
    #display_num_p22 = np.array([0.0]*num_points_in_model // display_step)#np.zeros(num_points_in_model // display_step, dtype=float)
    display_num_p22 = np.zeros(num_points_in_model//display_step, dtype=np.float64) #[0.0]*num_points_in_model
    Rabi_freq_div2 = Rabi_freq / 2 
    display_count = 0
    for i in range(0, num_points_in_model):
        d_rho[0] = 1j * Rabi_freq_div2  * (rho[1] - rho[2]) * time_step #+ decay * rho[3]
        d_rho[3] = -1j * Rabi_freq_div2  * (rho[1] - rho[2]) * time_step #- decay * rho[3]
        d_rho[1] = 1j * Rabi_freq_div2  * (rho[0] - rho[3]) * time_step #- (gamma + 1j * delta) * rho[1] #(gamma + 1j * (delta + nu_noise[i])) * rho[1]
        d_rho[2] = -1j * Rabi_freq_div2 * (rho[0] - rho[3]) * time_step#- (gamma - 1j * delta) * rho[2] #(delta + nu_noise[i])) * rho[2]
        rho += d_rho #rho[0] += d_rho[0]         rho[1] += d_rho[1]        rho[2] += d_rho[2]        rho[3] += d_rho[3]
        if (i % display_step == 0):
            display_num_p22[display_count] = abs(rho[3])
            display_count += 1
            
    return display_num_p22
    
print(timeit.timeit('model_evolution(rho_0, time_step, Rabi_freq)', globals=globals(), number = 1000))
#print(timeit.timeit('model_evolution(rho_0, time_step, Rabi_freq)', setup = 'from __main__ import model_evolution', number = 1000))
print('А здесь плохо, хотя rho_0 нигде не используется: ', rho_0)
'''display = model_evolution(rho_0, time_step)
plt.scatter(np.arange(len(display)), display)
plt.show()
'''

