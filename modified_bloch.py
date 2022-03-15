import math as m 
import numpy as np 
import matplotlib.pyplot as plt
import csv 
import random
import timeit
from numba import njit, types, typed
import copy 

time_step = 2e-5 #параметры модели
display_step = 500
num_cycles = 1000
#density_matrix: eg -> 1,2

delta = 0
Rabi_freq = 2 * np.pi #1 MHz
delta_freq = 500

'''rho_0 = np.array([1.0+0.0j, 0.0+0.0j,0.0+0.0j,0.0+0.0j])
print('Вот здесь все хорошо: ', rho_0)
rho_transfer = copy.deepcopy(rho_0)

#@jit(nopython=False)
@njit('float64[:](complex128[:], float64, float64)', cache=True, nogil=False, fastmath=True, parallel=False)
def model_evolution(initial_state, time_step, Rabi_freq):
    #print('initial state:', initial_state)
    rho = initial_state #np.array([1+0j, 0+0j,0+0j,0+0j])
    time = 0 
    d_rho = np.array([0+0j,0+0j,0+0j,0+0j])
    num_points_in_model = int(num_cycles*2*np.pi/((Rabi_freq**2+delta**2)**0.5 * time_step))
    #display_num_p22 = np.array([0.0]*num_points_in_model // display_step)#np.zeros(num_points_in_model // display_step, dtype=float)
    display_num_p22 = np.zeros(num_points_in_model//display_step + 1, dtype=np.float64) #[0.0]*num_points_in_model
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
    #print(rho_0)
    return display_num_p22

#for i in range(100):
#    display = model_evolution(rho_transfer, time_step, Rabi_freq)
print(timeit.timeit('model_evolution(rho_transfer, time_step, Rabi_freq)', globals=globals(), number = 1000))
#print(timeit.timeit('model_evolution(rho_0, time_step, Rabi_freq)', setup = 'from __main__ import model_evolution', number = 1000))
print('А здесь плохо, хотя rho_0 нигде не используется: ', rho_0)
display = model_evolution(rho_0, time_step)
plt.scatter(np.arange(len(display)), display)
plt.show()
'''

freq = []
intens = []
with open(r"C:\Users\А\Desktop\Учеба\Научка\Шум лазера\servos\red\spec_highres_min10dBm.csv", newline='') as csvfile: #ampl_noise\red_wide2_5av.csv
    spamreader = csv.reader(csvfile, delimiter=',')
    data_flag = 0
    for row in spamreader:
        if (data_flag):
            freq.append(float(row[0]))
            intens.append(float(row[1]))
        elif (row[0] == 'DATA'):
            print('YES')
            data_flag = 1

freqs = np.array(freq) / 1e6 - 360 + delta_freq
print(freqs[0]) 
signal = np.array(intens)
amplitudes = (np.power(10, signal/10))**0.5
amplitudes = amplitudes/ max(amplitudes)


@njit('float64[:](complex128[:], float64, float64, float64[:], float64[:])', cache=True, nogil=False, fastmath=True, parallel=False)
def model_evolution(initial_state, time_step, Rabi_freq, amplitudes, freqs):
    
    state = initial_state #np.array([1+0j, 0+0j,0+0j,0+0j])
    time = 0 
    #d_rho = np.array([0+0j,0+0j,0+0j,0+0j])
    num_points_in_model = int(num_cycles*2*np.pi/((Rabi_freq**2+delta**2)**0.5 * time_step))
    #display_num_p22 = np.array([0.0]*num_points_in_model // display_step)#np.zeros(num_points_in_model // display_step, dtype=float)
    display_num_1 = np.zeros(num_points_in_model//display_step + 1, dtype=np.float64) #[0.0]*num_points_in_model
    display_num_2 = np.zeros(num_points_in_model//display_step + 1, dtype=np.float64)
    display_num_3 = np.zeros(num_points_in_model//display_step + 1, dtype=np.float64)
    Rabi_freq_div2 = Rabi_freq / 2 
    display_count = 0

    # Генерация шума 
    rd_phases = 2*np.pi * np.random.random_sample((len(freqs),)) 

    for i in range(0, num_points_in_model):

        int_Rabi_spectra_1 = np.pi * np.sum(amplitudes * np.exp(1j * (time * freqs + rd_phases)))
        int_Rabi_spectra_2 = 1.5 * int_Rabi_spectra_1 #np.dot(amplitudes, np.exp(1j * time * freqs))

        state[0] += -1j * int_Rabi_spectra_1 * state[2] * time_step
        state[1] += -1j * int_Rabi_spectra_2 * state[2] * time_step
        state[2] += -1j * (np.conj(int_Rabi_spectra_1) * state[0] + np.conj(int_Rabi_spectra_2) * state[1]) * time_step

        #print(state[0], state[1], state[2], int_Rabi_spectra_1)

        if (i % display_step == 0):
            display_num_1[display_count] = abs(state[0])**2
            display_num_2[display_count] = abs(state[1])**2
            display_num_3[display_count] = abs(state[2])**2
            display_count += 1

        time += time_step

    #print(rho_0)
    #result = np.array([display_num_1,display_num_2,display_num_3])
    return display_num_2

state_0 = np.array([0.0+1.0j, 0.0+0.0j,0.0+0.0j])
result = model_evolution(copy.deepcopy(state_0), time_step, Rabi_freq, amplitudes, freqs)
fig, ax = plt.subplots()
timeline = np.linspace(0, num_cycles, len(result))
plt.plot(timeline, result)
plt.show()
