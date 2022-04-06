import math as m 
import numpy as np 
import matplotlib.pyplot as plt
import csv 
import random
import timeit
from numba import njit, types, typed
import copy 


fig, ax = plt.subplots()

num_simul = 20
time_step = 2e-6 #параметры модели
display_step = int(1/time_step / 1000)
num_cycles = 80
num_spectra_points = 100
#density_matrix: eg -> 1,2

delta = 0
Rabi_freq = 2 * np.pi #1 MHz
delta_freq = 500.0

freq = []
intens = []
with open(r"C:\Users\А\Desktop\Учеба\Научка\Шум лазера\servos\red\spec_centre.csv", newline='') as csvfile: #ampl_noise\red_wide2_5av.csv
    spamreader = csv.reader(csvfile, delimiter=',')
    data_flag = 0
    for row in spamreader:
        if (data_flag):
            if (float(row[1]) > -65):
                freq.append(float(row[0]))
                intens.append(float(row[1]))
        elif (row[0] == 'DATA'):
            print('YES')
            data_flag = 1

freqs = np.array(freq) / 1e6 - 360.01 + delta_freq
print('После отсекания осталось ', len(freqs), " частот")
signal = np.power(10, np.array(intens)/10)
signal = signal / np.sum(signal[145:165])
amplitudes = (signal)**0.5
#plt.scatter(freqs[145:165], signal[145:165])
#plt.show()

mod_freqs = []
Rabi_amplitudes = []
for i in range(int(num_spectra_points/2.2)):
    mod_freqs.append(freqs[i * (len(freqs) // num_spectra_points)])
    Rabi_amplitudes.append(amplitudes[i * (len(freqs) // num_spectra_points)])
    mod_freqs.append(freqs[-i * (len(freqs) // num_spectra_points)])
    Rabi_amplitudes.append(amplitudes[-i * (len(freqs) // num_spectra_points)])
mod_freqs.append(delta_freq)
#print(mod_freqs)
#print((np.sum(signal[450:550]))**0.5)
Rabi_amplitudes.append(1.0)
mod_freqs = np.array(mod_freqs)
Rabi_amplitudes = np.array(Rabi_amplitudes)
'''
main_peak = np.sum(signal[450:550])**0.5
servo_bump1 = np.sum(signal[100:180])**0.5
servo_bump2 = np.sum(signal[820:900])**0.5 #np.sum(signal[800:900])**0.5
print(servo_bump1 / main_peak, servo_bump2 / main_peak, 'left = ', 1-main_peak**2-servo_bump1**2-servo_bump2**2 )'''

plt.scatter(mod_freqs, Rabi_amplitudes)
plt.show()

#@njit('float64[:](complex128[:], float64, float64, float64[:], float64[:], float64[:])', cache=True, nogil=False, fastmath=True, parallel=False)
def model_evolution(initial_state, time_step, Rabi_freq, amplitudes, freqs, rd_phases): #Rabi_freq, amplitudes, freqs, rd_phases):
    #rd_phase = rd_phases
    #phi = 2 * np.pi * (1000 * rd_phase - int(1000 * rd_phase))
    #print('.') 
    state = initial_state #np.array([1+0j, 0+0j,0+0j,0+0j])
    time = 0 
    #d_rho = np.array([0+0j,0+0j,0+0j,0+0j])
    num_points_in_model = int(num_cycles*2*np.pi/((Rabi_freq**2+delta**2)**0.5 * time_step))
    num_cycle = int(2*np.pi/((Rabi_freq**2+delta**2)**0.5 * time_step))
    #display_num_p22 = np.array([0.0]*num_points_in_model // display_step)#np.zeros(num_points_in_model // display_step, dtype=float)
    
    length_result = num_points_in_model//display_step + 1
    #length_result2 = 2 * length_result
    display_num = np.zeros( length_result, dtype=np.float64) #[0.0]*num_points_in_model
    '''display_num_1 = np.zeros(num_points_in_model//display_step + 1, dtype=np.float64)
    display_num_2 = np.zeros(num_points_in_model//display_step + 1, dtype=np.float64)
    display_num_3 = np.zeros(num_points_in_model//display_step + 1, dtype=np.float64)'''
    Rabi_freq_div2 = Rabi_freq / 2 
    '''servo_rel_ampl_1 = 4 * servo_bump1 / main_peak
    servo_rel_ampl_2 = 2 * servo_bump2 / main_peak'''
    display_count = 0

    # Генерация шума 
    #rd_phases = 2*np.pi * np.random.random_sample((len(freqs),)) 

    for i in range(0, num_points_in_model):

        coeff = time_step * np.pi * np.sum(Rabi_amplitudes * (np.cos(time * mod_freqs + rd_phases) + 1j * np.sin(time * mod_freqs + rd_phases)))#np.exp(1j * (time * freqs + rd_phases)))
        #int_Rabi_spectra_2 = 1.5 * int_Rabi_spectra_1 #np.dot(amplitudes, np.exp(1j * time * freqs))

        #int_Rabi_spectra_div2 = Rabi_freq_div2 * (1 +  * np.exp(1j * (2.12 * time + rd_phase)))
        
        #coeff = -1j * Rabi_freq_div2 * (1 + servo_rel_ampl * np.exp(1j * (2.12 * time + rd_phase))) * time_step
    #coeff = Rabi_freq_div2 * (1 + servo_rel_ampl_1 * (np.cos(2.12 * time + rd_phase) + 1j * np.sin(2.12 * time + rd_phase)) ) * time_step #+ servo_rel_ampl_2 * (np.cos(2.12 * time + rd_phase + phi) + 1j * np.sin(2.12 * time + rd_phase + phi))) * time_step
        
        state[0] += -1j * coeff * state[2] #-1j * int_Rabi_spectra_div2 * state[1] * time_step
        state[1] += -1j * (coeff) * state[2] # np.conj #-1j * int_Rabi_spectra_div2 * state[0] * time_step
        state[2] += -1j * (np.conj(coeff) * state[0] + np.conj(coeff) * state[1])

        #print(state[0], state[1], state[2], int_Rabi_spectra_1)

        if (i % display_step == 0):
            display_num[display_count] = abs(state[1])**2
            #display_num[length_result + display_count] = abs(state[1])**2
            #display_num[length_result2 + display_count] = abs(state[2])**2
            
            '''display_num_1[display_count] = abs(state[0])**2
            display_num_2[display_count] = abs(state[1])**2
            display_num_3[display_count] = abs(state[2])**2'''
            display_count += 1

        #if (i % num_cycle == 0):
        #    print(i // num_cycle)

        time += time_step

    #print(rho_0)
    #result = np.array([display_num_1,display_num_2,display_num_3])
    return display_num

state_0 = np.array([1.0+0.0j, 0.0+0.0j,0.0+0.0j])

#rd_phases = 2 * np.pi * (np.random.random_sample(num_simul))

rd_phases = 2*np.pi * np.random.random_sample((len(mod_freqs),)) 

#print(timeit.timeit('model_evolution(copy.deepcopy(state_0), time_step, Rabi_freq, amplitudes, freqs, rd_phases[0])', globals=globals(), number = 10))

display = model_evolution(copy.deepcopy(state_0), time_step, Rabi_freq, amplitudes, freqs, rd_phases)
timeline = np.linspace(0, num_cycles, len(display))
plt.plot(timeline, (1-np.cos(Rabi_freq**2/2/delta_freq * timeline))/2)
plt.plot(timeline, display, color = 'red', linewidth = 0.5)

for i in range(num_simul-1):
    print(i+1)
    rd_phases = 2*np.pi * np.random.random_sample((len(mod_freqs),)) 
    result = model_evolution(copy.deepcopy(state_0), time_step, Rabi_freq, Rabi_amplitudes, mod_freqs, rd_phases)
    display += result
    plt.plot(timeline, result, color = 'red', linewidth = 0.5)
    
plt.plot(timeline, display/num_simul, color = 'blue', label = f'step = {time_step}, noise_pts = {num_spectra_points}')
plt.legend()
plt.show()


'''
for i in range(1,len(amplitudes)):
    print(i)
    rd_phases = 2*np.pi * np.random.random_sample((1,)) 
    result = model_evolution(copy.deepcopy(state_0), time_step, Rabi_freq, amplitudes[i], freqs[i], rd_phases)
    plt.plot(timeline, result, color = 'red', linewidth = 0.5)
plt.show()
'''

'''result = model_evolution(copy.deepcopy(state_0), time_step, Rabi_freq, amplitudes, freqs, rd_phases)
#result_wo_numba = model_evolution_wo_numba(copy.deepcopy(state_0), time_step, Rabi_freq, amplitudes, freqs, rd_phases)
#print(timeit.timeit('model_evolution(copy.deepcopy(state_0), time_step, Rabi_freq, amplitudes, freqs, rd_phases)', globals=globals(), number = 10))

timeline = np.linspace(0, num_cycles, len(result)//3)
plt.plot(timeline, result[:len(result)//3], label = 'ground')
plt.plot(timeline, result[len(result)//3 : 2 * len(result)//3], label = 'excited')
#plt.plot(timeline, result[2 * len(result)//3:], label = 'rydberg')
#plt.plot(timeline, result_wo_numba[:len(result_wo_numba)//3], label = 'ground_wo_numba')
#plt.plot(timeline, result_wo_numba[len(result_wo_numba)//3 : 2 * len(result_wo_numba)//3], label = 'excited_wo_numba')
#plt.plot(timeline, result_wo_numba[2 * len(result_wo_numba)//3:], label = 'rydberg_wo_numba')
plt.legend()
plt.show()
'''