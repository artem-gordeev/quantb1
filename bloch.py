import math as m 
import numpy as np 
import matplotlib.pyplot as plt

num_modeling = 50
time_step = 0.00001 #параметры модели
display_step = 50
#total_time = 10
num_cycles = 4
#density_matrix: eg -> 1,2
#rho = np.array([[0.5+0j, 0+0.5j],[0-0.5j,0.5+0j]], dtype=complex)

gamma = 0  
decay = 0
delta = 0.1
Rabi_freq = 1
#noise
sigma_e = 0
sigma_nu = 0 #0.1
sigma_Rabi = 0.7

def exp_i(phi): 
    return np.cos(phi) + 1j * np.sin(phi) #* (np.cos((delta + nu_noise[i])) + 1j * np.sin() )

'''rho1 = np.array([[1+0j, 0+0j],[0+0j,0+0j]], dtype=complex)
rho2 = np.array([[1+0j, 0+0j],[0+0j,0+0j]], dtype=complex)
print(rho1+rho2 * 2)'''

fig, ax = plt.subplots()

#display = np.zeros(int(num_cycles*2*np.pi/((Rabi_freq**2+delta**2)**0.5 * time_step))//display_step + 1) #int(num_cycles*2*np.pi/((Rabi_freq**2+delta**2)**0.5 * time_step) / display_step))

display = [[] for i in range(num_modeling)] #массив, хранящий данные о разных моделях
for k in range(num_modeling):
    print(k)
    rho = np.array([[1+0j, 0+0j],[0+0j,0+0j]], dtype=complex)
    d_rho = np.array([[0+0j,0+0j],[0+0j,0+0j]], dtype=complex)
    display_num_p11 = []
    display_anal_p11 = []
    display_anal_p22 = []
    display_num_p22 = []
    display_time = []
    time = 0 
    phase = 0

    #rotating frame:
    #e_plus = 
    #e_minus(time): 
    #nu_noise = np.random.normal(0, sigma_nu, int(num_cycles*2*np.pi/((Rabi_freq**2+delta**2)**0.5 * time_step)))
    #noisy_Rabi = np.random.normal(Rabi_freq, sigma_Rabi, int(num_cycles*2*np.pi/((Rabi_freq**2+delta**2)**0.5 * time_step)))
    #noise_freq = 0.05
    #noisy_Rabi = 1 + sigma_Rabi * np.cos(noise_freq * time_step * np.arange(int(num_cycles*2*np.pi/((Rabi_freq**2+delta**2)**0.5 * time_step))))
    #print(j, 'mean = ', np.mean(nu_noise) ,'std_error = ', abs(sigma_nu - np.std(nu_noise, ddof=1)))
    
    #phase noise
    freqs = np.arange(0.05, 1.0, 0.005)
    rd_phases = 2*np.pi * np.random.random_sample((len(freqs),))
    #spectra = 0.001 * np.square(freqs)

    # Генерация шума 
    nu_noise = []
    for i in range(0, int(num_cycles*2*np.pi/((Rabi_freq**2+delta**2)**0.5 * time_step))):
        noise = 0
        for j in range(len(freqs)):
            noise += 0.0316*freqs[j]*np.cos(freqs[j]*i*time_step + rd_phases[j]) #спектр шума из статьи 
        nu_noise.append(noise) 
       
    # Моделирование (delta p = dp/dt * delta t)
    for i in range(0, int(num_cycles*2*np.pi/((Rabi_freq**2+delta**2)**0.5 * time_step))):
        d_rho[0,0] = 1j * Rabi_freq / 2 * (rho[0,1] - rho[1,0]) #+ decay * rho[1,1]
        d_rho[1,1] = -1j * Rabi_freq / 2 * (rho[0,1] - rho[1,0]) #- decay * rho[1,1]
        d_rho[0,1] = 1j * Rabi_freq / 2 * (rho[0,0] - rho[1,1]) - (gamma + 1j * (delta + nu_noise[i])) * rho[0,1]
        d_rho[1,0] = -1j * Rabi_freq / 2 * (rho[0,0] - rho[1,1]) - (gamma - 1j * (delta + nu_noise[i])) * rho[1,0]
        ''' #frequency noise
        d_rho[0,0] = 1j * Rabi_freq / 2 * (rho[0,1]*exp_i(phase) - rho[1,0]*exp_i(-phase)) #+ decay * rho[1,1]
        d_rho[1,1] = -1j * Rabi_freq / 2 * (rho[0,1]*exp_i(-phase) - rho[1,0]*exp_i(phase)) #- decay * rho[1,1]
        d_rho[0,1] = 1j * Rabi_freq / 2 * exp_i(phase) * (rho[0,0] - rho[1,1])# - (gamma + 1j * (delta + 0 *nu_noise[i])) * rho[0,1]
        d_rho[1,0] = -1j * Rabi_freq / 2 * exp_i(-phase) * (rho[0,0] - rho[1,1])# - (gamma - 1j * (delta + 0 *nu_noise[i])) * rho[1,0]
        '''
        rho = rho + d_rho * time_step
        if (i % display_step == 0):
            #display_num_p11.append(abs(rho[0,0]))
            display_num_p22.append(abs(rho[1,1]))
            #display_anal_p11.append((1+np.cos((Rabi_freq**2+delta**2)**0.5*time))/2)

            # Обычные осцилляции Раби для сравнения
            display_anal_p22.append((1-np.cos((Rabi_freq**2+delta**2)**0.5*time))/2) 
            display_time.append(time)
        time += time_step
        phase += (delta + nu_noise[i]) * time_step 
    #display = display + np.array(display_num_p22)/num_modeling
    display[k] = display_num_p22

# Вывод данных

#print(rho)

 #, linewidth=0.3)
#plt.plot(display_time, display_num_p11, linewidth=0.3)
#plt.plot(display_time, display_num_p22, linewidth=0.3)
#plt.plot(display_time, display_anal_p11, linewidth = 0.5)
plt.plot(display_time, display_anal_p22, linewidth = 0.5)
#display_time = np.linspace(0, 12.566359999862602, 12567)
#plt.plot(display_time, display, linewidth = 0.5)
#print(f'fidelity = {display[int(np.pi/(time_step*(Rabi_freq**2+delta**2)**0.5)/display_step)]}')
avg = np.zeros(len(display[0]))
for i in range(num_modeling):
    plt.plot(display_time, display[i], 'r', linewidth = 0.4)
    avg += np.array(display[i])/num_modeling

plt.plot(display_time,avg, 'b', linewidth = 1)
plt.show()

#Louvillian = np.array([[]
 #                      []])