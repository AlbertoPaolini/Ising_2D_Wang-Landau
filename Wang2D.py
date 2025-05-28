import numpy as np
import matplotlib.pyplot as plt
import random
import math
import os

class Spins:
    def __init__(self, coordinate, spin):
        self.coordinate = coordinate
        self.spin = spin

# Function to check if the histogram is "flat", i.e. if the 90% of the data-points are above the average
def is_flat(H, threshold=0.9):
    nonzero = H[H > 0]
    if len(nonzero) == 0:
        return False
    min_H = np.min(nonzero)
    avg_H = np.mean(nonzero)
    return min_H >= threshold * avg_H


def Wang_Landau(Lattice, n, nt, f_in, f_threshold):
    Lattice_Size = n * nt
    num_energy_levels = 4 * Lattice_Size  # range of possible energies
    offset = 2 * Lattice_Size # this offset is to avoid negative arguments in log(g)

    H = np.zeros(num_energy_levels)
    log_g = np.zeros(num_energy_levels)
    f = f_in

    # Initial energy of the system
    current_energy = compute_total_energy(Lattice, n, nt)

    while f > f_threshold:
        H.fill(0)
        steps = 0

        while True:
            for _ in range(1000000):
                i = random.randint(0, nt * n - 1) # Randomly pick one spin in the lattice
                x=Lattice[i].coordinate[0]
                y=Lattice[i].coordinate[1]

                x_m = (x -1 + n)%n
                y_m = (y -1 + nt)%nt
                x_p = (x + 1)%n
                y_p = (y + 1)%nt

                x_m_pos= x_m * nt + y 
                y_m_pos= x * nt + y_m 
                x_p_pos= x_p * nt + y 
                y_p_pos= x * nt + y_p 

                # Propose spin flip
                dE = 2 * Lattice[i].spin* ( Lattice[x_p_pos].spin + Lattice[y_p_pos].spin + Lattice[x_m_pos].spin + Lattice[y_m_pos].spin )
                Lattice[i].spin *= -1
                new_energy = current_energy - dE

                e_old_idx = int(current_energy + offset)
                e_new_idx = int(new_energy + offset)

                if 0 <= e_old_idx < num_energy_levels and 0 <= e_new_idx < num_energy_levels:
                    delta_log_g = log_g[e_old_idx] - log_g[e_new_idx]
                    try:
                        accept_prob = min(1.0, math.exp(delta_log_g))
                    except OverflowError:
                        accept_prob = 1.0 

                    if random.random() < accept_prob:
                        # Accept move
                        current_energy = new_energy
                        current_index = e_new_idx
                    else:
                        # Reject move
                        Lattice[i].spin *= -1
                        current_index = e_old_idx

                    log_g[current_index] += math.log(f)
                    H[current_index] += 1

                steps += 1

            if is_flat(H) or steps > 2000 * Lattice_Size: # The second condition is useful in case H fails to reach the flatness condition after many cycles. In that case, the code accepts the current state and proceeds anyway.
                print("Flat-condition not respected.")
                break

        f = math.sqrt(f)

    return log_g


# Function to perform the energy in the Wang-Landau algorithm. In this function, energy is the sum of the product of the near-neighbour spins.
def compute_total_energy(Lattice, n, nt):
    energy = 0.0
    for i in range(len(Lattice)):
        x=Lattice[i].coordinate[0]
        y=Lattice[i].coordinate[1]
        
        # Periodic boundary conditions implemented using the modulo operator
        x_p = (x + 1)%n
        y_p = (y + 1)%nt

        x_p_pos= x_p * nt + y 
        y_p_pos= x * nt + y_p

        energy += - Lattice[i].spin * (Lattice[x_p_pos].spin + Lattice[y_p_pos].spin) 
    
    return energy

    
# Function to compute <E>
def mean_energy_from_log_g(log_g, beta, L, Lt):
    energies = np.arange(len(log_g)) - 2 * L * Lt  #Because the offset in the Wang-Landau algorithm, energy is shifted
    log_weights = log_g - beta * energies

    valid = log_g > 1e-6
    log_weights = log_weights[valid]
    energies = energies[valid]

    log_weights -= np.max(log_weights) # Shift in the weights to avoid numerical underflow

    weights = np.exp(log_weights)
    Z = np.sum(weights)
    E_avg = np.sum(energies * weights) / Z
    return E_avg

# Function to compute <E^2>
def mean_energy2_from_log_g(log_g, beta, L, Lt):
    energies = np.arange(len(log_g)) - 2 * L * Lt  
    log_weights = log_g - beta * energies

    valid = log_g > 1e-6
    log_weights = log_weights[valid]
    energies = energies[valid]

    log_weights -= np.max(log_weights)

    weights = np.exp(log_weights)
    Z = np.sum(weights)
    E2_avg = np.sum(weights * energies **2) / Z
    return E2_avg
    
#----------------This is the main------------

for _ in range(20): # This first for is to perform the simulation 20 times in order to have more data-points
    Size = [20,14,8]
    All_Energies = []
    All_Heath = []
    All_Variance = []

    for s in Size:
        Lenght = s
        Height = s
        Lattice_Size= s**2

        SpecificHeat = []
        Average_Energy = []
        Average_Energy2 = []
        Variance = []

        t=0.5
        K_b = 1
        beta = 1
        Temperature = []
        Lattice = []

        for i in range(Lenght):
            for j in range(Height):
                pos=[i , j]
                spin=int(np.random.choice([-1, 1]))
                new_site=Spins(pos , spin)
                Lattice.append(new_site)
        f = np.exp(1.0)
        Threshold = 1.0000000001
        log_g = Wang_Landau(Lattice, Lenght, Height, f, Threshold)
        energies = np.arange(len(log_g)) - 2 * Lattice_Size

        for i in range(60):

            beta = 1.0 / ( K_b * t)
            E_avg = mean_energy_from_log_g(log_g, beta, Lenght, Height)
            
            print(f"Energia totale a beta = {beta}: {E_avg}")
            E2_avg = mean_energy2_from_log_g(log_g, beta, Lenght, Height) 
            c = E2_avg - E_avg**2
            Variance.append(c)
            c = c / (K_b * t**2)
            Average_Energy2.append(E2_avg)
            Average_Energy.append(E_avg / (Lattice_Size))
            SpecificHeat.append(c / (Lattice_Size))
            Temperature.append(t)
            t += 0.06
        
        All_Energies.append(Average_Energy)
        All_Heath.append(SpecificHeat)
        All_Variance.append(Variance)



    for i, s in enumerate(Size):
        plt.plot(Temperature, All_Heath[i], label=f"{s}x{s}")
    plt.xlabel('Temperature')
    plt.ylabel('Specific Heat per Spin')
    plt.title('Specific Heat for different lattice sizes')
    plt.legend()
    plt.grid(True)
    plt.show()

    for i, s in enumerate(Size):
        plt.plot(Temperature, All_Energies[i], label=f"{s}x{s}")
    plt.xlabel('Temperature')
    plt.ylabel('Average Energy per spin')
    plt.title('Average Energy per spin for different lattice sizes')
    plt.legend()
    plt.grid(True)
    plt.show()


    output_file = os.path.join("specific_heat_data.txt")
    file_exists = os.path.isfile(output_file)
    with open(output_file, "a") as f:
        if not file_exists:
            f.write("# Size\tTemperature\tSpecificHeat\n")
        for i, s in enumerate(Size):
            for j in range(len(Temperature)):
                line = f"{s}\t{Temperature[j]:.4f}\t{All_Heath[i][j]:.6f}\n"
                f.write(line)

    output_file = os.path.join("Variance_energy_data.txt")
    file_exists = os.path.isfile(output_file)
    with open(output_file, "a") as f:
        if not file_exists:
            f.write("# Size\tTemperature\tVariance\n")
        for i, s in enumerate(Size):
            for j in range(len(Temperature)):
                line = f"{s}\t{Temperature[j]:.4f}\t{All_Variance[i][j]:.6f}\n"
                f.write(line)



