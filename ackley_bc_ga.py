import random
import math
import numpy as np

random.seed(30)

def ackley(x):
    n = len(x)
    sum1 = 0
    sum2 = 0
    for i in range(n):
        sum1 += x[i]**2
        sum2 += math.cos(2*math.pi*x[i])
    return -20 * math.exp(-0.2 * math.sqrt(sum1/n)) - math.exp(sum2/n) + 20 + math.e

def binary_decode(binary, lower, upper, n):
    decoded = [0] * n
    for i in range(n):
        range_ = upper - lower
        total = 0
        for j in range(len(binary[i])):
            total += binary[i][j] * 2**(len(binary[i])-j-1)
        decoded[i] = total * range_ / (2**len(binary[i])-1) + lower
    return decoded

def binary_encode(decoded, lower, upper, n, num_bits):
    binary = [[0] * num_bits for i in range(n)]
    for i in range(n):
        range_ = upper - lower
        decimal = int((decoded[i] - lower) * (2**num_bits - 1) / range_)
        for j in range(num_bits):
            binary[i][num_bits-j-1] = decimal % 2
            decimal = decimal // 2
    return binary

def initial_population(num_bits, lower, upper, n, size):
    population = []
    for i in range(size):
        decoded = [random.uniform(lower, upper) for j in range(n)]
        population.append(binary_encode(decoded, lower, upper, n, num_bits))
    return population

def fitness(population, lower, upper, n, num_bits):
    fitness = []
    for i in range(len(population)):
        decoded = binary_decode(population[i], lower, upper, n)
        fitness.append(ackley(decoded))
    return fitness

def natural_selection(fitness, size, random_num):
    fitness = np.array(fitness)
    print('avg:', np.average(fitness))
    max_fitness = max(fitness)
    fitness = [max_fitness-fitness[i] for i in range(10)]
    print("fitness", fitness)
    fmax = max(fitness)
    fmin = min(fitness)
    fitness = (fitness-fmin)/(fmax-fmin)
    print('scaled_fitness', fitness)

    total_fitness = sum(fitness)
    probabilities = [fitness[i] / total_fitness for i in range(len(fitness))]
    print('prob', probabilities)
    cumsum = np.cumsum(probabilities)
    print('cumsum', cumsum)

    selected = []
    for rn in random_num:
        j = 0
        while(rn > cumsum[j]):
            j=j+1
        selected.append(j)

    return selected

def split_num(random_num_site):
    if random_num_site >=100:
        random_num_list = [int(s) for s in str(random_num_site)]
    elif random_num_site<100 and random_num_site >=10:
        random_num_list = [0] + [int(s) for s in str(random_num_site)]
    else:
        random_num_list = [0] + [0] + [int(s) for s in str(random_num_site)]
    return random_num_list

def crossover(parent1, parent2, n, num_bits, size, random_num_site):
    random_num_list = split_num(random_num_site)
        
    child1 = []
    child2 = []
    for j in range(n):
        random_num = random_num_list[j]
        child1.append(parent1[j][:random_num] + parent2[j][random_num:])
        child2.append(parent2[j][:random_num] + parent1[j][random_num:])
    return child1, child2

def mutate(candidate, random_num_mut):
    random_num_mut_list = split_num(random_num_mut)
    for j in range(3):
        candidate[j][random_num_mut_list[j]] = 1 if candidate[j][random_num_mut_list[j]]==0 else 0
    return candidate


#=============MAIN 

lower = -20
upper = 20
n = 3
num_bits = 10
pop_size = 10
init_pop = initial_population(num_bits, lower, upper, n, pop_size)

#====Generation 1
print('>>>>>>>GEN1<<<<<<')
func_val = fitness(init_pop, lower, upper, n, num_bits)
print('fitness_score', func_val)
random_num=[0.731, 0.537, 0.503, 0.859, 0.331, 0.967, 0.891, 0.154, 0.335, 0.725]
selected = natural_selection(func_val, pop_size, random_num)
# print table
for i in range(pop_size):
    print(f' {i} | {init_pop[i]} | {binary_decode(init_pop[i], lower, upper, n)} | {round(func_val[i], 2)} ' ) #

print('selection pool', selected)
# selection
init_pop = [init_pop[i] for i in selected]

print('selection')
for i in range(pop_size):
    print(f' {i} | {init_pop[i]} ') #{binary_decode(init_pop[i], lower, upper, n)} |

random_num_site = [834, 639, 657, 880, 922]
random_num_mut = [82, 516, 430, 954, 463, 721, 488, 533, 135, 143]

#crossover
new_pop = init_pop.copy()
print('crossover_random', random_num_site)
for i in range(5):
    ix1 = i*2
    ix2 = i + 1
    ch1, ch2 = crossover(init_pop[ix1], init_pop[ix2], n, num_bits, pop_size, random_num_site[i])
    new_pop[ix1] = ch1
    new_pop[ix2] = ch2

for i in range(pop_size):
    print(f' {i} | {new_pop[i]}')

# mutate
print('mutate')
for i in range(pop_size):
    new_pop[i] = mutate(new_pop[i], random_num_mut[i])

# print table
for i in range(pop_size):
    print(f' {i} |{random_num_mut[i]}|{new_pop[i]}')

init_pop = new_pop

# Generation 2:
print('>>>>>>>GEN2<<<<<<')
func_val = fitness(init_pop, lower, upper, n, num_bits)
print('fitness_score', func_val)
random_num=[0.173, 0.964, 0.569, 0.010, 0.576, 0.800, 0.46, 0.211, 0.871, 0.619]
selected = natural_selection(func_val, pop_size, random_num)
# print table
for i in range(pop_size):
    print(f' {i} | {init_pop[i]} | {binary_decode(init_pop[i], lower, upper, n)} | {round(func_val[i], 2)} ' ) #

print('selection pool', selected)
# selection
init_pop = [init_pop[i] for i in selected]

print('selection')
for i in range(pop_size):
    print(f' {i} | {init_pop[i]} ') #{binary_decode(init_pop[i], lower, upper, n)} |

random_num_site = [187, 570, 171, 947, 852] 
random_num_mut = [523, 808, 12, 789, 875, 852, 845, 674, 874, 237] 

#crossover
print('crossover_random', random_num_site)
new_pop = init_pop.copy()
for i in range(5):
    ix1 = i*2
    ix2 = i + 1
    ch1, ch2 = crossover(init_pop[ix1], init_pop[ix2], n, num_bits, pop_size, random_num_site[i])
    new_pop[ix1] = ch1
    new_pop[ix2] = ch2

for i in range(pop_size):
    print(f' {i} | {new_pop[i]}')

# mutate
print('mutate')
for i in range(pop_size):
    new_pop[i] = mutate(new_pop[i], random_num_mut[i])

# print table
for i in range(pop_size):
    print(f' {i} |{random_num_mut[i]}|{new_pop[i]}')

init_pop = new_pop

# Generation 3:
print('>>>>>>>GEN3<<<<<<')
func_val = fitness(init_pop, lower, upper, n, num_bits)
print('fitness_score', func_val)
random_num=[.935, .728, .899, .649, .791, .136, .516, .151, .243, .884] 
selected = natural_selection(func_val, pop_size, random_num)
# print table
for i in range(pop_size):
    print(f' {i} | {init_pop[i]} | {binary_decode(init_pop[i], lower, upper, n)} | {round(func_val[i], 2)} ' ) #

print('selection pool', selected)
# selection
init_pop = [init_pop[i] for i in selected]

print('selection')
for i in range(pop_size):
    print(f' {i} | {init_pop[i]} ') #{binary_decode(init_pop[i], lower, upper, n)} |

random_num_site = [276, 11, 594, 360, 998]  
random_num_mut = [240, 565, 699, 471, 17, 78, 493, 268, 602, 476] 

#crossover
new_pop = init_pop.copy()
print('crossover_random', random_num_site)
for i in range(5):
    ix1 = i*2
    ix2 = i + 1
    ch1, ch2 = crossover(init_pop[ix1], init_pop[ix2], n, num_bits, pop_size, random_num_site[i])
    new_pop[ix1] = ch1
    new_pop[ix2] = ch2

for i in range(pop_size):
    print(f' {i} | {new_pop[i]}')

# mutate
print('mutate')
for i in range(pop_size):
    new_pop[i] = mutate(new_pop[i], random_num_mut[i])

# print table
for i in range(pop_size):
    print(f' {i}|{random_num_mut[i]}|{new_pop[i]}')

init_pop = new_pop

print('>>>>>>>>>>>>FINAL')
func_val = fitness(init_pop, lower, upper, n, num_bits)
print('final avg', np.average(np.array(func_val)))
for i in range(pop_size):
    print(f' {i} | {init_pop[i]} | {binary_decode(init_pop[i], lower, upper, n)} |{round(func_val[i], 2)} ' ) #