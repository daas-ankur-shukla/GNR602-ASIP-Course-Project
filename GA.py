# importing packages
import os
import numpy as np
import scipy.io as sio
from scipy.misc import imshow
from scipy.misc import imsave
from scipy.misc import toimage
import matplotlib.pyplot as plt
from scipy.stats import entropy
from sklearn.cluster import KMeans
from sklearn.metrics import mutual_info_score
from sklearn.metrics import adjusted_mutual_info_score


# Genetic Algorithm Code

# Crossover
# Single Point Crossover
def crossover(candidates):
    set_size=np.shape(candidates)
    cross_points=np.random.randint(0,set_size[1], size=len(candidates))
    for i in range(0,int(set_size[0]/2)-1):
        if(candidates[i+1][cross_points[i]] not in candidates[i]):
            temp=candidates[i][cross_points[i]]
            candidates[i][cross_points[i]]=candidates[i+1][cross_points[i]]
            candidates[i+1][cross_points[i]]=candidates[i][cross_points[i]]

    return candidates

# Mutatation
# Single Point Mutation
def mutation(candidates):
    set_size=np.shape(candidates)
    #print('mutation',set_size)
    mut_points=np.random.randint(0,set_size[1], size=2)
    for i in range(0,set_size[0]-1):
        if(candidates[i+1][mut_points[1]] not in candidates[i]):
            temp=candidates[i][mut_points[0]]
            candidates[i][mut_points[0]]=candidates[i+1][mut_points[1]]
            candidates[i+1][mut_points[1]]=temp

    return candidates

# Fitness
# f() = Entropy
def fitness(candidate):
    shape=np.shape(candidate)
    actual_candidate=np.zeros((16,220))

    for i in range(0,shape[0]):
        actual_candidate[:,i]=clstr_mat[:,candidate[i]]

    ent=np.zeros((1,shape[0]-1))
    for i in range(0,(shape[0]-1)):
        ent[0][i]=entropy(actual_candidate[:,i],actual_candidate[:,i+1])

    return np.linalg.norm(ent[0])


# Selection
# Rank Based Selection
def selection(population,size):
    l=len(population)
    fit=np.zeros(l)


    for i in range(0,l):
        fit[i]=fitness(population[i])

    rankers = [c for f,c in sorted(zip(fit,population))]

    return rankers[::-1][:size], np.linalg.norm(fit[::-1][:size])

#Convergence Criterion
def convergence(rms0,rms1,thresh):
    print(rms0,rms1)
    if((rms1-rms0)<thresh and (rms1>rms0)):
        return True
    else:
        return False


# Fetching Hyperspectral Cube

ip='path\to\cube'

ip_data=np.zeros((145*145,220))

for i in range(220):
    ip_data[:,i]=np.load(ip+'\sample_'+str(i)+'.npy').flatten()

#Loading Labels
ip_gt=sio.loadmat('E:\GitHub\MTP\dataset\ip_gt.mat')
gt=ip_gt['indian_pines_gt']
labels=gt.flatten()

#Preparing Cluster mean matrix
kmeans_ip = KMeans(n_clusters=16, init='k-means++', random_state=0).fit(ip_data)

clstr_mat=kmeans_ip.cluster_centers_
print(clstr_mat.shape)


# Genetic Algorithm Calling Code

# hyperparameters
num_bands=220
crossProb=0.8
mutProb=0.4
cand_size=55
pop=int(num_bands/cand_size)
band_seq=np.arange(0,220)
np.random.shuffle(band_seq)
conv_thresh=0.001

population=[]
temp_band=[]
low=0
high=cand_size

for i in range(1,pop+1):
    temp_cand=band_seq[low:high].T.tolist()
    population.append(temp_cand)
    low=high
    high=high+cand_size


gen_size=len(population)
iterations=30 #no. of iterations
cross_cand=[]
mut_cand=[]
new_generation=population
new_gen_rms=np.zeros((1,iterations))
generation=[]

for i in range(iterations):
    generation=new_generation
    cross_cand=[]
    mut_cand=[]
    crossed_cand=[]
    mutated_cand=[]
    #calculating probability
    prob=np.random.rand(1, pop)

    for j in range(0,gen_size):
        if prob[0][j]<crossProb:
            cross_cand.append(generation[j])

        if prob[0][j]<mutProb:
            mut_cand.append(generation[j])

    if len(cross_cand)>0:
        crossed_cand=crossover(cross_cand)
    if len(mut_cand)>0:
        mutated_cand=mutation(mut_cand)

    print('length of gen before alteration',len(generation))
    generation.extend(crossed_cand)
    generation.extend(mutated_cand)
    print('length of gen after alteration',len(generation))

    new_generation,new_gen_rms[0][i]=selection(generation,gen_size)
    print(new_gen_rms[0][i])

#Plotting the fitness value of population
plt.plot(new_gen_rms[0])
plt.savefig('ip_iterations.png')
plt.show()
print(new_generation)

#Points for analysis
#1. Can store reports on the error matrix, rms errors
#2. Can store and see whats the difference between the classification accuracy obtained from the best combination of band
# and with the first band in the generation with best fitness
#3. Can play with hyhperparameters
#4. Can play with more datasets


# Making the new cube
best_can=new_generation[-1]
cube=np.zeros((145*145,len(best_can)))
for i in range(len(best_can)):
    cube[:,i]=ip_data[:,best_can[i]]
print(cube.shape)


# Calculating K-means for the reduced cube
kmeans_ga = KMeans(n_clusters=16, random_state=0).fit(cube)
result_ga=kmeans_ga.labels_
plt.imshow(result_ga.reshape(145,145))
# plt.savefig('ip_ga_kmeans.png')
plt.show()


kmeans_full = KMeans(n_clusters=10, random_state=0).fit(ip_data)
result_full=kmeans_full.labels_
plt.imshow(result_full)
plt.show()

# Calculating K-Means metrics
score_full=mutual_info_score(labels,result_full)
score_ga=mutual_info_score(labels,result_ga)
ad_full=adjusted_mutual_info_score(labels,result_full)
ad_ga=adjusted_mutual_info_score(labels,result_ga)
print(score_ga)
print(ad_ga)
