import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
from scipy.stats import special_ortho_group


# Setup

def chooseNextVec(P,sol_inds,sol_vecs,sol_norms):
    n = len(P)
    residues = dict()
    best_norm = 0
    best_ind = -1
    best_vec = None
    for i in range(n):
        if i in sol_inds: continue
        w = np.copy(P[i])
        for v in sol_vecs:
            c = np.dot(w,v)/np.dot(v,v)
            w = w - c * v
        w_norm = np.dot(w,w)
        if best_ind == -1 or w_norm > best_norm:
            best_norm = w_norm
            best_ind = i
            best_vec = w
    sol_inds.append(best_ind)
    sol_vecs.append(best_vec)
    sol_norms.append(best_norm)
    return [sol_inds,sol_vecs,sol_norms]

def greedy(P,k):
    sol_inds,sol_vecs,sol_norms = [],[],[]
    for i in range(k):
        sol_inds, sol_vecs, sol_norms = chooseNextVec(P,sol_inds,sol_vecs,sol_norms)
    return [sol_inds, sol_vecs, sol_norms]

def chooseNextVec2(P,sol_inds,sol_vecs,sol_norms):
    n = len(P)
    P = np.array(list(P))
    residues = dict()
    best_norm = 0
    best_ind = -1
    best_vec = None
    A = P[sol_inds,].T
    for i in range(n):
        if i in sol_inds: continue
        w = np.copy(P[i])
        proj_matrix = A @ np.linalg.inv(A.T @ A) @ A.T
        proj_w = proj_matrix @ w
        w = w - proj_w
        w_norm = np.dot(w,w)
        if best_ind == -1 or w_norm > best_norm:
            best_norm = w_norm
            best_ind = i
            best_vec = w
    sol_inds.append(best_ind)
    sol_vecs.append(best_vec)
    sol_norms.append(best_norm)
    return [sol_inds,sol_vecs,sol_norms]

def greedy2(P,k):
    sol_inds,sol_vecs,sol_norms = [],[],[]
    for i in range(k):
        sol_inds, sol_vecs, sol_norms = chooseNextVec2(P,sol_inds,sol_vecs,sol_norms)
    return [sol_inds, sol_vecs, sol_norms]

def genRandomData(n,d):
    P = np.random.randn(n,d)
    norms = np.linalg.norm(P, axis=1).reshape(n,1)
    P /= norms
    return P

def load_mnist():
    df = pd.read_csv('mnist.csv', sep=',', header=None)
    P = np.array(df)
    P = P[:,1:]
    print("MNIST Dataset Loaded")
    return P

def load_random_mnist():
    print("Random MNIST Dataset Loaded")
    return genRandomData(60000,784)

def load_genes():
    df = pd.read_csv('genes.csv', sep=',', header=None)
    P = np.array(df)
    print("GENES Dataset Loaded")
    return P

def load_random_genes():
    print("Random GENES Dataset Loaded")
    return genRandomData(7778,331)


def computeDet(A):
    u, s, vh = np.linalg.svd(A, full_matrices=True)
    res = 1
    for val in s: res *= val
    return res

def getLocalSearchParam(P,sol_inds):
    n = len(P)
    P = np.array(P)
    swap_inds = set([i for i in range(n)])
    for i in sol_inds: swap_inds.remove(i)

    sol_P = P[sol_inds,:]
    sol_P_t = np.transpose(sol_P)
    M = np.dot(sol_P,sol_P_t)
    sol_vol = computeDet(M)**(0.5)

    max_local_search_val = 1
    for i in sol_inds:
        cur_inds = set(sol_inds)
        cur_inds.remove(i)
        cur_inds = list(cur_inds)
        cur_inds.append(-1)
        for j in swap_inds:
            cur_inds[-1] = j
            cur_P = P[cur_inds,:]
            cur_P_t = np.transpose(cur_P)
            M = np.dot(cur_P,cur_P_t)
            vol = computeDet(M)**(0.5)
            local_search_val = vol/sol_vol
            if local_search_val > max_local_search_val:
                max_local_search_val = local_search_val
    return max_local_search_val

#############################################################################################################
# Experiments

debug = False
do_experiment_1 = False
do_experiment_2 = True


# Experiment 1: Epsilon Values against K on Real and Random Datasets
if do_experiment_1:
    print("Starting Experiment 1")
    numIters = 5
    num_streams = [10]
    datasets = ["genes.csv","mnist.csv","genes_random","mnist_random"]
    k_step = 10
    num_points_in_data = None #only 3000 points from each dataset for computational reasons, set to None otherwise

    if debug: 
        numIters = 1
        num_streams = [5]
        k_step = 2

    allResults = dict()
    for dataset in datasets: 
        allResults[dataset] = dict()
        for m in num_streams:
            allResults[dataset][m] = dict()
            allResults[dataset][m]['avg'] = None
            allResults[dataset][m]['max'] = None



    for dataset in datasets:
    # for dataset in ["genes.csv"]:
        data = None

        if dataset == 'mnist.csv':
            data = load_mnist()
        elif dataset == "genes.csv":
            data = load_genes()
        elif dataset == "genes_random":
            data = load_random_genes()
        elif dataset == "mnist_random":
            data = load_random_mnist()
        if debug: data = data[:100,]
        if num_points_in_data != None: data = data[:num_points_in_data]
        n,d = data.shape

        # Normalize data for numerical reasons
        max_entry = np.amax(data)
        # data = (1/10000) * data
        data = (1/max_entry) * data
        # data = list(data)
        
        for m in num_streams:
            points_per_chunk = 3000
            # max_k = points_per_chunk
            # ks_to_run = [1]
            # for k in range(k_step,max_k + 1,k_step): ks_to_run.append(k)
            ks_to_run = [k for k in range(2,21,2)]
            max_k = max(ks_to_run)
            ks_to_run_set = set(ks_to_run)

            results = dict()
            for k in ks_to_run: results[k] = []

            for iter in range(1,numIters+1):
                print("Starting Iteration ", iter)
                # np.random.shuffle(data)
                # P_stream = [data[i*points_per_chunk:(i+1)*points_per_chunk] for i in range(m)]
                P_stream = [list(data[np.random.choice(n, points_per_chunk, replace=False),:]) for i in range(m)]
                
                eps_streams = dict()
                for k in ks_to_run: eps_streams[k] = []

                for i in tqdm(range(m)):
                    sol_inds, sol_vecs, sol_norms = [],[],[]
                    for k in tqdm(range(1,max_k+1)):
                        sol_inds, sol_vecs, sol_norms = chooseNextVec(P_stream[i],sol_inds,sol_vecs,sol_norms)
                        if k in ks_to_run_set:
                            eps = getLocalSearchParam(P_stream[i],sol_inds)
                            eps_streams[k].append(eps)
                
                for k in ks_to_run:
                    eps_max = max(eps_streams[k])
                    results[k].append(eps_max)

            results_max = [max(results[k]) for k in ks_to_run]
            results_avg = [sum(results[k])/len(results[k]) for k in ks_to_run]

            allResults[dataset][m]['max'] = results_max
            allResults[dataset][m]['avg'] = results_avg
            allResults[dataset][m]['ks_to_run'] = ks_to_run

            print("n:",n)
            print("d:",d)
            print("m:",m)
            print("points per chunk:",points_per_chunk)
            print("number of iterations:",numIters)
            print("results max")
            print(results_max)
            print("results avg")
            print(results_avg)


    for exp in ["MNIST","GENES"]:
        dataset = None
        dataset_random = None
        if exp == "MNIST":
            dataset = datasets[0]
            dataset_random = datasets[2]
        else:
            dataset = datasets[1]
            dataset_random = datasets[3]

        for m in num_streams:
            plot_title = "Dataset: " + exp
            plt.title(plot_title)

            X = allResults[dataset][m]['ks_to_run']
            
            Y = allResults[dataset][m]['avg']
            Y = [1 + y for y in Y]
            plt.plot(X,Y,label = "Real Dataset")

            Y = allResults[dataset_random][m]['avg']
            Y = [1 + y for y in Y]
            plt.plot(X,Y,label = "Random Dataset")

            Y = [1 + i**(0.5) for i in X]
            plt.plot(X,Y,label = "Theoretical Bound: sqrt(k)")

            plt.xlabel("K")
            plt.ylabel("Local Optimality: 1 + Epsilon")
            plt.legend()

            plt.savefig("images_final/" + plot_title + "Exp1_3000_5iters"'.jpg')
            plt.clf()



#############################################################################################################

# Experiment 2: Fix K, and plot local optimality epsilon against number of points in base set. 
do_random = False
if do_experiment_2:
    print("Starting Experiment 2")
    numIters = 5
    num_points_in_chunk = [500*i for i in range(1,9)]
    datasets = ["genes.csv","mnist.csv","genes_random","mnist_random"]
    ks_to_run = [5,10,15,20]

    if debug: 
        numIters = 2
        num_points_in_chunk = [10*i for i in range(1,11)]
        ks_to_run = [2,4]

    ks_to_run_set = set(ks_to_run)
    max_k = max(ks_to_run)

    allResults = dict()
    for dataset in datasets: 
        allResults[dataset] = dict()
        for M in num_points_in_chunk:
            allResults[dataset][M] = dict()
            for k in ks_to_run:
                allResults[dataset][M][k] = 0


    for dataset in datasets:
    # for dataset in ["genes.csv"]:
        data = None

        if do_random == False and dataset.endswith("random"): continue

        if dataset == 'mnist.csv':
            data = load_mnist()
        elif dataset == "genes.csv":
            data = load_genes()
        elif dataset == "genes_random":
            data = load_random_genes()
        elif dataset == "mnist_random":
            data = load_random_mnist()
        if debug: data = data[:100,]

        n,d = data.shape

        # Normalize data for numerical reasons
        max_entry = np.amax(data)
        data = (1/max_entry) * data
        
        for M in num_points_in_chunk:
            print("Number of points in chunk: ",M)
            for max_k in ks_to_run:
                results = dict()
                for iter in range(1,numIters+1):
                    P = list(data[np.random.choice(n, M, replace=False),:])
                    # for i in tqdm(range(M)):
                    sol_inds, sol_vecs, sol_norms = [],[],[]
                    for k in tqdm(range(1,max_k+1)):
                        sol_inds, sol_vecs, sol_norms = chooseNextVec(P,sol_inds,sol_vecs,sol_norms)
                        # if k in ks_to_run_set:
                    eps = getLocalSearchParam(P,sol_inds) - 1
                    allResults[dataset][M][max_k] += eps
                allResults[dataset][M][max_k] /= numIters




    for exp in ["MNIST","GENES"]:
        print("Experiment: ",exp)
        dataset = None
        dataset_random = None
        if exp == "MNIST":
            dataset = datasets[0]
            dataset_random = datasets[2]
        else:
            dataset = datasets[1]
            dataset_random = datasets[3]

        plot_title = "Dataset: " + exp
        plt.title(plot_title)
        X = num_points_in_chunk
        print("X: num_points_in_chunk", X)

        for k in ks_to_run:
            print("k = ",k)
            Y1,Y2 = [],[]
            for M in num_points_in_chunk:
                Y1.append(allResults[dataset][M][k]+1)
                if do_random: Y2.append(allResults[dataset_random][M][k]+1)
            plt.plot(X,Y1,label="k = " +str(k))
            print("Y: Epsilons for Real Dataset")
            print(Y1)
            if do_random: 
                plt.plot(X,Y2,label="Random Dataset, k= " +str(k))
                print("Y: Epsilons for Random Dataset")
                print(Y2)
        plt.xlabel("Number of Points in P")
        plt.ylabel("Local Optimality: 1 + Epsilon")
        plt.legend()
        plt.savefig("images_final/" + plot_title +'_pointsetsize.jpg')
        plt.clf()
