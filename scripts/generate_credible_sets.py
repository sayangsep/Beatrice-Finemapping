import torch
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pickle
import pandas as pd
from scripts.convert_to_gpu import gpu
from scripts.convert_to_gpu_and_tensor import gpu_t
from scripts.convert_to_gpu_scalar import gpu_ts
from scripts.convert_to_cpu import cpu
import os

matplotlib.use('Agg')

def save_object(obj, filename):
    with open(filename, 'wb') as output:  # Overwrites any existing file.
        pickle.dump(obj, output, 4)
        
def load_object(filename):
    with open(filename, 'rb') as input:  # Overwrites any existing file.
        obj = pickle.load(input)
    return obj





def cond_prob(M, K, bp):
    total = 0
    for k in M:
        if len(k)==K:
            total += np.squeeze(M[k])
    pip = np.zeros(bp)

    for k in M:
        if len(k)==K:
            for j in k:
                pip[j] += np.squeeze(M[k])
    pip = pip/total
    
    return pip

def find_cond_prob_constrained_causal_no_dup(given, n_caus, M, bp, cred_set):
    
    cred = []
    for c in cred_set:
        cred+=c
    cred = set(cred)    
    total = 0
    ss = given[:]
    p = np.array([0]*bp)
    
    for k in M:
        tag = True
        # for cr in cred_set:
        #     if len(np.intersect1d(cr,k))!=1:
        #         tag = False
        if tag:
            if all([s in k for s in ss]):
                total+=M[k]
                for kk in k:
                    if kk not in given and kk not in cred:                   
                        p[kk] += M[k]
        
   
    p = p/total
    
    p = p.squeeze()
    index_max = np.argmax(p)
    
    return index_max, p[index_max],p 

def find_cond_prob_constrained_causal(given, n_caus, M, bp):
    
    total = 0
    ss = given[:]
    p = np.array([0]*bp)
    
    for k in M:
            if all([s in k for s in ss]):
                total+=M[k]
                for kk in k:
                    if kk not in given:
                        p[kk] += M[k]
        
   
    p = p/total
    
    p = p.squeeze()
    index_max = np.argmax(p)
    
    return index_max, p[index_max],p 


def cond_stepwise_causal(M, pip, prior_causal, threshold, Z, LD, n_sub, sigma_sq, p0, S, bp, allow_dup):
    """ Create key set.
    """
    
    ind       = np.argsort(pip)[::-1]
        
    start_set = sorted(ind[:1])
    
    for nn in range(prior_causal-1):
    
        
        add(M, Z, LD, n_sub, sigma_sq, p0, S, start_set, range(bp))
        if allow_dup:
            new_index_snp, index_prob,_ = find_cond_prob_constrained_causal(start_set, nn+2, M, bp)
        else:
            new_index_snp, index_prob,_ = find_cond_prob_constrained_causal_no_dup(start_set, nn+2, M, bp, [])
        if index_prob>threshold:
            start_set.append(new_index_snp)
            start_set = sorted(start_set)
        else:
            return sorted(start_set)
    return sorted(start_set)
    

                
def find_credible_set(LD, M, start_set, bp, ths, prob_ths, allow_dup):
    cred_prob = []
    cred_set =  []
    start_set = list(start_set)

    
    for i, s1 in enumerate(start_set):

        ss = start_set[:]
        ss.pop(i)
       
        
        # local cred sets
        cr_set = [s1]
        p_set = []
        
        # posterior cond prob.
        if allow_dup:
            _,_, p = find_cond_prob_constrained_causal(ss, 0, M, bp)
        else:
           
            _,_, p = find_cond_prob_constrained_causal_no_dup(ss, 0, M, bp, cred_set)
        
        ind_cred = np.argsort(p)[::-1]
        
        p_set.append(np.round(p[s1], decimals=3))
        
        # threshold tracker.
        prob = p[s1]
        
        for it in ind_cred:
            if prob > ths:
                break
            else:                
                if it != s1 and p[it]>prob_ths:
                    cr_set.append(it)
                    p_set.append(np.round(p[it], decimals=3))
                    prob += p[it]
        
        LLD  = LD[:,cr_set]
        LLD = LLD[cr_set,:]
        LLD =np.abs(cpu(LLD).data.numpy())
        #if np.min(LLD)>0.5:        
        
        cred_set.append(cr_set)
        cred_prob.append(p_set)
    return cred_set, cred_prob


def abf( z, ld, memo, n_sub, sigma_sq, p0, ind, S):
        z = z.unsqueeze(1)
        ind_m  = tuple(ind)
        cc = gpu(torch.ones(len(z)))
        if len(ind)>0:
            if ind_m in memo:
                return memo[ind_m]
        
            U =  n_sub*torch.diag(sigma_sq*cc)[:,ind]
            V = ld[ind,:]
                
            inv            = torch.inverse(gpu(torch.eye(len(ind))) + torch.mm(V,U))
                
            sigma_inv      = torch.mm(torch.mm(U,inv),V)
                
            sigma          = gpu(torch.eye(len(ind))) + torch.mm(V,U)
                
            sigma2         = torch.matmul(torch.matmul(z.T, sigma_inv),S)/2
            
            prior = 1 - p0
            prior[ind] = p0[ind]
        
            res =  min(torch.tensor(10**10),torch.exp(-torch.logdet(sigma)/2 + sigma2 + torch.sum(torch.log(prior)) )) 
        
        
            memo[ind_m] = cpu(res).data.numpy()
        
            return res
        else:
            return            
        
def delete(memo, z, ld, n_sub, sigma_sq, p0, S, st_Set, search_Set):
    
        abf(z, ld, memo, n_sub, sigma_sq, p0, st_Set, S)
    
        K = list(st_Set)

        for i in range(len(K)):
            kk = K[:]
            kk.pop(i)
            kk = sorted(kk)
            abf(z, ld, memo, n_sub, sigma_sq, p0, kk,S)
        return

def add(memo, z, ld, n_sub, sigma_sq, p0, S, st_Set, search_set):
        abf(z, ld, memo, n_sub, sigma_sq, p0, st_Set, S)
    
        K = list(st_Set)
        for i in search_set:
            kk = K[:]
            if i not in K:
                kk.append(i)
            kk = sorted(kk)
            
            abf(z, ld, memo, n_sub, sigma_sq, p0, kk, S)
        return
    
def change(memo, z, ld, n_sub, sigma_sq, p0, S, st_Set,search_set):
    
        abf(z, ld, memo, n_sub, sigma_sq, p0, st_Set, S)
    
        K = list(st_Set)
        
        for j, k_con in enumerate(K):
            
            tem_K = K[:]
            tem_K.pop(j)
            for i in search_set:
                kk = tem_K[:]
                if i not in K and i !=k_con:
                    kk.append(i)
                elif i!=k_con:
                    rem_id = kk.index(i)
                    kk.pop(rem_id)
                kk = sorted(kk)
            
                abf(z, ld, memo, n_sub, sigma_sq, p0, kk, S)
        return          
    
def calculate_pip(memo,bp):
    
    pip = np.zeros(bp)
    tot = 0
    for k in memo:
        tot+= memo[k]
        for i in k:
            pip[i]+= memo[k]
            
    return np.squeeze(pip/tot)    


def regularize_ld(LD):
    LD = (LD + LD.T)/2
    s, w = np.linalg.eig(cpu(LD).data.numpy())
    s = np.real(s)
    s_new = torch.zeros(len(s))
    if min(s)<10**-3:
        s_new = torch.ones(len(s))*(min(s)-10**-3)   
        print("\n Adding a constant {} to regularize LD".format(-min(s)+10**-3))
    LD = LD - gpu(torch.diag(s_new))    
    return LD


def unique_sets(cred_set, cred_prob):
    D = []
    P = []
    C = []
    for i, xx in enumerate(cred_set):
        ids = np.argsort(xx)[::-1]
        x = np.array(xx)[ids]
        if  list(x) not in D:
            D += [list(x)]
            C += [xx]
            P += [cred_prob[i]]
    return C, P
    
    
    x = D

def main(options):
    ########################################################################################################################################################
    
    Z  = gpu_t(pd.read_table(options['z'],  sep=' ', header=None).to_numpy()[:,1].astype(float))
    LD = gpu_t(pd.read_table(options['LD'], sep=' ', header=None).to_numpy())
    LD = regularize_ld(LD)
    
    S = torch.matmul(torch.inverse(LD),Z.unsqueeze(1))
    try:
        prior_loc = options['prior_location']
        p0 = gpu_t(pd.read_table(prior_loc,  sep=' ', header=None).to_numpy()[:,1].astype(float))
        
    except:    
        p0 = gpu_t(np.array([1/len(Z)]*len(Z)))
        
    sigma_sq =  gpu_ts(0.22)**2
    bp = len(Z)
    m = load_object(os.path.join(options['target'],'res'))['memo']
    

    ###############################################################################################################################################
    pip = calculate_pip(m, bp)
    threshold_causal = options['key_thres']
    prior_n_causal   = min(bp,options['n_causal'] )
    n_sub = options['n_sub']
    allow_dup = options['allow_duplicates']
    start_set = cond_stepwise_causal(m, pip, prior_n_causal, threshold_causal, Z, LD, n_sub, sigma_sq, p0, S, bp, allow_dup)
                    
    ######################################################
    
    for ij, ji in enumerate(start_set):
        ss = start_set[:]
        ss.pop(ij)
        add(m, Z, LD, n_sub, sigma_sq, p0, S,ss, range(bp))

    ######################################################
    ths = options['coverage_ths']
    prob_ths = options['selection_prob']
    cred_set, cred_prob = find_credible_set(LD, m, start_set, bp, ths, prob_ths, allow_dup)
    
    pip = calculate_pip(m, bp)
    
    df = {'variant_index':list(range(bp)),'pip':pip, 'variant_names':options['names']}
    df = pd.DataFrame(df)
    df.to_csv(os.path.join(options['target'],'pip.csv'), index=False)
    
    cred_str = []
    for item in cred_set:
        s = ''
        for ii in item:
            s += str(ii) + ' '
        cred_str.append(s[:-1])
    
    cred_p = []
    for item in cred_prob:
        s = ''
        for ii in item:
            s += str(ii) + ' '
        cred_p.append(s)
    
    
    cred_set, cred_prob = unique_sets(cred_set, cred_prob)
    f = open(os.path.join(options['target'],'credible_set.txt'),'w')
    f.write('\n'.join(cred_str))
    f.close()
    
    
    f = open(os.path.join(options['target'],'conditional_credible_variants_probability.txt'),'w')
    f.write('\n'.join(cred_p))
    f.close()
    
    C= [np.zeros(3)]
    for i in range(len(cred_set)):
        C.append(np.random.rand(3))
    c =[C[0] for i in range(bp)]
    for ii, i in enumerate(cred_set):
        for j in i:
            c[j] = C[ii+1]
    
    plt.scatter(range(bp), pip, c=c)
    plt.xlabel('variants')
    plt.ylabel('PIP')
    plt.savefig(os.path.join(options['target'],'credible_set.pdf'))
    plt.close()
                
        
        
        
        
        
