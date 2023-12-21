import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib
import shutil
import os
import glob
from scripts.convert_to_gpu import gpu
from scripts.convert_to_gpu_and_tensor import gpu_t
from scripts.convert_to_gpu_scalar import gpu_ts
from scripts.convert_to_cpu import cpu
import matplotlib.pyplot as plt
import pickle
import torch.nn.functional as F
from torch.autograd import Variable
import pandas as pd
import imageio
import seaborn as sn
from tqdm.auto import tqdm
import scripts.generate_credible_sets as gen_cred
import pandas as pd
import time

matplotlib.use('Agg')



def save_object(obj, filename):
    with open(filename, 'wb') as output:  # Overwrites any existing file.
        pickle.dump(obj, output, 4)
        
def load_object(filename):
    with open(filename, 'rb') as input:  # Overwrites any existing file.
        obj = pickle.load(input)
    return obj

class network(nn.Module):
    def __init__(self,K,f_dim,n_l,A,x):
        """Initialization of the neural network.
        """
        super(network, self).__init__()
        
  
        self.softmax  = nn.Softmax(dim=1)
        self.imp      =[]
        self.sig = nn.Sigmoid()
        self.rel = nn.ReLU()
        self.tan = nn.Tanh()
        
        self.sig           = nn.Sigmoid()

        
        self.L1 = nn.Linear(f_dim[0], f_dim[1], bias=False)
        self.N1 = nn.LayerNorm(K)
        self.A1 = nn.Sequential(
            nn.ReLU(),
            nn.Linear(f_dim[1], f_dim[2], bias=False)
            )
        self.N2 = nn.LayerNorm(K)
        self.A2 = nn.Sequential(
            nn.ReLU(),
            nn.Linear(f_dim[2], f_dim[3], bias=False)
            )
        self.N3 =  nn.LayerNorm(K)
        
        self.A3 = nn.Sequential(
            nn.ReLU()           
            )
        

        self.conc = nn.Linear(f_dim[n_l], 2,bias=False)

        
        self.A = A
        self.x = x
        self.degree = []
        self.variance = nn.Parameter(torch.rand(1))
        
        
    def gumbel(self,alpha,t):
        """ Generate Binary Concrete Vectors."""
        u = (-torch.log(-torch.log(gpu(torch.rand(alpha.size())))) + alpha)/t
        return F.softmax(u,dim=1)
    
        
    def forward(self, T, samples):
        """ The inference module which generates the parameters of the binary 
        concrete distribution and generate samples of binary concrete vectors.      
        """

        eps = gpu_ts(10**-7)
        X   = self.x.unsqueeze(1)        
        out = self.conc(self.A3(self.N3(self.A2(self.N2(self.A1(self.N1(self.L1(X).T).T).T).T).T).T)  )

        imp     = gpu(torch.exp(out))
        imp_o   = imp[:,1]/torch.sum(imp,dim=1)
        self.imp = cpu(imp_o.detach()).data.numpy()
        
        eps = gpu_ts(10**-6)
        if self.training:
                 z_N     = self.gumbel(torch.log(imp.repeat(samples, 1)+eps), T) 
                 z_N1    = self.gumbel(torch.log(imp.repeat(1, 1)+eps), T) 
                 z_N2    = self.gumbel(torch.log(imp.repeat(1, 1)+eps), T)
                 if torch.isnan(torch.max(z_N)):
                     print(torch.max(z_N))

                 bin_concrete =  z_N[:,1].reshape(samples,len(imp_o))
                 bin_concrete1 = z_N1[:,1].reshape(1,len(imp_o))
                 bin_concrete2 = z_N2[:,1].reshape(1,len(imp_o))
        return bin_concrete,bin_concrete1,bin_concrete2, imp_o

          
        

class finemapper():
    def __init__(self, model, opt, sch):
        self.model = model
        self.opt = opt
        self.scheduler = sch
    

    def abf(self, z, ld, memo, n_sub, sigma_sq, cc, p0, K_C, eps):

        id_sort = np.argsort(cpu(cc).data.numpy())[::-1]
        id_sort = id_sort[:K_C]
        
        cc_t = cc[list(id_sort)]
        
        ind = sorted(id_sort[cpu(torch.where(cc_t>eps)[0]).data.numpy()])
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
                
            sigma2         = torch.matmul(torch.matmul(z.T, sigma_inv),self.S)/2
        
            prior = 1 - p0
            prior[ind] = p0[ind]
        
            res =  -torch.logdet(sigma)/2 + sigma2 + torch.sum(torch.log(prior)) 
        
        
            memo[ind_m] = cpu(res).data.numpy()
        
            return res
        else:
            return
        
        
        
        
        
    def train(self, z_score, ld, temp, n_samples, sigma_sq, n_sub, p_0, num_iter,memo, epp, K_C, gamma):
        """ A training loop.
        Args:
            z_score: Normalized effect sizes obtained from GWAS summary statistics.
            ld: LD of variants present in the locus.
            temp: Temperature variable of the binary concrete distribution.
            n_sample: Number of monte carlo samples for integration.
            n_sub: Number of subjects.
            sigma_sq: Variance of causal variants.
            p_0: Prior probability of the underlying probability maps of binary concrete.
            num_iter: Number of times the training loop will run.
            memo: Dictionary that contains the probable causal configurations.
            epp: Epoch Number
            K_C: Sparsifying threshold.
            gamma: Threshold to ceates reduced set of binary vectors.
        
        Output:
            total loss
            likelihood loss
            kl loss (regularization loss)
        """
        sigma_sq =  gpu_ts(sigma_sq)
        eps = gpu_ts(10**-7)
        ll_lik=[]
        ll_kl=[]
        ll_total=[]
        M = len(z_score)
        for n_b in range(num_iter):

            Z  =  Variable(z_score)            
            LD = Variable(ld)
            
            self.opt.zero_grad()
            # sets gradient for all parameters
            for param in self.model.parameters():
                param.requires_grad = True
            
            c,c1,c2, imp = self.model(temp,n_samples)
            loss = gpu_ts(0)
            loss_f = gpu_ts(0)
            Z = Z.unsqueeze(1)
            
            lik_loss = gpu_ts(0)
            kl_loss = gpu_ts(0)
            s2 = gpu_ts(0)
            for i in range(n_samples):
                (u,ind) = torch.topk(c[i],K_C)
                ind = ind[cpu(torch.where(u>0.01)[0]).data.numpy()]
                K_C = len(ind)
                if K_C ==0:
                    return [],[],[]
                cc = c[i]
                if epp>0:      
                    self.abf(Z, ld, memo, n_sub, sigma_sq, cc, p_0, K_C, gamma)
                
                U =  n_sub*torch.diag(sigma_sq*cc)[:,ind]

                
                V = LD[ind,:]

                
                inv            = torch.inverse(gpu(torch.eye(K_C)) + torch.mm(V,U))
                
                sigma_inv      = gpu(torch.eye(M)) - torch.mm(torch.mm(U,inv),V)
                
                sigma          = gpu(torch.eye(K_C)) + torch.mm(V,U)
                

                
                sigma2         = -torch.matmul(torch.matmul(Z.T, sigma_inv),self.S)/2
                

                if torch.isnan(torch.logdet(sigma)):
                    print(torch.logdet(sigma))
                log_likelihood =  -torch.logdet(sigma)/2 + sigma2
                            
                
                lik_loss += -log_likelihood.squeeze()

                loss     += -log_likelihood.squeeze() 
                
                # Maybe sample seperately????????????
                
            # Analytic KL    
            x2 = imp[ind]
            x1 = p_0[ind]
            s1  = torch.sum(x2 * (torch.log(x2+eps) - torch.log(x1+eps)))
            s2 += torch.sum((1 - x2) * (torch.log(1 - x2+eps) - torch.log(1 - x1+eps))) + s1    
            kl_loss = s2
             
            loss_f = loss/n_samples  + kl_loss
            loss_f.backward()

            self.opt.step()
            ll_lik.append(cpu(lik_loss.detach()).data.numpy()/n_samples)
            ll_kl.append(cpu(kl_loss.detach()).data.numpy())
            ll_total.append(cpu(loss_f.detach()).data.numpy())
        
        return [np.mean(ll_total)], [np.mean(ll_lik)], [np.mean(ll_kl)]
            
 
           
        

##################################################################


def calculate_pip(M,bp):
    """Calculate posterior inclusion probabilities.
    """
    pip = np.zeros(bp)
    tot = 0
    for k in M:
        tot+= M[k]
        for i in k:
            pip[i]+= M[k]
            
    return np.squeeze(pip/tot)

def make_gif(M, bp, loc, store, ep):
    tr      = np.zeros(bp)
    tr[loc] = 1
    pip = []
    for k in M:   
        pip+= list(k)
    sn.histplot(pip, stat='count', bins=100,element='poly')
    #plt.stem(tr*(ep/2))
    plt.savefig(store+'/'+str(ep)+'.png')
    plt.close()
    
    list_of_files = filter( os.path.isfile,
                        glob.glob(store+ '/*.png') )
    # Sort list of files based on last modification time in ascending order
    filenames = sorted( list_of_files,
                        key = os.path.getmtime)

    images = []
    for filename in filenames:
        images.append(imageio.imread(filename))
    imageio.mimsave(store+'/movie.gif', images)
    return 
    
def regularize_ld(LD):
    """Regularize LD to make it non-singular
    """
    LD = (LD + LD.T)/2
    s, w = np.linalg.eig(cpu(LD).data.numpy())
    s = np.real(s)
    s_new = torch.zeros(len(s))
    if min(s)<10**-3:
        s_new = torch.ones(len(s))*(min(s)-10**-3)   
        print("\n Adding a constant {} to regularize LD".format(-min(s)+10**-3))
    LD = LD - gpu(torch.diag(s_new))    
    return LD
    
    
def reformat_memo(memo):
    m0 = np.mean([val for val in memo.values()])
    for key in memo:
        memo[key] = min(10**15,np.exp(min(np.log(10**15),memo[key]-m0)))
    return m0
    
def main(options):    
    """
    options: A dictionary of hyper-parameters. 
    
    
    """
    
    start_time = time.time()
    ###################
    # Creat a folder to store figures.
    fig_location = os.path.join(options['target'],'figures')
    if os.path.exists(fig_location):
            shutil.rmtree(fig_location)
    os.mkdir(fig_location)
    ###################
    try:
        names = list(pd.read_table(options['z'],  sep=' ', header=None).to_numpy()[:,0])
        options['names'] = names
        Z  = gpu_t(pd.read_table(options['z'],  sep=' ', header=None).to_numpy()[:,1].astype(float))
        if torch.max(Z)==torch.inf:
            print('Z vector has inf as an element, converting it to 200')
            Z[torch.where(Z==torch.inf)[0]] = 200
            
        LD = gpu_t(pd.read_table(options['LD'], sep=' ', header=None).to_numpy())
    
    except BaseException as be:
        print(be)
        return
    if LD.size()[0]!=LD.size()[1]:
        return print('\n LD is not a square matrix')
    if LD.size()[0]!=Z.size()[0]:
        return print("\n Dimension of Z and dimension of LD are not same. Dim of Z = {}, dim of LD = {}".format(list(Z.size()), list(LD.size()) ))
    
    bp  = len(Z) # Number of variants..
    LD = regularize_ld(LD) 
    
    n_sub    = gpu_ts(options['n_sub'])
    
    if len(options['loc_true'])!=0:
        loc      = options['loc_true']
    else:
        loc = []
    

    ## Hyperparamters
    n_samples = options['MCMC_samples']
    sigma_sq =  options['sigma_sq']    
    n_epochs = options['max_iter']
    temp_lower_bound = gpu_ts(options['temp_lower_bound'])
    K_C = min(bp,options['sparsity_cl'])
    gamma_sp = options['gamma']
    num_iter = 1
    
    try:
        prior_loc = options['prior_location']
        p_0 = gpu_t(pd.read_table(prior_loc,  sep=' ', header=None).to_numpy()[:,1].astype(float))
        
    except:    
        p_0 = gpu_t(np.array([1/len(Z)]*len(Z)))

    # Initialize model.
    model = gpu(network(len(Z),[1]+options['NN'],3,LD,Z)   )

    # init optimizer
    opt_j       = optim.Adam(model.parameters(), lr=0.002, betas=(0.9, 0.999), weight_decay=0)
    scheduler_j = torch.optim.lr_scheduler.StepLR(opt_j, step_size=1000, gamma = 0.5) # this will decrease the learning rate by factor of 0.1

    F_map = finemapper(model, opt_j, scheduler_j)
    F_map.S = torch.matmul(torch.inverse(LD),Z.unsqueeze(1))
    F_map.logdetLD = torch.logdet(LD)

    Loss = []
    Loss_lik=[]
    Loss_kl=[]
    memo ={}
    pip = np.zeros(len(Z))
    for n in tqdm(range(n_epochs+1)):
        temp  = torch.max(temp_lower_bound,gpu_ts(np.exp(-0.0001*n)))
        ll, ll_lik,ll_kl = F_map.train(Z, LD, temp, n_samples, sigma_sq,\
                                       n_sub, p_0, num_iter, memo, n, K_C, gamma_sp)
        F_map.scheduler.step()
        Loss.extend(ll)
        Loss_lik.extend(ll_lik)
        Loss_kl.extend(ll_kl)
        
            
        if n==n_epochs:  
            mean_memo = reformat_memo(memo)
            res_to_save={'loss':Loss,'lik_loss':Loss_lik,'kl_loss':Loss_kl, 'imp':F_map.model.imp,'loc':loc, 'pip':pip,'memo':memo, 'mean_memo':mean_memo}               
            pip = calculate_pip(memo, bp)  
            save_object(res_to_save, os.path.join(options['target'],'res'))
                
        
        if (n==(n_epochs//2) and n>0 and options['plot_loss'])or (n==n_epochs):
            real = np.zeros(len(Z))
            if len(options['loc_true'])!=0:
                real[loc] = 1
                plt.stem(real, linefmt='r-', markerfmt='ro')
            plt.stem(real, linefmt='r-', markerfmt='ro')
            plt.stem(pip)
            plt.xlabel('variants')
            plt.ylabel('PIP')
            plt.savefig(os.path.join(fig_location, 'pip.pdf'))
            plt.close()  

            plt.plot(Loss)
            plt.xlabel('epochs')
            plt.ylabel('Total Loss')
            plt.title('total loss')
            plt.savefig(os.path.join(fig_location,'total_loss.pdf'))
            plt.close()
            
            plt.plot(Loss_lik)
            plt.xlabel('epochs')
            plt.ylabel('Likelihood Loss')
            plt.title('lik loss')
            plt.savefig(os.path.join(fig_location,'lik_loss.pdf'))
            plt.close()
        
            plt.plot(Loss_kl)
            plt.title('kl loss')
            plt.xlabel('epochs')
            plt.ylabel('KL Regularization Loss')
            plt.savefig(os.path.join(fig_location,'kl_loss.pdf'))
            plt.close()

            real = np.zeros(len(Z))
            if len(options['loc_true'])!=0:
                real[loc] = 1
                plt.stem(real, linefmt='r-', markerfmt='ro')
            plt.stem(F_map.model.imp)
            plt.savefig(os.path.join(fig_location,'binary_concrete_prob.pdf'))
            plt.close()
            
            

    if options['get_cred']:
        gen_cred.main(options) 
    else:
        df = {'variant_index':list(range(bp)),'pip':pip, 'variant_names':names}
        df = pd.DataFrame(df)
        df.to_csv(os.path.join(options['target'],'pip.csv'), index=False)
    
    
    finish_time = time.time()
    
    f = open(os.path.join(options['target'],'time'),'w')
    f.write(str(finish_time-start_time))
    f.close()    



    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
