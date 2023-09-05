########## coupled opinion-disease dynamics ############
##########Created by: Sanjukta Krishnagopal#########
##################Sept 2022#######################


# neighborhood bounded confidence models (NBCM)

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random
import copy
#from matplotlib import pyplot as plt
from matplotlib.pyplot import *
import networkx as nx
import ndlib.models.ModelConfig as mc
import ndlib.models.epidemics as ep
import ndlib.models.opinions as op
from ndlib.viz.mpl.DiffusionTrend import DiffusionTrend
from scipy.spatial import distance
import itertools
# We change the fontsize of minor ticks label 
rcParams['xtick.labelsize'] = 13
rcParams['ytick.labelsize'] = 13
rcParams['legend.title_fontsize']=13


#####opinion model algorithmic bounded confidence
def alg_bound_conf(opn,A, sigma, epsilon, zeta, disc, n, rew_strat):
     #update opinions using NBC-DW 
     #One iteration changes the opinion of N dyads/agent pairs. For each dyad
     #opinions are compared. If distance of other agent is close enough, update opinion
     #if either node has opinion too far from the other, sever edge
     #rewire edge using homophily-induced strategy
     m=len(opn)
     n_ = random.choices(np.arange(m), k=n) # select n individuals randomly
     rewired=0
     opn_updated = 0
     for i in n_:
          count=0
          setj=list(np.where(A[i,:]!=0)[0])
          if len(setj)==0:
               continue
          j = random.choices(setj)[0]
          diff = np.abs(opn[i]-opn[j])
          diff_i = np.abs(opn[i]-A[j]@opn/np.sum(A[j]))
          diff_j = np.abs(opn[j]-A[i]@opn/np.sum(A[i]))
          
          if sigma*diff+(1-sigma)*diff_i <epsilon: 
               opn[i] += gamma*(opn[j]-opn[i])
               opn_updated+=1
          if sigma*diff+(1-sigma)*diff_j <epsilon:   
               opn[j] += gamma*(opn[i]-opn[j])
               opn_updated+=1
          if min(sigma*diff+(1-sigma)*diff_i, sigma*diff+(1-sigma)*diff_j)> zeta: #or < zeta, like in Kan et al?
               A[i,j]=0
               A[j,i]=0
               rewired+=1
               i_choice = random.choice([i,j]) #pick which node to rewire
               setj_complmnt=list(np.where(A[i_choice,:]==0)[0])
               #setj_complmnt.remove(i)
               if len(setj_complmnt)==0:
                    continue
               if rew_strat == 3:
                    continue
               elif rew_strat ==2:
                    j_choice = random.choice(setj_complmnt)
               else: #rewiring based on opinion difference between the two nodes
                    dist1 = np.array([np.abs(opn[i_choice]-opn[j]) for j in setj_complmnt])
                    dist2 =  np.array([np.abs(opn[i_choice]-A[j]@opn/max(1,np.sum(A[j]))) for j in setj_complmnt])
                    dist = sigma*dist1+ (1-sigma)*dist2
                    prob = 1-np.array(dist) + 0.001
                    prob/=sum(prob)
                    j_choice = random.choices(setj_complmnt, weights=prob)[0]
               A[i_choice,j_choice]=1
               A[j_choice,i_choice]=1
     return opn, A,rewired/n, opn_updated/(2*n)

def discordant_edges(A,opn,sigma):
     edges = 0
     for i in range(len(A)):
          for j in range(i):
               diff = np.abs(opn[i]-opn[j])
               diff_i = np.abs(opn[i]-A[j]@opn/max(1,np.sum(A[j])))
               diff_j = np.abs(opn[j]-A[i]@opn/max(1,np.sum(A[i])))
               if A[i,j]!=0 and min(sigma*diff+(1-sigma)*diff_i, sigma*diff+(1-sigma)*diff_j)>zeta:
                    edges+=1
     return edges
                    

def number_groups(op_state):
     split = []
     split_ind=[]
     sorted_final_op = np.sort(op_state)
     sorted_final_op_pos = np.argsort(op_state)
     for i in range(len(sorted_final_op)-1):
          if sorted_final_op[i+1]-sorted_final_op[i] > zeta:
               split_ind.append(i+1)
               split.append((sorted_final_op[i+1] + sorted_final_op[i])/2)
     return len(split)+1


#count and pad discourdant edges array
def pad(l, content, width):
    l.extend([content] * (width - len(l)))
    return l


def disease_adj_update(A, sigma, all_changes, all_vals, sus, inf, rem,e, opn, rew_strat):
     for c,v in zip(all_changes[-1*tau], all_vals[-1*tau]):
          #if person becomes infected, randomly remove $m$ I-S connections with probability given by homophilic rules
          if v ==1:
               f = np.where(A[c]==1)[0]
               per = list(set(f).intersection(set(sus)))
               if len(per) < e:
                    n_ = per
               else:
                    if rew_strat ==2:
                         n_ = random.choices(per, k=n) #e is the number of I-S connections to be severed
                    else: #rewiring based on opinion difference between the two nodes
                         dist1 = np.array([np.abs(opn[c]-opn[j]) for j in per])
                         dist2 =  np.array([np.abs(opn[c]-A[j]@opn/max(1,np.sum(A[j]))) for j in per])
                         dist = sigma*dist1+ (1-sigma)*dist2
                         prob = np.array(dist) +0.001
                         prob/=sum(prob)
                         n_ = np.random.choice(per,size=int(e),replace=False, p=prob) 
               for j in n_:
                    A[c,j] = 0
                    A[j,c] = 0
          #if person becomes recovered, randomly add e R-R or R-S connections following strategy:
          r = np.random.random_sample()
          if v==2:
               f = list(np.where(A[c]==0)[0])
               #f.remove(c)
               per = list(set(f) - set(inf))
               if len(per) < e:
                    n_ = per
               else:
                    if rew_strat ==2:
                         n_ = random.choices(per, k=n) #e is the number of I-S connections to be severed
                    else: #rewiring based on opinion difference between the two nodes
                         dist1 = np.array([np.abs(opn[c]-opn[j]) for j in per])
                         dist2 =  np.array([np.abs(opn[c]-A[j]@opn/max(1,np.sum(A[j]))) for j in per])
                         dist = sigma*dist1+ (1-sigma)*dist2
                         prob = 1-np.array(dist) + 0.001
                         prob/=sum(prob)
                         n_ = np.random.choice(per,size=int(e),replace=False, p=prob) 
               for j in n_:
                    A[c,j] = 1
                    A[j,c] = 1
     return A



def SIR_init(A):
     f=nx.from_numpy_matrix(A)
     model_dis = ep.SIRModel(f)
     cfg = mc.Configuration()
     cfg.add_model_parameter('beta', 0.01)
     cfg.add_model_parameter('gamma', 0.05)
     cfg.add_model_parameter('alpha', 0.5)
     cfg.add_model_parameter("fraction_infected", 0.05)
     model_dis.set_initial_status(cfg)
     state = model_dis.iteration_bunch(2)
     sus = list(np.where(np.array(list(state[0]['status'].values()))==0)[0])
     inf = list(np.where(np.array(list(state[0]['status'].values()))==1)[0])
     rem = list(np.where(np.array(list(state[0]['status'].values()))==2)[0])
     changes = list(state[1]['status'].keys())
     vals = list(state[1]['status'].values())
     return model_dis,changes, vals, state, sus, inf, rem

def SIR(A, sus, inf,rem):
     f=nx.from_numpy_matrix(A)
     model_dis = ep.SIRModel(f)
     cfg = mc.Configuration()
     cfg.add_model_parameter('beta', 0.01)
     cfg.add_model_parameter('gamma', 0.05)
     cfg.add_model_parameter('alpha', 0.5)
     cfg.add_model_initial_configuration("Susceptible", sus)
     cfg.add_model_initial_configuration("Infected", inf)
     cfg.add_model_initial_configuration("Removed", rem)
     model_dis.set_initial_status(cfg)
     state = model_dis.iteration_bunch(2)
     changes = list(state[1]['status'].keys())
     vals = list(state[1]['status'].values())
     for c,v in zip(changes, vals):
          if c in sus:
               sus.remove(c)
          if c in inf:
               inf.remove(c)
          #if c in exp:
          #     exp.remove(c)
          if c in rem:
               rem.remove(c)
          if v == 0:
               sus.append(c)
          if v == 1:
               inf.append(c)
          #if v == 2:
          #     exp.append(c)
          if v == 2:
               rem.append(c)
     return model_dis,changes, vals, state, sus, inf, rem


no_trials=20
dis_changing_A = True #quantities that vary with time, only plotting for one hyperparameter value set
disc_edges=[]
frac_rew=[]
opinion_updated=[]
spectral_gap=[]
conn_comp_A=[]
assort=[]

t_max = 2000
nodes = 50
tau = 1 #time delay for changing adjacency matrix
gamma = 0.3 #how quickly opinions come closer in DW model #0.1 to 0.5 is fine 
epsilon_vals =[0.1]#np.arange(0.0,0.6,0.1) # np.arange(0.05,0.4,0.03) #between 0.1 and 0.4
zeta_vals = [0.2]#np.arange(0,1.1,0.1)# [0.2]
#zeta_vals = np.arange(0.0,1,0.1)#0.2 #distance for them to be discordant - only matters for how you count them/changes scale #zeta between 0.3 and 0.5 typically this should be > epsilon
#if zeta<=epsilon, there will always be convergence if synchronous updating - if alpha >0.5, then clusters will have at least epsilon distance between then, if alpha<0.5, then neighborhood aggreg distance willl be greater than epsilon, but distance between nodes can be anything.
#if zeta>epsilon, some non-influencing friends are likely to always be friends.
#things to plot $zeta/epsilon vs sigma$
sigma_vals = [0,0.5,1]#np.arange(0,1.1,0.1)# in [0,0.5] #between 0.4 (non-convergent) and 0.7 (convergent)
pval=0.1 #number of edges to remove and add in disease-based adaptation
seeding = np.arange(0.1,1.1,0.1) #number of interactions each iteration
n_vals = [int(nodes*0.2)]#[int(nodes*s) for s in seeding]
rew_strat = 1 #1 for homophilic rewiring, 2 for random rewiring

m = len(sigma_vals) #tochange
k = len(epsilon_vals) #tochange
tau = 1 #time delay in effect of disease on changing connectivity
no_clusters =np.zeros((m,k,no_trials)) 
convergence_times =np.zeros((m,k,no_trials))
inf_peak =np.zeros((m,k,no_trials))
inf_peak_val =np.zeros((m,k,no_trials))
dispersion_coeff =np.zeros((m,k,no_trials))
disease_len =np.zeros((m,k,no_trials))
final_op = np.zeros((m,k,no_trials,nodes))
num_clusters = np.zeros((m,k,no_trials))
size_largest_cluster = np.zeros((m,k,no_trials))
size_second_largest_cluster = np.zeros((m,k,no_trials))
size_rec = np.zeros((m,k,no_trials))

for trial in range(no_trials):
     print (trial)
     random.seed(trial)
     np.random.seed(trial)
     #run
     g = nx.erdos_renyi_graph(nodes, 0.3, seed=42)
     no_edges = len(g.edges())
     disc_edges.append([])
     frac_rew.append([])
     spectral_gap.append([])
     opinion_updated.append([])
     assort.append([])
     conn_comp_A.append([])
     opn0 = np.random.rand(nodes)
     for types,eps in itertools.product(range(len(sigma_vals)),range(len(epsilon_vals))): #tochange
          sigm = sigma_vals[types]
          epsil = epsilon_vals[eps]
          zeta = zeta_vals[0]
          n = n_vals[0]
          print (zeta,epsil)
          opn = copy.deepcopy(opn0)
          A=nx.to_numpy_array(g)
          disc=discordant_edges(A,opn,sigm)
          eigvals = np.real(np.linalg.eigvals(A))
          spectral = np.sort(eigvals)[-1]-np.sort(eigvals)[-2]
          disc_edges[-1].append([disc])
          frac_rew[-1].append([])
          opinion_updated[-1].append([])
          spectral_gap[-1].append([spectral])
          conn_comp_A[-1].append([nx.number_connected_components(g)])
          assort[-1].append([nx.degree_assortativity_coefficient(g)])
          #initialize opinions
          opinion_states=[]
          disease_states=[]
          all_changes=[]
          all_vals=[]
          count=0
          for t in range(t_max):
               #update opinion model
               opn_old=copy.deepcopy(opn)
               opn,A, frac_rewired, opn_updated= alg_bound_conf(opn, A, sigm, epsil,zeta,disc_edges[trial][types][-1],n, rew_strat)
               opn_new=copy.deepcopy(opn)
               frac_rew[trial][types].append(copy.deepcopy(frac_rewired))
               opinion_updated[trial][types].append(copy.deepcopy(opn_updated))
               disc=discordant_edges(A,opn_new,sigm)
               disc_edges[trial][types].append(copy.deepcopy(disc))
               assort[trial][types].append(nx.degree_assortativity_coefficient(nx.from_numpy_array(A)))
               conn_comp_A[trial][types].append(nx.number_connected_components(nx.from_numpy_array(A)))
               eigvals = np.real(np.linalg.eigvals(A))
               spectral = np.sort(eigvals)[-1]-np.sort(eigvals)[-2]
               spectral_gap[trial][types].append(spectral)
               #disease model
               if t==0:
                    model_dis,changes, vals,state, sus, inf, rem = SIR_init(copy.deepcopy(A))
               else:
                    model_dis,changes,vals,state,sus, inf, rem = SIR(copy.deepcopy(A), sus, inf,rem)
               all_changes.append(changes)
               all_vals.append(vals)
               disease_states.append(state[1])
               #update adjacency matrix based on disease state
               e=pval*nodes
               if dis_changing_A and t> tau: 
                    A = disease_adj_update(A, sigm,all_changes, all_vals, sus, inf, rem,e, opn, rew_strat)
               #check for termination
               opinion_states.append(opn_new)
               if np.linalg.norm(opn_old - opn_new) < 0.001:# and len(changes)==0:
               #if len(changes) ==0:
                    count+=1
               else:
                    count = 0
               if count==200:
                    convergence_times[types,eps,trial]=t-200
                    break
          trends = model_dis.build_trends(disease_states)
          disease_states=np.array(disease_states)
          opinion_states=np.array(opinion_states)
          '''max_inf = np.argmax(trends[0]['trends']['node_count'][1])
          max_inf_val = np.max(trends[0]['trends']['node_count'][1])/nodes
          dis_len = np.argmax(trends[0]['trends']['node_count'][2])
          sz_rec = trends[0]['trends']['node_count'][2][-1]/nodes
          size_rec[types,eps,trial] = sz_rec
          inf_peak[types,eps,trial] = max_inf
          inf_peak_val[types,eps,trial] = max_inf_val
          disease_len[types,eps,trial] = dis_len'''
          #no_groups[types,eps,trial] = number_groups(opinion_states[-1])
          final_op[types,eps,trial]=opn_new
          #find sizes of opinion clusters
          sort_op = np.sort(opn_new)
          s = sort_op[1:]-sort_op[:-1]
          b = list(np.where(s>epsil)[0])
          b.insert(0,0)
          b.append(nodes)
          b=np.array(b)
          sz = b[1:]-b[:-1]
          sz = np.sort(sz)[::-1]
          no_clusters[types,eps,trial] = len(sz)
          dispersion_coeff[types,eps,trial] = np.sum(sz**2)/np.sum(sz)**2
          size_largest_cluster[types,eps,trial] = sz[0]/nodes
          if len(sz)>1:
               size_second_largest_cluster[types,eps,trial] = sz[1]/nodes
          else:
               size_second_largest_cluster[types,eps,trial] = 0
          #disease trends for varying parameters:
          rcParams['xtick.labelsize'] = 16
          rcParams['ytick.labelsize'] = 16
          rcParams['legend.title_fontsize']=16
          viz = DiffusionTrend(model_dis, trends)
          figure()
          plot(np.array(trends[0]['trends']['node_count'][0])/nodes, label = 'Susceptible')
          plot(np.array(trends[0]['trends']['node_count'][1])/nodes, label = 'Infected')
          plot(np.array(trends[0]['trends']['node_count'][2])/nodes, label = 'Recovered')
          legend()
          ylabel('fraction of population', fontsize=24)
          xlabel('time', fontsize=24)
          savefig('Disease_evolution'+str(sigm)+'.jpg', bbox_inches = 'tight')
          
##########plot evolution of opinion#############
#coloring
dif = np.where(np.diff(np.sort(opinion_states[-1])) > epsil)[0]
vals = np.sort(opinion_states[-1])[dif]
vals = np.append(vals,1)
vals = np.insert(vals,0,0)

colors = plt.cm.Dark2(np.linspace(0,1,6))
nodecol = []
for n in range(nodes):
     for v in range(len(vals)-1):
          if opinion_states[-1,n] > vals[v] and opinion_states[-1,n] <=vals[v+1]:
               nodecol.append(colors[v])
               continue

rcParams['xtick.labelsize'] = 16
rcParams['ytick.labelsize'] = 16
rcParams['legend.title_fontsize']=16
figure()
for n in range(nodes):
     plot(np.linspace(0,np.shape(opinion_states)[0]-1,np.shape(opinion_states)[0]),opinion_states[:,n], color = nodecol[n])
xlabel("time", fontsize=24)
ylabel('opinion', fontsize=24)
#legend(loc="best", fontsize=18)
savefig('Opinion_evolution_'+str(trial)+'.jpg',bbox_inches='tight')


###############plot evolution of disease############

viz = DiffusionTrend(model_dis, trends)
figure()
plot(np.array(trends[0]['trends']['node_count'][0])/nodes, label = 'Susceptible')
plot(np.array(trends[0]['trends']['node_count'][1])/nodes, label = 'Infected')
plot(np.array(trends[0]['trends']['node_count'][2])/nodes, label = 'Recovered')
legend()
ylabel('fraction of individuals', fontsize=24)
xlabel('time', fontsize=24)
savefig('Disease_evolution.jpg', bbox_inches = 'tight')

rcParams['xtick.labelsize'] = 13
rcParams['ytick.labelsize'] = 13
rcParams['legend.title_fontsize']=13

#plot all quantities that don't change with time, i.e., 1 scalar value per trial


'''colors = plt.cm.viridis(np.linspace(0,1,11))

a_all = [inf_peak, inf_peak_val, disease_len,size_rec,convergence_times,no_clusters,size_largest_cluster,size_second_largest_cluster, dispersion_coeff ]
a_all = np.array(a_all)
labels = ['peak infection time', 'infection peak value', 'disease time','fraction of population infected','convergence time', 'number of opinion clusters','fraction largest opinion cluster','fraction second largest opinion cluster','dispersion coefficient']

i=0
for a in a_all:
     figure()
     for n in range(len(zeta_vals)): #tochange
          plot(np.array(epsilon_vals),np.mean(a[n,:,:], axis=1),label = round(zeta_vals[n],1), alpha = 0.7, color = colors[n]) 
          error = np.std(a[n,:,:], axis=1)/np.sqrt(no_trials)
          fill_between(np.array(epsilon_vals),np.mean(a[n,:,:], axis=1)-error,np.mean(a[n,:,:], axis=1)+error, alpha=0.2, color = colors[n])
     legend(title = r'$\zeta$',loc=1, fontsize = 13) #tochange
     ylabel(str(labels[i]), fontsize = 15)
     xlabel(r'confidence bound ($\epsilon$)', fontsize = 15)
     savefig(str(labels[i])+'_zeta_.pdf') #tochange
     i+=1'''


#plot all time varying quantities (do this only for one trial)

l=[]
for i in range(len(disc_edges)):
     l.append([len(x) for x in disc_edges[i]])
max_len = np.max(l)

vals = list(itertools.product(sigma_vals,epsilon_vals))
for ite in range(len(disc_edges)):
     for val in range(len(vals)): #it aggregates time-series for different sigma into single list
          opinion_updated[ite][val] = pad(opinion_updated[ite][val], np.nan, max_len)
          disc_edges[ite][val] = pad(disc_edges[ite][val],np.nan, max_len)
          frac_rew[ite][val] = pad(frac_rew[ite][val],np.nan, max_len)
          conn_comp_A[ite][val] = pad(conn_comp_A[ite][val],np.nan, max_len)
          spectral_gap[ite][val] = pad(spectral_gap[ite][val],np.nan, max_len)
          assort[ite][val]= pad(assort[ite][val],np.nan, max_len)


          
a_all = [disc_edges,frac_rew,spectral_gap,conn_comp_A,assort, opinion_updated]
labels = ['discordant edges','fraction of edges rewired','spectral gap','connected components','assortativity','fraction of opinions updated']

i=0
for v in range(len(a_all)):
     figure()
     for l in range(len(vals)):
          plot(np.arange(max_len), np.mean(a_all[v],axis=0)[l],label = vals[l], alpha = 0.7, color=colors[l])
          error = np.std(a_all[v],axis=0)[l]/np.sqrt(no_trials)
          fill_between(np.arange(max_len), np.mean(a_all[v],axis=0)[l]-error, np.mean(a_all[v],axis=0)[l]+error, color=colors[l], alpha=0.2)
     legend(title = r'$  \quad (\sigma, \epsilon)$',loc=1, fontsize=13)
     ylabel(str(labels[i]), fontsize = 15)
     xlabel('time', fontsize = 15)
     savefig(str(labels[i])+'_withdisease_.pdf',bbox_inches='tight')
     i+=1






     
close('all')
     
     
     
