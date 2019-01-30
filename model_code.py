import numpy as np
import pandas as pd

## basic implementation of static algorithm for assortment optimization

# reading and collecting data
data = pd.read_csv("WA_Sales_Products_2012-14.csv")
data.columns
category = 'Watches'
data.groupby('Product type')
data.groupby('Product type').count()
data.groupby(['Product type', 'Product']).count()
watch_data = data.loc[data['Product type']==category]
watch_data.to_csv('data_category.csv')
watch_data = watch_data.drop('Product type',1)
#define the costs column
#gross_margin = (Revenue-Cost)/Revenue => Cost = (1-gross_margin)*Revenue
watch_data['Cost'] = watch_data['Revenue']*(1-watch_data['Gross margin'])
watch_data = watch_data.drop('Gross margin',1)
watch_data.to_csv('data_cat_saved.csv')
watch_data = watch_data.drop(['Retailer country', 'Order method type', 'Retailer type',
       'Product line','Year','Quarter'],1)
aggregation_functions = {'Product': 'first', 'Revenue': 'sum', 'Quantity': 'sum', 'Cost' : 'sum'}
df = watch_data.groupby(watch_data['Product']).aggregate(aggregation_functions)
df['Gross margin'] = (df['Revenue']-df['Cost'])/df['Revenue']
df.to_csv('data_cat_to_use.csv')
df_mat = df.iloc[:,1:].values


def sort_list(list_given, order_given):
    output1 = list()
    for i in range(len(list_given)):
        output1.append(list_given[order_given[i]])
    return output1

# initialization of parameters
w_vec = df_mat[:,3] #weights w_i assigned as profit margins
w_vec = w_vec/min(w_vec) #to scale weights such that max(w_i)>=1
mu_vec = df_mat[:,1] #utilities
mu_vec = mu_vec/max(mu_vec) #scaling utilities appropriately
prod_list = df['Product'].to_list() #list of products

N = len(prod_list) #total number of products
C = 5 #capacity

#adding no choice option
a = np.zeros([1,])
w_vec = np.concatenate((a,w_vec),0)
mu_vec = np.concatenate((a,mu_vec),0)
a = list()
a.append('No choice')
prod_list = a + prod_list

v_vec = np.exp(mu_vec) #v_i = exp(mu_i)
sort_order = np.argsort(v_vec) #to keep v_i's in the increasing order
v_vec = np.array(v_vec)[sort_order]
mu_vec = np.array(mu_vec)[sort_order]
w_vec = np.array(w_vec)[sort_order]
prod_list = sort_list(prod_list,sort_order.tolist())

#the main algorithm starts here

I_mat = np.zeros([N+1,N+1]) #matrix containing intersection points
I_mat[:,:] = np.inf
I_mat[0,:] = w_vec
I_mat[0,0] = np.inf
# I_mat[:,0] = w_vec
for i in range(1,N+1):
    for j in range(i+1,N+1):
        I_mat[i,j] = (v_vec[i]*w_vec[i]-v_vec[j]*w_vec[j])/(v_vec[i]-v_vec[j])

I_mat_saved = np.copy(I_mat)

K = int((N+1)*N/2) #K = N+1 choose 2
I_sorted = np.zeros([2,K])
#vectors containing indices i_l, j_l of products such that I is in increasing order
for k in range(K):
    temp = np.where(I_mat==np.min(I_mat))
    i_l = int(temp[0])
    j_l = int(temp[1])
    I_mat[i_l,j_l] = np.inf
    I_sorted[:,k] = [i_l,j_l]

I_sorted = I_sorted.astype(int)

sigma = []
temp = np.flip(np.argsort(v_vec)).tolist()
sigma.append(temp)
G = []
G.append(sigma[0][0:C])
A = []
A.append(G[0])
B = []
B.append([])

#the looping start here
for t in range(0,K): #for all the
    i_t = I_sorted[0,t]
    j_t = I_sorted[1,t]
    if i_t!=0:
        new_sigma = sigma[t].copy()
        temp = new_sigma[i_t]
        new_sigma[i_t] = new_sigma[j_t]
        new_sigma[j_t] = temp
        sigma.append(new_sigma)
        new_B = B[t].copy()
        B.append(new_B)
        new_G = sigma[t+1].copy()
        new_G = new_G[0:C]
        G.append(new_G)
        new_A = new_G.copy()
        A.append(new_A)
    else:
        new_sigma = sigma[t].copy()
        sigma.append(new_sigma)
        new_B = B[t].copy()
        new_B.append(j_t)
        B.append(new_B)
        new_G = sigma[t+1].copy()
        new_G = new_G[0:C]
        G.append(new_G)
        new_A = G[t+1].copy()
        if j_t in new_A:
            new_A.remove(j_t)
        A.append(new_A)

#routing that computes the expected profit associated with an assortment
def compute_fS(S):
    sum_up, sum_down = 0, 0
    for j in range(0,int(len(S))):
        temp = S[j]
        sum_up += w_vec[temp]*v_vec[temp]
        sum_down += v_vec[temp]
    return sum_up/(1+sum_down)

assortment_best = A[0]
profit_best = compute_fS(assortment_best)

for i in range(1,K+1):
    profit_this = compute_fS(A[i])
    print(profit_this)
    if profit_this>profit_best:
        assortment_best = A[i]
        profit_best = compute_fS(assortment_best)

products_best = list(prod_list[i] for i in assortment_best)

print('Best assortment = ',assortment_best)
print('Best products = ', products_best)
print('Best profit = ',profit_best)
print(prod_list)
print(w_vec)