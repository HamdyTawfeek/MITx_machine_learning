import numpy as np
import em
import common
import naive_em
import pandas as pd
#X = np.loadtxt("test_incomplete.txt")
#X_gold = np.loadtxt("test_complete.txt")

#K = 4
#n, d = X.shape
#seed = 0

# Question 4
# X = np.loadtxt('toy_data.txt')
# for K in range(1,5):
#         logs = []
#         for seed in range(5):
#                 mixture, post = common.init(X,K, seed)
#                 mu_s, var_s, p_s = mixture.mu, mixture.var, mixture.p
#                 mixture, post, LL = naive_em.run(X , mixture, post)
#                 common.plot(X, mixture, post, f"k:{K}, seed: {seed}")
#                 logs.append(LL)
#         print('############## K = ',K)
#         print('Log likelihood: ', np.max(logs))
 

# Question 5

X = np.loadtxt('toy_data.txt')
results = []
for K in range(1,5):
        for seed in range(5):
                mixture, post = common.init(X,K, seed)
                mu_s, var_s, p_s = mixture.mu, mixture.var, mixture.p
                mixture, post, LL = naive_em.run(X , mixture, post)
                BIC = common.bic(X, mixture, LL)
                results.append([K ,seed, BIC])


output = pd.DataFrame(results, columns=['K', 'seed', 'BIC'])
print(output)
max_bic_row = output['BIC'].idxmax()
print('Answer')
print(output.iloc[[max_bic_row]])
