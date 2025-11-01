import numpy as np
import random
import math

def random_weights(n):
    w = np.random.rand(n)
    return w / np.sum(w)

def portfolio_return(weights, returns):
    return np.dot(weights, returns)

def portfolio_risk(weights, covariance):
    return math.sqrt(np.dot(weights.T, np.dot(covariance, weights)))

def score(ret, risk):
    if risk == 0:
        return 0
    return ret / risk

def small_change(weights):
    i, j = random.sample(range(len(weights)), 2)
    delta = random.uniform(-0.1, 0.1)
    new = weights.copy()
    new[i] += delta
    new[j] -= delta
    new = np.clip(new, 0, 1)
    return new / np.sum(new)

#5 stocks example with RCM 
num_assets = 5
returns = np.array([0.11, 0.18, 0.07, 0.13, 0.15])
covariance = np.random.rand(num_assets, num_assets)
covariance = (covariance + covariance.T) / 2
np.fill_diagonal(covariance, 0.02 + np.random.rand(num_assets) * 0.03)

best_weights = random_weights(num_assets)
best_score = score(portfolio_return(best_weights, returns),
                   portfolio_risk(best_weights, covariance))

print("Initial score:", round(best_score, 4))

for iteration in range(17000):
    new_weights = small_change(best_weights)
    new_return = portfolio_return(new_weights, returns)
    new_risk = portfolio_risk(new_weights, covariance)
    new_score = score(new_return, new_risk)

    if new_score > best_score or random.random() < 0.007:
        best_weights = new_weights
        best_score = new_score

    if iteration % 1000 == 0:
        print(f"Iteration {iteration}: Score = {round(best_score, 4)}")

print("\n")
print("Best portfolio weights:", np.round(best_weights, 3))
print("Total return:", round(portfolio_return(best_weights, returns), 4))
print("Total risk:", round(portfolio_risk(best_weights, covariance), 4))
print("Return/Risk ratio:", round(best_score, 4))
