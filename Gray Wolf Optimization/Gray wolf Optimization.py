import numpy as np



def fake_model_accuracy(params):
    lr, layers = params
 
    accuracy = np.exp(-((lr - 0.05)**2) * 200) * np.exp(-((layers - 5)**2) / 3)
    return accuracy

def fitness_function(params):
    return 1 - fake_model_accuracy(params)  
num_wolves = 6
num_iterations = 15
dim = 2
lb = np.array([0.001, 1])
ub = np.array([0.1, 10])

wolves = np.random.uniform(lb, ub, (num_wolves, dim))
alpha, beta, delta = np.zeros(dim), np.zeros(dim), np.zeros(dim)
alpha_score, beta_score, delta_score = float('inf'), float('inf'), float('inf')

for iter in range(num_iterations):
    for i in range(num_wolves):
        fitness = fitness_function(wolves[i])
        if fitness < alpha_score:
            delta_score, delta = beta_score, beta.copy()
            beta_score, beta = alpha_score, alpha.copy()
            alpha_score, alpha = fitness, wolves[i].copy()
        elif fitness < beta_score:
            delta_score, delta = beta_score, beta.copy()
            beta_score, beta = fitness, wolves[i].copy()
        elif fitness < delta_score:
            delta_score, delta = fitness, wolves[i].copy()

    a = 2 - iter * (2 / num_iterations)
    for i in range(num_wolves):
        for j in range(dim):
            r1, r2 = np.random.rand(), np.random.rand()
            A1 = 2 * a * r1 - a
            C1 = 2 * r2
            D_alpha = abs(C1 * alpha[j] - wolves[i][j])
            X1 = alpha[j] - A1 * D_alpha

            r1, r2 = np.random.rand(), np.random.rand()
            A2 = 2 * a * r1 - a
            C2 = 2 * r2
            D_beta = abs(C2 * beta[j] - wolves[i][j])
            X2 = beta[j] - A2 * D_beta

            r1, r2 = np.random.rand(), np.random.rand()
            A3 = 2 * a * r1 - a
            C3 = 2 * r2
            D_delta = abs(C3 * delta[j] - wolves[i][j])
            X3 = delta[j] - A3 * D_delta

            wolves[i][j] = (X1 + X2 + X3) / 3

        wolves[i] = np.clip(wolves[i], lb, ub)

    best_acc = 1 - alpha_score
    print(f"Iteration {iter+1}/{num_iterations} | Best simulated accuracy: {best_acc:.4f}")

best_lr, best_layers = alpha
best_acc = 1 - alpha_score

print("\n🎯 Best Hyperparameters Found by Gray Wolf Optimization (Simulated):")
print(f"Learning Rate: {best_lr:.4f}")
print(f"Number of Layers: {best_layers:.2f}")
print(f"Simulated Best Accuracy: {best_acc:.4f}")
