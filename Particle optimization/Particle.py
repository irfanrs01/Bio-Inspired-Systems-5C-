import random

def fitness(airfoil):
    smoothness = sum(abs(airfoil[i+1]-airfoil[i]) for i in range(len(airfoil)-1))
    camber = sum(airfoil)
    return camber - smoothness

def pso_airfoil(n_points=5, n_particles=10, max_iters=50, w=0.5, c1=1.5, c2=1.5):
    particles = [[random.uniform(0,1) for _ in range(n_points)] for _ in range(n_particles)]
    velocities = [[0 for _ in range(n_points)] for _ in range(n_particles)]
    
    pbest = particles[:]
    pbest_scores = [fitness(p) for p in particles]
    
    gbest_idx = pbest_scores.index(max(pbest_scores))
    gbest = pbest[gbest_idx][:]
    gbest_score = pbest_scores[gbest_idx]
    
    for it in range(max_iters):
        for i, particle in enumerate(particles):
            for j in range(n_points):
                r1 = random.random()
                r2 = random.random()
                velocities[i][j] = (w*velocities[i][j] +
                                    c1*r1*(pbest[i][j] - particle[j]) +
                                    c2*r2*(gbest[j] - particle[j]))
                particle[j] += velocities[i][j]
            
            f = fitness(particle)
            if f > pbest_scores[i]:
                pbest[i] = particle[:]
                pbest_scores[i] = f
                
        max_idx = pbest_scores.index(max(pbest_scores))
        if pbest_scores[max_idx] > gbest_score:
            gbest = pbest[max_idx][:]
            gbest_score = pbest_scores[max_idx]
        
        print(f"Iteration {it+1}: Best Fitness = {gbest_score:.4f}")
    
    return gbest, gbest_score

if __name__ == "__main__":
    n_points = int(input("Enter number of control points for airfoil: "))
    n_particles = int(input("Enter number of particles in swarm: "))
    max_iters = int(input("Enter number of iterations: "))

    best_airfoil, best_score = pso_airfoil(n_points=n_points,
                                           n_particles=n_particles,
                                           max_iters=max_iters)
    
    print("\nOptimized Airfoil Control Points:", best_airfoil)
    print("Best Fitness (L/D estimate):", best_score)