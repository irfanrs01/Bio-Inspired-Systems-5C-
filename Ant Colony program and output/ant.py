import random
from copy import deepcopy

def initialize_pheromone(n_jobs, m_machines, initial_tau=1.0):
    return [[initial_tau for _ in range(m_machines)] for _ in range(n_jobs)]

def heuristic(job_time, machine_load):
    return 1.0 / (machine_load + job_time + 1e-9)

def choose_machine_for_job(job_idx, job_time, loads, pheromone_row, alpha, beta):
    m = len(loads)
    probs = []
    denom = 0.0
    for k in range(m):
        tau = pheromone_row[k]
        eta = heuristic(job_time, loads[k])
        val = (tau ** alpha) * (eta ** beta)
        probs.append(val)
        denom += val
    if denom == 0:
        return random.randrange(m)
    probs = [p / denom for p in probs]
    r = random.random()
    cum = 0.0
    for k, p in enumerate(probs):
        cum += p
        if r <= cum:
            return k
    return m - 1

def build_solution(jobs, m_machines, pheromone, alpha, beta):
    n_jobs = len(jobs)
    loads = [0] * m_machines
    assignment = [-1] * n_jobs
    for j in range(n_jobs):
        machine = choose_machine_for_job(j, jobs[j], loads, pheromone[j], alpha, beta)
        assignment[j] = machine
        loads[machine] += jobs[j]
    makespan = max(loads)
    return assignment, loads, makespan

def evaporate_pheromone(pheromone, rho):
    for j in range(len(pheromone)):
        for k in range(len(pheromone[0])):
            pheromone[j][k] *= (1.0 - rho)
            if pheromone[j][k] < 1e-9:
                pheromone[j][k] = 1e-9

def deposit_pheromone(pheromone, solutions, Q=1.0):
    for assignment, makespan in solutions:
        deposit = Q / makespan
        for j, k in enumerate(assignment):
            pheromone[j][k] += deposit

def aco_job_scheduling(jobs, m_machines, n_ants=10, max_iters=50, alpha=1.0, beta=2.0, rho=0.1, Q=1.0):
    pheromone = initialize_pheromone(len(jobs), m_machines)
    global_best = None

    for it in range(max_iters):
        ant_solutions = []
        for _ in range(n_ants):
            assignment, loads, makespan = build_solution(jobs, m_machines, pheromone, alpha, beta)
            ant_solutions.append((assignment, makespan))
            if global_best is None or makespan < global_best[2]:
                global_best = (deepcopy(assignment), deepcopy(loads), makespan)
        evaporate_pheromone(pheromone, rho)
        deposit_pheromone(pheromone, ant_solutions)
        # elitist reinforcement
        g_assignment, _, g_makespan = global_best
        extra = Q / g_makespan
        for j, k in enumerate(g_assignment):
            pheromone[j][k] += extra
        print(f"Iter {it+1}/{max_iters} Best makespan so far: {global_best[2]}")
    return global_best

# -------------------------------
# Take user input
# -------------------------------
if __name__ == "__main__":
    random.seed(42)
    n_jobs = int(input("Enter number of jobs: "))
    jobs = []
    for i in range(n_jobs):
        p = int(input(f"Processing time of job {i}: "))
        jobs.append(p)
    m_machines = int(input("Enter number of machines: "))

    best = aco_job_scheduling(jobs, m_machines,
                              n_ants=10,
                              max_iters=50,
                              alpha=1.0,
                              beta=2.0,
                              rho=0.1,
                              Q=1.0)

    assignment, loads, makespan = best
    print("\nFinal Best Makespan:", makespan)
    for k in range(m_machines):
        assigned_jobs = [i for i, m in enumerate(assignment) if m == k]
        print(f"Machine {k}: jobs {assigned_jobs} -> load = {sum(jobs[j] for j in assigned_jobs)}")