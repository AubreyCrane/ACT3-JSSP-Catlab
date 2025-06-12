# Comprehensive Job Shop Scheduling Model with CP-SAT
#
# This script implements an extended Job Shop Scheduling Problem (JSSP) using Google's OR-Tools CP-SAT solver,
# modeling a system-level scheduling scenario inspired by Air Force mission coordination. The mathematical
# framework is a constraint satisfaction problem (CSP) with multi-objective optimization, incorporating:
# - Task sequencing within jobs (linear order constraints).
# - Disjunctive constraints for machine sharing (no overlap).
# - Cumulative constraints for resource allocation.
# - Time window constraints for task scheduling.
# - Cross-job precedence constraints for inter-job dependencies.
# - Multi-objective optimization minimizing makespan and weighted tardiness.
#
# The model is designed with modularity to support compositional reasoning, aligning with category-theoretic
# principles for future translation to Julia using Catlab.jl. Key mathematical structures include:
# - **Task Sequences**: Each job is a totally ordered set of tasks, forming a chain in a poset (partially ordered set),
#   which can be modeled as a sequence of morphisms in a category where objects are tasks and morphisms are
#   precedence relations.
# - **Resource Constraints**: Resource allocation is modeled as a cumulative constraint, interpretable as a functor
#   from the category of tasks to the category of resources, mapping tasks to their resource demands.
# - **Cross-Job Dependencies**: These form a directed acyclic graph (DAG) across jobs, representable as morphisms
#   between objects in different job categories, enabling composition of scheduling subproblems.
# - **Multi-Objective Optimization**: The objective function is a weighted sum of makespan (maximum completion time)
#   and total tardiness (sum of delays past due dates), formulated as a linear combination in a constrained
#   optimization problem.
#
# **Categorical Connections for Catlab.jl**:
# - **Objects and Morphisms**: Tasks are objects, and precedence constraints (within and across jobs) are morphisms.
#   Jobs can be modeled as categories, with task sequences as composable morphisms.
# - **Functors**: Resource constraints map tasks to resource demands, acting as functors between task and resource
#   categories. In Catlab.jl, these can be represented as wiring diagrams.
# - **Monoidal Structure**: The composition of jobs and resources suggests a monoidal category, where parallel jobs
#   are tensored, and resource constraints enforce compatibility. Catlab.jl’s symmetric monoidal categories can model this.
# - **Subproblem Decomposition**: The modular structure supports decomposing the problem into subcategories (e.g., per job),
#   with solutions composed via natural transformations, implementable in Catlab.jl using functor composition.
#
# The code is structured into modular functions for data loading, model construction, solving, solution extraction,
# visualization, and profiling, facilitating translation to Julia. Visualization uses Plotly for Gantt charts, and
# diagnostics provide solver performance metrics.
#
# Dependencies: ortools, pandas, plotly
# Future Work: Translate to Catlab.jl by mapping tasks to objects, constraints to morphisms, and resources to functors,
# using wiring diagrams for composition.

######################################################################################################

import collections
from ortools.sat.python import cp_model
import pandas as pd
import plotly.express as px
from datetime import datetime, timedelta


###############  Lets reframe the job shop problem as a compostion of cateogries  ##################


# Load or generate realistic problem data
def load_data():

    #Each job is a finite sequence (totally ordered set) of tasks, denoted J_i = {T_i1, T_i2, ..., T_in},
    # where T_ij is the j-th task of job i. Each task T_ij is a tuple (m_ij, d_ij, p_ij, u_ij, w_ij, r_ij),
    # where:
    # - m_ij: Machine (resource) required, an element of a finite set M of machines.
    # - d_ij: Duration, a positive integer representing processing time.
    # - p_ij: Priority, a positive integer influencing objective weights.
    # - u_ij: Due date, a non-negative integer for tardiness calculation.
    # - w_ij: Time window, a pair (s_min, s_max) constraining the start time s_ij ∈ [s_min, s_max].
    # - r_ij: Resource demands, a map r_ij: R → ℤ_+ assigning non-negative demand to each resource in a finite set R.
    # Cross-job dependencies form a DAG, D = (V, E), where V is the set of all tasks, and E ⊆ V × V represents
    # precedence constraints (T_ik, T_jl) such that T_jl starts after T_ik completes.
    # Resources are a set R with capacities c_r ∈ ℤ_+, constraining cumulative demand at any time t.
    #
    # Categorical questions for Catlab.jl:
    # - are Jobs categories C_i, with tasks as objects and precedence as morphisms (T_ij → T_i,j+1)?
    # - are Cross-job dependencies morphisms between objects in different categories C_i → C_j?
    # - is The dataset a structured object in a product category Π_i C_i, representable in Catlab.jl as a wiring diagram?
    # - Are Tasks morphisms in a poset, forming a chain of morphisms within each job category?
    # - Resource demands can be a functor F: C_i → R, where R is a category of resources with objects as resource types (assuming first question is correctly modeled)
    # - Can we model the scheduling problem as a monoidal category, where jobs are tensored and resources are constraints?

    """
    Define job/task data mimicking Air Force mission scheduling with realistic attributes.
    """
    jobs_data = {
        'Mission_A': [
            {'task': 'PreFlightCheck', 'machine': 'Hangar1', 'duration': 2, 'priority': 2, 'due_date': 8, 'time_window': (0, 5), 'resources': {'SeniorOperator': 1}},
            {'task': 'Refuel', 'machine': 'RefuelBay1', 'duration': 3, 'priority': 1, 'due_date': 10, 'time_window': (2, 8), 'resources': {'FuelCrew': 2}},
            {'task': 'Takeoff', 'machine': 'Runway1', 'duration': 1, 'priority': 3, 'due_date': 12, 'time_window': (5, 15), 'resources': {}},
        ],
        'Mission_B': [
            {'task': 'Maintenance', 'machine': 'Hangar2', 'duration': 4, 'priority': 1, 'due_date': 10, 'time_window': (0, 7), 'resources': {'SeniorOperator': 1}},
            {'task': 'Refuel', 'machine': 'RefuelBay2', 'duration': 2, 'priority': 2, 'due_date': 12, 'time_window': (3, 10), 'resources': {'FuelCrew': 1}},
            {'task': 'Launch', 'machine': 'Runway1', 'duration': 1, 'priority': 3, 'due_date': 15, 'time_window': (5, 20), 'resources': {}},
        ],
        'Mission_C': [
            {'task': 'Briefing', 'machine': 'CommandCenter', 'duration': 1, 'priority': 1, 'due_date': 5, 'time_window': (0, 4), 'resources': {'Officer': 1}},
            {'task': 'LoadCargo', 'machine': 'Hangar1', 'duration': 3, 'priority': 2, 'due_date': 10, 'time_window': (2, 8), 'resources': {'LoadCrew': 2}},
            {'task': 'Takeoff', 'machine': 'Runway1', 'duration': 1, 'priority': 3, 'due_date': 15, 'time_window': (5, 20), 'resources': {}},
        ],
    }
    resources = {
        'SeniorOperator': 1,  # Limited senior operators
        'FuelCrew': 3,       # Multiple fuel crew members
        'LoadCrew': 2,       # Cargo loading crew
        'Officer': 2         # Command officers
    }
    cross_job_dependencies = [
        (('Mission_A', 'Refuel'), ('Mission_B', 'Launch')),  # Mission A refuels before B launches
        (('Mission_C', 'Briefing'), ('Mission_A', 'Takeoff'))  # Briefing before takeoff
    ]
    return jobs_data, resources, cross_job_dependencies

# Build the constraint model
def build_model(jobs_data, resources, cross_job_dependencies):
    """
    Construct CP-SAT model with realistic constraints and multi-objective optimization.
    """
    model = cp_model.CpModel()
    horizon = sum(task['duration'] for job in jobs_data for task in jobs_data[job])
    task_intervals = {}
    all_machines = set(task['machine'] for job in jobs_data for task in jobs_data[job])

    # Define task intervals with time windows
    for job in jobs_data:
        for task in jobs_data[job]:
            name = f"{job}_{task['task']}"
            start = model.NewIntVar(task['time_window'][0], task['time_window'][1], f'start_{name}')
            duration = task['duration']
            end = model.NewIntVar(0, horizon, f'end_{name}')
            interval = model.NewIntervalVar(start, duration, end, f'interval_{name}')
            task_intervals[(job, task['task'])] = interval

    # Sequence constraints within jobs
    # 1. **Sequence Constraints**: For each job i, tasks are ordered: e_i,j ≤ s_i,j+1 for all j.
    #    Mathematically, this forms a chain in a poset: s_i1 ≤ e_i1 ≤ s_i2 ≤ e_i2 ≤ ... ≤ e_in.
    # 2. **Machine Constraints**: For each machine m ∈ M, tasks assigned to m do not overlap.
    #    For tasks T_ij, T_kl assigned to m, either e_ij ≤ s_kl or e_kl ≤ s_ij.
    #    This is a disjunctive constraint, modeled as a no-overlap condition on intervals I_ij.
    # 3. **Resource Constraints**: For each resource r ∈ R with capacity c_r, the cumulative demand
    #    at time t, Σ_{i,j: I_ij contains t} r_ij(r), is at most c_r.
    # 4. **Time Window Constraints**: For each task T_ij, s_ij ∈ [s_min, s_max].
    # 5. **Cross-Job Dependencies**: For each (T_ik, T_jl) ∈ E, e_ik ≤ s_jl.

# Categorical note: are Tasks objects in a category, with morphisms representing precedence?
    # assuming yes, that would meant the tasks would form a chain of morphisms within a job category. (which could totally be represented with wiring diagrams in Catlab.jl)
    # - **Task Sequences**: The sequence constraints form a category C_i per job, with tasks as objects
    #   and precedence as morphisms (T_ij → T_i,j+1). In Catlab.jl, this is a finite category.
    # - **Machine Constraints**: No-overlap constraints can be modeled as a coproduct of task intervals
    #   on each machine, ensuring disjoint timelines. In Catlab.jl, use a coproduct diagram.
    # - **Resource Constraints**: The cumulative constraint is a functor F: C → R, where C is the
    #   category of all tasks, and R is the resource category with morphisms constraining total demand.
    #   In Catlab.jl, implement as a wiring diagram with resource ports.
    # - **Cross-Job Dependencies**: These are morphisms between categories C_i and C_j, representable
    #   in Catlab.jl as arrows in a product category Π_i C_i.
    # - **Objective**: The weighted sum can be modeled as a natural transformation between functors
    #   evaluating makespan and tardiness, implementable in Catlab.jl via optimization over a monoidal structure.

    for job in jobs_data:
        for i in range(len(jobs_data[job]) - 1):
            task1 = jobs_data[job][i]
            task2 = jobs_data[job][i + 1]
            model.Add(task_intervals[(job, task2['task'])].StartExpr() >= task_intervals[(job, task1['task'])].EndExpr())

    # Machine constraints: no overlap on shared machines
    for machine in all_machines:
        intervals = [task_intervals[(job, task['task'])] for job in jobs_data for task in jobs_data[job] if task['machine'] == machine]
        model.AddNoOverlap(intervals)

    # Resource constraints with skill levels
    # Categorical note: Resource allocation as a functor from tasks to resource categories.
    for resource, capacity in resources.items():
        intervals = []
        demands = []
        for job in jobs_data:
            for task in jobs_data[job]:
                if resource in task['resources']:
                    intervals.append(task_intervals[(job, task['task'])])
                    demands.append(task['resources'][resource])
        model.AddCumulative(intervals, demands, capacity)

    # Cross-job precedence constraints
    for (task1_job, task1_name), (task2_job, task2_name) in cross_job_dependencies:
        model.Add(task_intervals[(task2_job, task2_name)].StartExpr() >= task_intervals[(task1_job, task1_name)].EndExpr())

    # Multi-objective: Minimize makespan, tardiness, and resource overuse
    makespan = model.NewIntVar(0, horizon, 'makespan')
    for job in jobs_data:
        last_task = jobs_data[job][-1]
        model.Add(makespan >= task_intervals[(job, last_task['task'])].EndExpr())

    tardiness_vars = {}
    for job in jobs_data:
        for task in jobs_data[job]:
            task_end = task_intervals[(job, task['task'])].EndExpr()
            due_date = task['due_date']
            tardiness = model.NewIntVar(0, horizon, f'tardiness_{job}_{task["task"]}')
            model.Add(tardiness >= task_end - due_date)
            tardiness_vars[(job, task['task'])] = tardiness

    total_tardiness = model.NewIntVar(0, horizon * len(tardiness_vars), 'total_tardiness')
    model.Add(total_tardiness == sum(tardiness_vars.values()))

    # Objective: Weighted sum of makespan and tardiness
    model.Minimize(makespan + 3 * total_tardiness)  # Higher weight on tardiness for mission-critical tasks

    return model, task_intervals, tardiness_vars

# Solve the model
def solve_model(model):
    """
    Execute the solver with detailed status tracking.
    """
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 15.0
    status = solver.Solve(model)
    return solver, status

# Extract and print detailed solution
def extract_solution(solver, status, jobs_data, task_intervals, tardiness_vars):
    """
    Extract and display detailed schedule, tardiness, and resource usage.
    """
    if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
        solution = []
        print("\n=== Mission Schedule ===")
        print(f"Objective Value: {solver.ObjectiveValue()}")
        print(f"Makespan: {solver.Value(task_intervals[max(task_intervals, key=lambda k: solver.Value(task_intervals[k].EndExpr()))].EndExpr())}")
        print(f"Total Tardiness: {solver.Value(sum(tardiness_vars.values()))}")
        for job in jobs_data:
            print(f"\n{job}:")
            for task in jobs_data[job]:
                start = solver.Value(task_intervals[(job, task['task'])].StartExpr())
                end = solver.Value(task_intervals[(job, task['task'])].EndExpr())
                tardiness = solver.Value(tardiness_vars[(job, task['task'])])
                resources = ', '.join([f"{k}: {v}" for k, v in task['resources'].items()]) or 'None'
                print(f"  {task['task']} on {task['machine']} from {start} to {end} (Tardiness: {tardiness}, Resources: {resources})")
                solution.append({
                    'Job': job,
                    'Task': task['task'],
                    'Machine': task['machine'],
                    'Start': start,
                    'Finish': end,
                    'Tardiness': tardiness
                })
        return pd.DataFrame(solution)
    else:
        print("\nNo solution found.")
        return None

# Visualize the schedule
def visualize_solution(solution):
    """
    Generate a Gantt chart with proper time formatting and annotations.
    """
    if solution is not None:
        # Convert start/finish to datetime for better visualization
        solution['StartTime'] = solution['Start'].apply(lambda x: datetime(2025, 6, 5) + timedelta(hours=x))
        solution['EndTime'] = solution['Finish'].apply(lambda x: datetime(2025, 6, 5) + timedelta(hours=x))
        solution['TaskLabel'] = solution.apply(lambda row: f"{row['Job']}: {row['Task']} (Tardiness: {row['Tardiness']})", axis=1)
        
        fig = px.timeline(
            solution,
            x_start='StartTime',
            x_end='EndTime',
            y='Machine',
            color='Job',
            text='TaskLabel',
            title='Mission Scheduling Gantt Chart'
        )
        fig.update_traces(textposition='inside')
        fig.update_layout(xaxis_title="Time", yaxis_title="Resource", showlegend=True)
        fig.show()

# Profile solver performance
def profile_solver(solver, status):
    """
    Output detailed solver diagnostics.
    """
    status_map = {cp_model.OPTIMAL: 'Optimal', cp_model.FEASIBLE: 'Feasible', cp_model.INFEASIBLE: 'Infeasible', cp_model.UNKNOWN: 'Unknown'}
    print("\n=== Solver Diagnostics ===")
    print(f"Status: {status_map.get(status, 'Unknown')}")
    print(f"Wall Time: {solver.WallTime():.2f} seconds")
    print(f"Conflicts: {solver.NumConflicts()}")
    print(f"Branches: {solver.NumBranches()}")

# Main execution
def main():
    jobs_data, resources, cross_job_dependencies = load_data()
    model, task_intervals, tardiness_vars = build_model(jobs_data, resources, cross_job_dependencies)
    solver, status = solve_model(model)
    solution = extract_solution(solver, status, jobs_data, task_intervals, tardiness_vars)
    visualize_solution(solution)
    profile_solver(solver, status)

if __name__ == '__main__':
    main()