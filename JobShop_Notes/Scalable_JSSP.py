import collections
from ortools.sat.python import cp_model

def create_model(jobs_data, horizon=None):
    """Initialize the CP-SAT model and compute basic parameters."""
    machines_count = 1 + max(task[0] for job in jobs_data for task in job)
    all_machines = range(machines_count)
    if horizon is None:
        horizon = sum(task[1] for job in jobs_data for task in job)
    model = cp_model.CpModel()
    return model, all_machines, horizon

def create_variables(model, jobs_data, horizon):
    """Create start, end, and interval variables for each task."""
    task_type = collections.namedtuple("task_type", "start end interval")
    all_tasks = {}
    machine_to_intervals = collections.defaultdict(list)
    for job_id, job in enumerate(jobs_data):
        for task_id, task in enumerate(job):
            machine, duration = task
            suffix = f"_{job_id}_{task_id}"
            start_var = model.new_int_var(0, horizon, "start" + suffix)
            end_var = model.new_int_var(0, horizon, "end" + suffix)
            interval_var = model.new_interval_var(start_var, duration, end_var, "interval" + suffix)
            all_tasks[job_id, task_id] = task_type(start=start_var, end=end_var, interval=interval_var)
            machine_to_intervals[machine].append(interval_var)
    return all_tasks, machine_to_intervals

def add_no_overlap_constraints(model, machine_to_intervals, all_machines):
    """Add constraints to prevent overlapping tasks on the same machine."""
    # Categorical note: This could represent a monoid of non-overlapping intervals per machine.
    for machine in all_machines:
        model.add_no_overlap(machine_to_intervals[machine])

def add_precedence_constraints(model, all_tasks, jobs_data):
    """Add within-job precedence constraints."""
    # Categorical note: Morphism ensuring sequential order within job objects.
    for job_id, job in enumerate(jobs_data):
        for task_id in range(len(job) - 1):
            model.add(all_tasks[job_id, task_id + 1].start >= all_tasks[job_id, task_id].end)

def add_cross_job_precedence(model, all_tasks, cross_job_dependencies):
    """Add cross-job precedence constraints.
    
    Args:
        cross_job_dependencies: List of tuples ((job_id1, task_id1), (job_id2, task_id2)),
        where task1 must finish before task2 starts.
    """
    # Categorical note: Natural transformation between job categories.
    for (job_id1, task_id1), (job_id2, task_id2) in cross_job_dependencies:
        model.add(all_tasks[job_id2, task_id2].start >= all_tasks[job_id1, task_id1].end)

def set_objective(model, all_tasks, jobs_data):
    """Set the makespan minimization objective."""
    # Categorical note: Functor mapping task endings to a global maximum.
    horizon = sum(task[1] for job in jobs_data for task in job)
    obj_var = model.new_int_var(0, horizon, "makespan")
    model.add_max_equality(obj_var, [all_tasks[job_id, len(job) - 1].end for job_id, job in enumerate(jobs_data)])
    model.minimize(obj_var)
    return obj_var

def solve_model(model):
    """Solve the model and return solver and status."""
    solver = cp_model.CpSolver()
    status = solver.solve(model)
    return solver, status

def print_solution(solver, status, all_tasks, jobs_data, all_machines):
    """Print the optimal schedule if a solution exists."""
    if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
        print("Solution:")
        assigned_task_type = collections.namedtuple("assigned_task_type", "start job index duration")
        assigned_jobs = collections.defaultdict(list)
        for job_id, job in enumerate(jobs_data):
            for task_id, task in enumerate(job):
                machine = task[0]
                assigned_jobs[machine].append(
                    assigned_task_type(
                        start=solver.value(all_tasks[job_id, task_id].start),
                        job=job_id,
                        index=task_id,
                        duration=task[1],
                    )
                )
        output = ""
        for machine in all_machines:
            assigned_jobs[machine].sort()
            sol_line_tasks = f"Machine {machine}: "
            sol_line = "           "
            for assigned_task in assigned_jobs[machine]:
                name = f"job_{assigned_task.job}_task_{assigned_task.index}"
                sol_line_tasks += f"{name:15}"
                start = assigned_task.start
                duration = assigned_task.duration
                sol_tmp = f"[{start},{start + duration}]"
                sol_line += f"{sol_tmp:15}"
            output += sol_line_tasks + "\n" + sol_line + "\n"
        print(f"Optimal Schedule Length: {solver.objective_value}")
        print(output)
    else:
        print("No solution found.")

def get_solver_stats(solver):
    """Return solver statistics."""
    return {
        "conflicts": solver.num_conflicts,
        "branches": solver.num_branches,
        "wall_time": solver.wall_time
    }

def main(jobs_data, cross_job_dependencies=None, horizon=None):
    """Solve the JSSP with optional cross-job dependencies."""
    model, all_machines, horizon = create_model(jobs_data, horizon)
    all_tasks, machine_to_intervals = create_variables(model, jobs_data, horizon)
    add_no_overlap_constraints(model, machine_to_intervals, all_machines)
    add_precedence_constraints(model, all_tasks, jobs_data)
    if cross_job_dependencies:
        add_cross_job_precedence(model, all_tasks, cross_job_dependencies)
    set_objective(model, all_tasks, jobs_data)
    solver, status = solve_model(model)
    print_solution(solver, status, all_tasks, jobs_data, all_machines)
    stats = get_solver_stats(solver)
    print("\nStatistics:")
    print(f"  - conflicts: {stats['conflicts']}")
    print(f"  - branches : {stats['branches']}")
    print(f"  - wall time: {stats['wall_time']}s")

if __name__ == "__main__":
    jobs_data = [
        [(0, 3), (1, 2), (2, 2)],  # Job0
        [(0, 2), (2, 1), (1, 4)],  # Job1
        [(1, 4), (2, 3)],  # Job2
    ]
    # Example: Task 1 of Job 0 must finish before Task 0 of Job 1 starts
    cross_job_dependencies = [((0, 1), (1, 0))]
    main(jobs_data, cross_job_dependencies)