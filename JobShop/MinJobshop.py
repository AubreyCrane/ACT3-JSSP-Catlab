# this is a minimal jobshop example using the Google OR-Tools CP-SAT solver. This example can be found at: https://developers.google.com/optimization/scheduling/job_shop
# comments ending in a "." are from the original example, other comments are added by me: willcrane2008@gmail.com

"""Minimal jobshop example."""
import collections
from ortools.sat.python import cp_model

def main() -> None:
    """Minimal jobshop problem."""
    # Data.
    jobs_data = [  # task = (machine_id, processing_time).
        [(0, 3), (1, 2), (2, 2)],  # Job0
        [(0, 2), (2, 1), (1, 4)],  # Job1
        [(1, 4), (2, 3)],  # Job2
    ]

    machines_count = 1 + max(task[0] for job in jobs_data for task in job)
    all_machines = range(machines_count)
    # Computes horizon dynamically as the sum of all durations.
    # This is the maximum time needed to complete all jobs.
    # It is also possible to set a fixed horizon, but this would require more knowledge about the problem.
    horizon = sum(task[1] for job in jobs_data for task in job) # since horizon is a type: int, we can can set this to 10, and no solution will be found if the jobs cannot be completed in 10 time units.

    # Create the model.
    # This is the main model that will be used to solve the job shop scheduling problem (specifically, the makespan minimization problem)
    model = cp_model.CpModel()

    # Named tuple to store information about created variables.
    task_type = collections.namedtuple("task_type", "start end interval")
    # Named tuple to manipulate solution information.
    assigned_task_type = collections.namedtuple(
        "assigned_task_type", "start job index duration"
    )

### Code is optimizing the makespan by modeling the job shop scheduling problem as a constraint satisfaction problem (CSP) ###
# code below is how the optimization works:
    # Create a list of all tasks and a mapping from machines to intervals. (this could be represented as morphisms, even though it would be overkill for this example)
    # This will hold the start, end and interval variables for each task.

    # Creates job intervals and add to the corresponding machine lists.
    all_tasks = {} # (job_id, task_id) -> task_type
    # Maps machines to their intervals.
    machine_to_intervals = collections.defaultdict(list)

    for job_id, job in enumerate(jobs_data):
        for task_id, task in enumerate(job):
            machine, duration = task
            suffix = f"_{job_id}_{task_id}"
            start_var = model.new_int_var(0, horizon, "start" + suffix)
            end_var = model.new_int_var(0, horizon, "end" + suffix)
            interval_var = model.new_interval_var( # representing the tasks duration on the machine
                start_var, duration, end_var, "interval" + suffix
            )
            all_tasks[job_id, task_id] = task_type(
                start=start_var, end=end_var, interval=interval_var
            )
            machine_to_intervals[machine].append(interval_var)

    # adding constraints to the model.

    # Create and add disjunctive constraints.
    # A machine can only work on one task at a time
    for machine in all_machines:
        model.add_no_overlap(machine_to_intervals[machine])

    # Precedences inside a job.
    # This adds constraints to ensure that each task in a job starts after the previous task ends (completed in order)
    for job_id, job in enumerate(jobs_data):
        for task_id in range(len(job) - 1):
            model.add(
                all_tasks[job_id, task_id + 1].start >= all_tasks[job_id, task_id].end
            )

    # Makespan objective.
    obj_var = model.new_int_var(0, horizon, "makespan")
    model.add_max_equality(
        obj_var,
        [all_tasks[job_id, len(job) - 1].end for job_id, job in enumerate(jobs_data)], # this is the last task of each job, so we are looking for the maximum end time in each job
    )
    model.minimize(obj_var) # solver is being told to minimize the makespan, which is the maximum end time of all jobs.

    # Creates the solver and solve.
    solver = cp_model.CpSolver() # explores possible assignments of start times to tasks to seek the assigment that results
                                    # in smallest possible makespan, while respecting the constraints defined above.
    status = solver.solve(model)

    if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
        print("Solution:")
        # Create one list of assigned tasks per machine.
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

        # Create per machine output lines.
        output = ""
        for machine in all_machines:
            # Sort by starting time.
            assigned_jobs[machine].sort()
            sol_line_tasks = "Machine " + str(machine) + ": "
            sol_line = "           "

            for assigned_task in assigned_jobs[machine]:
                name = f"job_{assigned_task.job}_task_{assigned_task.index}"
                # add spaces to output to align columns.
                sol_line_tasks += f"{name:15}"

                start = assigned_task.start
                duration = assigned_task.duration
                sol_tmp = f"[{start},{start + duration}]"
                # add spaces to output to align columns.
                sol_line += f"{sol_tmp:15}"

            sol_line += "\n"
            sol_line_tasks += "\n"
            output += sol_line_tasks
            output += sol_line

        # Finally print the solution found.
        print(f"Optimal Schedule Length: {solver.objective_value}")
        print(output)
    else:
        print("No solution found.")

    # Statistics.
    print("\nStatistics")
    print(f"  - conflicts: {solver.num_conflicts}")
    print(f"  - branches : {solver.num_branches}")
    print(f"  - wall time: {solver.wall_time}s")


if __name__ == "__main__":
    main()