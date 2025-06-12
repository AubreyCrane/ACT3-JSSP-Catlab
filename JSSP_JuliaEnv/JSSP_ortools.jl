using Graphs
using DataStructures

# Define a Task struct to hold job_id, machine_id, and duration
struct Task
    job_id::Int
    machine_id::Int
    duration::Int
end

# Define JobShopProblem struct to hold the problem instance
struct JobShopProblem
    jobs::Vector{Vector{Task}}
    num_machines::Int
    num_jobs::Int
    name::String
end

# Parse JSSP data from a string (mimicking the or-tools example format)
function parse_jssp_data(data::String)
    lines = split(strip(data), "\n")
    # Skip header lines until we reach the data
    data_start = findfirst(line -> occursin(r"^\d+\s+\d+", line), lines)
    if isnothing(data_start)
        error("Invalid JSSP data format")
    end
    # Extract number of jobs and machines
    nums = parse.(Int, split(lines[data_start]))
    num_jobs, num_machines = nums[1], nums[2]
    jobs = Vector{Vector{Task}}(undef, num_jobs)
    
    # Parse each job's tasks
    for i in 1:num_jobs
        job_line = lines[data_start + i]
        values = parse.(Int, split(strip(job_line)))
        tasks = Task[]
        for j in 1:2:length(values)
            machine_id = values[j] + 1  # Adjust for 1-based indexing
            duration = values[j + 1]
            push!(tasks, Task(i - 1, machine_id, duration))
        end
        jobs[i] = tasks
    end
    
    name = occursin("instance", lines[1]) ? split(lines[1])[2] : "unnamed"
    return JobShopProblem(jobs, num_machines, num_jobs, name)
end

# Create disjunctive graph
function create_disjunctive_graph(jssp::JobShopProblem)
    # Number of tasks (vertices, excluding start/end)
    num_tasks = sum(length(job) for job in jssp.jobs)
    g = SimpleDiGraph(num_tasks + 2)  # +2 for start and end vertices
    start_vertex = num_tasks + 1
    end_vertex = num_tasks + 2
    vertex_to_task = Dict{Int, Tuple{Int, Int}}()  # Maps vertex to (job_id, task_idx)
    task_to_vertex = Dict{Tuple{Int, Int}, Int}()  # Maps (job_id, task_idx) to vertex
    vertex_weights = zeros(Int, num_tasks + 2)  # Processing times
    machine_tasks = [Vector{Tuple{Int, Int}}() for _ in 1:jssp.num_machines]  # Tasks per machine
    
    # Assign vertices to tasks and set weights
    v = 1
    for job_id in 1:jssp.num_jobs
        for (task_idx, task) in enumerate(jssp.jobs[job_id])
            vertex_to_task[v] = (job_id - 1, task_idx - 1)
            task_to_vertex[(job_id - 1, task_idx - 1)] = v
            vertex_weights[v] = task.duration
            push!(machine_tasks[task.machine_id], (job_id - 1, task_idx - 1))
            v += 1
        end
    end
    vertex_weights[start_vertex] = 0
    vertex_weights[end_vertex] = 0
    
    # Add conjunctive arcs (job precedence)
    for job_id in 1:jssp.num_jobs
        job = jssp.jobs[job_id]
        # From start to first task
        add_edge!(g, start_vertex, task_to_vertex[(job_id - 1, 0)])
        # Between consecutive tasks
        for task_idx in 1:length(job)-1
            v1 = task_to_vertex[(job_id - 1, task_idx - 1)]
            v2 = task_to_vertex[(job_id - 1, task_idx)]
            add_edge!(g, v1, v2)
        end
        # From last task to end
        add_edge!(g, task_to_vertex[(job_id - 1, length(job) - 1)], end_vertex)
    end
    
    # Collect disjunctive arcs (to be oriented later)
    disjunctive_arcs = Tuple{Int, Int, Int}[]  # (v1, v2, machine_id)
    for machine_id in 1:jssp.num_machines
        tasks = machine_tasks[machine_id]
        for i in 1:length(tasks)
            for j in i+1:length(tasks)
                v1 = task_to_vertex[tasks[i]]
                v2 = task_to_vertex[tasks[j]]
                push!(disjunctive_arcs, (v1, v2, machine_id))
                push!(disjunctive_arcs, (v2, v1, machine_id))  # Both directions
            end
        end
    end
    
    return g, start_vertex, end_vertex, vertex_to_task, task_to_vertex, vertex_weights, disjunctive_arcs, machine_tasks
end

# Compute makespan (longest path from start to end)
function compute_makespan(g::SimpleDiGraph, start_vertex::Int, end_vertex::Int, weights::Vector{Int})
    # Create a weight matrix where weight[i, j] is the weight of edge i->j, or Inf if no edge
    n = nv(g)
    weight_matrix = fill(typemax(Int), n, n)
    for e in edges(g)
        src_v = e.src
        dst_v = e.dst
        weight_matrix[src_v, dst_v] = weights[src_v]
    end
    # Set diagonal to zero
    for i in 1:n
        weight_matrix[i, i] = 0
    end
    # Use dijkstra_shortest_paths with the weight matrix
    dist = dijkstra_shortest_paths(g, start_vertex, weight_matrix).dists
    return dist[end_vertex]
end

# Simple solver: Try a sequence of job orders per machine
function solve_jssp(jssp::JobShopProblem)
    g, start_vertex, end_vertex, vertex_to_task, task_to_vertex, vertex_weights, disjunctive_arcs, machine_tasks = create_disjunctive_graph(jssp)
    
    # Group disjunctive arcs by machine
    machine_arcs = Dict{Int, Vector{Tuple{Int, Int}}}()
    for (v1, v2, machine_id) in disjunctive_arcs
        if !haskey(machine_arcs, machine_id)
            machine_arcs[machine_id] = Tuple{Int, Int}[]
        end
        push!(machine_arcs[machine_id], (v1, v2))
    end
    
    # Try a simple ordering: process jobs in order (0, 1, 2, ...)
    oriented_g = copy(g)
    for machine_id in 1:jssp.num_machines
        tasks = sort([(task_to_vertex[(job_id, task_idx)], (job_id, task_idx)) for (job_id, task_idx) in machine_tasks[machine_id]], by=x -> x[2][1])
        for i in 1:length(tasks)-1
            v1 = tasks[i][1]
            v2 = tasks[i + 1][1]
            add_edge!(oriented_g, v1, v2)
        end
    end
    
    # Check for acyclicity
    if !is_cyclic(oriented_g)
        makespan = compute_makespan(oriented_g, start_vertex, end_vertex, vertex_weights)
        return oriented_g, makespan
    else
        error("Generated schedule contains a cycle")
    end
end

# Example usage with the or-tools 3x3 instance
jssp_data = """
instance tutorial_first_jobshop_example
Simple instance of a job-shop problem in JSSP format
to illustrate the working of the or-tools library
3 3
0 3 1 2 2 2
0 2 2 1 1 4
1 4 2 3
"""
jssp = parse_jssp_data(jssp_data)
oriented_g, makespan = solve_jssp(jssp)
println("Instance: $(jssp.name)")
println("Makespan: $makespan")