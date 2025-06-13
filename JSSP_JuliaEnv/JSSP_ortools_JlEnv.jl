using Graphs
using DataStructures
using GLMakie
using GraphMakie
using Combinatorics

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
    conjunctive_arcs = Tuple{Int, Int}[]  # Track conjunctive arcs for visualization
    
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
        push!(conjunctive_arcs, (start_vertex, task_to_vertex[(job_id - 1, 0)]))
        # Between consecutive tasks
        for task_idx in 1:length(job)-1
            v1 = task_to_vertex[(job_id - 1, task_idx - 1)]
            v2 = task_to_vertex[(job_id - 1, task_idx)]
            add_edge!(g, v1, v2)
            push!(conjunctive_arcs, (v1, v2))
        end
        # From last task to end
        add_edge!(g, task_to_vertex[(job_id - 1, length(job) - 1)], end_vertex)
        push!(conjunctive_arcs, (task_to_vertex[(job_id - 1, length(job) - 1)], end_vertex))
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
    
    # Vertex labels for visualization
    vertex_labels = ["($(vertex_to_task[v][1]),$(vertex_weights[v]))" for v in 1:num_tasks]
    push!(vertex_labels, "s")
    push!(vertex_labels, "t")

    # Improved 2D layout: jobs as rows, tasks as columns
    vertex_positions = Dict{Int, Point2f}()  # Use Point2f from Makie
    max_tasks = maximum(length.(jssp.jobs))
    for job_id in 1:jssp.num_jobs
        for (task_idx, task) in enumerate(jssp.jobs[job_id])
            v = task_to_vertex[(job_id - 1, task_idx - 1)]
            x = 1.5 + task_idx  # columns: 1,2,3,...
            y = -2.0 * (job_id - 1)  # rows: 0,-2,-4,...
            vertex_positions[v] = Point2f(x, y)
        end
    end
    # Place start and end nodes
    vertex_positions[start_vertex] = Point2f(0.0, -1.0)
    vertex_positions[end_vertex] = Point2f(max_tasks + 2.5, -1.0)
    return g, start_vertex, end_vertex, vertex_to_task, task_to_vertex, vertex_weights, disjunctive_arcs, machine_tasks, conjunctive_arcs, vertex_labels, vertex_positions
end

# Compute makespan (longest path from start to end)
function compute_makespan(g::SimpleDiGraph, start_vertex::Int, end_vertex::Int, weights::Vector{Int})
    n = nv(g)
    # Use Bellman-Ford for longest path (maximize by negating weights)
    dist = fill(-typemax(Int), n)
    dist[start_vertex] = 0
    for _ in 1:n-1
        for e in edges(g)
            src_v, dst_v = e.src, e.dst
            if dist[src_v] != -typemax(Int) && dist[src_v] + weights[src_v] > dist[dst_v]
                dist[dst_v] = dist[src_v] + weights[src_v]
            end
        end
    end
    # Check for negative cycles (not needed for makespan, but included for completeness)
    for e in edges(g)
        src_v, dst_v = e.src, e.dst
        if dist[src_v] != -typemax(Int) && dist[src_v] + weights[src_v] > dist[dst_v]
            error("Negative cycle detected")
        end
    end
    return dist[end_vertex]  # Remove negation, return the correct makespan
end

# Visualize the graph with Makie
function visualize_graph!(ax, g, vertex_labels, conjunctive_arcs, oriented_arcs, weights, vertex_positions, title="Disjunctive Graph")
    num_edges = ne(g)
    
    # Initialize edge attributes
    edge_colors = fill(RGB(0.7, 0.7, 0.7), num_edges)  # Light gray for unoriented
    edge_styles = fill(:dot, num_edges)  # Dotted for unoriented

    # Map edges to indices
    edge_index = Dict{Tuple{Int, Int}, Int}()
    for (i, e) in enumerate(edges(g))
        edge_index[(e.src, e.dst)] = i
    end

    # Assign colors and styles
    for (src, dst) in conjunctive_arcs
        if haskey(edge_index, (src, dst))
            i = edge_index[(src, dst)]
            edge_colors[i] = RGB(0, 0, 0)  # Black for conjunctive
            edge_styles[i] = :solid
        end
    end
    for (src, dst) in oriented_arcs
        if haskey(edge_index, (src, dst))
            i = edge_index[(src, dst)]
            edge_colors[i] = RGB(1, 0, 0)  # Red for oriented disjunctive
            edge_styles[i] = :solid
        end
    end

    # Update plot with fixed positions
    empty!(ax)
    ax.title = title
    hidespines!(ax)
    hidedecorations!(ax)
    positions_vec = [vertex_positions[v] for v in 1:nv(g)]
    GraphMakie.graphplot!(ax, g,
        nlabels=vertex_labels,
        nlabels_distance=10,
        nlabels_fontsize=14,
        node_size=[12 for _ in 1:nv(g)],
        node_attr=(; color=:white, strokecolor=:black, strokewidth=1.5),
        edge_color=edge_colors,
        edge_attr=(; linestyle=edge_styles),
        edge_width=[2 for _ in 1:num_edges],
        layout=positions_vec,
        edge_plottype=:beziersegments  # <-- fix here
    )
    # Add job labels on the right
    max_x = maximum(p[1] for p in values(vertex_positions))
    for job_id in 1:length(vertex_positions) - 2  # Exclude start/end
        y = -2.0 * (job_id - 1)
        Makie.text!(ax, "job $(job_id-1)", position=Point2f(max_x + 0.5, y), align=(:left, :center), fontsize=14)
    end
end

# Solver with step-by-step visualization
function solve_jssp(jssp::JobShopProblem)
    g, start_vertex, end_vertex, vertex_to_task, task_to_vertex, vertex_weights, disjunctive_arcs, machine_tasks, conjunctive_arcs, vertex_labels, vertex_positions = create_disjunctive_graph(jssp)
    
    best_makespan = typemax(Int)
    best_graph = nothing
    best_oriented_arcs = nothing
    best_orders = nothing
    
    # Initialize figure for animation outside the loop
    f = Figure(size=(800, 600))
    ax = Axis(f[1, 1], title="Initial Graph (No Disjunctive Arcs Oriented)")
    hidespines!(ax)
    hidedecorations!(ax)
    
    # Initial plot
    visualize_graph!(ax, g, vertex_labels, conjunctive_arcs, [], vertex_weights, vertex_positions, "Initial Graph (No Disjunctive Arcs Oriented)")
    
    # Record animation
    record(f, "jssp_optimization_exact.gif", Iterators.product([permutations(1:length(machine_tasks[m])) for m in 1:jssp.num_machines]...); framerate=2) do machine_perms
        oriented_g = copy(g)
        oriented_arcs = Tuple{Int, Int}[]
        is_acyclic = true
        machine_orders = Dict{Int, Vector{Int}}()  # Store job order per machine
        
        # Orient disjunctive arcs for each machine
        for (machine_id, perm) in enumerate(machine_perms)
            tasks = machine_tasks[machine_id]
            ordered_tasks = [tasks[i] for i in perm]
            machine_orders[machine_id] = [vertex_to_task[task_to_vertex[t]][1] for t in ordered_tasks]
            for i in 1:length(ordered_tasks)-1
                v1 = task_to_vertex[ordered_tasks[i]]
                v2 = task_to_vertex[ordered_tasks[i + 1]]
                add_edge!(oriented_g, v1, v2)
                push!(oriented_arcs, (v1, v2))
            end
        end
        
        # Visualize each machine's orientation step
        temp_arcs = Tuple{Int, Int}[]
        for machine_id in 1:jssp.num_machines
            tasks = machine_tasks[machine_id]
            ordered_tasks = [tasks[i] for i in machine_perms[machine_id]]
            for i in 1:length(ordered_tasks)-1
                v1 = task_to_vertex[ordered_tasks[i]]
                v2 = task_to_vertex[ordered_tasks[i + 1]]
                push!(temp_arcs, (v1, v2))
                order_str = join(["M$m: $(join(machine_orders[m], ','))" for m in 1:machine_id], "\n")
                title = "Orienting Machine $machine_id\n$order_str"
                visualize_graph!(ax, oriented_g, vertex_labels, conjunctive_arcs, temp_arcs, vertex_weights, vertex_positions, title)
                yield()
            end
        end
        
        # Check acyclicity and compute makespan
        makespan = typemax(Int)
        if !is_cyclic(oriented_g)
            makespan = compute_makespan(oriented_g, start_vertex, end_vertex, vertex_weights)
            order_str = join(["M$m: $(join(machine_orders[m], ','))" for m in 1:jssp.num_machines], "\n")
            title = "Feasible Schedule\n$order_str\nMakespan: $makespan"
            if makespan < best_makespan
                best_makespan = makespan
                best_graph = copy(oriented_g)
                best_oriented_arcs = copy(oriented_arcs)
                best_orders = deepcopy(machine_orders)
            end
        else
            order_str = join(["M$m: $(join(machine_orders[m], ','))" for m in 1:jssp.num_machines], "\n")
            title = "Infeasible Schedule (Cyclic)\n$order_str"
        end
        
        visualize_graph!(ax, oriented_g, vertex_labels, conjunctive_arcs, oriented_arcs, vertex_weights, vertex_positions, title)
        yield()
    end

    if !isnothing(best_graph)
        println("Best job orders per machine:")
        for m in 1:jssp.num_machines
            println("Machine $m: $(best_orders[m])")
        end
        return best_graph, best_makespan
    else
        error("No feasible schedule found")
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
println("Makespan: $makespan")
