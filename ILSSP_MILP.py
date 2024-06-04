# ----------------------------------------------------------------------------------------------------------------------
# Libraries
# ----------------------------------------------------------------------------------------------------------------------
from docplex.mp.model import Model
from docplex.mp.progress import *
import matplotlib.pyplot as plt
import time
import json

# ----------------------------------------------------------------------------------------------------------------------
# Settings
# ----------------------------------------------------------------------------------------------------------------------

# Read instance data from: "Instance_xx"
instance_name = "Instance_01"

# Results display:
gantt_machine = True
gantt_reticle = False

# Model Parameters:
solving_time_limit = 3600
min_opt_gap = 0
presolve = 1

# ----------------------------------------------------------------------------------------------------------------------
# Data and Sets
# ----------------------------------------------------------------------------------------------------------------------

# Read instance data
json_path = "Instances/" + instance_name + ".json"
with open(json_path, "r") as file:
    data = json.load(file)

    # Extract data from .json file
    I = data["nb_items"]
    R = data["nb_reticles"]
    M = data["nb_machines"]
    T = data["nb_periods"]
    L = data["L"]
    D = data["D"]
    B = data["B"]
    PT = data["PT"]
    SC = data["SC"]
    HC = data["HC"]
    BC = data["BC"]
    Transport = data["TT"]
    Setup = data["ST"]

# Create Sets
Items = range(0, I)         # Set of Items
Reticles = range(0, R)      # Set of Reticles
Machines = range(0, M)      # Set of Machines
Periods = range(0, T)       # Set of Periods

S = int(max(L / PT[i] for i in Items))      # Number of Positions
Positions = range(0, S)                     # Set of Positions

# Create Matrices
TT = {(m1, m2): Transport if m1 != m2 else 0 for m1 in Machines for m2 in Machines}     # Tranport Times Matrix
ST = {(r1, r2): Setup if r1 != r2 else 0 for r1 in Reticles for r2 in Reticles}         # Setup Times Matrix
Req = {i: [i] for i in Items}


# ----------------------------------------------------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------------------------------------------------
def main():

    # Initalize model and set timer
    model = Model()
    start_time = time.time()

    # Generate model
    model, x, y, z, m_start, m_end, r_start, r_end, setup_cost, holding_cost, backlog_cost = generate_model(model)
    model.print_information()
    print("\n")

    # Solve model and return total elapsed time
    solution = solve(model)
    end_time = time.time()
    print("\nElapsed Time:", end_time - start_time, "\n")

    # Print Solution
    print("Total Cost:", solution.get_objective_value(), "\n")
    print("Setup Costs", solution.get_value(setup_cost))
    print("Backlog Costs:", solution.get_value(backlog_cost))
    print("Inventory Costs:", solution.get_value(holding_cost))

    # Print Gantt charts
    if gantt_machine:
        display_machine_gantt(solution, x, m_start, m_end)

    if gantt_reticle:
        display_reticle_gantt(solution, y, r_start, r_end)


# ----------------------------------------------------------------------------------------------------------------------
# Build the model
# ----------------------------------------------------------------------------------------------------------------------
def generate_model(mdl):

    # Model Parameters
    mdl.add_progress_listener(TextProgressListener())
    mdl.set_time_limit(solving_time_limit)
    mdl.parameters.preprocessing.presolve(presolve)

    # Decision Variables
    x = {(i, m, s, t): mdl.binary_var(name="X_{0}_{1}_{2}_{3}".format(i, m, s, t)) for i in Items
         for m in Machines for s in Positions for t in Periods}
    y = {(i, r, s, t): mdl.binary_var(name="Y_{0}_{1}_{2}_{3}".format(i, r, s, t)) for i in Items
         for r in Reticles for s in Positions for t in Periods}
    z = {(i, m, t): mdl.integer_var(name="Z_{0}_{1}_{2}".format(i, m, t)) for i in Items for m in Machines
         for t in Periods}
    q = {(i, t): mdl.integer_var(name="I_{0}_{1}".format(i, t)) for i in Items for t in Periods}
    u = {(i, t): mdl.integer_var(name="U_{0}_{1}".format(i, t)) for i in Items for t in Periods}

    # Resultant variables
    m_start = {(m, s, t): mdl.continuous_var(name="m_start_{0}_{1}_{2}".format(m, s, t), lb=0) for m in Machines
               for s in Positions for t in Periods}
    m_end = {(m, s, t): mdl.continuous_var(name="m_end_{0}_{1}_{2}".format(m, s, t), lb=0) for m in Machines
             for s in Positions for t in Periods}
    r_start = {(r, s, t): mdl.continuous_var(name="r_start_{0}_{1}_{2}".format(r, s, t), lb=0) for r in Reticles
               for s in Positions for t in Periods}
    r_end = {(r, s, t): mdl.continuous_var(name="r_end_{0}_{1}_{2}".format(r, s, t), lb=0) for r in Reticles
             for s in Positions for t in Periods}

    # Decision Expressions
    setup_cost = sum(x[i, m, s, t] * SC[i] for i in Items for t in Periods for m in Machines for s in Positions)
    holding_cost = sum(q[i, t] * HC[i] for i in Items for t in Periods)
    backlog_cost = sum(u[i, t] * BC[i] for i in Items for t in Periods)
    total_cost = setup_cost + holding_cost + backlog_cost

    # Objective
    mdl.minimize(total_cost)

    # Constraints
    for i in Items:
        for t in Periods:
            if t > 0:
                mdl.add_constraint(q[i, t - 1] + sum(z[i, m, t] for m in Machines) + u[i, t] == D[i][t] + q[i, t]
                                   + u[i, t - 1], ctname="Balance")
            else:
                mdl.add_constraint(sum(z[i, m, t] for m in Machines) + u[i, t] == D[i][t] + q[i, t],
                                   ctname="Initial Balance")

    for i in Items:
        for m in Machines:
            for t in Periods:
                mdl.add_constraint(0 >= z[i, m, t] - B * sum(x[i, m, s, t] for s in Positions),
                                   ctname="Machine assignment")

    for i in Items:
        for t in Periods:
            mdl.add_constraint(sum(x[i, m, s, t] for m in Machines for s in Positions)
                               == sum(y[i, r, s, t] for r in Reticles for s in Positions),
                               ctname="Reticle assignment")

    for m in Machines:
        for s in range(1, S):
            for t in Periods:
                mdl.add_constraint(sum(x[i, m, s-1, t] for i in Items)
                                   >= sum(x[i, m, s, t] for i in Items),
                                   ctname="Machine postion sequencing")

    for m in Machines:
        for s in range(1, S):
            for t in Periods:
                mdl.add_constraint(m_start[m, s, t] >= m_end[m, s-1, t],
                                   ctname="Machine time sequencing")

    for m in Machines:
        for s in Positions:
            for t in Periods:
                mdl.add_constraint(sum(x[i, m, s, t] for i in Items) <= 1,
                                   ctname="Machine capacity")

    for r in Reticles:
        for s in Positions:
            for t in Periods:
                mdl.add_constraint(sum(y[i, r, s, t] for i in Items) <= 1,
                                   ctname="Reticle Capacity")

    for r in Reticles:
        for t in Periods:
            for s in range(1, S):
                mdl.add_constraint(sum(y[i, r, s-1, t] for i in Items)
                                   >= sum(y[i, r, s, t] for i in Items),
                                   ctname="Reticle position sequencing")
    for r in Reticles:
        for s in range(1, S):
            for t in Periods:
                mdl.add_constraint(r_start[r, s, t] >= r_end[r, s-1, t],
                                   ctname="Reticle time sequencing")

    for i in Items:
        for m in Machines:
            for s in Positions:
                for t in Periods:
                    mdl.add_constraint(m_end[m, s, t] >= m_start[m, s, t] + z[i, m, t] * PT[i] + B*(x[i, m, s, t] - 1))
                    mdl.add_constraint(m_end[m, s, t] <= m_start[m, s, t] + z[i, m, t] * PT[i] - B*(x[i, m, s, t] - 1),
                                       ctname="Processing Time")

    for m in Machines:
        for s in Positions:
            for t in Periods:
                mdl.add_constraint(m_end[m, s, t] <= L,
                                   ctname="Period time limit")

    for i in Items:
        for r in Reticles:
            for s in Positions:
                for t in Periods:
                    if r not in Req[i]:
                        mdl.add_constraint(y[i, r, s, t] <= 0,
                                           ctname="Reticle compatibility")

    # Machine - Reticle syncronization
    for i in Items:
        for m in Machines:
            for r in Reticles:
                for s1 in Positions:
                    for s2 in Positions:
                        for t in Periods:
                            mdl.add_constraint(m_start[m, s1, t] >= r_start[r, s2, t]
                                               + B * (x[i, m, s1, t] + y[i, r, s2, t] - 2))
                            mdl.add_constraint(m_start[m, s1, t] <= r_start[r, s2, t]
                                               - B * (x[i, m, s1, t] + y[i, r, s2, t] - 2))
                            mdl.add_constraint(m_end[m, s1, t] >= r_end[r, s2, t]
                                               + B * (x[i, m, s1, t] + y[i, r, s2, t] - 2))
                            mdl.add_constraint(m_end[m, s1, t] <= r_end[r, s2, t]
                                               - B * (x[i, m, s1, t] + y[i, r, s2, t] - 2))

    for s in range(1, S):
        for i in Items:
            for i2 in Items:
                for m in Machines:
                    for r1 in Reticles:
                        for r2 in Reticles:
                            for s1 in Positions:
                                for s2 in Positions:
                                    for t in Periods:
                                        mdl.add_constraint(m_start[m, s, t] >= m_end[m, s - 1, t] + ST[(r2, r1)]
                                                           - B * (4 - x[i2, m, s-1, t] - x[i, m, s, t]
                                                                    - y[i2, r2, s1, t] - y[i, r1, s2, t]),
                                                           ctname="Setup")

    for s in Positions:
        if s == 0:
            for t in range(1, T):
                for i in Items:
                    for i2 in Items:
                        for m in Machines:
                            for r1 in Reticles:
                                for r2 in Reticles:
                                    for s1 in Positions:
                                        for s2 in Positions:
                                            for s3 in Positions:
                                                mdl.add_constraint(m_start[m, s, t] >= ST[(r2, r1)]
                                                                   + m_end[m, s3, t - 1] - L
                                                                   - B * (4 - x[i2, m, s3, t - 1]
                                                                            - x[i, m, s, t]
                                                                            - y[i2, r2, s1, t - 1]
                                                                            - y[i, r1, s2, t]),
                                                                   ctname="Setup between micro periods")

    for s in range(1, S):
        for i in Items:
            for i2 in Items:
                for m1 in Machines:
                    for m2 in Machines:
                        for r in Reticles:
                            for s1 in Positions:
                                for s2 in Positions:
                                    for t in Periods:
                                        mdl.add_constraint(r_start[r, s, t] >= r_end[r, s-1, t] + TT[(m2, m1)]
                                                           - B * (4 - x[i2, m2, s1, t] - x[i, m1, s2, t]
                                                                    - y[i2, r, s-1, t] - y[i, r, s, t]),
                                                           ctname="Transportation")

    for s in Positions:
        if s == 0:
            for t in range(1, T):
                for i in Items:
                    for i2 in Items:
                        for m1 in Machines:
                            for m2 in Machines:
                                for r in Reticles:
                                    for s1 in Positions:
                                        for s2 in Positions:
                                            for s3 in Positions:
                                                mdl.add_constraint(r_start[r, s, t] >= TT[(m2, m1)]
                                                                   + r_end[r, s3, t - 1] - L
                                                                   - B * (4 - x[i2, m2, s1, t - 1]
                                                                            - x[i, m1, s2, t]
                                                                            - y[i2, r, s3, t - 1]
                                                                            - y[i, r, s, t]),
                                                                   ctname="Transportation between micro periods")

    return mdl, x, y, z, m_start, m_end, r_start, r_end, setup_cost, holding_cost, backlog_cost


# ----------------------------------------------------------------------------------------------------------------------
# Solve the model
# ----------------------------------------------------------------------------------------------------------------------
def solve(mdl):

    # Solve the model
    solution = mdl.solve()

    return solution


# ----------------------------------------------------------------------------------------------------------------------
# Display Gantt Chart for the Machines
# ----------------------------------------------------------------------------------------------------------------------
def display_machine_gantt(solution, x, m_start, m_end):

    # Print the solution
    if solution:

        # Define colors for each machine
        colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple',
                  'tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple',]

        # Create a figure and axis
        fig, ax = plt.subplots(figsize=(15, 6))

        # Plot each lot on the Gantt chart
        for i in Items:
            for m in Machines:
                for s in Positions:
                    for t in Periods:
                        start = solution.get_var_value(x[i, m, s, t]) * (L*t + solution.get_var_value(m_start[m, s, t]))
                        end = solution.get_var_value(x[i, m, s, t]) * (L*t + solution.get_var_value(m_end[m, s, t]))
                        if start >= 0 and end != 0:
                            ax.barh(m, end - start, left=start, color=colors[m], align='center', alpha=0.6)
                            ax.text((start + end) / 2, m, f'z[{i+1},{m+1},{t+1}]',
                                    ha='center', va='center', color='black')

        # Plot micro period dividers
        for t in range(T + 1):
            plt.axvline(x=t * L, color='gray', linestyle='--')

        # Set labels and title
        ax.set_xlabel('Time [min]')
        ax.set_title('Gantt Chart of Lots')
        ax.set_yticks(Machines)
        ax.set_yticklabels([f'Machine {m + 1}' for m in Machines])
        ax.invert_yaxis()  # Invert y-axis to have Machine 1 at the top

        # Show the plot
        plt.show()


# ----------------------------------------------------------------------------------------------------------------------
# Display Gantt Chart for the Reticles
# ----------------------------------------------------------------------------------------------------------------------
def display_reticle_gantt(solution, y, r_start, r_end):

    if solution:

        # Colors
        fig, ax = plt.subplots(figsize=(15, 6))
        colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple',
                  'tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple']

        # Plot each lot on the Gantt chart
        for i in Items:
            for r in Reticles:
                for s in Positions:
                    for t in Periods:
                        for m in Machines:
                            start = (solution.get_var_value(y[i, r, s, t]) *
                                     (L * t + solution.get_var_value(r_start[r, s, t])))

                            end = (solution.get_var_value(y[i, r, s, t]) *
                                   (L * t + solution.get_var_value(r_end[r, s, t])))

                            if end > 0:
                                ax.barh(r, end - start, left=start, color=colors[r], align='center', alpha=0.6)
                                ax.text((start + end) / 2, r, f'z[{i + 1},{m + 1},{t + 1}]', ha='center', va='center',
                                        color='black')

        # Plot micro period dividers
        for t in range(T + 1):
            plt.axvline(x=t * L, color='gray', linestyle='--')

        # Set labels and title
        ax.set_xlabel('Time [min]')
        ax.set_title('Gantt Chart of Lots')
        ax.set_yticks(Reticles)
        ax.set_yticklabels([f'Reticles {r + 1}' for r in Reticles])
        ax.invert_yaxis()  # Invert y-axis to have Machine 1 at the top

        # Show the plot
        plt.show()


# Main Guard
if __name__ == '__main__':
    main()
