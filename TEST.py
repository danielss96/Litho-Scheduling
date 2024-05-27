# ----------------------------------------------------------------------------------------------------------------------
# Libraries
# ----------------------------------------------------------------------------------------------------------------------
from docplex.mp.model import Model
from docplex.mp.progress import *
import matplotlib.pyplot as plt
import time

# ----------------------------------------------------------------------------------------------------------------------
# Data, Sets and Parameters
# ----------------------------------------------------------------------------------------------------------------------
I = 3           # Number of items
R = 3           # Number of reticles
M = 2           # Number of machines
T = 2           # Number of periods
L = 100         # Lenght of periods
B = L * 2       # Large positive number

Items = range(0, I)         # Set of Items
Reticles = range(0, R)      # Set of Reticles
Machines = range(0, M)      # Set of Machines
Periods = range(0, T)       # Set of Periods

PT = [25, 30, 25]           # Processing Times
PC = [20, 20, 20]           # Setup Costs
IC = [50, 70, 50]           # Holding Costs
BC = [100, 200, 100]        # Backlog Costs

S = int(max(L / PT[i] for i in Items))     # Number of Positions
Positions = range(0, S)                    # Set of Positions

TT = {(m1, m2): 10 if m1 != m2 else 0 for m1 in Machines for m2 in Machines}    # Tranport Times Matrix
ST = {(r1, r2): 5 if r1 != r2 else 0 for r1 in Reticles for r2 in Reticles}     # Setup Times Matrix

D = [
    [3, 2],
    [1, 2],
    [2, 1]
]

Req = {
    0: {0},
    1: {1},
    2: {2}
}


# ----------------------------------------------------------------------------------------------------------------------
# Build the model
# ----------------------------------------------------------------------------------------------------------------------
def create_problem(model, Items, Reticles, Machines, Positions, Periods, PT, PC, IC, BC, D, Req, TT, ST):

    # Model Paramters
    model.add_progress_listener(TextProgressListener())
    model.set_time_limit(60*40)

    # Decision Variables
    x = {(i, m, s, t): model.binary_var(name="X_{0}_{1}_{2}_{3}".format(i, m, s, t)) for i in Items for m in Machines
         for s in Positions for t in Periods}
    y = {(i, r, s, t): model.binary_var(name="Y_{0}_{1}_{2}_{3}".format(i, r, s, t)) for i in Items for r in Reticles
         for s in Positions for t in Periods}
    z = {(i, m, t): model.integer_var(name="Z_{0}_{1}_{2}".format(i, m, t)) for i in Items for m in Machines
         for t in Periods}
    q = {(i, t): model.integer_var(name="I_{0}_{1}".format(i, t)) for i in Items for t in Periods}
    u = {(i, t): model.integer_var(name="U_{0}_{1}".format(i, t)) for i in Items for t in Periods}

    # Resultant variables
    M_start = {(m, s, t): model.integer_var(name="M_start_{0}_{1}_{2}".format(m, s, t), lb=0) for m in Machines
               for s in Positions for t in Periods}
    M_end = {(m, s, t): model.integer_var(name="M_end_{0}_{1}_{2}".format(m, s, t), lb=0) for m in Machines
             for s in Positions for t in Periods}
    R_start = {(r, s, t): model.integer_var(name="R_start_{0}_{1}_{2}".format(r, s, t), lb=0) for r in Reticles
               for s in Positions for t in Periods}
    R_end = {(r, s, t): model.integer_var(name="R_end_{0}_{1}_{2}".format(r, s, t), lb=0) for r in Reticles
             for s in Positions for t in Periods}

    # Decision Expressions
    Setup_Cost = model.sum(x[i, m, s, t] * PC[i] for i in Items for t in Periods for m in Machines for s in Positions)
    Inventory_Cost = model.sum(q[i, t] * IC[i] for i in Items for t in Periods)
    Backlog_Cost = model.sum(u[i, t] * BC[i] for i in Items for t in Periods)
    Total_Cost = Setup_Cost + Inventory_Cost + Backlog_Cost

    # Objective
    model.minimize(Total_Cost)

    # Constraints
    for i in Items:
        for t in Periods:
            if t == 0:
                model.add_constraint(model.sum(z[i, m, t] for m in Machines) + u[i, t] == D[i][t] + q[i, t],
                                     ctname="Item balance for period 0")
            if t >= 1:
                model.add_constraint(q[i, t-1] + model.sum(z[i, m, t] for m in Machines) + u[i, t] == D[i][t] + q[i, t]
                                     + u[i, t-1], ctname="Item balance for period >= 1")

    for i in Items:
        for m in Machines:
            for t in Periods:
                model.add_constraint(z[i, m, t] <= B * model.sum(x[i, m, s, t] for s in Positions),
                                     ctname='Ensures Lots Quantity Z[i][t] is allocated to a machine within period t')

    for m in Machines:
        for s in Positions:
            for t in Periods:
                if s > 0:
                    model.add_constraint(model.sum(x[i, m, s-1, t] for i in Items)
                                         >= model.sum(x[i, m, s, t] for i in Items),
                                         ctname="Ensures correct sequencing for machines")

    for m in Machines:
        for s in Positions:
            for t in Periods:
                if s > 0:
                    model.add_constraint(M_start[m, s, t] >= M_end[m, s-1, t],
                                         ctname="Calculates start time positions for all machines")

    for i in Items:
        for m in Machines:
            for s in Positions:
                for t in Periods:
                    model.add_constraint(M_end[m, s, t] >= M_start[m, s, t] + z[i, m, t] * PT[i] + B*(x[i, m, s, t] - 1),
                                         ctname="Calculate end time positions")

    for m in Machines:
        for s in Positions:
            for t in Periods:
                model.add_constraint(M_end[m, s, t] <= L, ctname="Ensure end time is smaller al period length")

    for m in Machines:
        for s in Positions:
            for t in Periods:
                model.add_constraint(model.sum(x[i, m, s, t] for i in Items) <= 1,
                                     ctname="Limits handle capacity of machine to one Job")

    for i in Items:
        for t in Periods:
            model.add_constraint(model.sum(x[i, m, s, t] for m in Machines for s in Positions)
                                 == model.sum(y[i, r, s, t] for r in Reticles for s in Positions),
                                 ctname="Ensures lot qty is assigned")

    for i in Items:
        for r in Reticles:
            for s in Positions:
                for t in Periods:
                    if r in Req[i]:
                        model.add_constraint(y[i, r, s, t] <= 1, ctname="Ensures reticle compatibility")
                    else:
                        model.add_constraint(y[i, r, s, t] <= 0)

    for r in Reticles:
        for s in Positions:
            for t in Periods:
                model.add_constraint(model.sum(y[i, r, s, t] for i in Items) <= 1,
                                     ctname="Limits reticle to handle one iob at a time")

    for r in Reticles:
        for t in Periods:
            for s in Positions:
                if s > 0:
                    model.add_constraint(model.sum(y[i, r, s-1, t] for i in Items)
                                         >= model.sum(y[i, r, s, t] for i in Items),
                                         ctname="Ensures correct sequencing for reticles")

    for r in Reticles:
        for t in Periods:
            for s in Positions:
                if s > 0:
                    model.add_constraint(R_start[r, s, t] >= R_end[r, s-1, t],
                                         ctname="Calculates start time of position for reticle")

    for i in Items:
        for m in Machines:
            for r in Reticles:
                for s in Positions:
                    for s1 in Positions:
                        for t in Periods:
                            model.add_constraint(M_start[m, s, t] >= R_start[r, s1, t]
                                                 + B * (x[i, m, s, t] + y[i, r, s1, t] - 2))
                            model.add_constraint(M_start[m, s, t] <= R_start[r, s1, t]
                                                 - B * (x[i, m, s, t] + y[i, r, s1, t] - 2))
                            model.add_constraint(M_end[m, s, t] >= R_end[r, s1, t]
                                                 + B * (x[i, m, s, t] + y[i, r, s1, t] - 2))
                            model.add_constraint(M_end[m, s, t] <= R_end[r, s1, t]
                                                 - B * (x[i, m, s, t] + y[i, r, s1, t] - 2))

    for i in Items:
        for i2 in Items:
            for m1 in Machines:
                for m2 in Machines:
                    for r in Reticles:
                        for s in Positions:
                            for s1 in Positions:
                                for s2 in Positions:
                                    for s3 in Positions:
                                        for t in Periods:
                                            if t != 0:
                                                if s > 0:
                                                    model.add_constraint(R_start[r, s, t] >= R_end[r, s-1, t] + TT[(m2, m1)]
                                                                         - B * (4 - x[i2, m2, s1, t] - x[i, m1, s2, t]
                                                                                  - y[i2, r, s-1, t] - y[i, r, s, t]))
                                                model.add_constraint(R_start[r, s, t] >= (TT[(m2, m1)] - L) +
                                                                     R_end[r, s3, t-1] - B * (4 - x[i2, m2, s1, t-1]
                                                                                              - x[i, m1, s2, t]
                                                                                              - y[i2, r, s3, t-1]
                                                                                              - y[i, r, s, t]))
                                            if s > 0:
                                                model.add_constraint(R_start[r, s, t] >= R_end[r, s - 1, t] + TT[(m2, m1)]
                                                                     - B * (4 - x[i2, m2, s1, t] - x[i, m1, s2, t]
                                                                            - y[i2, r, s - 1, t] - y[i, r, s, t]))
    for i in Items:
        for i2 in Items:
            for m in Machines:
                for r1 in Reticles:
                    for r2 in Reticles:
                        for s in Positions:
                            for s1 in Positions:
                                for s2 in Positions:
                                    for s3 in Positions:
                                        for t in Periods:
                                            if t != 0:
                                                if s > 0:
                                                    model.add_constraint(M_start[m, s, t] >= M_end[m, s - 1, t]
                                                                         + ST[(r2, r1)] - B * (4 - x[i2, m, s-1, t]
                                                                                               - x[i, m, s, t]
                                                                                               - y[i2, r2, s1, t]
                                                                                               - y[i, r1, s2, t]))
                                                model.add_constraint(M_start[m, s, t] >= (ST[(r2, r1)] - L)
                                                                     + M_end[m, s3, t-1] - B * (4 - x[i2, m, s3, t-1]
                                                                                                - x[i, m, s, t]
                                                                                                - y[i2, r2, s1, t-1]
                                                                                                - y[i, r1, s2, t]))
                                            if s > 0:
                                                model.add_constraint(M_start[m, s, t] >= M_end[m, s - 1, t] + ST[(r2, r1)]
                                                                     - B * (4 - x[i2, m, s-1, t] - x[i, m, s, t]
                                                                            - y[i2, r2, s1, t] - y[i, r1, s2, t]))

    return model, x, z, M_start, Setup_Cost, Inventory_Cost, Backlog_Cost


# ----------------------------------------------------------------------------------------------------------------------
# Solve the model
# ----------------------------------------------------------------------------------------------------------------------
def solve(model):

    # Set Timer
    start_time = time.time()

    # Solve the model
    solution = model.solve()

    # Get solving Time
    end_time = time.time()
    elapsed_time = end_time - start_time
    print("\nElapsed Time:", elapsed_time, "\n")

    return solution


# ----------------------------------------------------------------------------------------------------------------------
# Display solution results and Gantt
# ----------------------------------------------------------------------------------------------------------------------
def display_solution(solution, x, z, M_start, Setup_Cost, Inventory_Cost, Backlog_Cost):
    # Print the solution
    if solution:
        print("Total Cost:", solution.get_objective_value())
        print()
        print("Setup Costs", solution.get_value(Setup_Cost))
        print("Backlog Costs:", solution.get_value(Backlog_Cost))
        print("Inventory Costs:", solution.get_value(Inventory_Cost))

        # Define colors for each machine
        colors = ['tab:blue', 'tab:orange', 'tab:green']

        # Create a figure and axis
        fig, ax = plt.subplots(figsize=(15, 6))

        # Plot each lot on the Gantt chart
        for i in Items:
            for m in Machines:
                for s in Positions:
                    for t in Periods:
                        start = solution.get_var_value(x[i, m, s, t]) * (L*t + solution.get_var_value(M_start[m, s, t]))
                        end = solution.get_var_value(x[i, m, s, t]) * (L*t + solution.get_var_value(M_start[m, s, t])
                                                                       + solution.get_var_value(z[i, m, t]) * PT[i])
                        if start >= 0 and end != 0:
                            ax.barh(m, end - start, left=start, color=colors[m], align='center', alpha=0.6)
                            ax.text((start + end) / 2, m, f'z[{i+1},{m+1},{t+1}]', ha='center', va='center', color='black')

        for t in range(T + 1):
            plt.axvline(x=t * L, color='gray', linestyle='--')

        # Set labels and title
        ax.set_xlabel('Time [min]')
        ax.set_title('Gantt Chart of Lots')
        ax.set_yticks(Machines)
        ax.set_yticklabels([f'Machine {m + 1}' for m in Machines])
        ax.invert_yaxis()  # Invert y-axis to have Machine 1 at the top

        plt.savefig('Test1', dpi=300)

        # Show the plot
        plt.show()


# ----------------------------------------------------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------------------------------------------------
def main():
    model = Model()
    model, x, z, M_start, Setup_cost, Inventory_cost, Backlog_cost = create_problem(model, Items, Reticles, Machines, Positions, Periods, PT, PC, IC, BC, D, Req, TT, ST)
    model.print_information()
    print("\n")
    solution = solve(model)
    display_solution(solution, x, z, M_start, Setup_cost, Inventory_cost, Backlog_cost)


if __name__ == '__main__':
    main()
