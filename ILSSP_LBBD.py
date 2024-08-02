
# ----------------------------------------------------------------------------------------------------------------------
# Libraries
# ----------------------------------------------------------------------------------------------------------------------
from docplex.mp.model import Model
from docplex.mp.progress import *
from docplex.cp.model import *
from docplex.cp.model import CpoParameters
import docplex.cp.utils_visu as visu
from pylab import rcParams
import time
import json

# ----------------------------------------------------------------------------------------------------------------------
# Read Data
# ----------------------------------------------------------------------------------------------------------------------

# Read data from Instance:"
instance_name = "Instance_71"

# Read instance data
json_path = "Instances/" + instance_name + ".json"
with open(json_path, "r") as file:
    data = json.load(file)

    # Extract data from loaded dictionary
    nb_items = data["nb_items"]
    nb_reticles = data["nb_reticles"]
    nb_machines = data["nb_machines"]
    nb_periods = data["nb_periods"]
    L = data["L"]
    D = data["D"]
    B = data["B"]
    PT = data["PT"]
    SC = data["SC"]
    HC = data["HC"]
    BC = data["BC"]
    ST = data["ST"]
    TT = data["TT"]
    Ri = data["Ri"]

# Sets
items = range(0, nb_items)          # Set of items
reticles = range(0, nb_reticles)    # Set of reticles
machines = range(0, nb_machines)    # Set of machines
periods = range(0, nb_periods)      # Set of periods

Ib = {(m, t): set() for m in machines for t in periods}        # Empty array of Items' sets for No_Good cuts
zb = {(i, m, t): 0 for i in items for m in machines for r in reticles for t in periods}   # Empty array for No-Good cuts
setup_matrix = [[ST if r1 != r2 else 0 for r2 in reticles] for r1 in reticles]            # Setup times
transport_matrix = [[TT if m1 != m2 else 0 for m2 in machines] for m1 in machines]        # Transport times


# ----------------------------------------------------------------------------------------------------------------------
# Main Algorithm
# ----------------------------------------------------------------------------------------------------------------------
def main():

    #Start Timer
    start_time = time.time()

    # Create an empty model
    master = Model(name='Master Problem')

    # Define Master Problem
    master, x, z, setup_cost, holding_cost, backlog_cost = master_model(master)

    # Flag to stop the iterations when "True"
    feasible_solution_found = False

    # Iteration loop
    while not feasible_solution_found:

        # Solve Master Problem
        master_solution = solve_master(master)

        # Greate an empty CP model
        cp = CpoModel(name='Sub-Problem')

        # Define Sub Problem
        cp, lot, stepper, reticle, makespan = subproblem_model(cp, master_solution, x, z)

        # Solve Sub-Problem
        cp_solution = solve_subprobblem(cp)

        # Validate if Solution is feasible.
        feasible_solution_found = validate_solution(master_solution, cp_solution, z, makespan, lot)

        # Add a Benders cut to the Master Problem if solution is not feasible
        if not feasible_solution_found:
            master = add_bender_cut(master, z)

    # Stop timer
    end_time = time.time()

    # Display Gantt Chart for the feasible solution
    dispaly_gantt(cp_solution, makespan, lot, stepper, reticle)

    # Print Timer
    print("\nElapsed Time:", end_time - start_time)

    # Print Objective Value
    print("\nTotal Cost:", master_solution.get_objective_value())


# ----------------------------------------------------------------------------------------------------------------------
# Define the Master Problem as MILP Model
# ----------------------------------------------------------------------------------------------------------------------
def master_model(master):

    # CPLEX Solver Parameters
    master.add_progress_listener(TextProgressListener())    # Print Log
    master.set_time_limit()                                 # Time Limit
    master.parameters.emphasis.memory = 0                   # Memory emphasis

    # Decision Variables
    x = {(i, m, r, t): master.binary_var(name="X_{0}_{1}_{2}_{3}".format(i, m, r, t)) for i in items
         for m in machines for r in reticles for t in periods}
    z = {(i, m, t): master.integer_var(name="Z_{0}_{1}_{2}".format(i, m, t)) for i in items for m in machines
         for t in periods}
    q = {(i, t): master.integer_var(name="I_{0}_{1}".format(i, t)) for i in items for t in periods}
    u = {(i, t): master.integer_var(name="U_{0}_{1}".format(i, t)) for i in items for t in periods}

    # Decision Expression
    setup_cost = master.sum(x[i, m, r, t] * SC[i] for i in items for t in periods for m in machines for r in reticles)
    holding_cost = master.sum(q[i, t] * HC[i] for i in items for t in periods)
    backlog_cost = master.sum(u[i, t] * BC[i] for i in items for t in periods)
    total_cost = setup_cost + holding_cost + backlog_cost

    # Objective
    master.minimize(total_cost)

    # Subject to:
    # Lot size balance
    for i in items:
        for t in periods:
            if t == 0:
                master.add_constraint(master.sum(z[i, m, t] for m in machines) + u[i, t] == D[i][t] + q[i, t])
            else:
                master.add_constraint(
                    q[i, t - 1] + master.sum(z[i, m, t] for m in machines) + u[i, t] == D[i][t] + q[i, t]
                    + u[i, t - 1])

    # Ensures x = 1 if: Lot I is processed in machine m, microperiod t, with a compatible reticle r
    # B = max demand plus 25% rounded up
    for i in items:
        for m in machines:
            for t in periods:
                master.add_constraint(0 >= z[i, m, t] - B * master.sum(x[i, m, r, t] for r in Ri[i]),
                                      ctname="Lot assignment with compatible reticle")

    # Limits the time capacity of a machine
    for m in machines:
        for t in periods:
            master.add_constraint(master.sum(z[i, m, t] * PT[i] for i in items)
                                  + (sum(x[i, m, r, t] for i in items for r in reticles) - 1)
                                  * ST <= L,
                                  ctname="Machine capacity")

    # Print Model Information
    master.print_information()
    print("\n")

    return master, x, z, setup_cost, holding_cost, backlog_cost


# ----------------------------------------------------------------------------------------------------------------------
# Solve the Model
# ----------------------------------------------------------------------------------------------------------------------
def solve_master(master):

    # Solve Model
    master_solution = master.solve()

    return master_solution


# ----------------------------------------------------------------------------------------------------------------------
# Define the Subproblem Model as CP
# ----------------------------------------------------------------------------------------------------------------------
def subproblem_model(cp, master_solution, x, z):

    # CP Optimizer Parameters
    param = CpoParameters()
    param.Workers = 8                           # Number of Workers (Default 8)
    param.RelativeOptimalityTolerance = 0.0     # Optimality Gap Criteria
    cp.set_parameters(param)

    if master_solution:

        # Interval Variables
        lot = {}
        for i in items:
            for m in machines:
                for t in periods:
                    for r in reticles:
                        lot[(i, m, r, t)] = interval_var(optional=True, name="Lot_{}_{}_{}_{}".format(i, m, r, t))

        # Define Sequence Variable for the Steppers (machines)
        stepper = {m: sequence_var([lot[(i, m, r, t)] for i in items for r in reticles for t in periods],
                                   types=[r for r in reticles for _ in items for _ in periods],
                                   name="S_{}".format(m)) for m in machines}

        # Define Sequence Variable for the reticles (auxiliary resource)
        reticle = {r: sequence_var([lot[(i, m, r, t)] for m in machines for t in periods],
                                   types=[m for m in machines for _ in periods],
                                   name="R_{}".format(r)) for i in items for r in Ri[i]}

        # Decision Expresion
        makespan = {t: integer_var(name="I_{0}".format(t)) for t in periods}
        period_makespan = sum(makespan[t] for t in periods)

        # Objective
        cp.minimize(period_makespan)

        # Constraints
        for i in items:
            for m in machines:
                for t in periods:
                    for r in reticles:
                        # Calculates the makespan of each period
                        cp.add(end_of(lot[i, m, r, t]) <= makespan[t])

                        # Ensures the precense of the interval variable if is x = 1 (master Problem)
                        if master_solution.get_var_value(x[i, m, r, t]) > 0.1:
                            cp.add(presence_of(lot[i, m, r, t]) == 1)
                        else:
                            cp.add(presence_of(lot[i, m, r, t]) == 0)

                        # Ensures the interval varaible is scheduled on its corresponding period
                        cp.add(start_of(lot[i, m, r, t]) >= L * t * master_solution.get_var_value(x[i, m, r, t]))

                        # Computes the lenght of the interval variable based on the lot-size (master Problem)
                        cp.add(length_of(lot[i, m, r, t]) == PT[i] * master_solution.get_var_value(z[i, m, t])
                               * master_solution.get_var_value(x[i, m, r, t]))

        # No Overlap for machines and setup matrix
        for m in machines:
            cp.add(no_overlap(stepper[m], setup_matrix, is_direct=True))

        # No overlap for reticles and transport matrix
        for r in reticles:
            cp.add(no_overlap(reticle[r], transport_matrix, is_direct=True))

        return cp, lot, stepper, reticle, makespan

    else:
        print("No Master Problem solution")


# ----------------------------------------------------------------------------------------------------------------------
# Solve the subproblem CP Model
# ----------------------------------------------------------------------------------------------------------------------
def solve_subprobblem(cp):

    # Note: Personal path to CP-Optimizer must be written here
    cpoptimizer_path = 'C:\\Program Files\\IBM\\ILOG\\CPLEX_Studio2211\\cpoptimizer\\bin\\x64_win64\\cpoptimizer.exe'

    # Solve Sub-Problem
    cp_solution = cp.solve(execfile=cpoptimizer_path)

    return cp_solution


# ----------------------------------------------------------------------------------------------------------------------
# Validate the feasibility of the Solution
# ----------------------------------------------------------------------------------------------------------------------
def validate_solution(master_solution, cp_solution, z, makespan, lot):

    # If a solution exists:
    if cp_solution:

        # Search for makespan infeasibilities (i.e. makespan[t] > L for each micro period)
        feasible_solution_found = True
        period_cut = set()
        for t in periods:
            if cp_solution.get_value(makespan[t]) > L * (t + 1):
                print(f"Infeasible makespan[{t}] = {cp_solution.get_value(makespan[t])}\n")
                feasible_solution_found = False
                period_cut.add(t)
                break

        # If an infeasibility exist:
        if not feasible_solution_found:

            # To Identify the machine(s) that is causing the infeasible microperiod...:
            for m in machines:
                for t in period_cut:

                    # ...the lot that causes the infeasibility must be found:
                    infeasible_lot = False
                    for r in reticles:
                        for i in items:
                            lot_var = lot[(i, m, r, t)]
                            if cp_solution.get_var_solution(lot_var).is_present():
                                endtime = cp_solution.get_var_solution(lot_var).get_end()
                                if endtime > L * (t + 1):
                                    # infeasible lot identified, hence also the machine
                                    infeasible_lot = True

                    # If infeasible lot and its machine are identified:
                    if infeasible_lot:
                        for r in reticles:
                            for i in items:
                                lot_var = lot[(i, m, r, t)]
                                if cp_solution.get_var_solution(lot_var).is_present():
                                    # Create a set with the items scheduled in this machine during the microperiod
                                    Ib[m, t].add(i)
                                    # Compute the total lot size of the scheduled items -1 to generate a cut
                                    zb[i, m, t] = master_solution.get_var_value(z[i, m, t])
                                    print(f'zb{i, m, t}:', zb[i, m, t])
                                    break

        return feasible_solution_found

    else:
        print("No solution found for Sub-Problem exists")


# ----------------------------------------------------------------------------------------------------------------------
# Display the final Solution of the Scheduling Problem
# ----------------------------------------------------------------------------------------------------------------------
def dispaly_gantt(cp_solution, makespan, lot, stepper, reticle):

    # Set Gantt parameters
    # Figure size based on the size of the problem (To be done manually)
    rcParams['figure.figsize'] = 8, 25
    visu.timeline('Reticles Solution for Lot Sizing Problem')
    visu.panel('Reticles')

    # Get machine numbers
    product = {}
    for m in machines:
        for i in items:
            for r in reticles:
                for t in periods:
                    product[lot[i, m, r, t].get_name()] = m

    # Get solution of the sequencing of interval variables
    seq = {r: [] for r in reticles}
    for r in reticles:
        seq[r] = cp_solution.get_var_solution(reticle[r])
        visu.sequence(name=f"Reticle {r+1}")
        vs = seq[r].get_value()

        # Add intervals to the visulization
        for v in vs:
            nm = v.get_name()
            visu.interval(v, product[nm] + 1, str(product[nm] + 1))

        # Add setup transport matrix to the visulization
        for i in range(len(vs) - 1):
            end = vs[i].get_end()
            i1 = product[vs[i].get_name()]
            i2 = product[vs[i + 1].get_name()]
            visu.transition(end, end + transport_matrix[i1][i2])

    # Display Reticles Gantt
    visu.show()

    rcParams['figure.figsize'] = 15, 5
    visu.timeline('Machines Solution for Lot Sizing Problem')
    visu.panel('Machines')

    # Get item numbers
    product = {}
    for m in machines:
        for i in items:
            for r in reticles:
                for t in periods:
                    product[lot[i, m, r, t].get_name()] = i

    # Get solution of the sequencing of interval variables
    seq = {m: [] for m in machines}
    for m in machines:
        seq[m] = cp_solution.get_var_solution(stepper[m])
        visu.sequence(name=f"Machine {m+1}")
        vs = seq[m].get_value()

        # Add intervals to the visulization
        for v in vs:
            nm = v.get_name()
            visu.interval(v, product[nm] + 1, str(product[nm] + 1))

        # Add setup transition matrix to the visulization
        for i in range(len(vs) - 1):
            end = vs[i].get_end()
            i1 = product[vs[i].get_name()]
            i2 = product[vs[i + 1].get_name()]
            visu.transition(end, end + setup_matrix[i1][i2])

    # Display Machines Gantt
    visu.show()

    # Print solutions of the Lots
    print("\n")
    for i in items:
        for m in machines:
            for t in periods:
                for r in reticles:
                    lot_var = lot[(i, m, r, t)]
                    if cp_solution.get_var_solution(lot_var).is_present():
                        print(cp_solution.get_var_solution(lot_var))

    # Print final makespan solution
    total_makespan = max(cp_solution.get_value(makespan[t]) for t in periods)
    print("\nTotal Makespan:", total_makespan, "\n")

    for (t), var in makespan.items():
        print(f"makespan[{t}] = {cp_solution.get_value(var)}")


# ----------------------------------------------------------------------------------------------------------------------
# Add a Benders Cut to the Master Problem if Solution is unfeasible
# ----------------------------------------------------------------------------------------------------------------------
def add_bender_cut(master, z):

    # Add cut pool to the Master Problem
    for m in machines:
        for t in periods:
            if sum(zb[i, m, t] for i in Ib[m, t]) > 0:
                master.add_constraint(sum(z[i, m, t] for i in Ib[m, t]) <= sum(zb[i, m, t] for i in Ib[m, t]) - 1,
                                      ctname="Benders Cut")

    return master


if __name__ == '__main__':
    main()
