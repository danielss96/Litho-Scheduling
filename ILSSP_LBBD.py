
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
import random
import json

# ----------------------------------------------------------------------------------------------------------------------
# Data, Sets and Parameters
# ----------------------------------------------------------------------------------------------------------------------
"""
instance_name = "Instance_43"

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

"""
# Data
nb_items = 60        # Number of items
nb_reticles = 60     # Number of reticles
nb_machines = 12     # Number of machines
nb_periods = 3      # Number of periods
L = 1440             # Length of periods
ST = 5             # Set up times
TT = 30
B = L * 2           # Large positive value

# Sets
items = range(0, nb_items)          # Set of items
reticles = range(0, nb_reticles)    # Set of reticles
machines = range(0, nb_machines)    # Set of machines
periods = range(0, nb_periods)      # Set of periods

# Parameters

D = [[random.randint(5, 10) for t in periods] for i in items]
PT = [random.randint(20, 25) for i in items]         # Processing times
SC = [random.randint(20, 50) for i in items]         # Setup Costs
HC = [random.randint(50, 75) for i in items]         # Holding Costs
BC = [random.randint(100, 200) for i in items]       # Backlog costs

Ib = {(m, t): set() for m in machines for t in periods}        # Empty array of Items' sets for No_Good cuts
zb = {(i, m, t): 0 for i in items for m in machines for r in reticles for t in periods}   # Empty array for No-Good cuts
setup_matrix = [[ST if r1 != r2 else 0 for r2 in reticles] for r1 in reticles]         # Setup times
transport_matrix = [[TT if m1 != m2 else 0 for m2 in machines] for m1 in machines]     # Transport times
Ri = {i: [i] for i in items}


# ----------------------------------------------------------------------------------------------------------------------
# Main Algorithm
# ----------------------------------------------------------------------------------------------------------------------
def main():

    # Print randomized Parameters
    print("Parameters:\n")
    print("D =", D)
    print("PT =", PT)
    print("SC =", SC)
    print("HC =", HC)
    print("BC =", BC, "\n")

    # Generate MILP and CP Model
    master = Model(name='Lot Sizing')
    cp = CpoModel(name='Scheduling')

    # Set Timer and Iterations Counter
    iteration = 1
    start_time = time.time()

    # Create the Master Problem as MILP (Lot sizing)
    master, x, z, setup_cost, holding_cost, backlog_cost = master_model(master)

    # Solve the Master Problem
    master_solution = solve_master(master)

    # Create the Subproblem as CP (Scheduling)
    cp, lot, stepper, reticle, makespan = subproblem_model(cp, master_solution, x, z)

    # Solve the Subproblem
    cp_solution = solve_subprobblem(cp)

    # Validate if Master Problem Solution is feasible
    infeasibility = validate_solution(master_solution, cp_solution, z, makespan, lot)

    while infeasibility:

        # Add a Benders cut to the Master Problem
        master = add_bender_cut(master, z)

        # Clean the previous CP model
        cp = CpoModel(name='Scheduling')

        # Solve the new Master Problem
        master_solution = solve_master(master)

        # Create the Subproblem with the new Master Problem Solution
        cp, lot, stepper, reticle, makespan = subproblem_model(cp, master_solution, x, z)

        # Solve the new Subproblem
        cp_solution = solve_subprobblem(cp)

        # Validate if new Master Problem Solution is feasible
        infeasibility = validate_solution(master_solution, cp_solution, z, makespan, lot)

        # Increase the iteration counter
        iteration += 1

    # Stop timer
    end_time = time.time()

    # Display Gantt Chart for the feasible solution
    dispaly_gantt(cp_solution, makespan, lot, stepper, reticle)

    # Print Results
    print("\nElapsed Time:", end_time - start_time)
    print("Total Iterations:", iteration)
    print("\nTotal Cost:", master_solution.get_objective_value())
    print("----------------------")
    print("Setup Cost:", master_solution.get_value(setup_cost))
    print("Holding Cost:", master_solution.get_value(holding_cost))
    print("Backlog Cost:", master_solution.get_value(backlog_cost))

    # Save instance parameters as .json file
    save_data()


# ----------------------------------------------------------------------------------------------------------------------
# Define the Master Problem as MILP Model
# ----------------------------------------------------------------------------------------------------------------------
def master_model(master):

    # CPLEX Solver Parameters
    master.add_progress_listener(TextProgressListener())
    master.parameters.mip.tolerances.integrality.set(1e-12)
    master.set_time_limit(320)
    master.parameters.emphasis.memory = 1

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
    param.RelativeOptimalityTolerance = 0.01
    cp.set_parameters(param)

    if master_solution:

        # Define Interval Variable
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

        # Define Sequence Variable for the reticles
        reticle = {r: sequence_var([lot[(i, m, r, t)] for m in machines for t in periods],
                                   types=[m for m in machines for _ in periods],
                                   name="R_{}".format(r)) for i in items for r in Ri[i]}

        # Objective Value
        makespan = {t: integer_var(name="I_{0}".format(t)) for t in periods}
        period_makespan = sum(makespan[t] for t in periods)
        cp.minimize(period_makespan)

        # Constraints
        for i in items:
            for m in machines:
                for t in periods:
                    for r in reticles:
                        # Calculates the makespan of each period
                        cp.add(end_of(lot[i, m, r, t]) <= makespan[t])

                        # Ensures the precense of the interval variable if is x = 1 (master Problem)
                        cp.add(presence_of(lot[i, m, r, t]) == master_solution.get_var_value(x[i, m, r, t]))

                        # Ensures the interval varaible takes place on its corresponding period
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
        print("No master solution found to solve the CP model.")


# ----------------------------------------------------------------------------------------------------------------------
# Solve the subproblem CP Model
# ----------------------------------------------------------------------------------------------------------------------
def solve_subprobblem(cp):

    # Solve Model
    cpoptimizer_path = 'C:\\Program Files\\IBM\\ILOG\\CPLEX_Studio2211\\cpoptimizer\\bin\\x64_win64\\cpoptimizer.exe'
    cp_solution = cp.solve(execfile=cpoptimizer_path)

    return cp_solution


# ----------------------------------------------------------------------------------------------------------------------
# Validate the feasibility of the Solution
# ----------------------------------------------------------------------------------------------------------------------
def validate_solution(master_solution, cp_solution, z, makespan, lot):

    # If a solution exists:
    if cp_solution:

        # Search for makespan infeasibilities (i.e. makespan[t] > L for each micro period)
        infeasibility = False
        period_cut = set()
        for t in periods:
            if cp_solution.get_value(makespan[t]) > L * (t + 1):
                print(f"Infeasible makespan[{t}] = {cp_solution.get_value(makespan[t])}\n")
                infeasibility = True
                period_cut.add(t)

        # If an infeasibility exist:
        if infeasibility:

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
                                    # zb[m][t] = sum(master_solution.get_var_value(z[i, m, t]) for i in items) - 1
                                    zb[i, m, t] = master_solution.get_var_value(z[i, m, t])
                                    print(f'zb{i, m, t}:', zb[i, m, t])
                                    break

        return infeasibility

    else:
        print("No solution found")


# ----------------------------------------------------------------------------------------------------------------------
# Display the final Solution of the Scheduling Problem
# ----------------------------------------------------------------------------------------------------------------------
def dispaly_gantt(cp_solution, makespan, lot, stepper, reticle):

    # Set Gantt parameters
    rcParams['figure.figsize'] = nb_reticles, nb_reticles * 0.8
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

    rcParams['figure.figsize'] = nb_periods * 10, nb_machines
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

    for m in machines:
        for t in periods:
            if sum(zb[i, m, t] for i in Ib[m, t]) > 0:
                master.add_constraint(sum(z[i, m, t] for i in Ib[m, t]) <= sum(zb[i, m, t] for i in Ib[m, t]) - 1,
                                      ctname="No-Good Cut")

    return master


# ----------------------------------------------------------------------------------------------------------------------
# Save the random generated data of the current instance
# ----------------------------------------------------------------------------------------------------------------------
def save_data():
    data = {
        "nb_items": nb_items,
        "nb_reticles": nb_reticles,
        "nb_machines": nb_machines,
        "nb_periods": nb_periods,
        "L": L,
        "ST": ST,
        "TT": TT,
        "B": B,
        "D": D,
        "PT": PT,
        "SC": SC,
        "HC": HC,
        "BC": BC
    }

    filename = "Instances/Instance_111.json"
    with open(filename, "w") as json_file:
        json.dump(data, json_file, indent=4)


if __name__ == '__main__':
    main()