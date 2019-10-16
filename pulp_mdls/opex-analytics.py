import datetime
import random

import pulp as plp

print(f'Running PuLP Test Data {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')

print(f'Creating Variables')
n = 10
m = 5
set_I = range(1, n + 1)
set_J = range(1, m + 1)
c = {(i, j): random.normalvariate(0, 1) for i in set_I for j in set_J}
a = {(i, j): random.normalvariate(0, 5) for i in set_I for j in set_J}
l = {(i, j): random.randint(0, 10) for i in set_I for j in set_J}
u = {(i, j): random.randint(10, 20) for i in set_I for j in set_J}
b = {j: random.randint(0, 30) for j in set_J}

print(f'Done Creating Variables')

print('Creating Model')

opt_model = plp.LpProblem(name="MIP Model")

print('Creating Variables')
# if x is Continuous
x_vars_cont = {(i, j):
                   plp.LpVariable(cat=plp.LpContinuous,
                                  lowBound=l[i, j], upBound=u[i, j],
                                  name="x_{0}_{1}".format(i, j))
               for i in set_I for j in set_J}
# if x is Binary
x_vars_bin = {(i, j):
                  plp.LpVariable(cat=plp.LpBinary, name="x_{0}_{1}".format(i, j))
              for i in set_I for j in set_J}
# if x is Integer
x_vars_int = {(i, j):
                  plp.LpVariable(cat=plp.LpInteger,
                                 lowBound=l[i, j], upBound=u[i, j],
                                 name="x_{0}_{1}".format(i, j))
              for i in set_I for j in set_J}
x_vars = x_vars_int
# Less than equal constraints
# constraints = {j:
#     plp.LpConstraint(
#         e=m(a[i, j] * x_vars[i, j] for i in set_I),
#         sense=plp.plp.LpConstraintLE,
#         rhs=b[j],
#         name="constraint_{0}".format(j))
#     for j in set_J}
# >= constraints
constraints = {j:
    plp.LpConstraint(
        e=plp.lpSum(a[i, j] * x_vars[i, j] for i in set_I),
        sense=plp.LpConstraintGE,
        rhs=b[j],
        name="constraint_{0}".format(j))
    for j in set_J}
# == constraints
# constraints = {j:
#     plp.LpConstraint(
#         e=plp.lpSum(a[i, j] * x_vars[i, j] for i in set_I),
#         sense=plp.LpConstraintEQ,
#         rhs=b[j],
#         name="constraint_{0}".format(j))
#     for j in set_J}

objective = plp.lpSum(x_vars[i,j] * c[i,j]
                    for i in set_I
                    for j in set_J)

# for maximization
opt_model.sense = plp.LpMaximize
# for minimization
opt_model.sense = plp.LpMinimize
opt_model.setObjective(objective)

# solving with CBC
opt_model.solve()
# solving with Glpk
# opt_model.solve(solver=plp.)