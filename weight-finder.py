from gurobipy import *

"""
x[1]: area
x[2]: volume
x[3]: compactness
x[4]: sphericity
x[5]: diameter
x[6]: abboxx volume
x[7]: rectangularity
x[8]: eccentricity
x[9]: A3
x[10]: D1
x[11]: D2
x[12]: D3
x[13]: D4
"""

def model():

    X = [i for i in range(1, 14)]

    mdl = Model('Weights')

    x = mdl.addVars(X, vtype=GRB.CONTINUOUS, name='x')

    mdl.addConstr(quicksum(x[i] for i in X) == 1, name="Sum_to_1")
    mdl.addConstr(2 * x[1] == x[9], name="Global_vs_Histogram")

    mdl.addConstr(x[1] == 3 * x[2], name="a")
    mdl.addConstr(x[3] == 1.5 * x[2], name="b")
    mdl.addConstr(x[4] == 1.5 * x[2], name="c")
    mdl.addConstr(x[7] == 1.5 * x[2], name="d")
    mdl.addConstr(x[8] == x[1], name="e")
    mdl.addConstr(x[6] == x[3], name="f")
    mdl.addConstr(x[5] == 2 * x[2], name="g")

    mdl.addConstr(x[9] == 3 * x[10], name="h")
    mdl.addConstr(x[13] == x[9], name="i")
    mdl.addConstr(x[12] == 2.5 * x[10], name="j")
    mdl.addConstr(x[11] == 1.5 * x[10], name="k")

    mdl.optimize()

    weights = [x[i].X for i in x]

    return weights


weights = model()
print(weights)

"""
Weights:
[0.08108108108108107, 0.027027027027027025, 0.040540540540540536, 0.04054054054054054, 0.05405405405405405, 0.040540540540540536, 0.040540540540540536, 0.08108108108108107, 0.16216216216216214, 0.05405405405405405, 0.08108108108108107, 0.13513513513513511, 0.16216216216216214]
"""