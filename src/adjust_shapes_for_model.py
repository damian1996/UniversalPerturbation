import numpy as np
import lucid
from lucid.optvis.render import import_model
import tensorflow as tf


def get_graph_def_tensor(model, name):
    for n in model.graph_def.node:
        if n.name == name:
            return n.attr.get('value').tensor


def fix_reshapes_rainbow(model):
    # Magic.
    # Do this after load_graphdef, before import_model.
    # It assumes input shape [None, 84, 84, 4] and architecture as in rainbow!

    # for n in model.graph_def.node:
    #     print(n)

    tensor = get_graph_def_tensor(model, 'Online/softmax/Shape')
    array = np.frombuffer(tensor.tensor_content, np.int32).copy()
    array[0] = -1
    tensor.tensor_content = array.tobytes()

    tensor = get_graph_def_tensor(model, 'Online/Flatten/flatten/Shape')
    array = np.frombuffer(tensor.tensor_content, np.int32).copy()
    array[0] = -1
    tensor.tensor_content = array.tobytes()

    tensor = get_graph_def_tensor(model, 'Online/Flatten/flatten/Reshape/shape/1')
    tensor.int_val[0] = 7744

def fix_reshapes_dqn(model):
    # for n in model.graph_def.node:
    #     print(n)
    
    tensor = get_graph_def_tensor(model, 'Online/Flatten/flatten/Shape')
    array = np.frombuffer(tensor.tensor_content, np.int32).copy()
    array[0] = -1
    tensor.tensor_content = array.tobytes()

    tensor = get_graph_def_tensor(model, 'Online/Flatten/flatten/Reshape/shape/1')
    tensor.int_val[0] = 7744

def fix_reshapes_impala(model):
    None

def fix_reshapes_a2c(model):
    None

def fix_reshapes_ga(model):
    None

def fix_reshapes_es(model):
    None

def fix_reshapes_apex(model):
    None

def fix_reshapes(model, algo):
    if algo == "rainbow":
        fix_reshapes_rainbow(model)
    elif algo == "dqn":
        fix_reshapes_dqn(model)
    elif algo == "impala":
        fix_reshapes_impala(model)
    elif algo == "a2c":
        fix_reshapes_a2c(model)
    elif algo == "ga":
        fix_reshapes_ga(model)
    elif algo == "es":
        fix_reshapes_es(model)
    elif algo == "apex":
        fix_reshapes_apex(model)
