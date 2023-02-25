import numpy as np
from time import perf_counter
import json
import ap_features as apf

def cart2pol(x, y):
    radius = np.sqrt(x**2 + y**2)
    angle = np.arctan2(y, x)
    return radius, angle

def pol2cart(angles, radius):
    x = radius * np.cos(angles)
    y = radius * np.sin(angles)
    return x, y

def load_model(path):
    f = open(path, "r")
    return json.loads(f.read())

def list_to_ndarray(model):
    model_new = {}
    for key in model:
        lm = np.array(model[key]['lm'])
        bw = model[key]['bw']
        coef = np.array(model[key]['coef'])
        model_new[key] = (lm, bw, coef)
    return model_new

def ndarray_to_list(model):
    model_new = {}
    for key in model:
        lm = model[key][0]
        bw = model[key][1]
        coef = model[key][2]

        model_new[key] = {}
        model_new[key]['lm'] = lm.tolist()
        model_new[key]['bw'] = bw
        model_new[key]['coef'] = coef.tolist()
    return model_new

def timer_func(func):
    # This function shows the execution time of
    # the function object passed
    def wrap_func(*args, **kwargs):
        t1 = perf_counter()
        result = func(*args, **kwargs)
        t2 = perf_counter()
        time = t2-t1
        print(f'Function {func.__name__!r} executed in {(t2-t1):.4f}s')
        return result, time
    return wrap_func

def _dist(p1, p2):
    """
    :param p1: 1-dim np array
    :param p2: 1-dim np array
    :return: distance between p1 and p2
    """
    return np.sqrt(np.sum((p1-p2)**2, axis=0))

def normalize(x):
    mean = np.mean(x, axis=0)
    std = np.std(x, axis=0)
    x_norm = (x-mean)/std
    return x_norm


def compute_cost_terms_trace(V, time):
    V_single = np.ascontiguousarray(V[0, :])
    cost_terms_trace = apf.cost_terms_trace(V_single.T, t=time)
    return cost_terms_trace

# For N traces, for voltage and calcium

@timer_func
def compute_cost_terms_traces(V, Ca, time):
    V_tmp = np.expand_dims(V, axis=1)
    Ca_tmp = np.expand_dims(Ca, axis=1)
    data = np.concatenate((V_tmp, Ca_tmp), axis=1)

    print(data.shape)
    print(time.shape)

    all_cost_terms = apf.all_cost_terms(arr=data.T, t=time)
    print(all_cost_terms.shape)

    # Filter out inf if present
    if np.any(np.isinf(all_cost_terms)):
        idx_cut = np.where((np.sum(all_cost_terms, axis=1)!=np.inf))[0]
        all_cost_terms = all_cost_terms[idx_cut, :]
    return all_cost_terms


def find_nearest(array, value):
    array = np.asarray(array)
    return (np.abs(array - value)).argmin()


from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d

class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        super().__init__((0,0), (0,0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def do_3d_projection(self, renderer=None):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
        self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))

        return np.min(zs)