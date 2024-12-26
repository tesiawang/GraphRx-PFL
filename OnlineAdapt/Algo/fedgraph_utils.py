import tensorflow as tf
import numpy as np
import copy
import cvxpy as cp

def flatten_model(model_weights):
    return tf.concat([tf.reshape(w, [-1]) for w in model_weights], axis=0)

# a = tf.constant([[1, 2, 3],[4,5,6],[7,8,9]])
# b = flatten_model(a)
# print(b)

def cal_model_diff(client_updated_model_para, initial_para, dist_metric):
    '''
    Compute the model difference matrix based on the distance_metric

    args:
        client_updated_model_para: a dictionary of client model parameters
        initial_para: the initial model parameter same for all the clients
        dist_metric: the metric to compute pair-wise model difference
    
    return:
        model_diff_mat: a matrix of model difference

    '''
    num_clients = len(client_updated_model_para)
    model_diff_mat = np.zeros((num_clients,num_clients))
    client_ids = list(client_updated_model_para.keys())
    flatten_initial_para = flatten_model(initial_para)

    for i in range(num_clients):
        for j in range(i, num_clients):
            flatten_model_i = flatten_model(client_updated_model_para[client_ids[i]])
            flatten_model_j = flatten_model(client_updated_model_para[client_ids[j]])
            
            model_update_i = flatten_model_i - flatten_initial_para # compared with the initial model
            model_update_j = flatten_model_j - flatten_initial_para

            if dist_metric == 'cosine':
                # difference/distance = - similarity
                # the negative sign is already taken in loss functions
                # diff: [-1,1]; -1: two models are the same; 1: two models are the opposite (the largest distance)
                diff = tf.keras.losses.cosine_similarity(model_update_i, model_update_j)
                if diff < - 0.9: # clip
                    diff = - 1.0
            elif dist_metric == 'l2':
                # diff >= 0 
                diff = tf.norm(model_update_i - model_update_j)
            elif dist_metric == 'l1':
                # diff >= 0
                diff = tf.norm(model_update_i - model_update_j, ord=1)
            else:
                raise ValueError("Consider a valid distance metric")
            
            # symmetric
            model_diff_mat[i, j] = diff
            model_diff_mat[j, i] = diff

    # ---------------------------- normalize if needed --------------------------- #
    if dist_metric == 'l2' or dist_metric == 'l1':
        # normalize the matrix: [0,1]
        max_element = np.max(model_diff_mat[np.nonzero(model_diff_mat)])
        model_diff_mat = model_diff_mat / max_element

    return model_diff_mat, client_ids



def update_graph_matrix_directed(graph_matrix, model_diff_mat, client_ids, alpha, opt_objective, hyper_c, p):
    '''
    Information on "alpha": a hyper parameter which is equal to 'lamba' in the source codes of pFedGraph
    In our method: the hyper-parameter is hyper_c, while alpha is not used.

    "p" is the quantity distribution vector, e.g., p = [1/6,..., 1/6]
    ''' 
    n = model_diff_mat.shape[0]
    sqrt_p = np.sqrt(p)
    P = alpha * np.identity(n)
    P = cp.atoms.affine.wraps.psd_wrap(P)
    G = - np.identity(n)
    h = np.zeros(n)
    A = np.ones((1, n))
    b = np.ones(1)

    for i in range(model_diff_mat.shape[0]):
        d = model_diff_mat[i]
        q = d - 2 * alpha * p
        x = cp.Variable(n) # solve the opt problem for every client

        if opt_objective == 0: # pFedGraph original objective function considering data quantity
            prob = cp.Problem(cp.Minimize(cp.quad_form(x, P) + q.T @ x),
                    [G @ x <= h, # positive weights
                    A @ x == b]) # sum of weights = 1

        elif opt_objective == 1:    # gen_error_bound with square root; align with the domain adaptation theory
            x_prime = cp.multiply(x, 1 / sqrt_p)  # element-wise multiplication
            prob = cp.Problem(cp.Minimize(cp.norm(x_prime, 2) * hyper_c + d.T @ x),
                            [G @ x <= h,
                            A @ x == b])
        else:
            raise ValueError("Consider a valid objective function")
        prob.solve(solver = cp.ECOS)
        graph_matrix[client_ids[i], client_ids] = x.value
    return graph_matrix



def update_graph_matrix_undirected(graph_matrix, model_diff_mat, client_ids, hyper_c, p):
    '''
    This function solves a global optimization problem to update the graph matrix
    Consider the graph matrix is symmetric

    "p" is the quantity distribution vector, e.g., p = [1/6,..., 1/6]
    '''

    n = model_diff_mat.shape[0] # n = num_total_client

    # p = quantity distribution vector
    sqrt_p = np.sqrt(p)
    
    # global objective function
    X = cp.Variable((n, n), symmetric=True) # automatically ensure X[i,j] = X[j,i]
    objective = 0

    for i in range(n):
        d = model_diff_mat[i]
        x = X[i,:] # get a row varible for client i by creating new reference
        x_prime = cp.multiply(x, 1 / sqrt_p)  # element-wise multiplication
        objective += cp.norm(x_prime, 2) * hyper_c + d.T @ x
    
    prob = cp.Problem(cp.Minimize(objective),
                      [cp.sum(X, axis=1) == 1, # each row sum =1
                       X >= 0]) # each element non-negative
    prob.solve(solver = cp.ECOS)
    graph_matrix = X.value

    return graph_matrix


def aggregation_by_graph(graph_matrix, client_updated_model_para):
    '''
    This function retruns the un-normed weighted aggregation model (direct aggregation)
    '''

    client_ids = client_updated_model_para.keys()
    unnorm_weighted_model_para = {}

    # initialize the un-normed weighted model parameters
    for c in client_ids:
        unnorm_weighted_model_para[c] = []
        for layer_idx in range(len(client_updated_model_para[c])):
            unnorm_weighted_model_para[c].append(np.zeros_like(client_updated_model_para[c][layer_idx]))
            
    # weighted aggregation
    for c in client_ids:
        # get the aggregation weight vector from the collaboration graph
        aggregation_weight_vector = graph_matrix[c]

        for neigh_id in client_ids:
            neigh_para = client_updated_model_para[neigh_id]
            # aggregation layer by layer
            # un-normed version:
            for layer_idx in range(len(neigh_para)):
                unnorm_weighted_model_para[c][layer_idx] += neigh_para[layer_idx] * aggregation_weight_vector[neigh_id]
    
    # return the un-normed weighted version as the weighted model parameters
    return unnorm_weighted_model_para


def aggregation_by_graph_norm(graph_matrix, client_updated_model_para):
    '''
    This function returns and the un-normed weighted aggregation model (direct aggregation), which is used as the local initialization point
    and the normed version of the weighted model parameters, which is used as the penalty term in local optimization
    '''
    # when we try to get the norm_weighted_model_para, we need to flatten the model first
    
    client_ids = client_updated_model_para.keys()
    unnorm_weighted_model_para = {}
    norm_weighted_model_para = {}

    # ------------ initialize the un-normed weighted model parameters ------------ #
    for c in client_ids:
        # initialize the un-normed weighted model parameters
        unnorm_weighted_model_para[c] = []
        for layer_idx in range(len(client_updated_model_para[c])):
            unnorm_weighted_model_para[c].append(np.zeros_like(client_updated_model_para[c][layer_idx]))
        
        # intialize the normed weighted model parameters
        norm_weighted_model_para[c] = np.zeros_like(flatten_model(client_updated_model_para[c]))
        
    # --------------------------- weighted aggregation --------------------------- #
    for c in client_ids:
        # get the aggregation weight vector from the collaboration graph
        aggregation_weight_vector = graph_matrix[c]

        for neigh_id in client_ids:
            neigh_para = client_updated_model_para[neigh_id]
            # aggregation layer by layer
            for layer_idx in range(len(neigh_para)):
                unnorm_weighted_model_para[c][layer_idx] += neigh_para[layer_idx] * aggregation_weight_vector[neigh_id]
        
        for neigh_id in client_ids:
            neigh_para = flatten_model(client_updated_model_para[neigh_id])
            norm_weighted_model_para[c] += neigh_para * (aggregation_weight_vector[neigh_id] / np.linalg.norm(neigh_para))
    
    # return the un-normed weighted version as the weighted model parameters
    return unnorm_weighted_model_para, norm_weighted_model_para



def aggregation_by_graph_norm_update(graph_matrix, client_updated_model_para, initial_para):
    '''
    This function returns and the un-normed weighted aggregation model (direct aggregation), which is used as the local initialization point
    and the normed version of the weighted model updates compared to inital_para, which is used as the penalty term in local optimization
    '''
    client_ids = client_updated_model_para.keys()
    unnorm_weighted_update = {}
    norm_weighted_update = {}
    client_update = {}

    # ----------------------- get the client model updates compared to the initial model ----------------------- #
    for c in client_ids:
        client_update[c] = []
        for layer_idx in range(len(client_updated_model_para[c])):
            client_update[c].append(client_updated_model_para[c][layer_idx] - initial_para[layer_idx])

    # ------------------------------- initializing ------------------------------- #
    for c in client_ids:
        unnorm_weighted_update[c] = []
        for layer_idx in range(len(client_update[c])):
            unnorm_weighted_update[c].append(np.zeros_like(client_update[c][layer_idx]))

        # flattened initialization
        norm_weighted_update[c] = np.zeros_like(flatten_model(client_update[c]))

            
    # --------------------------- weighted aggregation --------------------------- #
    for c in client_ids:
        # get the aggregation weight vector from the collaboration graph
        aggregation_weight_vector = graph_matrix[c]

        for neigh_id in client_ids:
            neigh_update = client_update[neigh_id]
            # aggregation layer by layer
            for layer_idx in range(len(neigh_update)):
                unnorm_weighted_update[c][layer_idx] += neigh_update[layer_idx] * aggregation_weight_vector[neigh_id]
                
        for neigh_id in client_ids:
            neigh_update = flatten_model(client_update[neigh_id])
            norm_weighted_update[c] += neigh_update* (aggregation_weight_vector[neigh_id] / np.linalg.norm(neigh_update))
    
    # ---------- Plus the initial_para to re-initialize model parameters --------- #
    unnorm_weighted_para = {}
    for c in client_ids:
        unnorm_weighted_para[c] = []
        for layer_idx in range(len(initial_para)):
            unnorm_weighted_para[c].append(initial_para[layer_idx] + unnorm_weighted_update[c][layer_idx])
            
    return unnorm_weighted_para, norm_weighted_update