import numpy as np
from joblib import Parallel, delayed

def aggregate_model_updates(client_model_updates: dict,
                            client_aggregation_weight: dict,
                            client_layermask: dict,
                            para_njobs: int=4) -> list:
    '''Aggregate the updates of models.
    
    Input: 
        `client_model_updates`: dict, the updates of models.
        Each key is the client id, the corresponding value is a list which are this client's model updates.
        
        `client_aggregation_weight`: dict, the aggregation weight for clients. 
        Each key is the client id, the corresponding value is a float which is the aggregation weight for this client.
        Note that the aggregation weights DOES NOT need to be normalized.
        
        `client_layermask`: dict, the layer mask vectors for clients.
        Each key is the client id, the corresponding value is a list which is the layer mask vector of this client's model updates.
        For instance, if NN model has three layers and the layer mask vector of client k is [1, 0, 1], it indicates that 
        only the first and third layers of client k's local model are updated during its local updating process in this round.
    
    Output:
        `aggregated_model_updates`: list, the aggregated model updates.
    '''

    assert len(client_aggregation_weight.keys())==len(client_model_updates.keys())
    assert len(client_aggregation_weight.keys())==len(client_layermask.keys())

    # ----------------------- Normalized aggregate weights ----------------------- #
    # sum_weights = np.array([0.0], dtype=np.float64)
    # for client_id in client_aggregation_weight.keys():
    #     sum_weights += client_aggregation_weight[client_id]
    # for client_id in client_aggregation_weight.keys():
    #     client_aggregation_weight[client_id] /= sum_weights

    # ---------------------------------------------------------------------------- #

    # ---------------------- Calculate layerwise denominator --------------------- #
    layerwise_denominator = np.zeros_like(client_layermask[list(client_layermask.keys())[0]], dtype=np.float64)
    for client_id in client_layermask.keys():
        this_client_layermask_vector = client_layermask[client_id]
        layerwise_denominator += client_aggregation_weight[client_id]*this_client_layermask_vector
    # ---------------------------------------------------------------------------- #

    # ----------------------- Initialize aggregated updates ---------------------- #
    aggregated_model_updates = []
    for layer_idx in range(len(client_model_updates[client_id])):
        aggregated_model_updates.append(np.zeros_like(client_model_updates[client_id][layer_idx], dtype=np.float32))
    # ---------------------------------------------------------------------------- #
    
    # ------------- Parallelly get weighted updates of client models ------------- #
    def __aggregate(client_id):
        for layer_idx in range(len(client_model_updates[client_id])):
            if layerwise_denominator[layer_idx]>0:
                aggregated_model_updates[layer_idx] += (client_model_updates[client_id][layer_idx]*client_layermask[client_id][layer_idx]*client_aggregation_weight[client_id]/layerwise_denominator[layer_idx]).astype(np.float32)
   
    Parallel(n_jobs=para_njobs, prefer="threads", require ='sharedmem')(delayed(__aggregate)(client_id) for client_id in client_model_updates.keys())
    # ---------------------------------------------------------------------------- #
    
    return aggregated_model_updates
