# Simplified MI computation code from https://github.com/ravidziv/IDNNs
# https://github.com/artemyk/ibsgd/tree/master
import numpy as np

def get_unique_probs(x):
    uniqueids = np.ascontiguousarray(x).view(np.dtype((np.void, x.dtype.itemsize * x.shape[1])))
    _, unique_inverse, unique_counts = np.unique(uniqueids, return_index=False, return_inverse=True, return_counts=True)
    return np.asarray(unique_counts / float(sum(unique_counts))), unique_inverse

from scipy.stats import entropy

# def bin_calc_information_new_mod(inputdata, layerdata, num_of_bins_input=5, num_of_bins_layer=5):
    
    
#     input_min = np.min(inputdata) # 
#     input_max = np.max(inputdata)
#     bin_input = np.linspace(input_min, input_max, num_of_bins_input, dtype='float32')
#     digitized_input = bin_input[np.digitize(np.squeeze(inputdata.reshape(1, -1)), bin_input)- 1].reshape(len(inputdata), -1)
#     #     print(digitized_input)
#     layer_min = np.min(layerdata)
#     layer_max = np.max(layerdata)
#     bin_layer = np.linspace(layer_min, layer_max, num_of_bins_layer, dtype='float32')
#     digitized = bin_layer[np.digitize(np.squeeze(layerdata.reshape(1, -1)), bin_layer) - 1].reshape(len(layerdata), -1)
#     #     print(digitized)
#     digitized_concat = np.concatenate((digitized_input, digitized), axis=1)
#     #     print(digitized_concat)
#     value, counts_input = np.unique(digitized_input, return_counts=True, axis=1)
#     value, counts_layer = np.unique(digitized, return_counts=True, axis=1)
#     value, counts_concat = np.unique(digitized_concat, return_counts=True, axis=1)
#     return entropy(counts_input, base=2) + entropy(counts_layer, base=2) - entropy(counts_concat, base=2)

def bin_calc_information_new_mod(inputdata, layerdata, num_of_bins_input=5, num_of_bins_layer=5):
    def digitize_data(data, num_of_bins):
        data_min = np.min(data)
        data_max = np.max(data)
        bins = np.linspace(data_min, data_max, num_of_bins + 1, dtype='float32')
        digitized = np.digitize(data, bins, right=False) - 1
        # Ensure digitized values are within the correct range
        digitized = np.clip(digitized, 0, num_of_bins - 1)
        return digitized, bins

    def calculate_entropy(data):
        value, counts = np.unique(data, return_counts=True, axis=0)
        return entropy(counts, base=2)

    # Digitize input and layer data
    digitized_input, bins_input = digitize_data(inputdata, num_of_bins_input)
    digitized_layer, bins_layer = digitize_data(layerdata, num_of_bins_layer)

    # Reshape digitized data for concatenation
    digitized_input = digitized_input.reshape(len(inputdata), -1)
    digitized_layer = digitized_layer.reshape(len(layerdata), -1)

    # Concatenate digitized input and layer data
    digitized_concat = np.concatenate((digitized_input, digitized_layer), axis=1)

    # Calculate entropies
    entropy_input = calculate_entropy(digitized_input)
    entropy_layer = calculate_entropy(digitized_layer)
    entropy_concat = calculate_entropy(digitized_concat)

    # Return mutual information
    mutual_information = entropy_input + entropy_layer - entropy_concat
    return mutual_information