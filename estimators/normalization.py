# using the gpu accelated 
import idtxl.estimators_opencl as est


def global_norm_old(data, C):
    # Reshape data as before
    data = np.reshape(data, (data.shape[0], int(data.size / data.shape[0])))
    
    # Subtract the mean
    means = np.mean(data, axis=0)
    data = data - means
    
    # Compute the normalization factor
    norm = np.sqrt(np.mean(np.sum(data ** 2, axis=1)))
    
    # Normalize the data
    data *= C / norm
    
    return data, norm ** 2

def local_norm_old(data,C):
    data = np.reshape(data, (data.shape[0],int(data.size/data.shape[0])))
    
    means =  np.mean(data, axis=0) # find the mean for each dimension 
    data = data - means # data - means for each dimension
    
    norm = np.tile(np.sqrt(np.mean(data ** 2 ,axis=0)),(data.shape[0],1))
#     norm =  np.sqrt(np.mean(np.sum(sqz,axis=1)))
    normalized_data = C*data / (norm+(0.0000001))
    
    return normalized_data,norm**2


def global_normalize_inf_norm(data_orig, C):
    # Reshape data as before
    data = np.reshape(data_orig, (data_orig.shape[0], int(data_orig.size / data_orig.shape[0])))
    
    # Subtract the mean
    means = np.mean(data, axis=0)
    data = np.abs(data - means)
    
    # Compute the normalization factor
    # norm = np.sqrt(np.mean(np.sum(data ** 2, axis=1)))
    # linalg.norm(x, ord=None, 
    norm = np.mean(np.max(data,1))
    #print(norm)

    # Normalize the data
    data = data_orig * C / norm
    
    return data


def global_normalize_mine(data_orig, C, p=2,dim_correction= False):
    # Reshape data as before
    data = np.reshape(data_orig, (data_orig.shape[0], int(data_orig.size / data_orig.shape[0])))
    
    # Subtract the mean
    means = np.mean(data, axis=0)
    data = np.abs(data - means)
    
    # Compute the normalization factor
    norm = np.sqrt(np.mean(np.sum(data ** 2, axis=1)))
    norm = norm/ np.sqrt(data.shape[1])
    # print(norm)
    # Normalize the data
    
    data = C*(data_orig-means) / norm
    return data

