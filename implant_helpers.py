import numpy as np
import pulse2percept as p2p
from pulse2percept.implants import DiskElectrode, ProsthesisSystem, ElectrodeArray
  
def build_implant(params, r=100):
    # There could be a smarter way to handle all these params
    '''
    Takes an array of tuples giving the electrode position
    '''

    electrodes = []
    for i in range(len(params) - 1):
          electrodes.append(p2p.implants.DiskElectrode(params[i], params[i+1], 0, r))
    
    implant = p2p.implants.ProsthesisSystem(p2p.implants.ElectrodeArray(electrodes))
    return implant

def get_num_effective(implant, model):
  
  percepts = []
  for name in implant.electrode_names:
    implant.stim = {name : 1}
    percept = model.predict_percept(implant)
    percepts.append(percept)

  allPixelData = []

  for percept in percepts:
    xy = percept.data
    F = xy.reshape(xy.shape[0], xy.shape[1])
    allPixelData.append(F)

  allPixelData = np.array(allPixelData)
  shape = allPixelData.shape
  allPixelData = allPixelData.reshape((shape[0], shape[1] * shape[2]))

  # Standardize:
  from sklearn.preprocessing import StandardScaler
  scaler = StandardScaler()
  scaler.fit(allPixelData)
  scaledData = scaler.transform(allPixelData)


  # 2. Apply PCA
  from sklearn.decomposition import PCA
  pca = PCA(n_components = 0.95)
  pca.fit(scaledData)
  
  return len(pca.explained_variance_ratio_)