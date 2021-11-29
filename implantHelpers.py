import numpy as np
from pulse2percept.implants import DiskElectrode, ProsthesisSystem, ElectrodeArray

def buildElectrodeArray (electrodePositions, radius):
    '''
    Takes an array of tuples giving the electrode position
    '''
    earray = ElectrodeArray({})
    for position in electrodePositions:
        x = position[0]
        y = position[1]
        # print(x,y)
        earray.add_electrode(f'{x}{y}', DiskElectrode(x, y, 0, r=radius))

    implant = ProsthesisSystem(earray)

    return implant

def numberOfEffectiveElectrodes (implant, model):
  
  percepts = []
  for name in implant.electrode_names:
    implant.stim = {name : 1}
    percept = model.predict_percept(implant)
    percepts.append(percept)

  allPixelData = []

  for percept in percepts:
    data = percept.data
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