from tycho_cdm.model.TychoCDM import TychoCDM
import numpy as np

model = TychoCDM('mars')

images = np.random.random((10, 416, 416,3))
#images = 'img_list/A_0_366.png'
output = model.inference(images)