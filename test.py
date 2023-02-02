from tycho_cdm.model.TychoCDM import TychoCDM
import numpy as np

model = TychoCDM('mars')

image = np.random.random((416, 416, 3))
#images = 'img_list/A_0_366.png'
output = model.single_inference(image)
