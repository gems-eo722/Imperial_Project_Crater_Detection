from tycho_cdm.model.TychoCDM import TychoCDM

model = TychoCDM('weights/epoch_80.pth')
output = model.batch_inference('img_list')