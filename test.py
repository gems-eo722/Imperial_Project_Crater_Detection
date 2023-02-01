from tycho_cdm.model.TychoCDM import TychoCDM

model = TychoCDM('weights/epoch_80.pth')
output = model.batch_inference('example/Mars_THEMIS_Training/images')