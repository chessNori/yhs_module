import yhs_module
# yhs_module.path_to_txt("..\\..\\LabProject\\dataset\\LibriSpeech\\train-clean-100\\",
#                        250000, 'libri_speech_over_15sec')
# yhs_module.path_to_txt("..\\..\\LabProject\\dataset\\demand\\", 250000, 'demand_over_15sec')

data1 = yhs_module.TimeDomainData(320)
data2 = yhs_module.SpectralDomainData(320, False, 'time')
data3 = yhs_module.SpectralDomainData(320, False, 'spectrum')
data4 = yhs_module.SpectralMagnitudeData(320, False, 'time')

data1.load_data(1, 'libri_speech_over_15sec.txt', 0, 16, 'Libri Speech')
data1.load_data(2, 'demand_over_15sec.txt', 0, 16, 'Demand')
data2.load_data(1, 'libri_speech_over_15sec.txt', 0, 16, 'Libri Speech')
data2.load_data(2, 'demand_over_15sec.txt', 0, 16, 'Demand')
data3.load_data(1, 'libri_speech_over_15sec.txt', 0, 16, 'Libri Speech')
data3.load_data(2, 'demand_over_15sec.txt', 0, 16, 'Demand')
data4.load_data(1, 'libri_speech_over_15sec.txt', 0, 16, 'Libri Speech')
data4.load_data(2, 'demand_over_15sec.txt', 0, 16, 'Demand')

x1, y1 = data1.dataset(5)
rx2, ix2, y2 = data2.dataset(5, form='numpy')
rx3, ix3, ry3, iy3 = data3.dataset(5, form='tensorflow')
mx4, px4, y4 = data4.dataset(5, form='log')
