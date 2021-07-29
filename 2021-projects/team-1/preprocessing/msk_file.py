import pandas as pd
import numpy as np

with open("/umbc/xfs1/cybertrn/sea-ice-prediction/data/region_n.msk", "rb") as f:
	header = f.read(300)
	mask = np.fromfile(f, dtype=np.uint8)
	print(mask.shape)
	mask = mask.reshape(448, 304)
	print(mask)
	mask = np.where(mask == 11, 0, 1)
	print(mask)

with open("y_land_mask_actual.npy", "wb") as f:
	np.save(f, mask)
