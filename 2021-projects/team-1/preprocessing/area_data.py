import numpy as np
with open("/umbc/xfs1/cybertrn/sea-ice-prediction/data/region_n.msk", "rb") as fr:
    hdr = fr.read(300)
    mask = np.fromfile(fr, dtype=np.uint8)
mask=mask.reshape(448,304)
np.save('region_n_mask.npy', mask)


### read data ###
area = np.fromfile('/umbc/xfs1/cybertrn/sea-ice-prediction/data/psn25area_v3.dat', dtype=np.dtype('<i'))
area=area.reshape(448,304)
area = area/1000.0
np.save('area_size.npy', area)
