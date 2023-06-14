from utils import asdreader
from utils import unpack_asd
import struct

old_file = r'G:\Other computers\My Computer\veg-sim-data\raw_data\MEYER-OKIN\s1t3ind.067'
new_old_file = r'G:\Other computers\My Computer\veg-calib\field\SHIFT_FALL\DPA-004\L1\Endmembers\DP_.000'

data = open(old_file, "rb").read()
meta_data_asd = unpack_asd.get_asd_binary(data)
print(meta_data_asd[3])

# new - old files ; asd 4
data = open(new_old_file, "rb").read()
meta_data_asd = unpack_asd.get_asd_binary(data)
print(meta_data_asd[3])
