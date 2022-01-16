import ctunet
import os
ctunet_path = os.path.split(os.path.split(ctunet.__file__)[0])[0]

params = ctunet.load_params(ctunet_path + '/examples/UNetSPDO/FlapRecSP2O_128.ini')
ctunet.Model(params=params)
