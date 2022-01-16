import ctunet
import os
ctunet_path = os.path.split(ctunet.__file__)[0]

params = ctunet.load_params('examples/autoimplant2020/UNetSP/AutoImplant2020_wShapePrior.ini')
ctunet.Model(params=params)
