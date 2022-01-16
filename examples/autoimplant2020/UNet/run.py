import ctunet
import os
ctunet_path = os.path.split(os.path.split(ctunet.__file__)[0])[0]

params = ctunet.load_params(ctunet_path + '/examples/autoimplant2020/UNet/AutoImplant2020_woShapePrior.ini')
ctunet.Model(params=params)
