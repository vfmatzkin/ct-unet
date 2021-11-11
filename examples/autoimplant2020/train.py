import ctunet
import os
ctunet_path = os.path.split(ctunet.__file__)[0]

train_files = '/home/fmatzkin/Code/datasets/archived/autoimplant-challenge/training_set/complete_skull/ext_renamed/prep_304_224/train.csv'
validation_files = '/home/fmatzkin/Code/datasets/archived/autoimplant-challenge/training_set/complete_skull/ext_renamed/prep_304_224/validation.csv'
test_files = '/home/fmatzkin/Code/datasets/archived/autoimplant-challenge/additional_test_set_for_participants/ext_renamed/prep_304_224/files.csv'

params = ctunet.load_params(ctunet_path + '/cfg/AutoImplant2020_wShapePrior.ini')
params.update({'train_flag': True,
               'test_flag': False,
               'train_files_csv': train_files,
               'validation_files_csv': validation_files,
               'test_files_csv': test_files})
ctunet.Model(params=params)
