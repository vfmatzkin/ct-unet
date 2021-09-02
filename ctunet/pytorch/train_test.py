# This file is part of the
#   ctunet Project (https://github.com/vfmatzkin/ctunet).
# Copyright (c) 2021, Franco Matzkin
# License: MIT
#   Full Text: https://github.com/vfmatzkin/ctunet/blob/main/LICENSE

""" ctunet trainer class."""
import sys
from collections import OrderedDict
from shutil import copyfile

import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from .ProblemHandler import *
from .models import *

torch.autograd.set_detect_anomaly(True)


class Model:
    def __init__(self, cfg_file=None, params=None):
        """Constructor of Model class. Read the cfg file and
        initialize the required variables training or
        testing a model.
        If the trainVal flag is True, it starts training the model.
        If the test flag is True, it makes the predictions for the test dataset
        from the trained model (wich could be trained right before or in
        previous moment).

        :param cfg_file: path of the cfg file with the parameters.
        """
        if cfg_file and params:
            params = None
            print("You provided both a cfg file a params dictionary. Only "
                  "the cfg file will be used")

        if cfg_file is None and (params is None):
            print("No configuration file provided.")

        # CLI gives list of args so in this case take the first element
        cfg_file = cfg_file[0] if type(cfg_file) is list else cfg_file
        if cfg_file and not os.path.exists(cfg_file):  # Check if cfg exists
            raise FileNotFoundError(
                f"The configuration file does not exists ({cfg_file})."
            )

        self.params = {  # Initialize required params by defalut
            # DEFAULT
            "train_flag": False,
            "test_flag": False,
            # MODEL
            "name": None,  # Trained model name
            "model_class": None,  # Model class name (located in models.py)
            "problem_handler": None,
            # Class that contains img_path reading/writing
            # TRAINING
            "device": None,  # gpu/cpu
            "n_epochs": None,  # number of epochs
            "batch_size": None,
            "dice_lambda": None,  # Weight used for the Dice loss.
            "ce_lambda": None,  # Weight used for the Cross Entropy loss.
            "acnn_path": None,
            "acnn_lambda": None,  # Weight used for the ACNN loss.
            "msel_lambda": None,  # Weight used for the MSE loss.
            # OPTIMIZER
            "optimizer": None,  # Optimizer used for training
            "learning_rate": None,
            "momentum": None,
            "weight_decay": None,
            # PATHS
            "single_file": None,
            "workspace_path": None,
            "train_files_csv": None,  # Datasets csv
            "validation_files_csv": None,
            "test_files_csv": None,
            'tensorboard_run_path': None,
            # MISC
            "autosave_epochs": None,  # Frequency of model checkpoint saving.
            "save_dice_plots": None,  # Save Dice coefficient plot
            "resume_model": None,  # Load the model in the provided path
            "show_model_summary": None,  # Show no. of params
            "n_workers": None,
            "force_resumed": False,  # Force use the resumed model on inference
        }

        # Overwrite the supplied params (this way I don't have to specify
        # all params when using this argument)
        if params is not None:
            for key in params:
                self.params[key] = params[key]

        # Read the file and set the params attribute
        if not params:
            self.params = utils.set_cfg_params(cfg_file, self.params)

        self.resolve_out_folder()

        self.problem_handler = eval(self.params["problem_handler"])()
        self.write_predictions = self.problem_handler.write_predictions
        self.comp_losses_metrics = self.problem_handler.comp_losses_metrics

        self.models = {
            "main": None,  # Trained model
            "acnn": None,  # ACNN model used for regularization (if neccesary)
        }

        self.data = {
            "train_dataloader": None,
            "validation_dataset": None,
            "validation_dataloader": None,
            "test_dataset": None,
            "test_dataloader": None,
        }

        self.cfg_path = cfg_file

        self.params["device"] = (
            torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if str(self.params["device"]) == "cuda"
            else torch.device("cpu")
        )

        self.load_datasets()  # Load the datasets provided in the csv files

        self.current_epoch = self.current_train_iteration = 0

        self.pt_loss = []  # Losses used by PyTorch

        # Losses & metrics stored as lists of floats for display in Tensorboard
        self.losses_and_metrics = {}

        self.writer = SummaryWriter(self.params['tensorboard_run_path'])

        if self.params["train_flag"] is True:
            self.train()

        # Test the trained model with the test image.
        if self.params["test_flag"] is True:
            self.test()

    @staticmethod
    def get_dataloader(dataset_class, dataset, batch_size=1,
                       pin_memory=True, shuffle=True, n_workers=0,
                       single_file=None):
        """
        Return a sampled Dataloader from a given csv and dataset class.

        From a csv file, instance the dataset and get the DataLoader associated
        with it. If it'sh required and the csv is not found, it will raise an
        Exception.


        :param dataset_class: Class inherited of torch.utils.image.Dataset,
        that is used for creating the DataLader.
        :param dataset: Path of the dataset folder or CSV file listing the
        files.
        :param batch_size: Batch size used in the DataLoader.
        :param pin_memory: Dataloader'sh pin_memory flag, for loading the image
        into CUDA. This lets your DataLoader allocate the samples in
        page-locked memory, which speeds-up the transfer.
        :param shuffle: Shuffle the image in the DataLoader
        :param n_workers: How many subprocesses to use for image loading.
        :param single_file: Predict only a single file. If None, it will
        predict the csv.
        :return:
        """
        dataset = dataset_class(dataset, "", single_file=single_file)

        r_sampler = torch.utils.data.RandomSampler(
            dataset, replacement=True, num_samples=min(92, len(dataset))
        )

        dataloader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            pin_memory=pin_memory,
            num_workers=n_workers,
            sampler=r_sampler,
            shuffle=shuffle,
        )
        return dataloader

    def load_datasets(self):
        """ Initialize the self.image attribute with the corresponding datasets.

        The keys of self.image will be train_dataloader, validation_dataloader
        and test_dataloader.
        """
        dataset_c_train = self.problem_handler.train_dataset_class
        dataset_c_test = self.problem_handler.test_dataset_class

        pin_mem = False if str(self.params["device"]) == "cpu" else True

        if self.params["train_flag"]:
            self.data["train_dataloader"] = self.get_dataloader(
                dataset_class=dataset_c_train,
                dataset=self.params["train_files_csv"],
                batch_size=self.params["batch_size"],
                pin_memory=pin_mem,
                n_workers=self.params["n_workers"],
                shuffle=False
            )
            self.data["validation_dataloader"] = self.get_dataloader(
                dataset_class=dataset_c_train,
                dataset=self.params["validation_files_csv"],
                batch_size=self.params["batch_size"],
                n_workers=self.params["n_workers"],
                shuffle=False
            )

        if self.params["test_flag"]:
            self.data["test_dataloader"] = self.get_dataloader(
                dataset_class=dataset_c_test,
                dataset=self.params["test_files_csv"],
                batch_size=1,
                shuffle=False,
                single_file=self.params['single_file']
            )

    def train(self):
        """
        Train the model using the previously set up parameters.
        """
        self.initialize_models()
        self.initialize_optimizer()
        utils.print_params_dict(self.params)

        self.current_train_iteration = 0
        for n_epoch in range(1, self.params["n_epochs"] + 1):
            ep_time = utils.tic()  # Begin measuring time
            self.current_epoch = n_epoch

            print("Epoch: ", n_epoch)
            self.forward_pass("train", self.data["train_dataloader"])
            self.update_plots_tensorboard_avg("train", n_epoch)

            self.forward_pass("val", self.data["validation_dataloader"])
            self.update_plots_tensorboard_avg("val", n_epoch)

            # Calculate remaining time (for display)
            utils.toc_eps(ep_time, n_epoch, self.params["n_epochs"])

            autosave_now = (n_epoch % self.params["autosave_epochs"]) == 0
            if self.params["train_flag"] and autosave_now:  # Save epoch chkp.
                self.save_main_model(self.cfg_path, n_epoch, True)
                if self.params["test_flag"]:
                    self.test()  # Predict after saving the checkpoint.

            # Update the model on each epoch
            self.save_main_model()

    def save_main_model(self, cfg_file=None, epoch=-1, save_checkpoint=False):
        """ Save the model being trained.

        This function can save also the checkpoints in a subfolder with that
        name.

        :param cfg_file: Configuration file, that will be saved in the first 
        epoch.
        :param epoch: Trained epochs of the model.
        :param save_checkpoint: The method is called for saving a separate.
        checkpoint (training continues).
        :return:
        """
        path = self.params["model_path"]
        dir_m, fname = os.path.split(path)
        utils.veri_folder(dir_m)  # Check if output pred_folder exists
        torch.save(self.models["main"].state_dict(), path)  # Overwrite

        # Save the model parameters alongside the model if is first epoch
        if cfg_file and epoch == 1:
            copyfile(cfg_file, path.replace(".pt", "_params.ini"))

        if save_checkpoint:
            dir_chk = os.path.join(dir_m, 'checkpoints')
            chk_p = os.path.join(dir_chk,
                                 fname.replace(".pt",
                                               "_ep" + str(epoch) + ".pt"))
            utils.veri_folder(dir_chk)
            torch.save(self.models["main"].state_dict(), chk_p)
            print("Checkpoint saved ({})".format(save_checkpoint))

        print("Model saved ({})".format(path))

    def test(self):
        """ Make a test forward pass.

        If the model isn't loaded, it will load it.

        :return:
        """
        # If the model isn't loaded yet, load it.
        if self.models["main"] is None and self.params["name"]:
            self.initialize_models(load_out=True)
        elif self.models["main"] is None and self.params["resume_model"]:
            self.initialize_models()

        if self.params["test_flag"] and not self.params["test_files_csv"] \
                and not self.params["single_file"]:
            print("No csv provided for testing")
        elif self.params["test_flag"] and self.params["single_file"]:
            self.forward_pass("test", self.data["test_dataloader"])
        else:  # Regular inference
            print(
                "Images to test: ",
                os.path.split(self.params["test_files_csv"])[0],
            )
            utils.print_params_dict(self.params)  # Print the params dictionary
            self.forward_pass("test", self.data["test_dataloader"])

    def forward_pass(self, phase, data_loader):
        """
        Make a forward pass on the network.

        :param phase: 'train', 'val', 'test'
        :param data_loader: The train/validation/test loader
        :return:
        """
        print(f"Phase: {phase}.")

        if phase == "train":
            self.models["main"].train()
            # Equivalent to 'with torch.no_grad()'
            torch.set_grad_enabled(True)
        else:  # eval mode (won't update parameters and disable dropout)
            self.models["main"].eval()
            torch.set_grad_enabled(False)

        for batch_idx, sample in enumerate(data_loader):
            input_img = sample["image"].to(self.params["device"])
            if 'target' in sample:
                if type(sample['target']) == list:
                    target = [e.to(self.params["device"])
                              for e in sample['target']]
                else:
                    target = sample['target'].to(self.params["device"])

            if phase == "train":
                input_img.requires_grad_()
            #     target.requires_grad_()

            # I do this here because I need an img sample for getting FLOPS n.
            # if c_op and (phase == "train" and self.current_epoch == 1 and self.params[
            #    "show_model_summary"]) or (phase == "test" and batch_idx == 1):
            #    print(summary(self.models["main"], input_img))
            #    print(f'\nFLOPS: {count_ops(self.models["main"], input_img)}')

            model_out = self.models["main"](input_img)  # Forward pass

            if phase in ['train', 'validation', 'val']:
                self.comp_losses_metrics(self, model_out, target,
                                         batch_idx, len(data_loader))
                if phase == 'train':
                    self.pt_loss.backward()
                    self.params["optimizer"].step()
                    # self.params["optimizer"].zero_grad()
                    for param in self.models["main"].parameters():
                        param.grad = None
            elif phase == "test":
                self.write_predictions(model_out, sample["filepath"],
                                       self.params["name"], input_img)

    def update_plots_tensorboard_avg(self, phase, i, type="epoch",
                                     print_to_console=False):
        """ Update averages for all computed losses and metrics.

        :param phase: Phase of the epoch (train/test).
        :param i: Position in the temporal axis.
        :param type: Category to group.
        :param print_to_console: Print to console the output
        :return:
        """
        for key in self.losses_and_metrics.keys():
            avg = sum(self.losses_and_metrics[key]) / len(
                self.losses_and_metrics[key]
            )
            self.writer.add_scalar(
                phase + "/" + type + "/" + key, float(avg), i
            )
            self.losses_and_metrics[key] = []
            if print_to_console:
                print("{} {} average: {}.".format(type, i, float(avg)))

    def resolve_out_folder(self):
        """ Create workspace pred_folder if not exists and set model path.

        The workspace will contain the saved models, training information
        and plots. The model predictions in the test phase will be always
        saved in a subfolder of the input images pred_folder.
        """
        if not self.params['workspace_path']:
            raise AttributeError('workspace_path not defined in the ini file.')
        wsp = self.params['workspace_path'] = \
            os.path.expanduser(self.params['workspace_path'])
        utils.veri_folder(wsp)

        # Model folder in the workspace path (Model + ProblemHandler)
        mc, hd = self.params['model_class'], self.params['problem_handler']
        run_name = mc + '_' + hd
        model_folder = os.path.join(os.path.expanduser(wsp), run_name, 'model')
        utils.veri_folder(model_folder)

        name, res_path = self.params['name'], self.params['resume_model']
        res_filename = os.path.splitext(os.path.split(res_path)[1])[0]

        if name in ["", None] and res_path in ["", None]:
            raise AttributeError("You should set at least a name or a path "
                                 "of a previously trained model for lookup.")

        self.params['model_path'] = res_path if res_path != '' else None
        self.params['name'] = res_filename if not name and res_path else name

        if not self.params['force_resumed']:
            # In cases such as training a new model or testing right after
            # training, the model path will be this new path, otherwise it
            # will be the model previously trained set in resume_model.
            new_name = os.path.join(model_folder, name + '.pt')
            self.params['model_path'] = new_name

        if self.params['tensorboard_run_path'] is None:
            tb_folder_name = run_name + '_' + self.params['name']
            self.params['tensorboard_run_path'] = os.path.join(wsp, 'runs',
                                                               tb_folder_name)

    def load_model(self, model_path):
        """ Load a previously trained model and return it.

        This method works with both the old way of saving models (with
        serialization) and as an OrderedDict.

        :param model_path: Model path.
        :return: Loaded model (from a class that inherits from nn.Module).
        """
        loaded = torch.load(os.path.expanduser(model_path),
                            map_location=str(self.params["device"]),
                            )
        if type(loaded) != OrderedDict:  # It'sh the serialized object
            return loaded
        else:  # It'sh the state dictionary: must instance the model first
            model = self.new_model()
            model.load_state_dict(loaded)
            return model

    def new_model(self):
        """ Instantiate the model depending of the indicated class in the cfg.

        It also checks how many GPUs are avaiable (if selected in the cfg).

        :return:
        """
        if (self.params["device"] == torch.device("cuda") and
                torch.cuda.device_count() > 1):
            print(f"Using {torch.cuda.device_count()} GPUs. Set "
                  f"CUDA_VISIBLE_DEVICVES otherwise.")
            model = eval(self.params["model_class"])()
            model = nn.DataParallel(model)
        else:
            model = eval(self.params["model_class"])()
        model.to(self.params["device"])

        return model

    def initialize_models(self, load_out=False):
        """ Initialize the model classes.

        Instance the model classes or resumes the training from the
        resume_model param.

        :param load_out: If no model is provided in the test phase, it will
        try to load the output file with this flag.
        :return:
        """
        if load_out:  # Load the output file.
            self.models["main"] = self.load_model(self.params["model_path"])
        elif self.params["resume_model"] not in ["", None]:
            self.models["main"] = self.load_model(self.params['resume_model'])
        else:  # Initialize a new model
            self.models["main"] = self.new_model()

    def initialize_optimizer(self):
        """ Initialize the optimizer
        """
        # Configure the optimizer
        if self.params["optimizer"] == "adam":
            self.params["optimizer"] = optim.Adam(
                self.models["main"].parameters(),
                lr=self.params["learning_rate"],
                weight_decay=self.params["weight_decay"],
                amsgrad=True,
            )
        elif self.params["optimizer"] == "rmsprop":
            self.params["optimizer"] = optim.RMSprop(
                self.models["main"].parameters(),
                lr=self.params["learning_rate"],
                weight_decay=self.params["weight_decay"],
                momentum=self.params["momentum"],
            )
        elif self.params["optimizer"] == "sgd":
            self.params["optimizer"] = optim.SGD(
                self.models["main"].parameters(),
                lr=self.params["learning_rate"],
                momentum=self.params["momentum"],
                weight_decay=self.params["weight_decay"],
            )


def load_ini_file(ini_file: str):
    """ Create instance of Model with the provided ini file path."""
    Model(ini_file)


def cli():
    """Run the headctools CLI interface."""
    if len(sys.argv) > 1:
        Model([sys.argv[1]])


if __name__ == "__main__":
    if len(sys.argv) > 1:
        Model([sys.argv[1]])
