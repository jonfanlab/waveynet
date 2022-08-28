import  torch, os
from    torch import optim
import  numpy as np
import  pandas as pd
import  argparse
from    torch.utils.data import random_split, DataLoader
from    learner import *
from    simulation_dataset import SimulationDataset
from    torch.cuda.amp import GradScaler, autocast

#Define the physical constants of the simulation problem
c_0 = 299792458.13099605
eps_0 = 8.85418782e-12
mu_0 = 1.25663706e-6
dL = 6.25e-9
wavelength = 1050e-9
omega = 2 * np.pi * c_0 / wavelength
n_sub=1.45

def Hz_to_Ex(Hz_R: np.array, Hz_I: np.array, dL: float, omega: float, eps_grid: np.array, \
             eps_0: float = eps_0) -> np.array:
    '''
    This function performs finite difference on Hz to obtain corresponding Ex field.

    Note that ceviche implements the finite differnce slightly different from the
    conventional setup. Hence, the difference between Hz[:, 1] and Hz[:, 0] produces
    Ex[:, 0] instead of Ex[:, 1]. That's why x[:, 0:-1] is divided instead of x[:, 1:].
    Due to the need of averaging, the required dimension of the material grid is 1
    dimension larger from the bottom than fields.
    ----------
    Parameters
    ----------
    Hz : np.array:
        Hz fields (-Hy).

    dL : float:
        step size.

    omega : float:
        angular frequency.

    eps_grid : np.array:
        material grid.

    eps_0 : float
        vaccum permittivity.

    -------
    Returns
    -------
    Ex : np.array:
        finite-differenced Ex fields.

    '''

    # Material averaging
    x = 1 / 2 * (eps_grid[:, :, 1:, :] + eps_grid[:, :, 0:-1, :])
    # The returned Ex is corresponding to Ex_ceviche[:, 0:-1]
    Ex_R = (Hz_I[:, 1:, :] - Hz_I[:, 0:-1, :])/dL/omega/eps_0/x[:, 0, 0:-1, :]
    Ex_I = -(Hz_R[:, 1:, :] - Hz_R[:, 0:-1, :])/dL/omega/eps_0/x[:, 0, 0:-1, :]
    return torch.stack((Ex_R, Ex_I), axis = 1)

def Hz_to_Ey(Hz_R: np.array, Hz_I: np.array, dL: float, omega: float, eps_grid: np.array, \
             eps_0: float = eps_0) -> np.array:
    '''
    This function performs finite difference on Hz to obtain corresponding Ey field.

    Note that ceviche implements the finite differnce slightly different from the
    conventional setup. Hence, the difference between Hz[1, :] and Hz[0, :] produces
    Ey[0, :] instead of Ex[1, :]. That's why y[0:-1, :] is divided instead of y[1:, :].
    Due to the periodic structure, the required dimension of the material grid is same as the fields.
    ----------
    Parameters
    ----------
    Hz : np.array:
        Hz fields (-Hy).

    dL : float:
        step size.

    omega : float:
        angular frequency.

    eps_grid : np.array:
        material grid.

    eps_0 : float
        vaccum permittivity.

    -------
    Returns
    -------
    Ey : np.array:
        finite-differenced Ey (Ez) fields.
    '''

    y = 1 / 2 * (eps_grid[:, :, 1:, :] + torch.roll(eps_grid[:, :, 1:, :], 1, dims = 3))

    # The returned Ey corresponds to Ey_ceviche[0:-1, :]
    Ey_R = -(torch.roll(Hz_I, -1, dims = 2) - Hz_I)/dL/omega/eps_0/y[:, 0, :, :]
    Ey_I = (torch.roll(Hz_R, -1, dims = 2) - Hz_R)/dL/omega/eps_0/y[:, 0, :, :]
    return torch.stack((Ey_R, Ey_I), axis = 1)

def E_to_Hz(Ey_R: np.array, Ey_I: np.array, Ex_R: np.array, Ex_I: np.array, dL: float, \
            omega: float, mu_0: float = mu_0) -> np.array:
    '''
    This function performs finite difference on Ey, Ex to obtain corresponding Hz field.
    The -1j in the denominator has been absorbed for Hz -> -Hy
    ----------
    Parameters
    ----------
    Ey : np.array:
        Ey fields (Ez).

    Ex : np.array:
        Ex fields.

    dL : float:
        step size.

    omega : float:
        angular frequency.

    mu_0 : float
        vaccum permeability.

    -------
    Returns
    -------
    Hz : np.array:
        finite-differenced Hz (-Hy) fields.
    '''

    Hz_R = ((Ey_I[:, 1:] - torch.roll(Ey_I[:, 1:], 1, dims = 2)) - (Ex_I[:, 1:] - \
             Ex_I[:, 0:-1]))/dL/omega/mu_0
    Hz_I = -((Ey_R[:, 1:] - torch.roll(Ey_R[:, 1:], 1, dims = 2)) - (Ex_R[:, 1:] - \
              Ex_R[:, 0:-1]))/dL/omega/mu_0
    return torch.stack((Hz_R, Hz_I), axis = 1)

def H_to_H(Hz_R: np.array, Hz_I: np.array, dL: float, omega: float, eps_grid: np.array, \
           eps_0: float = eps_0, mu_0: float = mu_0) -> np.array:

    '''
    This function calls FD_Ex, FD_Ez, and subsequently FD_H to implement the
    Helmholtz equation for the H field.
    '''

    FD_Ex = Hz_to_Ex(Hz_R, Hz_I, dL, omega, eps_grid, eps_0)
    FD_Ez = Hz_to_Ey(Hz_R, Hz_I, dL, omega, eps_grid, eps_0)
    FD_H = E_to_Hz(FD_Ez[:, 0, :-1], FD_Ez[:, 1, :-1], FD_Ex[:, 0], FD_Ex[:, 1], dL, omega, mu_0)
    return FD_H

def regConstScheduler(epoch, args, last_epoch_data_loss, last_epoch_physical_loss):
    '''
    This function scales the physical regularization scaling constant over
    each epoch.
    '''
    if(epoch<1):
        return 0
    else:
        return args.ratio*last_epoch_data_loss/last_epoch_physical_loss

def main(args):

    torch.manual_seed(222)
    torch.cuda.manual_seed_all(222)
    np.random.seed(222)

    print(args)

    device = torch.device('cuda')

    #Select the neural network architecture to use
    model = None
    if args.arch == "UNet":
        model = UNet(args).to(device)
    else:
        raise("architectures other than Unet hasn't been added!!")

    #Initialize the optimizer
    model.optimizer = optim.Adam(model.parameters(), lr=args.lr, eps=1e-7, \
                                 amsgrad=True, weight_decay=args.weight_decay)
    model.lr_scheduler = optim.lr_scheduler.ExponentialLR(model.optimizer, args.exp_decay)

    tmp = filter(lambda x: x.requires_grad, model.parameters())
    num = sum(map(lambda x: np.prod(x.shape), tmp))
    print(model)

    print('Total trainable tensors:', num, flush=True)

    model_path = args.model_save_path + args.model_name + "_batch_size_" + \
                 str(args.batch_size) + "_lr_" + str(args.lr)

    if not os.path.isdir(model_path):
        os.mkdir(model_path)

    #Initialize the datasets
    ds = SimulationDataset(args.data_folder, args.local_data, total_sample_number = args.total_sample_number)
    means = [1e-3, 1e-3]
    print("means: ", means)
    torch.manual_seed(42)

    #Initialize the train and test data loaders. Based on the data pulled from Metanet
    #(see simulation_dataset.py), the first 90% of the data corresponds to the training
    #data, and the last 10% corresponds to the test data. This is thus loaded as
    #subsets accordingly.
    train_ds =  torch.utils.data.Subset(ds, range(int(0.9*len(ds))))

    test_ds = torch.utils.data.Subset(ds, range(int(0.9*len(ds)), len(ds)))

    #Uncomment this code if you would like to instead randomly split the data into
    #training and test datasets.
    # train_ds, test_ds = random_split(ds, [int(0.9*len(ds)), len(ds) - int(0.9*len(ds))])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)

    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)

    train_mean = 0
    test_mean = 0

    # first get the mean-absolute-field value:
    for sample_batched in train_loader:
        train_mean += torch.mean(torch.abs(sample_batched["field"]))
    for sample_batched in test_loader:
        test_mean += torch.mean(torch.abs(sample_batched["field"]))
    train_mean /= len(train_loader)
    test_mean /= len(test_loader)

    print("total training samples: %d, total test samples: %d, train_abs_mean: %f, test_abs_mean: %f" \
          % (len(train_ds), len(test_ds), train_mean, test_mean), flush=True)

    #Initialize the statistics collection, using a pandas dataframe
    df = pd.DataFrame(columns=['epoch','train_loss', 'train_phys_reg', 'test_loss', 'test_phys_reg'])

    train_loss_history = []
    train_phys_reg_history = []

    test_loss_history = []
    test_phys_reg_history = []

    start_epoch=0
    if (args.continue_train):
        print("Restoring weights from ", model_path+"/last_model.pt", flush=True)
        checkpoint = torch.load(model_path+"/last_model.pt")
        start_epoch=checkpoint['epoch']
        model = checkpoint['model']
        model.lr_scheduler = checkpoint['lr_scheduler']
        model.optimizer = checkpoint['optimizer']
        df = read_csv(model_path + '/'+'df.csv')

    scaler = GradScaler()

    best_loss = float('inf')
    last_epoch_data_loss = 1.0
    last_epoch_physical_loss = 1.0
    for step in range(start_epoch, args.epoch):
        print("epoch: ", step, flush=True)
        reg_norm = regConstScheduler(step, args, last_epoch_data_loss, last_epoch_physical_loss)

        # training
        for sample_batched in train_loader:
            model.optimizer.zero_grad()

            x_batch_train, y_batch_train, eps_distr = sample_batched['structure'].to(device), \
                                                      sample_batched['field'].to(device), \
                                                      sample_batched['eps_distr'].to(device)
            with autocast():
                logits = model(x_batch_train, bn_training=True)
                #calculate the loss using the ground truth
                loss = model.loss_fn(logits, y_batch_train)

                logits = logits[:,:,1:-1, :]

                # Calculate physical residue
                pattern = torch.cat((torch.ones([x_batch_train.shape[0], 1, 1, 256], \
                                     dtype = torch.float32, device = device)*n_sub**2, \
                                     x_batch_train), dim=2)

                # predicted fields [Hy_R, Hy_I, Ex_R, Ex_I, Ez_R, Ez_I]
                fields = torch.cat((y_batch_train[:, :, 0:1, :], logits, y_batch_train[:, :, -1:, :]), dim=2)
                FD_Hy = H_to_H(-fields[:, 0]*means[0], -fields[:, 1]*means[1], dL, omega, pattern)

                #Calculate the physical residue for the real and imaginary components
                phys_regR = model.loss_fn(FD_Hy[:, 0]/means[0], logits[:, 0])*reg_norm
                phys_regI = model.loss_fn(FD_Hy[:, 1]/means[1], logits[:, 1])*reg_norm

                #Add the real and imaginary physical residue components to yield the total loss
                loss += phys_regR + phys_regI

                #Backpropagate the loss
                scaler.scale(loss).backward()
                scaler.step(model.optimizer)
                scaler.update()

        #Save the weights at the end of each epoch
        checkpoint = {
                        'epoch': step,
                        'model': model,
                        'optimizer': model.optimizer,
                        'lr_scheduler': model.lr_scheduler
                     }
        torch.save(checkpoint, model_path+"/last_model.pt")


        # evaluation
        train_loss = 0
        train_phys_reg = 0
        for sample_batched in train_loader:
            x_batch_train, y_batch_train, eps_distr = sample_batched['structure'].to(device), \
                                                      sample_batched['field'].to(device), \
                                                      sample_batched['eps_distr'].to(device)

            with torch.no_grad():
                logits = model(x_batch_train, bn_training=False)
                loss = model.loss_fn(logits, y_batch_train)

                logits = logits[:, :, 1:-1, :]

                # Calculate physical residue
                pattern = torch.cat((torch.ones([x_batch_train.shape[0], 1, 1, 256], \
                                     dtype = torch.float32, device = device)*n_sub**2, \
                                     x_batch_train), dim=2)

                # predicted fields [Hy_R, Hy_I, Ex_R, Ex_I, Ez_R, Ez_I]
                fields = torch.cat((y_batch_train[:, :, 0:1, :], logits, y_batch_train[:, :, -1:, :]), dim=2)
                FD_Hy = H_to_H(-fields[:, 0]*means[0], -fields[:, 1]*means[1], dL, omega, pattern)

                phys_regR = model.loss_fn(FD_Hy[:, 0]/means[0], fields[:, 0, 1:-1, :])*reg_norm
                phys_regI = model.loss_fn(FD_Hy[:, 1]/means[1], fields[:, 1, 1:-1, :])*reg_norm

                train_loss += loss
                train_phys_reg += 0.5*(phys_regR + phys_regI)

        train_loss /= len(train_loader)
        train_phys_reg /= len(train_loader)

        # testing
        test_loss = 0
        test_phys_reg = 0
        for sample_batched in test_loader:
            x_batch_test, y_batch_test, eps_distr_test = sample_batched['structure'].to(device), \
                                                         sample_batched['field'].to(device), \
                                                         sample_batched['eps_distr'].to(device)

            with torch.no_grad():
                logits = model(x_batch_test, bn_training=False)
                loss = model.loss_fn(logits, y_batch_test)


                logits = logits[:, :, 1:-1, :]
                # Calculate physical residue

                pattern = torch.cat((torch.ones([x_batch_test.shape[0], 1, 1, 256], \
                                     dtype = torch.float32, device = device)*n_sub**2, \
                                     x_batch_test), dim=2)

                # predicted fields [Hy_R, Hy_I, Ex_R, Ex_I, Ez_R, Ez_I]
                fields = torch.cat((y_batch_test[:, :, 0:1, :], logits, y_batch_test[:, :, -1:, :]), dim=2)
                FD_Hy = H_to_H(-fields[:, 0]*means[0], -fields[:, 1]*means[1], dL, omega, pattern)

                phys_regR = model.loss_fn(FD_Hy[:, 0]/means[0], fields[:, 0, 1:-1, :])
                phys_regI = model.loss_fn(FD_Hy[:, 1]/means[1], fields[:, 1, 1:-1, :])

                test_loss += loss
                test_phys_reg += 0.5*(phys_regR + phys_regI)

        test_loss /= len(test_loader)
        test_phys_reg /= len(test_loader)
        last_epoch_data_loss = test_loss
        last_epoch_physical_loss = test_phys_reg.detach().clone()
        test_phys_reg *= reg_norm

        print('train loss: %.5f, test loss: %.5f' % (train_loss, test_loss), flush=True)

        model.lr_scheduler.step()

        df = df.append({'epoch': step+1, 'lr': str(model.lr_scheduler.get_last_lr()),
                        'train_loss': train_loss.item(),
                        'train_phys_reg': train_phys_reg.item(),
                        'test_loss': test_loss.item(),
                        'test_phys_reg': test_phys_reg.item(),
                       }, ignore_index=True)

        df.to_csv(model_path + '/'+'df.csv',index=False)

        if(test_loss<best_loss):
            best_loss = test_loss
            checkpoint = {
                            'epoch': step,
                            'model': model,
                            'optimizer': model.optimizer,
                            'lr_scheduler': model.lr_scheduler
                         }
            torch.save(checkpoint, model_path+"/best_model.pt")


if __name__ == '__main__':

    # Define all the arguments that can be passed to the script
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--epoch', type=int, help='number of training epochs', default=200)
    argparser.add_argument('--imgc', type=int, help='number of input channels of the network', default=1)
    argparser.add_argument('--outc', type=int, help='number of output channels of the network', default=2)
    argparser.add_argument('--batch_size', type=int, help='batch size', default=32)
    argparser.add_argument('--lr', type=float, help='initial learning rate', default=1e-4)
    argparser.add_argument("--data_folder", type=str, help='folder for the data', default="./")
    argparser.add_argument("--total_sample_number", type=int, \
                           help="total number of training and testing samples to \
                           take from the npy file (in case you don't want to use \
                           all the data there)", default=None)
    argparser.add_argument("--arch", type=str, help='architecture of the learner', default="UNet")
    argparser.add_argument('--num_down_conv', type=int, help='number of down conv \
                           blocks in Unet', default=6)
    argparser.add_argument("--hidden_dim", type=int, help='width of Unet, i.e. \
                           number of kernels of first block', default=16)
    argparser.add_argument("--model_save_path", type=str, help="the root dir \
                           to save checkpoints", default="./network_weights/")
    argparser.add_argument("--model_name", type=str, help="name for the model, \
                           used for storing under the model_save_path", \
                           default="waveynet_acsphotonics")
    argparser.add_argument("--exp_decay", type=float, help="exponential decay of \
                            learning rate, update per epoch", default=0.98)
    argparser.add_argument("--continue_train", type=bool, help = "if true, continue \
                            train from continue_epoch", default=False)
    argparser.add_argument("--alpha", type=float, help="negative slope of leaky relu", default=0.3)
    argparser.add_argument("--ratio", type=float, help="ratio weight for physical loss", default=0.5)
    argparser.add_argument("--weight_decay", type=float, help="l2 regularization coeff", default=3e-3)
    argparser.add_argument("--reg_norm", type=float, help="normalization for the \
                            physical regularizer", default = 5)
    argparser.add_argument("--local_data", type=bool, help="If False, data is pulled from Metanet.\
                            If True, it is assumed that the data has been pre-downloaded from\
                            Metanet and it is directly imported into memory from the same directory\
                            as this script.", default=False)
    args = argparser.parse_args()

    # Call the training function, passing the arguments
    main(args)
