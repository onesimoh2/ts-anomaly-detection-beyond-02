import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader, Subset
from torch.nn import functional as F
from utilities import DateUtils, variance
from fft_functions import fourier_extrapolation, fourierPrediction
import matplotlib.pyplot as plt

class FeatureDatasetFromDf(Dataset):


    def auto_inc(self, ipos):
        self.iPos += 1
        return ipos

    def __init__(self, *args):
        if len(args) == 7:
            df, scaler, fit_dat, columns_names, dateName, ser_pos, n_df = args[0], args[1], args[2], args[3], args[4], args[5], args[6]
        else:
            df, scaler, fit_dat, columns_names, dateName, ser_pos, n_df, self.extrpl, self.x_freqdom, self.f, self.p, self.indexes, self.n_train = args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7], args[8], args[9], args[10], args[11], args[12]

        x = df 
        x = x.reset_index()
        
        self.iPos = ser_pos
        self.x_train = np.array(x[columns_names].values)
        if(fit_dat == 'true'):
            # create the complex coefficients and the rest of the required parameters 
            x_in = df[columns_names].values.flatten()           
            self.extrpl, self.x_freqdom, self.f, self.p, self.indexes, self.n_train = fourier_extrapolation(x_in, 0)


        # use the coefficients and the rest of the parameters to calculate the nonlinear tend and the seasonality part 
        forcastVals, restored_sig, trend = fourierPrediction(x, self.x_freqdom, self.f, self.p, self.indexes, self.n_train, columns_names)
        
        axFFT = plt.axes()
        tt = np.arange(0, len(trend))
        axFFT.plot(tt, np.abs(trend))
        axFFT.plot(tt, np.abs(restored_sig))
        plt.show()        
        #nDfLast = DateUtils.calc_day(x[dateName].iloc[-1]) 
        #nDf1 = int(round(nDfLast * 0.033115)) #int(nDf*0.066)
        nDf1 = n_df
        #biasRand = [xrand_position[DateUtils.calcorderFromDay(item)] for item in x[dateName]]
        
        
      


        i = 0
                     
        
        #forcastValsR = np.array(forcastVals).reshape(-1, 1)
        trend_reshape = np.array(trend).reshape(-1, 1)
        seasonal_part = np.array(restored_sig).reshape(-1, 1)
        
        self.x_train = np.append(self.x_train, trend_reshape, axis=1)
        self.x_train = np.append(self.x_train, seasonal_part, axis=1)

        if(fit_dat == 'true'):            
            self.x_train[:,[0,1,2]] = scaler.fit_transform(self.x_train[:,[0,1,2]].reshape(-1, 3)).reshape(-1, 3)            
        else:
            self.x_train[:,[0,1,2]] = scaler.transform(self.x_train[:,[0,1,2]].reshape(-1, 3)).reshape(-1, 3)               

        #biasAddedSeed = [((item[0]*item[0]) + (item[1]*item[1])  + (item[2]*item[2])) for item in self.x_train]
        data_vs_trend = [np.abs(item[0]-item[1]) for item in self.x_train]
        data_vs_trend_reshape = np.array(data_vs_trend).reshape(-1, 1)
        self.x_train = np.append(self.x_train, data_vs_trend_reshape, axis=1)

        #x_train have; data, train estimate, stationary estimate and difference between the first two 

        #trainn.to_csv('C:/Users/ecbey/Downloads/x_train.csv')  
        self.X_train = torch.tensor(self.x_train, dtype=torch.float32)

    def __len__(self):
        return len(self.x_train)
    
    def __getitem__(self, idx):
        return self.X_train[idx]


#definition of all the features for creating a model, training, testing and execution. 
class autoencoder(nn.Module):

    #iitialize weights
    def initialize_weights(self):
        for n in self.modules():
            if isinstance(n, nn.Linear):
                nn.init.kaiming_uniform_(n.weight)
                nn.init.constant_(n.bias, 0)


    def __init__(self, epochs=15, batchSize=10, learningRate=1e-3, weight_decay=1e-5, layer_reduction_factor = 1.6, number_of_features = 29, seed=15000):
        super(autoencoder, self).__init__()
        #seed = 15000
        torch.manual_seed(seed)
        np.random.seed(seed)

        self.epochs = epochs
        self.batchSize = batchSize
        self.learningRate = learningRate
        self.weight_decay = weight_decay
        self.number_0f_features = number_of_features

        #defining the structure of the autoencoder, this is a general method that should fit different structure depending on the number of input nodes 
        self.first_encode_layer = nn.Linear(4, 3)
        self.get_mean = nn.Linear(3, 2)
        self.get_std = nn.Linear(3, 2)
        self.first_decode_layer = nn.Linear(2, 3)
        self.second_decode_layer = nn.Linear(3, 4)
        self.initialize_weights()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learningRate, weight_decay=self.weight_decay)     
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=5, factor=0.1, mode='min', verbose=True)

        
    def encoder(self, x):
        x = F.relu(self.first_encode_layer(x))
        return self.get_mean(x), self.get_std(x)

    def decoder(self, z):
        h3 = F.relu(self.first_decode_layer(z))
        return self.second_decode_layer(h3)
            
    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5*logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu
        
         # Reconstruction + KL divergence losses summed over all elements and batch
    def loss_function(self,recon_x, x, mu, logvar):
        #BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')
        BCE = rr = F.mse_loss(recon_x, x)
        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        return BCE + KLD   
        
        #self.loss = nn.MSELoss()

    def loss_function_reduction_none(self,recon_x, x, mu, logvar):
        #BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')
        BCE = rr = F.mse_loss(recon_x, x)
        test_loss_list = rr = F.mse_loss(recon_x, x, reduction='none')
        numCols = test_loss_list.size(dim=1)
        last_epoch_individual_loss = []
        
        sumAll = 0.0 #calculate individual loss
        for xsqTen in test_loss_list:
            sumAll = 0.0
            for xsq in xsqTen:
                sumAll = sumAll + xsq

            indivAve = float(sumAll/numCols)
            last_epoch_individual_loss.append(indivAve)
        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        return BCE + KLD, last_epoch_individual_loss   
    
    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar


    def train_only(self, train_sample, max_training_loss_var_num):                               
        train_ave = []
        last_epoch_loss = []
        last_epoch_individual_loss_all = []
        max_training_loss = 0
        #criterion_no_reduced = nn.MSELoss(reduction = 'none')

        for epoch in range(self.epochs):
            self.train() 
            train_epc = 0.0
            train_num = 0.0
            for data in train_sample:
                #predict
                recon_batch, mu, logvar =  self.forward(data)
                #output = self(data)
                # find loss
                if epoch + 1 == self.epochs:
                    loss, last_epoch_individual_loss = self.loss_function_reduction_none(recon_batch, data, mu, logvar)
                    last_epoch_individual_loss_all.extend(last_epoch_individual_loss)
                else:
                    loss = self.loss_function(recon_batch, data, mu, logvar)
                
                lossTrainData = loss.data.item() 
                train_epc = train_epc + lossTrainData
                train_num =  train_num + 1
                # perform back propagation
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            loss_train_data = loss.data.item() 
            #average of the losses for a given epoch
            epoc_t_ave = train_epc/train_num
            #add all losses in an array
            train_ave.append(float(epoc_t_ave))
            
            print(f'******epoch {epoch + 1}, loss: {train_ave[epoch]:.4f}')
            
            #supply the current loss for the scheduler
            min_loss_round = round(train_ave[epoch], 4)
            self.scheduler.step(min_loss_round)
            #accumulae the losses for each element
            train_epc = train_epc + loss_train_data
            train_num =  train_num + 1
             

            #supply the current loss for the scheduler
            # min_loss_round = round(train_ave[epoch], 4)
            # self.scheduler.step(min_loss_round)
       

        #calculate the theashold to detect anomalies
        mean, var, sig = variance(last_epoch_individual_loss_all)
        max_training_loss = mean +  (max_training_loss_var_num * sig)
        return max_training_loss, train_ave



    def execute_evaluate(self, feature_sample, max_training_loss, index_df):
        self.eval()
        indx = 0
        test_epc = 0.0
        test_num = 0.0
        detected_anomalies = []
        
        with torch.no_grad(): # Run without Autograd
            for original in feature_sample:
                recon_batch, mu, logvar = self.forward(original)  # model can't use test to learn
                test_loss = self.loss_function(recon_batch, original, mu, logvar)

                
            
                test_epc = test_epc + test_loss
                test_num = test_num + 1

                indx1 = index_df.iloc[[indx], [0]]
                print('test_loss=', test_loss, ' Indx=', indx)

                if test_loss > (1 * max_training_loss) :                    
                    item = [test_loss, int(indx1['ID123'])]
                    detected_anomalies.append(item)
                    #print('          test_loss=', test_loss, ' Indx=', indx1['ID123'])
                    #pd.DataFrame(original)
                

                indx = indx + 1
    
        print('max_training_loss=', max_training_loss )
        test_loss = (test_epc/test_num)
        pcent_anomalies_detected = (len(detected_anomalies) / len(feature_sample)) * 100
        #print(f'     Validate_loss: {test_loss:.4f}')
        return detected_anomalies, pcent_anomalies_detected, test_loss

