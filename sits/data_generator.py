import numpy as np
import matplotlib.pyplot as plt
import mysrc.constants as cst

# -----------------------------------------------------------------------
def computingAbsMinMaxPerImagePerBand(Xt):
    Xt_shape = Xt.shape
    min_per_image = np.reshape(np.amin(Xt, axis=(1, 2)),(Xt_shape[0],1,1,Xt_shape[3]))
    max_per_image =  np.reshape(np.amax(Xt, axis=(1, 2)),(Xt_shape[0],1,1,Xt_shape[3]))
    return min_per_image, max_per_image

# -----------------------------------------------------------------------
def normMinMaxPerImagePerBand(X, min_per, max_per, back=False):
    if back == True:
        return X * (max_per - min_per) + min_per
    else:
        return (X - min_per) / (max_per - min_per)
# -----------------------------------------------------------------------
class DG(object):
    """
    Data augmentation. First data are systematically shifted left and right of xshift ( xshift = [1,2]). We thus have the orginal histo plus
    the 4 shifted (5 in total). On this we add gaussian noise (norm, gauss add, mask, norm back). In total we have 10 samples from 1 histo
    After that we add gaussian to the corresponding yield as well. This will double the samples (20 from 1)

    Assumption: histograms are raw data or mormalized betwen 0 and 0, i.e. zeros are true zeros (to be masked)
    """

    def __init__(self, Xt_full, region_ohe, y, Xshift=False, Xnoise=False, Xmin_max_standardized_noiseSD=0.05,
                 Ynoise=False, Ymin_max_standardized_noiseSD=0.05):
        """Instantiates the class with metadata"""
        self.X = Xt_full         # the full set histograms to be augmented (must be complete histo, i.e. 36 dekads, shape (510 samples,64 y-bins,36 x-time-steps,4-bands)
        self.X_augmented = None
        self.region_ohe = region_ohe
        self.region_ohe_augmented = None
        self.y = y              # the var to be estimated, can be yield or area
        self.y_augmented = None
        self.Xshift = Xshift
        self.Xnoise = Xnoise
        self.Xmin_max_standardized_noiseSD = Xmin_max_standardized_noiseSD
        self.Ynoise = Ynoise
        self.Ymin_max_standardized_noiseSD = Ymin_max_standardized_noiseSD

    def generate(self, lenTS, subset_bool):
        # lenTS: the length of the time series to return (starting from index 3 that is 1st of Sep)
        # subset_bool: train samples to be augmented

        # set augmented arrays to the original data arrays
        self.X_augmented = self.X[subset_bool, :, :, :]
        X_current = self.X_augmented.copy()
        self.region_ohe_augmented = self.region_ohe[subset_bool,:]
        #self.groups_augmented = self.groups[subset_bool]
        self.y_augmented = self.y[subset_bool,:]

        if self.Xshift == True:
            # 1 - shift left (no matter if we leave the last deks unchanged, they will not be used)
            self.X_augmented = np.concatenate((self.X_augmented, np.roll(X_current, -1, axis=2)), axis=0)
            self.X_augmented = np.concatenate((self.X_augmented, np.roll(X_current, -2, axis=2)), axis=0)
            # 2 - shift right
            self.X_augmented = np.concatenate((self.X_augmented, np.roll(X_current, 1, axis=2)), axis=0)
            self.X_augmented = np.concatenate((self.X_augmented, np.roll(X_current, 2, axis=2)), axis=0)
            # add unchanged data for the other variables
            self.region_ohe_augmented = np.tile(self.region_ohe_augmented, (5,1))
            #self.groups_augmented = np.tile(self.groups_augmented, 5)
            self.y_augmented = np.tile(self.y_augmented, (5, 1))
            if False:
                variables = ['NDVI', 'Radiation', 'Rainfall', 'Temperature']
                fig, axs = plt.subplots(2, 4, figsize=(16.5, 7))
                cmaps = ['Greens', 'Blues', 'Purples', 'Reds']
                for col in range(len(cmaps)):
                    ax = axs[0,col]
                    plt.sca(ax)
                    pcm = ax.imshow(np.flipud( self.X_augmented[0,:, :, col]), cmap=cmaps[col])
                    fig.colorbar(pcm, ax=ax)
                    plt.title(variables[col])
                    ax = axs[1, col]
                    plt.sca(ax)
                    pcm = ax.imshow(np.flipud(self.X_augmented[510, :, :, col]), cmap=cmaps[col])
                    fig.colorbar(pcm, ax=ax)
                    plt.title(variables[col])
                plt.tight_layout()
                plt.show()

        if self.Xnoise == True:
            n_before_noise = self.X_augmented.shape[0]
            # X data can come normalize min max (min hard coded to 0) to 0-1 (so min is actually 0 count) or not
            # so I normalize again here (if it is already norm has no effect).
            # I have to normalize count (0 to n)  in [0,1] to apply a gaussian noise with 0 mean and SD
            # But we don't want to add noise in 0 count grid cell, so I have to mask the zeros and keep them zeros
            # Normalize
            min_per_image, max_per_image = computingAbsMinMaxPerImagePerBand(self.X_augmented)
            X0 = normMinMaxPerImagePerBand(self.X_augmented, min_per_image, max_per_image)
            # add noise
            X0 = X0 + np.random.normal(0, self.Xmin_max_standardized_noiseSD, X0.shape)
            # set back to zero those that were 0
            X0[self.X_augmented == 0] = 0
            # adding noise can result in negative values, clip to zeros if there are negative values
            X0[X0 < 0] = 0
            # now denormalize back and add to augmented sample
            self.X_augmented = np.concatenate((self.X_augmented, normMinMaxPerImagePerBand(X0, min_per_image, max_per_image, back=True)), axis=0)
            # add data for the other variables
            #self.region_ohe_augmented = np.tile(self.region_ohe_augmented, 2)
            self.region_ohe_augmented = np.tile(self.region_ohe_augmented, (2, 1))
            #self.groups_augmented = np.tile(self.groups_augmented, 2)
            self.y_augmented = np.tile(self.y_augmented, (2, 1))
            if False:
                id2plt = 1000 # refers to a sample before adding noise (must be < n_before_noise)
                fig, axs = plt.subplots(3, 4, figsize=(16.5, 8))
                cmaps = ['Greens', 'Blues', 'Purples', 'Reds']
                for col in range(len(cmaps)):
                    ax = axs[0,col]
                    plt.sca(ax)
                    pcm = ax.imshow(np.flipud(self.X_augmented[id2plt,:, :, col]), cmap=cmaps[col])
                    fig.colorbar(pcm, ax=ax)
                    zeros = self.X_augmented[id2plt, :, :, col].copy()
                    zeros[zeros !=0] = np.nan
                    zeros[zeros == 0] = 1
                    pcm = ax.imshow(np.flipud(zeros), cmap='gray')
                    variables = ['NDVI', 'Radiation', 'Rainfall', 'Temperature']
                    plt.title(variables[col])

                    ax = axs[1, col]
                    #pcm = ax.imshow(np.flipud(X_noisy[id2plt, :, :, col]), cmap=cmaps[col])plt.sca(ax)
                    pcm = ax.imshow(np.flipud(self.X_augmented[id2plt+n_before_noise, :, :, col]), cmap=cmaps[col])
                    fig.colorbar(pcm, ax=ax)
                    zeros = self.X_augmented[id2plt+n_before_noise, :, :, col].copy() #X_noisy[id2plt, :, :, col].copy()
                    zeros[zeros != 0] = np.nan
                    zeros[zeros == 0] = 1
                    pcm = ax.imshow(np.flipud(zeros), cmap='gray')
                    variables = ['noisy_NDVI', 'noisy_Radiation', 'noisy_Rainfall', 'noisy_Temperature']
                    plt.title(variables[col])

                    ax = axs[2, col]
                    plt.sca(ax)
                    #noise = X_noisy[id2plt,:, :, col] - self.X_augmented[id2plt,:, :, col]
                    noise = self.X_augmented[id2plt+n_before_noise, :, :, col] - self.X_augmented[id2plt, :, :, col]
                    pcm = ax.imshow(np.flipud(noise), cmap=cmaps[col])
                    fig.colorbar(pcm, ax=ax)
                    zeros = noise.copy()
                    zeros[zeros != 0] = np.nan
                    zeros[zeros == 0] = 1
                    pcm = ax.imshow(np.flipud(zeros), cmap='gray')
                    variables = ['noise_NDVI', 'noise_Radiation', 'noise_Rainfall', 'noise_Temperature']
                    plt.title(variables[col])
                plt.tight_layout()
                plt.show()
        if self.Ynoise == True:
            # as we did for X we normalize 0-1, add noise, clip to >= 0 and back to value
            # Normalize
            min_per_crop = np.amin(self.y_augmented, axis=0)
            max_per_crop = np.amax(self.y_augmented, axis=0)
            y0 = normMinMaxPerImagePerBand(self.y_augmented, min_per_crop, max_per_crop)
            # add noise
            y0 = y0 + np.random.normal(0, self.Ymin_max_standardized_noiseSD, y0.shape)
            # be care it can give negative values, clip to zeros if there were negative values
            y0[y0 < 0] = 0
            # now denormalize back and add to sample
            self.y_augmented = np.concatenate((self.y_augmented, normMinMaxPerImagePerBand(y0, min_per_crop, max_per_crop, back=True)), axis=0)
            # add data for the other variables
            #self.region_id_augmented = np.tile(self.region_id_augmented, 2)
            self.region_ohe_augmented = np.tile(self.region_ohe_augmented, (2, 1))
            #self.groups_augmented = np.tile(self.groups_augmented, 2)
            self.X_augmented = np.tile(self.X_augmented, (2,1,1,1))
        # adjust dimension of lenTS
        first = (cst.first_month_input_local_year) * 3
        return self.X_augmented[:,:,first:first+lenTS,:], self.region_ohe_augmented, self.y_augmented

