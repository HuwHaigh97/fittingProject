import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as tck
from iminuit import Minuit
import scipy
import math
import random


class PeakAnalyzer:
    """
    This class should allow to find and fit multiple peaks in a given signal.
     
    Methods
    -------
    __init__()
        Constructor - add arguments as needed
    fit()
        Fit signal y(x) and return list of centroids (= central channel of a
        peak) and the fitted function = a sum of multiple Gaussians, one for each
        peak in the signal. The fit parameters for a single Gaussian are
        * the centroid mu
        * the area
        * the width sigma
    evaluate()
        Evaluate fitted function for a given channel.
    """

    def __init__(self, x_data, y_data):
        """
        Constructor - add arguments as needed.

        Parameters:
        x_data      (np.array/list): X values.
        y_data      (np.array/list): Y values.

        """
        self.x_data = x_data
        self.y_data = y_data

        #Estimate number of peaks and initial guesses on centroids and widths
        self.peakEstimates, self.widthEstimates = self.estimatePeaks(width=[4,50], prominence=[40., None])
        self.numPeaks = len(self.peakEstimates)


    def model(self, x, **parameters):
        """
        Model of linear background plus estimated number of gaussian peaks.

        Parameters:
        **parameters (dict): linear (m, c) + gaussian parameters (amplitude, mean, width).
        
        """
        
        linearPart = parameters["a"] * x + parameters["b"]
        
        #Add individual gaussians to full model
        gaussPart = 0
        for i in range(self.numPeaks):
            A = parameters[f"A{i}"]
            mu = parameters[f"mu{i}"]
            sigma = parameters[f"sigma{i}"]
            gaussPart += A*np.exp(-np.power(x - mu, 2.) / (2 * np.power(sigma, 2.)))
        
        return linearPart + gaussPart


    def residual_sum_of_squares(self, *params):
        """
        Calculate squared sum of residuals from model prediction

        Parameters:
        *params     (list/array): model parameters.
        """

        parameters = {"a":params[0], "b":params[1]}
        for i in range(self.numPeaks):
            parameters[f"A{i}"] = params[2 + i * 3]
            parameters[f"mu{i}"] = params[3 + i * 3]
            parameters[f"sigma{i}"] = params[4 + i * 3]
        
        #Get predicted values 
        y_pred = self.model(self.x_data, **parameters)

        return np.sum((self.y_data - y_pred) ** 2)


    def estimatePeaks(self, width=None, prominence=None):
        """ 
        Estimate central values and widths of peaks to provide initial params for fit. 

        Parameters:
        width       (list/tuple): upper and lower limits of widths of possible peaks.
        prominence  (list/tuple): upper and lower limits of prominence (height wrt neighbouring values) of possible peaks.

        returns:
        peaks       (list): The estimated peak centroids across the set of data.
        widths      (list): The estimated peak widths across the set of data.

        """
        peaks, peakDict = scipy.signal.find_peaks(self.y_data, width=width, prominence = prominence)

        return(peaks, peakDict['widths'])


    def fit(self, a_init=1.0, b_init=0.0, amplitudeEstimate=150.):
        """Identify peaks in signal y and fit with multiple Gaussian peaks.

        Parameters
        ----------
        a_init      (float): Initial estimate of background slope. 
        b_init      (float): Initial estimate of background y-intersect. 


        returns
        ------
        params_dict (dict): Dictionary containing all fitted parameters.
        """

        #Create parameter dictionary to be passed to minuit for fitting
        params = {"a": a_init, "b": b_init}

        for i in range(self.numPeaks):
            params[f"A{i}"] = amplitudeEstimate
            params[f"mu{i}"] = self.peakEstimates[i]
            params[f"sigma{i}"] = self.widthEstimates[i] 

        # Set up model to fit
        m = Minuit(self.residual_sum_of_squares, name=list(params.keys()), **params)

        #Set limits for parameters
        m.limits["a"] = (-5, 5)
        m.limits["b"] = (-100, 100)

        #Keep the mean and width limited to sensible values around the estimates
        for i in range(self.numPeaks):
            m.limits[f"A{(i)}"] =  (25, 500)
            m.limits[f"mu{(i)}"] = (max(self.peakEstimates[i]-4*self.widthEstimates[i], 0), min(self.peakEstimates[i]+4*self.widthEstimates[i], 512))
            m.limits[f"sigma{(i)}"] = (0.1*self.widthEstimates[i], 2*self.widthEstimates[i])

        #Run the fit 
        m.migrad()

        #Confirm fit is valid, could also implement re-running with modified initial values or range here
        if m.fmin.is_valid:
            print("Fit converged accurately!")    
        else:
            print("Fit did not converge.")

        #Print the fitted parameters 
        print(m.params)

        params_dict = {name: m.values[name] for name in m.parameters}
        return params_dict


    def plot(self, fittedParams=None):
        """
        Plot the scatter data and the fitted model with individual components.

        Parameters:
        fittedParams     (dictionary): fitted model parameters.

        """

        #Ensure fit has actually been conducted first
        if not fittedParams:
            raise ValueError("Fit the model first before plotting, pass the fit parameter dictionary.")

        fig, axs = plt.subplots(1,1,figsize=(9,7))

        # Scatter plot of the data
        axs.scatter(self.x_data, self.y_data, color='black', label="Data", s=5)

        # Plot the fitted model (linear + Gaussians)
        xRange = np.linspace(min(self.x_data), max(self.x_data), 2000)

        y_fit = self.model(xRange, **fittedParams)

        #Plot total model                         
        axs.plot(xRange, y_fit, ls='-', color='tab:red', lw=2, label=f"Total PDF")

        #Plot bakcground component
        axs.plot(xRange, fittedParams['a']*xRange + fittedParams['b'], ls='--', color='gray', lw=2, 
        label=f"y = {np.round(fittedParams['a'], 2)}x + {np.round(fittedParams['b'],2)}")

        #Plot each of the gaussian peaks
        for i in range(self.numPeaks):
            gauss = fittedParams[f"A{i}"] * np.exp(-0.5 * ((xRange - fittedParams[f"mu{i}"]) / fittedParams[f"sigma{i}"]) ** 2)
            axs.plot(xRange, gauss, ls='--', lw=2, 
                    label=r"$\mu$ ="+str(np.round(fittedParams[f"mu{i}"], 2)) + " | " + r"$\sigma$ ="+str(np.round(fittedParams[f"sigma{i}"], 2)))

        #Make the plot look a bit better
        axs.yaxis.set_minor_locator(tck.AutoMinorLocator())
        axs.xaxis.set_minor_locator(tck.AutoMinorLocator())
        axs.tick_params(axis='both', direction='in', which='both', labelsize=20, top=True, right=True, length=5)
        axs.tick_params(axis='both', which='major', top=True, right=True, length=6)
        axs.set_xlabel('Channel', fontsize=26)
        axs.set_ylabel('Signal', fontsize=26)
        axs.legend(frameon=False, fontsize=14)
        plt.show()



def gaussian(x, mu, sig):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))

#Linear background function 
def linear_bgd(x, m, c):
    return(m*x+c)

if __name__ == "__main__":

    n = 512
    max_area = 200
    min_area = 50
    noise_amp = 5.0
    max_nPeaks = 4

    # Random peak parameters
    peaks = [(n * np.random.rand(), min_area + (max_area - min_area) * np.random.rand())
             for p in range(np.random.randint(1, max_nPeaks+1))]

    # Build signal to analyze
    y = np.zeros(n)
    x = np.linspace(0, n - 1, n)
    for p in peaks:
        centroid = p[0]
        area = p[1]
        
        #Added some randomisation to the width
        sigma = random.uniform(3,15)
        y = y + area * gaussian(x, centroid, sigma)

    noise = np.random.normal(0.0, noise_amp, y.shape)

    #Adding a linear component to the background just for a bit of interest, this could also go to higher orders. 
    m = np.random.normal(0, 0.05)
    c = np.random.randint(100)
    y += noise + linear_bgd(x, m, c)
    y = np.clip(y, 0, 1e6)

    # Find centroids and fit Gaussian peaks
    analyzer = PeakAnalyzer(x, y)
    fitPars = analyzer.fit(a_init=0.1, b_init=40.0)
    print(fitPars)

    analyzer.plot(fitPars)