#!/usr/bin/env python
# coding: utf-8

# In[1]:


# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# from scipy.fft import fft, fftfreq
# from scipy.signal import find_peaks
# import calendar
# import statsmodels.api as sm
# from datetime import datetime


# In[2]:


class ExploratoryDataAnalysis:
    def __init__(self, data):
        self.data = data
        
    def CleaningData(self):
        # re-name the columns of data
        self.data.columns = ['Date', 'Temp']
        
        # change the temperature string type to float
        try :    
            self.data["Temp"] = self.data["Temp"].str.replace(',', '.', regex=True)
            self.data["Temp"] = self.data["Temp"].astype(float)
        except :
            pass
        
        # make the date column a pandas datetime type
        self.data['Date'] = pd.to_datetime(self.data['Date'],dayfirst=True)
        
        # Define the upper and lower bounds for extreme values
        upper_bound = self.data['Temp'].mean() + (5 * self.data['Temp'].std())
        lower_bound = self.data['Temp'].mean() - (5 * self.data['Temp'].std())

        # Replace values outside the bounds with NaN values
        self.data.loc[(self.data['Temp'] > upper_bound) | (self.data['Temp'] < lower_bound), 'Temp'] = np.nan
        self.DataClean = self.data.copy()
        return self.data
    
    def RemovingLeafDays(self):
        
        self.DataClean = self.CleaningData()
        
        # create column indicating whether each date is in a leap year or not
        self.DataClean['is_leap_year'] = self.DataClean['Date'].dt.is_leap_year

        # filter out rows corresponding to the 29th of February in leap years
        self.DataClean = self.DataClean[~((self.DataClean['Date'].dt.month == 2) & (self.DataClean['Date'].dt.day == 29) & self.DataClean['is_leap_year'])]
        return self.DataClean
    
    def DailySeasonality(self):
        # extracts the time component from the 'Date' column
        self.DataClean['time'] = self.DataClean['Date'].dt.time
        
        # The following line calculates a decimal representation of time for each row based on the hour, minute, and second
        # of the 'Date' column. This allows for aggregation by hour using the 'groupby' function:
        self.DataClean['decimal_time'] = self.DataClean['Date'].dt.hour / 24 + self.DataClean['Date'].dt.minute / (24 * 60) + self.DataClean['Date'].dt.second / (24 * 60 * 60)
        
        # This line groups the DataFrame by the 'decimal_time' column and calculates the mean temperature for each group:
        df_hourly = self.DataClean.groupby('decimal_time')['Temp'].mean()
        
        #Plotting Mean Temperature for Each Time of Day
        fig, ax = plt.subplots(figsize=(15, 8))
        ax.plot(df_hourly.index, df_hourly)
        # Add x- and y-axis labels
        ax.set_xlabel('Time of Day')
        ax.set_ylabel('Mean Temperature')
        # Add a title
        ax.set_title('Mean Temperature for Each Time of Day')
        
    def AnnualSeasonality(self):
        # This line groups the 'data' DataFrame by month, calculates the mean temperature for each month,
        # and stores the result in a new DataFrame 'df_monthly':
        df_monthly = self.DataClean.groupby(self.DataClean['Date'].dt.month)['Temp'].mean()

        # Convert the resulting groupby object to a DataFrame
        df_monthly = df_monthly.reset_index()

        # Rename the columns to 'month' and 'avg_temperature'
        df_monthly = df_monthly.rename(columns={'Date': 'month', 'Temp': 'avg_temp'})
        
        #creates a line plot of the 'month' and 'avg_temp' columns in the 'df_monthly' DataFrame:
        fig, ax = plt.subplots(figsize=(15, 8))
        # This line creates a line plot of the 'month' and 'avg_temp' columns in the 'df_monthly' DataFrame:
        ax.plot(df_monthly['month'], df_monthly['avg_temp'])
        # Set the title for the plot
        ax.set_title('Average Monthly Temperature')

        # Set the x-axis label
        plt.set_xlabel('Month')

        # Set the y-axis label
        plt.set_ylabel('Temperature (°C)')

        
    def AnnuelTrending(self):
        # This line extracts the year from the 'Date' column of the 'data' DataFrame and creates a new 'Year' column:
        self.DataClean['Year'] = self.DataClean['Date'].dt.year

        # groups the 'data' DataFrame by year, calculates the mean temperature for each year, and 
        # stores the result in a new DataFrame 'df_yearly':
        df_yearly = self.DataClean.groupby('Year')['Temp'].mean().reset_index()

        # creates a new figure with size 15x7 and returns a tuple of the figure and its axes:
        fig, axs = plt.subplots(figsize=(15, 7))
        # sets the data for the x and y axes to the 'Year' and 'Temp' columns of the 'df_yearly' DataFrame:
        x = df_yearly['Year']
        y = df_yearly['Temp']
        # creates a scatter plot of the x and y values:
        axs.plot(x, y, 'o')

        # fits a linear polynomial to the data and plots it in red:
        z = np.polyfit(x, y, 1)
        p = np.poly1d(z)
        axs.plot(x, p(x), 'r--')

        # sets the x-axis label for the plot:
        plt.xlabel('Year')
        # sets the y-axis label for the plot:
        plt.ylabel('Average temperature')
        plt.show()

    def InteractionDayYear(self):
        # creates a new column 'hour' in the 'data' DataFrame by extracting the hour from the 'Date' column:
        self.DataClean['hour'] = self.DataClean['Date'].dt.hour

        # groups the 'data' DataFrame by month and hour, calculates the mean temperature for each group, 
        # and stores the result in a new DataFrame 'result':
        result = self.DataClean.groupby([self.DataClean["Date"].dt.month, "hour"])["Temp"].mean().reset_index()

        # renames the columns in the 'result' DataFrame for readability:
        result.columns = ["month", "hour", "average_temperature"]

        # creates a new figure with size 20x10 and returns a tuple of the figure and its axes:
        fig, axe = plt.subplots(figsize=(20, 10))

        # This loop plots the average temperature by hour of day for each month:
        for i in range(1,13):
            month = list(result[result["month"] == i]["average_temperature"])
            axe.plot(np.arange(len(month)) + 1 + (i-1)*8, month, label=f"Month {i}")

        # sets the x-axis label for the plot:
        axe.set_xlabel("Hour of the day")
        # sets the y-axis label for the plot:
        axe.set_ylabel("Average temperature (°C)")
        # sets the title of the plot:
        axe.set_title("Average temperature by hour of day for each month")
        # adds a legend to the plot:
        axe.legend()
        # displays the plot:
        plt.show()

    def FastFourierTransform(self):
        df6 = self.DataClean.dropna()
        # Define the time step and length of the data
        dt = 1
        n = len(df6)

        # Calculate the Fourier Transform of the temperature data
        x = df6['Temp'].values
        F = fft(x)

        # Calculate the frequency range of the FFT coefficients
        w = fftfreq(n, dt)

        # Select the positive frequencies and corresponding FFT coefficients
        indices = w > 0
        w_pos = w[indices]
        F_pos = abs(F[indices])

        # Plot the Fourier spectrum
        fig, axs = plt.subplots(2, 1, figsize=(20, 20))
        
        # Plot the first 5 yeaes of the data
        x = list(range(len(self.DataClean.index)))
        y = self.DataClean.Temp
        axs[0].plot(x[:365*8*5],y[:365*8*5],linewidth=0.9)
        axs[0].set_ylabel('temperature (°C)')
        axs[0].set_xlabel('hour x dyas')
        axs[0].set_title('Evolution of temperature in the first 5 years')
        
        # Plot the Fourier spectrum
        axs[1].plot(w_pos, F_pos)
        axs[1].set_xlabel('Frequency (Hz)')
        axs[1].set_ylabel('Amplitude')
        axs[1].set_title('Fourier spectrum of temperature data')
        
        # Find the peaks in the Fourier spectrum
        peaks, _ = find_peaks(F_pos, height=30000)

        # Get the frequencies corresponding to the peaks
        peak_frequencies = w_pos[peaks]

        # Print the peak frequencies
        print(1/peak_frequencies/8)

    def TemperatureDistributionMonth(self):
        self.DataClean['month'] = self.DataClean['Date'].dt.month
        # Create a color palette using HUSL color space
        palette = sns.color_palette('husl', n_colors=12)

        # Create a 4 by 3 subplot of the temperature distribution for each month
        fig, axes = plt.subplots(nrows=4, ncols=3, figsize=(12, 12))

        for i, ax in enumerate(axes.flatten()):
            # Get the name of the current month
            month_name = calendar.month_name[i + 1]

            # Filter the data for the current month
            month_data = self.DataClean[self.DataClean['month'] == (i + 1)]

            # Plot the temperature distribution as a histogram
            sns.histplot(data=month_data, bins=10, x='Temp', kde=True, ax=ax, color=palette[i])

            # Set the title for the subplot
            ax.set_title(month_name)

            # Set the x-axis label
            ax.set_xlabel('Temperature')


        # Add a main title to the figure
        fig.suptitle('Temperature Distribution by Month', fontsize=16)

        # Adjust the spacing between subplots
        fig.tight_layout()
    
    def Autocorolation(self):
        df = self.DataClean.copy()
        # Set the date column as the index
        df.set_index('Date', inplace=True)

        # Calculate hourly and daily mean temperatures
        hourly_mean = df.groupby(df.index.hour).mean()
        hourly_mean.fillna(hourly_mean.mean(), inplace=True)
        
         # Create daily autocorrelation plot
        daily_mean = df.groupby(df.index.date).mean()
        daily_mean.fillna(hourly_mean.mean(), inplace=True)

        fig, axs = plt.subplots(2, 1, figsize=(20, 20))
        sm.graphics.tsa.plot_acf(hourly_mean['Temp'], ax=axs[0])
        axs[0].set(title='Hourly Autocorrelation', xlabel='Lag (hours)', ylabel='Autocorrelation')

        sm.graphics.tsa.plot_acf(daily_mean['Temp'] ,ax=axs[1])
        axs[1].set(title='Daily Autocorrelation', xlabel='Lag (days)', ylabel='Autocorrelation')


# In[3]:


class TemperatureCalibration:
    def __init__(self, data):
        self.data = data
        self.datacalibration = self.data.copy()
        hourly_mean_temp = self.datacalibration.groupby(self.datacalibration['Date'].dt.hour)['Temp'].transform('mean')
        self.datacalibration['Temp'] = self.datacalibration['Temp'].fillna(hourly_mean_temp)
        self.datacalibration.index = np.array([ i for i in range(len(self.datacalibration))])
        self.datacalibration['day'] = self.datacalibration['Date'].dt.day
        min_hour = self.datacalibration['Date'].min().hour
        self.__d = self.datacalibration['Date'].apply(lambda x: x.timetuple().tm_yday) # day number ranging from 1 to 31
        self.__h = self.datacalibration['Date'].apply(lambda x: ((x.hour - min_hour) % 24) // 3 + 1 ) # hour number ranging from 1 to 8
        self.__T = self.datacalibration['Temp'].values # temperature values
        self.datacalibration['d'] = self.__d
        self.datacalibration['h'] = self.__h
    def CreateFourierMatrix(self):
        self.S = pd.DataFrame()
        self.S['b0'] = np.array([1 for i in self.__d])
        for k in range(3):
            idcos = 2*k + 1 
            idsin = 2*(k + 1)
            self.S['b'+str(idcos)] = np.cos(2 * np.pi * (k+1) * self.__d / 365) 
            self.S['b'+str(idsin)] = np.sin(2 * np.pi * (k+1) * self.__d / 365)
        for k in range(3):
            idcos = 2*k + 1 
            idsin = 2*(k + 1)
            self.S['b'+str(idcos+6)] = np.cos(2 * np.pi * (k+1) * self.__h / 8) 
            self.S['b'+str(idsin+6)] = np.sin(2 * np.pi * (k+1) * self.__h / 8)
        for k in range(3):
            for l in range(3):
                idx = 12 * k + 4 * l + 13
                self.S['b'+str(idx)] = np.cos(2 * np.pi * (k+1) * self.__d / 365) * np.cos(2 * np.pi * (l+1) * self.__h / 8)
                self.S['b'+str(idx+1)] = np.cos(2 * np.pi * (k+1) * self.__d / 365) * np.sin(2 * np.pi * (l+1) * self.__h / 8)
                self.S['b'+str(idx+2)] = np.sin(2 * np.pi * (k+1) * self.__d / 365) * np.cos(2 * np.pi * (l+1) * self.__h / 8)
                self.S['b'+str(idx+3)] = np.sin(2 * np.pi * (k+1) * self.__d / 365) * np.sin(2 * np.pi * (l+1) * self.__h / 8)
        return self.S
    
    def Calculate_m(self):
        while True:
            try:
                # create some sample data
                Xm = self.S.values
                ym = self.__T.copy()

                # create a linear regression model and fit it to the data
                model_m = LinearRegression(fit_intercept=False).fit(Xm, ym)
                self.__betas_m = model_m.coef_
                self.m = model_m.predict(Xm)
                self.datacalibration['m'] = self.m
                # break out of the while loop if the code executes successfully
                return self.m

            except AttributeError:
                # redefine self.S using the CreateFourierMatrix() method
                self.S = self.CreateFourierMatrix()
    
    def PlotOneYear_m(self):
        fig, axf = plt.subplots(figsize=(20, 10))
        y = self.m
        x = list(range(len(y)))
        axf.plot(x[:365*8],y[:365*8],linewidth=1)
        plt.ylabel('mt')
        plt.xlabel('hour x dyas')
        
    def PlotAverageYear_m(self):
        fig, axf = plt.subplots(figsize=(20, 10))
        daily_means = self.datacalibration.groupby('d')['m'].mean()[:-1]
        y = daily_means
        x = list(range(len(y)))
        axf.plot(x,y,linewidth=1)
        plt.ylabel('mt')
        plt.xlabel('hour x dyas')
        
    def Calculate_s(self):
        # create some sample data
        Xs = self.S.values
        ys = (self.__T.copy() - self.m)**2

        # create a linear regression model and fit it to the data
        model_s = LinearRegression(fit_intercept=False).fit(Xs, ys)
        self.__betas_s = model_s.coef_
        s2 = model_s.predict(Xs)
        self.s = np.sqrt(s2)
        self.Tprime = (self.__T - self.m)/self.s
        self.datacalibration['s'] = self.s
        self.datacalibration['Tprime'] = self.Tprime
        return self.s
    
    def Calculate_phi(self):
        # prepare the matrix for phi
        S_1 = self.S[7:-1].multiply(self.Tprime[7:-1], axis=0)
        S_8 = self.S[:-8].multiply(self.Tprime[:-8], axis=0)
        Tprime_8 = self.Tprime[8:]
        S_8.columns = ['b'+str(i) for i in range(49,98)]
        S_1.index = [i for i in range(len(S_1))]
        S_8.index = [i for i in range(len(S_8))]
        S_phi = pd.concat([S_1, S_8], axis=1)
        
        # create some sample data
        Xphi = S_phi.values
        yphi = Tprime_8

        # create a linear regression model and fit it to the data
        model_phi = LinearRegression(fit_intercept=False).fit(Xphi, yphi)
        self.__betas_phi1 = model_phi.coef_[:49]
        self.__betas_phi8 = model_phi.coef_[49:]
        self.phi1 = np.array(self.S@np.array(self.__betas_phi1))
        self.phi8 = np.array(self.S@np.array(self.__betas_phi8))
        self.datacalibration['phi1'] = self.phi1
        self.datacalibration['phi8'] = self.phi8
        return self.phi1, self.phi8
    
    def Calculate_sigma(self):
        Tprime_8 = self.Tprime[8:]
        
        # create some sample data
        Xsig = self.S[8:].values
        ysig = (Tprime_8-self.phi1[7:-1]*self.Tprime[7:-1]-self.phi8[:-8]*self.Tprime[:-8])**2

        # create a linear regression model and fit it to the data
        model_sig = LinearRegression(fit_intercept=False).fit(Xsig, ysig)
        self.betas_sigma = model_sig.coef_
        sigma2 = model_sig.predict(self.S)
        treshold = -sigma2[sigma2<0].min()
        self.sigma = np.sqrt(sigma2+treshold)
        self.datacalibration['sigma'] = self.sigma
        return self.sigma


# In[15]:


class TemperatureSimulation:
    def __init__(self, data):
        self.data = data
        self.Tprime = np.array(data['Tprime'])
        self.m = np.array(data['m'])
        self.s = np.array(data['s'])
        self.phi1 = np.array(data['phi1'])
        self.phi8 = np.array(data['phi8'])
        self.sigma = np.array(data['sigma'])
        self.data['counter'] = (self.data.index % 2920) + 1
        self.m_mean_hour = np.array(data.groupby('counter')['m'].mean())
        self.m_mean_day = np.array(data.groupby('d')['m'].mean()[:-1])
        self.s_mean = np.array(data.groupby('counter')['s'].mean())
        
    def WarmingSimulation(self, epsi):        
        Tprev = np.zeros(365*8)
        Tprev[0] = self.sigma[0]*epsi[0]
        for i in range(1,8) : 
            Tprev[i] = self.phi1[i]*Tprev[i-1] + self.sigma[i]*epsi[i]
        for t in range(8,365*8):
            Tprev[t] = Tprev[t-1]*self.phi1[t] + Tprev[t-8]*self.phi8[t] + self.sigma[t]*epsi[t]
        return Tprev
    
    def Simulation(self, n, y):
            matrice = []
            for itera in range(n):
                epsilonWarming = np.random.randn(365*8)
                Tprime_prev = self.WarmingSimulation(epsilonWarming)
                epsilonSimulation = np.random.randn(365*8*y)
                T_prime_model = np.zeros(365*8*y)
                T_prime_model[:8] = Tprime_prev[-8:]
                for t in range(8,365*8*y):
                    T_prime_model[t] = T_prime_model[t-1]*self.phi1[t] + T_prime_model[t-8]*self.phi8[t] + self.sigma[t]*epsilonSimulation[t]
                prediction = self.m[:365*8*y] + T_prime_model*self.s[:365*8*y]
                matrice.append(prediction)
            self.simulated_temperature = np.array(matrice).mean(0)
            return self.simulated_temperature
        
    def SimulationOfPeriode(self, n, start_date_str, end_date_str):
            start_date = datetime.strptime(start_date_str, '%Y-%m-%d').date()
            end_date = datetime.strptime(end_date_str, '%Y-%m-%d').date()
            
            years_between = (end_date - start_date).days // 365
            
            jan1_date = datetime.strptime(str(start_date.year)+"-01-01", '%Y-%m-%d').date()
            days_from_jan1 = (start_date - jan1_date).days
            dec31_date = datetime.strptime(str(end_date.year)+"-12-31", '%Y-%m-%d').date()
            days_until_dec31 = (dec31_date - end_date).days
            
            temperature_simulated = self.Simulation(n, years_between+1)
            return temperature_simulated[days_from_jan1*8:-days_until_dec31*8]
    
    def GroupByday(self, listOfT):
        arr_reshaped = listOfT.reshape(-1, 8)
        arr_means = np.mean(arr_reshaped, axis=1)
        return arr_means
    
    def PlotComparasionSimulation(self, temperature):
        
        years = int(len(temperature)/(365*8))
        if years == 1 :
            fig, axf = plt.subplots(figsize=(20, 10))
            real_temperature = self.m_mean_day
            simulated_temperature = self.GroupByday(temperature)
            x = list(range(len(real_temperature)))
            axf.plot(x,real_temperature,linewidth=1)
            axf.plot(x,simulated_temperature,linewidth=1)
            plt.ylabel('temperature')
            plt.xlabel('3hours x dyas')
        else :
            minidata = self.data[:365*8*years].copy()
            minidata['counter'] = (minidata.index % 2920) + 1
            minidata['temperature'] = temperature
            mean_temperature = np.array(minidata.groupby('counter')['temperature'].mean())
            self.PlotComparasionSimulation(mean_temperature)
            
    def PlotSimulationAverage(self, temperature):
        years = int(len(temperature)/(365*8))
        minidata = self.data[:365*8*years].copy()
        minidata['counter'] = (minidata.index % 2920) + 1
        minidata['temperature'] = temperature
        mean_temperature = np.array(minidata.groupby('counter')['temperature'].mean())
        
        fig, axf = plt.subplots(figsize=(20, 10))
        average_temperature = self.GroupByday(mean_temperature)
        x = list(range(len(average_temperature)))
        axf.plot(x,average_temperature,linewidth=1)
        plt.ylabel('Average Temperature')
        plt.xlabel('3hours x dyas')

    def PlotSimulation(self, temperature):
        
        fig, axf = plt.subplots(figsize=(20, 10))
        average_temperature = self.GroupByday(temperature)
        x = list(range(len(average_temperature)))
        axf.plot(x,average_temperature,linewidth=1)
        plt.ylabel('Average Temperature')
        plt.xlabel('3hours x dyas')        
        
    def PlotSimulationOfPeriode(self, temperature):
        fig, axf = plt.subplots(figsize=(20, 10))
        x = list(range(len(temperature)))
        axf.plot(x,temperature,linewidth=1)
        plt.ylabel('Temperature')
        plt.xlabel('3hours x dyas')  
        
    def PlotSimulationOfAveragePeriode(self, temperature):
        fig, axf = plt.subplots(figsize=(20, 10))
        average_temperature = self.GroupByday(temperature)
        x = list(range(len(average_temperature)))
        axf.plot(x,average_temperature,linewidth=1)
        plt.ylabel('Temperature')
        plt.xlabel('3hours x dyas') 

