import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import time
from numba import jit


def time_it(method):
    def timed(*args, **kw):
        start_time = time.time()
        result = method(*args, **kw)
        end_time = time.time()
        print('%r (%r, %r) %2.2f sec' % \
              (method.__name__, args, kw, end_time - start_time))
        return result

    return timed


class BSMpathSimulator():

    def __init__(self, S0=100., Npath=1000, Nt=200, mu=0.25, sig=0.28, T=1.):
        self.Npath = int(Npath)
        self.Nt = int(Nt)
        self.S0 = S0
        self.mu = mu
        self.sig = sig
        self.dt = T / Nt
        self.T = T
        self.paths = np.zeros((int(Nt), int(Npath))) + S0

    def genrn(self):
        self.rn = np.random.normal(loc=0, scale=1, size=(self.Nt, self.Npath))
        return self.rn

    @time_it
    def genpath(self, newrn=True, S0=None, sigma=None):

        if newrn:
            self.genrn()
        if S0 is not None:
            self.S0 = S0
        if sigma is not None:
            self.sig = sigma
        incre = np.r_[np.zeros((1, self.Npath)) + self.S0, np.exp((self.mu - self.sig ** 2 / 2) * self.dt \
                                                                  + self.sig * np.sqrt(self.dt) * self.rn)]
        self.paths = np.cumprod(incre, axis=0)

        return self.paths

    def plotpath(self, n=1):
        for i in range(n):
            plt.plot(self.paths.T[i])
        plt.show()

    @time_it
    def gamma_scalping_PnL(self, type_option="call", r=0.2, q=0.02, hedgeIV=0.15, K=99.):
        # Need to optimize
        # This function is for vol surf course HW3 problem 1.

        PVPnL = np.zeros(self.Npath)
        gamma = np.vectorize(BSMpathSimulator.BMS_gamma)
        for i in range(1, self.Nt + 1):
            dS = self.paths[i] - self.paths[i - 1]
            PVPnL += 0.5 * np.exp(-r * i * self.dt) * gamma(type_option, self.paths[i - 1], K, r, q, hedgeIV, self.T,
                                                            (i - 1) * self.dt) * (
                             dS ** 2 - (hedgeIV ** 2) * (self.paths[i - 1] ** 2) * self.dt)
        return PVPnL.mean(), PVPnL.std(ddof=1), PVPnL

    @time_it
    def discontinuous_hedge_Price(self, type_option="call", r=0., q=0., hedgeIV=0.3, K=100., cost=0.002):
        # Need to optimize
        # This function is for vol surf course HW3 problem 4.

        delta = np.vectorize(BSMpathSimulator.BMS_delta)
        deltalst = delta(type_option, self.paths[0], K, r, q, hedgeIV, self.T)
        deltalst_pre = deltalst
        PV = self.paths[0] * deltalst * (1 - cost)
        dPnL = np.full_like(PV, fill_value=0.)

        for i in range(1, self.Nt):
            deltalst = delta('call', self.paths[i], K, r, q, hedgeIV, self.T, i * self.dt)
            dPnL = (deltalst - deltalst_pre) * np.exp(-r * i * self.dt) * self.paths[i]
            PV += (dPnL - np.abs(dPnL) * cost)

            deltalst_pre = deltalst

        f = np.vectorize(lambda x: x - K if x > K else 0.)

        PV -= (deltalst * self.paths[-1]) * (1 + cost) * np.exp(-r * self.T)
        PV += f(self.paths[-1]) * np.exp(-r * self.T)

        return PV.mean(), PV.std(ddof=1), PV

    @time_it
    def delayed_start_change_option_pricing(self, r=0.015, q=0.005, t1=1.5, H=1250, l1=1.15, l2=0.85,
                                            method="ConditionalMC"):
        # This function is for MC course Assignment2 problem 9.

        if t1 >= self.T:
            print("Bad input t1.")
            return
        sigma = self.sig

        if method == "ConditionalMC":
            P1 = lambda x: self.BMS_price('put', x, l2 * x, r, q, sigma, self.T - t1) * np.exp(
                -r * t1 / 2) if x < H else 0.
            P2 = lambda x: self.BMS_price('call', x, l1 * x, r, q, sigma, self.T - t1) * np.exp(
                -r * t1 / 2) if x >= H else 0.
            Pricefunc = np.vectorize(lambda x: P1(x) + P2(x))
            idx = int(np.round(t1 / self.dt))
            price = Pricefunc(self.paths[idx])
        else:
            idx1, idx2 = int(np.round(t1 / self.dt)), -1
            Pricefunc = np.vectorize(
                lambda x, y: ((y - l1 * x if y > l1 * x else 0) if x >= H \
                                  else (l2 * x - y if y < l2 * x else 0)) \
                             * np.exp(-r * self.T))
            price = Pricefunc(self.paths[idx1], self.paths[idx2])

        return price.mean(), price.std(ddof=1), price

    def delayed_start_option_pricing(self, type_option="call", r=0.015, q=0.03, sigma=0.25, \
                                     t1=9 / 12, l=1.1, method="NormalMC"):
        # This function is for MC course Assignment3 problem 2.

        if method == "ConditionalMC":
            Pricefunc = np.vectorize(
                lambda x: self.BMS_price(type_option, x, l * x, r, q, sigma, self.T - t1) * np.exp(
                    -r * t1 / 2))
            idx = int(np.round(t1 / self.dt))
            price = Pricefunc(self.paths[idx])
        else:
            idx1, idx2 = int(np.round(t1 / self.dt)), -1
            if type_option == "call":
                Pricefunc = np.vectorize(
                    lambda x, y: (y - l * x if y > l * x else 0) * np.exp(-r * self.T))
            elif type_option == "put":
                Pricefunc = np.vectorize(
                    lambda x, y: (l * x - y if y < l * x else 0) * np.exp(-r * self.T))
            else:
                print("Unsupportable option type!")
                return
            price = Pricefunc(self.paths[idx1], self.paths[idx2])
        return price.mean(), price.std(ddof=1), price

    @time_it
    def delayed_start_option_greeks(self, type_option="call", r=0.015, q=0.03, sigma=0.25, t1=9 / 12, l=1.1,
                                    greek="delta", method="CRN", d=1):
        # This function is for MC course Assignment3 problem 2.

        if method == "CRN":
            if greek == "delta":
                self.genpath(newrn=False, S0=self.S0 + d)
                Vplus, stdplus, _ = self.delayed_start_option_pricing(type_option=type_option, r=r, q=q, t1=t1, l=l)
                self.genpath(newrn=False, S0=self.S0 - 2 * d)
                Vminus, stdminus, _ = self.delayed_start_option_pricing(type_option=type_option, r=r, q=q, t1=t1, l=l)
                self.genpath(newrn=False, S0=self.S0 + d)
                return (Vplus - Vminus) / (2 * d)

            if greek == "gamma":
                self.genpath(newrn=False, S0=self.S0 + d)
                Vplus, stdplus, _ = self.delayed_start_option_pricing(type_option=type_option, r=r, q=q, t1=t1, l=l)
                self.genpath(newrn=False, S0=self.S0 - 2 * d)
                Vminus, stdminus, _ = self.delayed_start_option_pricing(type_option=type_option, r=r, q=q, t1=t1, l=l)
                self.genpath(newrn=False, S0=self.S0 + d)
                Vmid, stdmid, _ = self.delayed_start_option_pricing(type_option=type_option, r=r, q=q, t1=t1, l=l)
                return (Vplus + Vminus - 2 * Vmid) / (d ** 2)

            if greek == "vega":
                self.genpath(newrn=False, sigma=self.sig + d)
                Vplus, stdplus, _ = self.delayed_start_option_pricing(type_option=type_option, r=r, q=q, t1=t1, l=l)
                self.genpath(newrn=False, sigma=self.sig - 2 * d)
                Vminus, stdminus, _ = self.delayed_start_option_pricing(type_option=type_option, r=r, q=q, t1=t1, l=l)
                self.genpath(newrn=False, sigma=self.sig + d)
                return (Vplus - Vminus) / (2 * d)

        if method == "PE":
            idx1, idx2 = int(np.round(t1 / self.dt)), -1

            if greek == "delta":
                PEfunc = np.vectorize(
                    lambda x, y: (y - l * x) / self.S0 * np.exp(-r * self.T) if y > l * x else 0.)
                greek_value = PEfunc(self.paths[idx1], self.paths[idx2])

            if greek == "gamma":
                PEfunc = np.vectorize(lambda s1, s2: 0.)
                greek_value = PEfunc(self.paths[idx1], self.paths[idx2])

            if greek == "vega":
                PEfunc = np.vectorize(
                    lambda s1, s2, z1, z2: (s2 * (-self.sig * self.T + np.sqrt(self.T) * z2) - l * s1 * (
                            -self.sig * t1 + np.sqrt(t1) * z1)) * np.exp(-r * self.T) if s2 > l * s1 else 0.)
                rn_cumsum = self.rn.cumsum(axis=0)
                greek_value = PEfunc(self.paths[idx1], self.paths[idx2], rn_cumsum[idx1 - 1] * np.sqrt(self.dt / t1),
                                     rn_cumsum[idx2] * np.sqrt(self.dt / self.T))

            return greek_value.mean(), greek_value.std(ddof=1), greek_value

        if method == "LR":
            idx1, idx2 = int(np.round(t1 / self.dt)), -1

            if greek == "delta":
                Payoff_func = np.vectorize(lambda s1, s2, z1: (s2 - l * s1) * np.exp(-r * self.T) * z1 / (
                        self.S0 * self.sig * np.sqrt(t1)) if s2 > l * s1 else 0.)
                greek_value = Payoff_func(self.paths[idx1], self.paths[idx2],
                                          self.rn.cumsum(axis=0)[idx1 - 1] * np.sqrt(self.dt / t1))

            if greek == "gamma":
                Payoff_func = np.vectorize(
                    lambda s1, s2, z1: (s2 - l * s1 if s2 > l * s1 else 0.) * np.exp(-r * self.T) * ((z1 ** 2 - 1) / (
                            self.S0 ** 2 * self.sig ** 2 * t1) - z1 / (self.S0 ** 2 * self.sig * np.sqrt(t1))))
                greek_value = Payoff_func(self.paths[idx1], self.paths[idx2],
                                          self.rn.cumsum(axis=0)[idx1 - 1] * np.sqrt(self.dt / t1))

            if greek == "vega":
                Payoff_func = np.vectorize(lambda s1, s2, z1, z2: (s2 - l * s1) * np.exp(-r * self.T) * (
                        (z1 ** 2 - 1) / self.sig - z1 * np.sqrt(t1) + (z2 ** 2 - 1) / self.sig - z2 * np.sqrt(
                    self.T - t1)) if s2 > l * s1 else 0.)
                rn_cumsum = self.rn.cumsum(axis=0)
                greek_value = Payoff_func(self.paths[idx1], self.paths[idx2],
                                          rn_cumsum[idx1 - 1] * np.sqrt(self.dt / t1),
                                          (rn_cumsum[idx2] - rn_cumsum[idx1 - 1]) * np.sqrt(self.dt / (self.T - t1)))

            return greek_value.mean(), greek_value.std(ddof=1), greek_value

        print("Unsupported method.")
        return

    @staticmethod
    @time_it
    def stratified_sampling_pricing(type_option="put", S0=1260, K=1100, T=0.25, r=0.0025, q=0.01, sigma=0.35, \
                                    interval=[-2, -1, 1, 2], trunc=10000, optimaln=True, Npilot=1000, Nsample=1e5):
        # This function is for MC course Assignment3 problem 1.

        probi = np.zeros(len(interval) + 1)
        probi[0] = norm.cdf(interval[0])
        for i in range(1, len(probi) - 1):
            probi[i] = norm.cdf(interval[i]) - norm.cdf(interval[i - 1])
        probi[-1] = 1 - norm.cdf(interval[-1])

        interval = [-trunc] + interval + [trunc]

        invnormcdf = np.vectorize(norm.ppf)
        Pricefunc = np.vectorize(lambda x: np.exp(-r * T) * max(K - x, 0))

        if optimaln:
            sigmapilot = np.zeros(len(probi))
            for i in range(len(probi)):
                uniform_sample = np.random.uniform(low=norm.cdf(interval[i]), high=norm.cdf(interval[i + 1]),
                                                   size=int(Npilot / len(probi)))
                sigmapilot[i] = Pricefunc(S0 * np.exp(
                    (r - q - 1 / 2 * sigma ** 2) * T + sigma * np.sqrt(T) * invnormcdf(uniform_sample))).std()
            ni = (probi * sigmapilot) / (probi * sigmapilot).sum() * Nsample
            ni = np.round(ni)
        else:
            ni = np.round(probi * Nsample)

        mean = 0
        var = 0
        for i, n in enumerate(ni):
            if int(n) != 0:
                uniform_sample = np.random.uniform(low=norm.cdf(interval[i]), high=norm.cdf(interval[i + 1]),
                                                   size=int(n))
                price = Pricefunc(S0 * np.exp(
                    (r - q - 1 / 2 * sigma ** 2) * T + sigma * np.sqrt(T) * invnormcdf(uniform_sample)))

                mean += (probi[i] * price).mean()
                var += probi[i] ** 2 * price.std(ddof=1) ** 2 / n

        return mean, np.sqrt(var)

    @staticmethod
    def BMS_d1(S, K, r, q, sigma, tau):
        ''' Computes d1 for the Black Merton Scholes formula '''
        d1 = 1.0 * (np.log(1.0 * S / K) + (r - q + sigma ** 2 / 2) * tau) / (sigma * np.sqrt(tau))
        return d1

    @staticmethod
    def BMS_d2(S, K, r, q, sigma, tau):
        ''' Computes d2 for the Black Merton Scholes formula '''
        d2 = 1.0 * (np.log(1.0 * S / K) + (r - q - sigma ** 2 / 2) * tau) / (sigma * np.sqrt(tau))
        return d2

    @staticmethod
    def BMS_price(type_option, S, K, r, q, sigma, T, t=0):
        ''' Computes the Black Merton Scholes price for a 'call' or 'put' option '''
        tau = T - t
        d1 = BSMpathSimulator.BMS_d1(S, K, r, q, sigma, tau)
        d2 = BSMpathSimulator.BMS_d2(S, K, r, q, sigma, tau)
        if type_option == 'call':
            price = S * np.exp(-q * tau) * norm.cdf(d1) - K * np.exp(-r * tau) * norm.cdf(d2)
        elif type_option == 'put':
            price = K * np.exp(-r * tau) * norm.cdf(-d2) - S * np.exp(-q * tau) * norm.cdf(-d1)
        return price

    @staticmethod
    def BMS_delta(type_option, S, K, r, q, sigma, T, t=0):
        ''' Computes the delta for a call or a put. '''
        tau = T - t
        d1 = BSMpathSimulator.BMS_d1(S, K, r, q, sigma, tau)
        if type_option == 'call':
            delta = np.exp(-q * tau) * norm.cdf(d1)
        elif type_option == 'put':
            delta = np.exp(-q * tau) * (norm.cdf(d1) - 1)
        return delta

    @staticmethod
    def BMS_gamma(type_option, S, K, r, q, sigma, T, t=0):
        ''' Computes the gamma for a call or a put. '''
        tau = T - t
        d1 = BSMpathSimulator.BMS_d1(S, K, r, q, sigma, tau)
        return np.exp(-q * tau) * norm.pdf(d1) / (S * sigma * np.sqrt(tau))

    @staticmethod
    @jit(nopython=True)
    def quick_mean_and_std(array, ddof=1):
        n = len(array)
        if n <= 1:
            print("Array length less or equal to 1.")
            return np.nan, np.nan
        squaresum = 0
        sum = 0
        for i in range(n):
            sum += array[i]
            squaresum += array[i] ** 2
        return sum / n, np.sqrt((squaresum - sum ** 2 / n) / (n - ddof))


if __name__ == "__main__":
    # MCmodel1 = BSMpathSimulator()
    # MCmodel1.genpath()
    # MCmodel1.plotpath(3)
    #
    # mean, std, _ = MCmodel1.gamma_scalping_PnL()
    # print(mean, std)

    # MCmodel2 = BSMpathSimulator(S0=1000., Npath=20000, Nt=100, mu=0.01, sig=0.32, T=3)
    # MCmodel2.genpath()
    # MCmodel2.plotpath(3)
    # mean, std, _ = MCmodel2.delayed_start_change_option_pricing()
    # print(mean, std)
    # mean, std, _ = MCmodel2.delayed_start_change_option_pricing(method="NormalMC")
    # print(mean, std)

    # MCmodel4 = BSMpathSimulator(S0=100., Npath=10000, Nt=30, mu=0., sig=0.3, T=1 / 12)
    # MCmodel4.genpath()
    # mean, std, _ = MCmodel4.discontinuous_hedge_Price()
    # print(mean,std)

    # spot = 1260
    # K = 1100
    # r = 0.0025
    # q = 0.01
    # sig = 0.35
    #
    # maturity = 1 / 4
    #
    # print(BSMpathSimulator.BMS_price('put', spot, K, r, q, sig, maturity))

    # mean, std = BSMpathSimulator.stratified_sampling_pricing()
    # print(mean, std)

    r = 0.015
    q = 0.03

    MCmodel3 = BSMpathSimulator(S0=1000., Npath=5e5, Nt=18, mu=r - q, sig=0.25, T=18 / 12)
    MCmodel3.genpath()
    MCmodel3.plotpath(3)

    # mean, std, _ = MCmodel3.delayed_start_option_pricing(type_option="call", r=0.015, q=0.03, sigma=0.25, \
    #                                                      t1=9 / 12, l=1.1, method="ConditionalMC")
    # print(mean, std)
    # mean, std, _ = MCmodel3.delayed_start_option_pricing(type_option="call", r=0.015, q=0.03, sigma=0.25, \
    #                                                      t1=9 / 12, l=1.1, method="NormalMC")
    # print(mean, std)

    greeks = ["delta", "gamma", "vega"]
    for greek in greeks:
        print(MCmodel3.delayed_start_option_greeks(type_option="call", r=0.015, q=0.03, sigma=0.25, t1=9 / 12, l=1.1,
                                                   greek=greek, method="CRN", d=0.001))

    #MCmodel3.genpath()
    for greek in greeks:
        mean, std, _ = MCmodel3.delayed_start_option_greeks(type_option="call", r=0.015, q=0.03, sigma=0.25, t1=9 / 12,
                                                            l=1.1,
                                                            greek=greek, method="PE")
        print(mean, std)

    #MCmodel3.genpath()
    for greek in greeks:
        mean, std, _ = MCmodel3.delayed_start_option_greeks(type_option="call", r=0.015, q=0.03, sigma=0.25, t1=9 / 12,
                                                            l=1.1,
                                                            greek=greek, method="LR")
        print(mean, std)
