import numpy as np
from pdb import set_trace
from statsmodels.multivariate.cancorr import CanCorr
from math import log, pow
from scipy.stats import chi2

EPSILON = 1e-9


class CCARankTester:
    def __init__(self, data, alpha=0.05):

        # Centre the data
        data = data - np.mean(data, axis=0)
        self.data = np.array(data)
        self.n = data.shape[0]
        self.alpha = alpha

    def test(self, pcols, qcols, r=1):
        """
        Test null hypothesis that rank of cov(X, Y) <= r.

        Returns True if reject null.
        """
        try:
            p = len(pcols)
            q = len(qcols)
            X = self.data[:, pcols]
            Y = self.data[:, qcols]

            # Rank will always be <= r if there are <= r columns
            if (X.shape[1] <= r) or (Y.shape[1] <= r):
                return False

            cca = CanCorr(X, Y, tolerance=1e-8)
            l = cca.cancorr[r:]

            testStat = 0
            for li in l:
                testStat += log(1 - pow(li, 2) + EPSILON)
            testStat = testStat * -(self.n - 0.5 * (p + q + 3))

            dfreedom = (p - r) * (q - r)
            criticalValue = chi2.ppf(1 - self.alpha, dfreedom)
        except:
            set_trace()
        # print(f"testStat: {testStat}, crit: {criticalValue}")

        return testStat > criticalValue

    def prob(self, pcols, qcols, r=1):
        p = len(pcols)
        q = len(qcols)
        X = self.data[:, pcols]
        Y = self.data[:, qcols]
        cca = CanCorr(X, Y)
        l = cca.cancorr[r:]

        testStat = 0
        for li in l:
            testStat += log(1 - pow(li, 2))
        testStat = testStat * -(self.n - 0.5 * (p + q + 3))

        dfreedom = (p - r) * (q - r)
        criticalValue = chi2.ppf(1 - self.alpha, dfreedom)
        # print(f"testStat: {testStat}, crit: {criticalValue}")

        return 1 - chi2.cdf(testStat, dfreedom)
