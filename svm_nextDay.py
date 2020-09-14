import numpy as np
from sklearn import svm

'''We use a support vector machine (SVM) to predict the next day's trend.'''

def myTradingSystem(DATE, OPEN, HIGH, LOW, CLOSE, VOL, OI, P, R, RINFO, exposure, equity, settings):
    nMarkets = len(settings['markets'])
    gap = settings['gap']

    pos = np.zeros((1, nMarkets), dtype='float')

    for i in range(nMarkets):
        try:
        #call predict() based on closing price
            pos[0,i] = predict(CLOSE[:,i].reshape(-1,1),gap,)
        #for NaN values
        except ValueError:
            pos[0,i] = 0

    return pos, settings

def predict(CLOSE, gap):
    lookback = CLOSE.shape[0]
    #samples (closing prices) over gap period
    x = np.concatenate([CLOSE[i: i + gap] for i in range(lookback - gap)], axis=1).T
    #y = 1 if price goes up, y = -1 if price goes down
    y = np.sign((CLOSE[gap: lookback] - CLOSE[gap - 1: lookback - 1]).T[0])
    y[y == 0] = 1

    #support vector classification: linearly separable data y = 1 and y = -1
    clf = svm.SVC()
    clf.fit(x,y)

    return clf.predict(CLOSE[-gap:].T)

def mySettings():
    settings = {}

    settings['markets'] = ['CASH', 'F_AD', 'F_BO', 'F_BP', 'F_C', 'F_CC', 'F_CD',
                               'F_CL', 'F_CT', 'F_DX', 'F_EC', 'F_ED', 'F_ES', 'F_FC', 'F_FV', 'F_GC',
                               'F_HG', 'F_HO', 'F_JY', 'F_KC', 'F_LB', 'F_LC', 'F_LN', 'F_MD', 'F_MP',
                               'F_NG', 'F_NQ', 'F_NR', 'F_O', 'F_OJ', 'F_PA', 'F_PL', 'F_RB', 'F_RU',
                               'F_S', 'F_SB', 'F_SF', 'F_SI', 'F_SM', 'F_TU', 'F_TY', 'F_US', 'F_W', 'F_XX',
                               'F_YM']
    
    settings['lookback'] = 252
    settings['budget'] = 10**6
    settings['slippage'] = 0.05

    #use past 5 days
    settings['gap'] = 5

    return settings


if __name__ == "__main__":
    import quantiacsToolbox
    results = quantiacsToolbox.runts(__file__)
