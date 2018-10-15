import pandas as pd
import numpy as np
from sklearn.datasets import load_boston, load_breast_cancer

# ------regression------
def load_community():
    """
    https://archive.ics.uci.edu/ml/datasets/Communities+and+Crime+Unnormalized
    """
    columns = ['state', 'county', 'community', 'communityname', 'fold', 'population: population for community', 'householdsize', 
          'racepctblack', 'racePctWhite',
     'racePctAsian', 'racePctHisp', 'agePct12t21', 'agePct12t29', 'agePct16t24', 'agePct65up', 'numbUrban', 'pctUrban', 'medIncome',
          'pctWWage', 'pctWFarmSelf',
     'pctWInvInc', 'pctWSocSec', 'pctWPubAsst', 'pctWRetire', 'medFamInc', 'perCapInc', 'whitePerCap', 'blackPerCap', 
          'indianPerCap', 'AsianPerCap',
     'OtherPerCap', 'HispPerCap', 'NumUnderPov', 'PctPopUnderPov', 'PctLess9thGrade', 'PctNotHSGrad', 'PctBSorMore', 
          'PctUnemployed', 'PctEmploy',
     'PctEmplManu', 'PctEmplProfServ', 'PctOccupManu', 'PctOccupMgmtProf', 'MalePctDivorce', 'MalePctNevMarr', 
          'FemalePctDiv', 'TotalPctDiv',
     'PersPerFam', 'PctFam2Par', 'PctKids2Par', 'PctYoungKids2Par', 'PctTeen2Par', 'PctWorkMomYoungKids', 'PctWorkMom', 'NumIlleg',
     'PctIlleg', 'NumImmig', 'PctImmigRecent', 'PctImmigRec5', 'PctImmigRec8', 'PctImmigRec10', 'PctRecentImmig', 'PctRecImmig5',
     'PctRecImmig8', 'PctRecImmig10', 'PctSpeakEnglOnly', 'PctNotSpeakEnglWell', 'PctLargHouseFam', 'PctLargHouseOccup', 
          'PersPerOccupHous',
     'PersPerOwnOccHous', 'PersPerRentOccHous', 'PctPersOwnOccup', 'PctPersDenseHous', 'PctHousLess3BR', 'MedNumBR',
          'HousVacant', 'PctHousOccup',
     'PctHousOwnOcc', 'PctVacantBoarded', 'PctVacMore6Mos', 'MedYrHousBuilt', 'PctHousNoPhone', 'PctWOFullPlumb', 
          'OwnOccLowQuart', 'OwnOccMedVal',
     'OwnOccHiQuart', 'RentLowQ', 'RentMedian', 'RentHighQ', 'MedRent', 'MedRentPctHousInc', 'MedOwnCostPctInc', 
          'MedOwnCostPctIncNoMtg',
     'NumInShelters', 'NumStreet', 'PctForeignBorn', 'PctBornSameState', 'PctSameHouse85', 'PctSameCity85', 
          'PctSameState85', 'LemasSwornFT',
     'LemasSwFTPerPop', 'LemasSwFTFieldOps', 'LemasSwFTFieldPerPop', 'LemasTotalReq', 'LemasTotReqPerPop', 
          'PolicReqPerOffic', 'PolicPerPop',
     'RacialMatchCommPol', 'PctPolicWhite', 'PctPolicBlack', 'PctPolicHisp', 'PctPolicAsian', 'PctPolicMinor', 
          'OfficAssgnDrugUnits', 'NumKindsDrugsSeiz',
     'PolicAveOTWorked', 'LandArea', 'PopDens', 'PctUsePubTrans', 'PolicCars', 'PolicOperBudg', 'LemasPctPolicOnPatr', 
          'LemasGangUnitDeploy', 'LemasPctOfficDrugUn',
     'PolicBudgPerPop', 'ViolentCrimesPerPop']

    data = pd.read_csv('./../data/communities.data', header=None, na_values='?')
    data.columns = columns
    data = data.dropna(axis=1)
    data = data.drop(['communityname', 'fold', 'state'], axis=1)
    X, y = data.iloc[:, :-1].as_matrix(), data.iloc[:, -1].as_matrix()
    X = (X - X.mean(axis=0)) / X.std(axis=0)
    return X, y


def load_forestfires():
    """
    https://archive.ics.uci.edu/ml/datasets/Forest+Fires
    """
    data = pd.read_csv('./../data/forestfires.csv')
    data.head()
    data = data.drop(['month', 'day'], axis=1)
    X, y = data.iloc[:, :-1].as_matrix(), data.iloc[:, -1].as_matrix()
    X = (X - X.mean(axis=0)) / X.std(axis=0)
    y = np.log(y + 1)
    return X, y


def load_bostonprices():
    """
    scikit-learn
    """
    X, y = load_boston(return_X_y=True)
    X = (X - X.mean(axis=0)) / X.std(axis=0)
    return X, y


def load_residentialprice():
    data = pd.read_excel('./../data/Residential-Building-Data-Set.xlsx', sheetname='Data', skiprows=[0])
    X, y = data.iloc[:, 4:-2].as_matrix(), data.iloc[:, -2].as_matrix()
    X = (X - X.mean(axis=0)) / X.std(axis=0)
    y = (y - y.mean()) / y.std()
    return X, y


def load_residentialcost():
    data = pd.read_excel('./../data/Residential-Building-Data-Set.xlsx', sheetname='Data', skiprows=[0])
    X, y = data.iloc[:, 4:-2].as_matrix(), data.iloc[:, -1].as_matrix()
    X = (X - X.mean(axis=0)) / X.std(axis=0)
    y = (y - y.mean()) / y.std()
    return X, y


# ------classification------
def load_breast():
    """
    scikit-learn
    """
    X, y = load_breast_cancer(return_X_y=True)
    X = (X - X.mean(axis=0)) / X.std(axis=0)
    return X, y


def load_cardiotocography():
    """
    https://archive.ics.uci.edu/ml/datasets/Cardiotocography
    """
    data = pd.read_excel('./../data/CTG.xls', sheetname='Data', header=1)
    columns = ['LB', 'AC', 'FM', 'UC', 'DL', 'DS',
                'DP', 'ASTV', 'MSTV', 'ALTV', 'MLTV',
                'Width', 'Min', 'Max', 'Nmax', 'Nzeros',
                'Mode', 'Mean', 'Median', 'Variance',
                'Tendency', 'NSP']
    data = data[columns]
    data = data.dropna()
    X, y = data.iloc[:, :-1].as_matrix(), (data['NSP'] > 1.5).astype(float).as_matrix()
    X = (X - X.mean()) / (X.std(axis=0))
    return X, y


def load_climate():
    """
    https://archive.ics.uci.edu/ml/datasets/Climate+Model+Simulation+Crashes
    """
    with open('./../data/pop_failures.dat') as f:
        columns = f.readline().split()
        data = []
        while True:
            line = f.readline()
            if line == '':
                break
            data.append(list(map(float, line.split())))
    data = np.array(data)
    X, y = data[:, 2:-1], data[:, -1]
    X = (X - X.mean(axis=0)) / X.std(axis=0)
    return X, y