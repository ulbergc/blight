import pandas as pd
import numpy as np

def blight_model():
    
    import pandas as pd
    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.metrics import precision_recall_curve, roc_curve, auc

    #### read model
    df=pd.read_csv('train.csv', encoding = 'ISO-8859-1', low_memory=False)
    df=df.set_index('ticket_id')
    dfOG=df

    #### define features to use and clean model
    target='compliance'
    usecols=['disposition', 'agency_name', 'late_fee', 'fine_amount']
    dum_col=['disposition','agency_name'] # make sure these are in usecols
    # usecols=['late_fee', 'fine_amount']
    # dum_col=[] # make sure these are in usecols

    newtmp=usecols.copy()
    newtmp.append(target)

    df=dfOG[newtmp]
    df=df.dropna()

    #### split into X and y
    colnames=df.columns
    X=df[colnames[:-1]]
    y=df[colnames[-1]]

    #### make dummy variables out of categorical features
    X_tmp=pd.get_dummies(X[dum_col])
    X_new=X[X.columns[np.in1d(X.columns,dum_col,invert=True)]].join(X_tmp)

    # X_new=X

    #### split into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X_new, y, random_state=0)

    #### scale the data so things are nice
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    # we must apply the scaling to the test set that we computed for the training set
    X_test_scaled = scaler.transform(X_test)

    #### apply the classifier to scaled data

    clf = GradientBoostingClassifier().fit(X_train_scaled, y_train)
    # clf = RandomForestClassifier().fit(X_train_scaled, y_train)
    # clf = DecisionTreeClassifier(max_depth=2).fit(X_train_scaled, y_train)
    # clf = LogisticRegression().fit(X_train_scaled, y_train)

    #### calculate predicted probabilities with the chosen classifier
    y_proba=clf.predict_proba(X_test_scaled)

    #### calculate stuff for precision recall curve
    precision, recall, thresholds = precision_recall_curve(y_test, y_proba[:,1])

    #### calculate stuff for roc curve
    fpr_lr, tpr_lr, _ = roc_curve(y_test, y_proba[:,1])
    roc_auc_lr = auc(fpr_lr, tpr_lr)

    #### print out auc value that this will be evaluated on
#     print('{:.3f}'.format(roc_auc_lr))
    
    #### do the final step
    dftest=pd.read_csv('test.csv')
    dftest=dftest.set_index('ticket_id')
    Xf=dftest[usecols]
    
    #### make dummy variables out of categorical features
    Xf_tmp=pd.get_dummies(Xf[dum_col])
    Xf_new=Xf[Xf.columns[np.in1d(Xf.columns,dum_col,invert=True)]].join(Xf_tmp)

    #### remove columns that weren't in the training
    id_rm=np.in1d(Xf_new.columns,X_train.columns)
    Xf_new2=Xf_new[Xf_new.columns[id_rm]]
    Xf_new2.columns

    #### add any columns that were in training and aren't in test?
    id_add=np.in1d(X_train.columns,Xf_new2.columns,invert=True)
    kwargs={} # ugly but I guess it works
    for colname in X_train.columns[id_add]:
        kwargs[colname]=0

    Xf_new2=Xf_new2.assign(**kwargs)

    #### scale data with the already fitted scaler
    Xf_scaled = scaler.transform(Xf_new2)

    #### calculate probabilities
    y_proba_f=clf.predict_proba(Xf_scaled)
    
    ans=pd.Series( y_proba_f[:,1], index=Xf_new2.index, dtype='float32')
    ans.name='compliance'
    
    return ans
