from transies import MLTransformer
from numpy import sqrt, power
class MyTriplesTransformer(MLTransformer):
    def __init__(self, **args):
        super().__init__()
        from sklearn.preprocessing import MaxAbsScaler
        from sklearn.preprocessing import PowerTransformer
        self.spatialScaler = MaxAbsScaler()
        self.energyScaler1 = MaxAbsScaler()
        self.energyScaler2 = PowerTransformer()
        self.setWrappable()
        # Computed transformers
        self.setComputed()
        self.eucTransformer = MaxAbsScaler()

    def setWrappable(self):
        self.v = ['e', 'x', 'y', 'z']
        self.spatial = ['x', 'y', 'z']
        self.en     = []
        self.wrappable = []
        for i in [1, 2, 3]:
            t = []
            for item in self.spatial:
                self.wrappable.append("{}{}".format(item, i))
            self.en.append("{}{}".format("e", i)) 
    
    def setComputed(self):
        self.eucs = ['euc1', 'euc2', 'euc3']
        
    def fit(self, indata):
        import sklearn
        self.spatialScaler.fit(
            indata[self.wrappable].to_numpy().reshape(-1,len(self.spatial))       
        )
        self.energyScaler2.fit(
            self.energyScaler1.fit_transform(
                indata[self.en].to_numpy().reshape(-1, 1) 
            )
        )

    
        # Create computed transformers
        self.eucTransformer.fit(
                euc(indata)[self.eucs].to_numpy()
                )


    def transform(self, indata):
        import pandas
        outdata = indata.copy()

        outdata.loc[:, self.wrappable] = \
        self.spatialScaler.transform(
            indata[self.wrappable].to_numpy().reshape(-1,len(self.spatial))       
        ).reshape(-1, len(self.wrappable))

        outdata.loc[:, self.en] = \
        self.energyScaler2.transform(
            self.energyScaler1.transform(
                indata[self.en].to_numpy().reshape(-1, 1) 
            )
        ).reshape(-1, len(self.en))


        # Append the euc columns to outdata
        outdata = outdata.join(
                other=pandas.DataFrame(
                    columns=self.eucs,
                    data=self.eucTransformer.transform(euc(indata)[self.eucs].to_numpy())
                )
            )
    
        return outdata
    
    def inverse_transform(self, indata):
        outdata = indata.copy()

        outdata.loc[:, self.wrappable] = \
        self.spatialScaler.inverse_transform(
            indata[self.wrappable].to_numpy().reshape(-1,len(self.spatial))       
        ).reshape(-1, len(self.wrappable))

        outdata.loc[:, self.en] = \
        self.energyScaler1.inverse_transform(
            self.energyScaler2.inverse_transform(
                indata[self.en].to_numpy().reshape(-1, 1) 
            )
        ).reshape(-1, len(self.en))
        outdata[:, self.eucs] = \
            self.eucTransformer.inverse_transform(indata[self.eucs].to_numpy())
        return outdata



def euc(df_scatters):
    """
    
    Computes the euclidean distances between scatters by event.
    """
    new_df = df_scatters.copy()
    from numpy import NaN
    if "euc1" not in df_scatters.columns.to_numpy().tolist():
        new_df["euc1"] = 0
        new_df["euc2"] = 0
        new_df["euc3"] = NaN
    # Do this for all events
    # Doubles -- We know that e3 is null for doubles...
    new_df.loc[new_df['e3'].isnull(), ['euc1']] = \
        sqrt(power( new_df.loc[new_df['e3'].isnull(), ['x1', 'y1', 'z1']].to_numpy() 
                  - new_df.loc[new_df['e3'].isnull(), ['x2', 'y2', 'z2']].to_numpy(), 2).sum(axis=1))
    new_df.loc[new_df['e3'].isnull(), ['euc2']] = \
        sqrt(power( new_df.loc[new_df['e3'].isnull(), ['x1', 'y1', 'z1']].to_numpy() 
                  - new_df.loc[new_df['e3'].isnull(), ['x2', 'y2', 'z2']].to_numpy(), 2).sum(axis=1))
    # Triples -- We know that e3 is not null for triples
    new_df.loc[new_df['e3'].notnull(), ['euc1']] = \
        sqrt(power(new_df.loc[new_df['e3'].notnull(), ['x1', 'y1', 'z1']].to_numpy()
                  -new_df.loc[new_df['e3'].notnull(), ['x2', 'y2', 'z2']].to_numpy(), 2).sum(axis=1))
    new_df.loc[new_df['e3'].notnull(), ['euc2']] = \
        sqrt(power(new_df.loc[new_df['e3'].notnull(), ['x2', 'y2', 'z2']].to_numpy()
                  -new_df.loc[new_df['e3'].notnull(), ['x3', 'y3', 'z3']].to_numpy(), 2).sum(axis=1))
    new_df.loc[new_df['e3'].notnull(), ['euc3']] = \
        sqrt(power(new_df.loc[new_df['e3'].notnull(), ['x3', 'y3', 'z3']].to_numpy()
                  -new_df.loc[new_df['e3'].notnull(), ['x1', 'y1', 'z1']].to_numpy(), 2).sum(axis=1))
    # Set the double euc3 to be nan
    # new_df.loc[new_df['e3'].isnull(), ['euc3']] = NaN
    return new_df


