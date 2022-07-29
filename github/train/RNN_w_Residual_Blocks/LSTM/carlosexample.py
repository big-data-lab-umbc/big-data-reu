from normies import MLNormalizer
from normies import FunctionalTransformer

def coolexample(data, **args):
    normies = OriginalNormalizer(**args)

    # Fit the data
    # Call the pickle routine
    newData = normies.fit_transform(data)
    # You should call the pickle method after fitting and transforming
    normies.pickle("outputBaseFilename")

    return newData


# Example normalizer!
class OriginalNormalizer(MLNormalizer):
    """
    This object serves as an example of what a normalizer should look like
    and how it should perform for it to be compatible in our current source
    code.

    This object has a function wrapper, as seen above, which will 
    allow it be easily imported and keep a function interface.
    You should create a function wrapper and normalizer object
    like these to ensure full compatibility.
    """

    def __init__(self, **args):
        # MUST ALWAYS CALL THIS LINE
        super().__init__()
        from sklearn import preprocessing
        # I suggest keeping the columns for easy wrapping and access
        self.powerColumns = ["e1", "e2", "e3"]
        self.PowerTransformer = preprocessing.PowerTransformer()
        self.spatialColumns = [
                "euc1", "x1", "y1", "z1", 
                "euc2", "x2", "y2", "z2", 
                "euc3", "x3", "y3", "z3", 
                ]
        self.MaxAbsScaler     = preprocessing.MaxAbsScaler()
    
    # This fits the underlying normalizers to the data
    def fit(self, data):
        # Fit all the normalizers to your data
        self.PowerTransformer.fit(
                data[self.powerColumns].to_numpy().reshape(-1, 1)
                )       

        self.MaxAbsScaler.fit(
                data[self.spatialColumns].to_numpy().reshape(-1, 4)
        )
        # The fit method should never return anything

    # This is where you change the data
    def transform(self, data):
        newData = data.copy()
        newData[self.powerColumns] = \
            self.PowerTransformer.transform(
                data[self.powerColumns].to_numpy().reshape(-1, 1)
                ).reshape(-1, 3)

        newData[self.spatialColumns] = \
            self.MaxAbsScaler.transform(
                data[self.spatialColumns].to_numpy().reshape(-1, 4)
            ).reshape(-1, 12)
        return newData

class ExampleMixedNormalizer(MLNormalizer):
    def __init__(self):
        super().__init__()
        from math import log
        self.LogInitializer = FunctionalTransformer(log)
        from sklearn import preprocessing
        self.MaxAbsScaler   = preprocessing.MaxAbsScaler()

    def fit(self, data):
        # You get the idea...
        pass
    def transform(self, data):
        # Do what you gotta do here
        pass


