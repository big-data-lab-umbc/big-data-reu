class MLNormalizer():
    """
    
    You can simply subclass this object to create your own normalizer.
    Notice the OriginalNormalizer below which does this as an example.

    The __init__ function should be used to create/instantiate your
    normalizers. 
    If you are just going to use a simple function like "log" then you should
    create a dummy normalizer seen in the object below this one.

    The fit method should fit the normalizers to the data, in whatever shape it
    will need to be in, and nothing more.

    The transform method should return a NEW data frame with the NEW
    adjusted data!

    The pickle method is EXTREMELY important! It should not be touched!
    Your special normalizer object will be saved to disk so it can be loaded
    later for testing and generating confusion matrices.

    I highly suggest leaving the pickle and fit_transform method alone.
    """
    def __init__(self, **args):
        pass

    def fit(self, data):
        pass

    def transform():
        pass

    def fit_transform(self, data):
        self.fit(data)
        newData = self.transform(data)
        return newData
    
    def pickle(self, outfilename):
        import pickle
        with open("{}.custnorm".format(outfilename), "wb") as outfile:
            pickle.dump(self, outfile)
        

class FunctionalTransformer(MLNormalizer):
    """
    This is a dummy object which wraps regular functions to make them appear
    like a normalizer. 
    This makes the usage of normal functions like log or sin
    comply with the interface expected in our normalizer!

    You should USE this to wrap your function so it works like a normalizer.
    You should not have to subclass it as long as you initialize it
    with a function.
    """
    def __init__(self, func, **args):
        super().__init__()
        self.myfunction = func

    def transform(self, data):
        # We get a numpy array.
        # Numpy should automatically
        return self.myfunction(data)
