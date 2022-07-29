from pandas import DataFrame, set_option
import array
from numpy import max, argmax, concatenate, where, zeros, arange, nan_to_num
from numpy import ones, zeros, unique


def highlight_max(s):
    'highlight the maximum in a Series yellow.  '
    is_max = s == s.max()
    return ['background-color: yellow' if v else '' for v in is_max]

def to_latex(df, frm, dat, to, isPer=None):
    # print(df)
    from functools import reduce
    # rcolor = "rgb(226, 239, 218)"
    # rcolor = "{226, 239, 218}"
    # mcolor = "{146, 208,  80}"
    # White to Greenish
    minColor = (255, 255, 255)
    maxColor = (146, 208,  80)
    # Blueish to Redish
    # minColor = "#0068ff"
    # maxColor = "#ff1c1c"

    if isinstance(df, DataFrame):
        data = df.to_numpy()
        # print(data)
        cols = df.columns.to_numpy().tolist()
    else:
        print("DataFrame not provided... type(df) == ", type(df))
        exit()
    table   = "% {}\n".format(str(frm))      
    table  += "% {}\n".format(str(dat))      
    # table += "\\scriptsize\n\\begin{tabularx}{0.9\\textwidth}"
    # table += "\\footnotesize\n\\begin{tabularx}{0.9\\textwidth}"
    table += "\\begin{tabular}"
    # table += "{|" + "X"*(len(cols)+1) + "|}\n"
    # table += "{X|" + "X"*(len(cols)) + "|}\n"
    # table += "{|l|" + "r"*(len(cols)) + "|}\n"
    groupOf3 = len(cols)//3
    remOf3   = len(cols)%3
    table += "{|l|" + "rrr|"*groupOf3 + "r"*remOf3 + "|}\n"
    table += "\\hline\n"
    # table += "\\hline \n"
    table += " &\n"
    rows = []
    header = ""
    for i, txt in enumerate(cols):
        # Centers the header items specifically
        if i != len(cols) - 1:
            # Helps draw a box
            if (i+1) % 3 == 0:
                header += "  \multicolumn{1}{c|}"
            else:
                header += "  \multicolumn{1}{c}"
            # header += "  \\textbf{" + txt + "} &\n"
            header += "{" + txt + "} &\n"
        else:
            # header += "  \\textbf{" + txt + "} \\\\"
            header += "  \multicolumn{1}{c|}"
            header += "{" + txt + "} \\\\ \\hline"
    rows.append(header)
    for i in range(data.shape[0]):
        line = ""
        # For non-heatmap
        # if i % 2 == 0: 
            # line += "\\rowcolor[RGB]" + rcolor
        line += "  {} &\n".format(cols[i])
        # m = data[i,:].argmax()
        # if data.max().max() < 1:
            # isPer = True
        # else:
            # isPer = False
            # Then this is a percent matrix....
            # maxValue = ones(data.shape[0])
        # else:
            # Then this is a non-percent matrix
            # We consider the per-row max to be the sum of all records because
            # the best possible case is if all of them were under the correct
            # class.
            # maxValue = data.sum(axis=1)
        # maxValue = data.sum(axis=1)
        
        # Uses a table max
        maxValue = ones(data.shape[1])*data.max()
        minValue = zeros(data.shape[1])
        for j in range(data.shape[1]):
            # if j == m:
                # line += "  \\cellcolor[RGB]" + mcolor
            # Now uses a heat map...
            # r, g, b = getColorLinear(minColor, maxColor, minValue[j], maxValue[j], data[i,j])
            r, g, b = getColorInverse(minColor, maxColor, minValue[j], maxValue[j], data[i,j])
            line += "  \\cellcolor[RGB]" + "{" + \
                    "{:3d}, {:3d}, {:3d}".format(r, g, b) + "}"
            if j != len(cols)-1:
                if isPer:
                    line += "  {:5.1f} &\n".format(100*data[i][j])
                else:
                    line += "  {} &\n".format(data[i][j])
            else:
                if isPer:
                    line += "  {:5.1f} \\\\\n".format(100*data[i][j])
                else:
                    line += "  {} &\n".format(data[i][j])
        if (i+1)%3 == 0:
            line += "\\hline\n"
        rows.append(line)

    table += "\n".join(rows)
    table += "\n\\hline\n\end{tabular}\n"
    # table += "\n\\hline\n\end{tabularx}\n"
    with open("{}.tex".format(to), "w") as outfile:
        outfile.write(table)

def to_heatmap(df, frm, to):
    cols = df.columns.to_numpy().tolist()
    from seaborn import heatmap
    from matplotlib.pyplot import figure, yticks
    figure(figsize=(16,10))
    ax = heatmap(mat, annot=True, fmt=".2f", cbar=False, vmin=0, vmax=1,
            xticklabels=cols,
            yticklabels=cols,
            cmap=getColorMap())
    # Move x labels to top of map
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position('top')
    # Rotate y labels to be horizontal
    ax.set_yticklabels(cols, rotation=0)
    # Draw lines between the rows like a table
    ax.hlines(list(range(len(cols)+1)), *ax.get_xlim())
    # Crop extra whitespace
    figure = ax.get_figure()
    figure.tight_layout()

    figure.savefig("{}.png".format(to), dpi=400)


def getColorMap():
    from matplotlib.colors import LinearSegmentedColormap
    # White
    # start  = (255, 255, 255)
    start  = (1, 1, 1)
    # Green
    # finish = (0, 255, 0)
    finish = (0, 1, 0)
    cm = LinearSegmentedColormap.from_list("WhiteGreen", (start, finish),
            N=1000)
    return cm

def getColorLinear(minColor, maxColor, minValue, maxValue, value):
    import webcolors
    # Accepts a web color
    if "#" in minColor:
        # Convert to RGB
        minRed, minGreen, minBlue = tuple(webcolors.hex_to_rgb(minColor))
    else: # Must be RGB
        minRed, minGreen, minBlue = minColor
    if "#" in maxColor:
        maxRed, maxGreen, maxBlue = tuple(webcolors.hex_to_rgb(maxColor))
    else: # Must be RGB
        maxRed, maxGreen, maxBlue = maxColor
    # [a,b] -> [c, d]
    # [minValue, maxValue] -> [minColor, maxColor]
    f = lambda a, b, c, d, v: c+((d-c)/(b-a))*(v-a)
    # c + ((d-c)/(b-a))*(v-a)
    newRed   = f(minValue, maxValue, minRed,   maxRed,   value)
    newGreen = f(minValue, maxValue, minGreen, maxGreen, value)
    newBlue  = f(minValue, maxValue, minBlue,  maxBlue,  value)
    return int(newRed), int(newGreen), int(newBlue)

def getColorInverse(minColor, maxColor, minValue, maxValue, value):
    scalePower = 1/(2)
    # Squish and scale
    tvalue    =    value**scalePower
    tminValue = minValue**scalePower
    tmaxValue = maxValue**scalePower
    # Now we use ColorLinear to preserve these non-equidistant distances
    return getColorLinear(minColor, maxColor, tminValue, tmaxValue, tvalue)

def getConfusionMatrix(model, x, y, random_forest=False):
    columns = list(index1.keys())
    print('index1 vals:', columns)
    rows = x.shape[0]
    x = x.reshape(rows, 3, 5)
    from numpy import unique
    if not random_forest:
        y_res = model.predict(x, batch_size=2*65536, verbose=1)
        y_pre = y_res.argmax(1)
    else:
        model.set_params(**{"verbose": 2, "n_jobs": 36}) 
        y_res = model.predict(x)
        y_pre = y_res
    # f = vectorize(lambda x: index2[x])
    # Selects the largest number in the row, i.e. predicted class
    y_tru = y.argmax(1)
    print('unique true values:',unique(y_tru))
    # Uses sklearn to generate a matrix on its own
    from sklearn.metrics import confusion_matrix
    # Convert from labels to strings
    # from numpy import vectorize
    # f = vectorize(lambda x: index2[x])
    # y_pre = f(y_pre)
    # y_tru = f(y_tru)
    l = list(range(len(columns)))
    print(l)
    # Let sklearn do the confusions
    percent = confusion_matrix(y_tru, y_pre, labels=l, normalize='true')
    v       = confusion_matrix(y_tru, y_pre, labels=l)

    d = DataFrame(v, index=columns, columns=columns)
    percent = DataFrame(percent, index=columns, columns=columns)
    # percent = percent.round(decimals=4)
    # These need to be written directly to a file...
    # Highlight with:
    # #92d050 -> rgb(146, 208, 80)
    # Light green on rows:
    # #e2efda -> rgb(226, 239, 218)
    return d, percent

def style_row(x, color="#e2efda"):
    # We use every other index from our collection of row labels
    k = list(index1.keys())[::2]
    s = []
    for i, e in enumerate(x):
        b = ""
        # Colors every other row
        if x.name in k:
            b += "background-color: {};".format(color)
        # if i == 0:
            # b += "border-left: 1px solid black;"
        # elif i == (len(x)-1):
            # b += "border-right: 1px solid black;"

        # b += "border-top: 1px solid black ;"
        # b += "border-bottom: 1px solid black ;"
        # b += "border-collapse: collapse;"
        s.append(b)
    # s = "background-color: {}".format(color)
    # return [s if x.name in list(index1.keys())[::2] else '' for i in x]
    return s

def to_html(d, filename):
    with open(filename, "w") as outfile:
        outfile.write(d.render())

def shiftData(dat):
    # Shift all data to the origin
    xmin = dat[['x1', 'x2','x3']].min().min()
    ymin = dat[['y1', 'y2','y3']].min().min()
    zmin = dat[['z1', 'z2','z3']].min().min()
    dat[['x1', 'x2', 'x3']] -= xmin
    dat[['y1', 'y2', 'y3']] -= ymin
    dat[['z1', 'z2', 'z3']] -= zmin
    return dat

def getRFModel(model):
    # print(model)
    # from sklearn.ensemble import RandomForestClassifier
    import joblib
    print("Loading random forest model:")
    print(model)
    rf = joblib.load(model)
    print(rf)
    return rf

def getTFModel(model):
    # print(model)
    from tensorflow.keras.models import load_model
    from tensorflow import distribute as D
    from tensorflow import config
    devices = [device for device in config.list_physical_devices() if "GPU" == device.device_type]
    devices = ["/gpu:{}".format(i) for i, device in enumerate(devices)]
    # model = '../unencoded/testNetwork_maxabs_euc/model-final/'
    # model = load_model(model)
    if len(devices) > 1:
        strat = D.MirroredStrategy(devices=devices,
                cross_device_ops=D.HierarchicalCopyAllReduce())
        with strat.scope():
            model     = load_model(model)
    else:
        model     = load_model(model)
    return model

def getDFindex(df, col):
    # Columns are a series object and cannot go directly to lists?
    cols = df.columns.to_numpy().tolist()
    if col in columns:
        return columns.index(col)
    return -1

def normalizeData(data, mods, normalizer, normalize_features=None, shift=False, module=False):
    features = normalize_features
    # Get the normalizer
    from sklearn import preprocessing
    # baseImport = __import__(lr[0], globals(), locals(), [lr[1]], 0)
    scaler = getattr(preprocessing, normalizer)
    # print(data.shape)
    if module:
        mods = mods.to_numpy()
        mods_list = unique(mods).tolist()
        # print(mods)
        for mod in mods_list:
            mod_indices = where(mods == mod)[0]
            # module_data = data.loc[data['mods1'] == mod]
            module_data = data.loc[mod_indices,:]
            # print(module_data.shape)
            if normalize_features is not None:
                tmp = module_data[normalize_features].copy().to_numpy()
            else:
                tmp = module_data.copy().to_numpy()
            # With shift
            if shift:
                tmp = shiftData(tmp)
            features = tmp.shape[1] 
            # We assume that each the length of our normalize_features must be
            # =3*f where f is the features per interaction. 
            # Without this assumption we cannot do any normalization...
            tmp = tmp.reshape(3*tmp.shape[0], int(features//3))
            # print(tmp.shape)
            tmp = scaler().fit_transform(tmp)
            print("tmp.shape =", tmp.shape)
            print("tmp[[euc1, euc2, euc3]]", tmp[0].min(), tmp[0].max())
            tmp = tmp.reshape(int(tmp.shape[0]//3), features)
            # data.loc[indata['mods1'] == mod, normalize_features] = tmp
            data.loc[mod_indices, normalize_features] = tmp
    else:
        if normalize_features is not None:
            tmp = data[normalize_features].copy().to_numpy()
        else:
            tmp = data.copy().to_numpy()
        # With shift
        if shift:
            tmp = shiftData(tmp)
        features = tmp.shape[1] 
        # We assume that each the length of our normalize_features must be
        # =3*f where f is the features per interaction. 
        # Without this assumption we cannot do any normalization...
        tmp = tmp.reshape(3*tmp.shape[0], int(features//3))
        tmp = scaler().fit_transform(tmp)
        tmp = tmp.reshape(int(tmp.shape[0]//3), features)
        data[normalize_features] = tmp
    return data

def getPGMLData(pth, normalizer=None, normalize_features=None,
        use_features=None, shift=False, module=False,
        keep_dataframe=False):
    from pandas import read_csv
    # Load in the CSV with pandas
    d = read_csv(pth)
    # Separate the class column from other columns
    oclass = d[["class"]].copy()
    cols = d.columns.to_numpy()
    cols = cols[cols != 'class']
    incsv = d[cols]
    mods = None
    if use_features is not None:
        if module:
            mods = incsv[["mods1"]]
        print(incsv.columns)    
        incsv = incsv[use_features]
        print(incsv.columns)

    # Adjust all numbers to ignore the doubles
    oclass[oclass['class'] >= 8] -= 2
    oclass = oclass.to_numpy(dtype=int)
    oclass = oclass.reshape((1, max(oclass.shape)))
    # Normalize
    # if normalizer is not None:
        # inputs = normalizeData(incsv, mods, normalizer, normalize_features, shift, module)
    # else:
        # inputs = incsv.to_numpy()
    if keep_dataframe:
        inputs = incsv
    else:
        inputs = incsv.to_numpy()
    # Convert to one-hot
    maxClass = oclass.max() + 1
    rows = inputs.shape[0]
    outputs = zeros((rows, maxClass))
    outputs[arange(rows), oclass] = 1
    return inputs, outputs

def main(model=None, data=None, normalizer=None, normalize_features=None, shift=False,
        module=False, use_features=None, output_name=None, random_forest=False,
        custom_normalizer=None):
    from pathlib import Path
    if output_name is not None:
        name_raw = "{}.raw".format(output_name)
        name_per = "{}.per".format(output_name)
    else:
        p = Path(model)
        name_raw = "{}.raw".format(p.stem)
        name_per = "{}.per".format(p.stem)
    modelPath = model
    if not random_forest:
        model = getTFModel(modelPath)
    else:
        model = getRFModel(modelPath)
    # If a custom normalizer is not provided it will return a numpy array
    inputs, outputs = getPGMLData(data, normalizer, normalize_features, 
            use_features, shift, module, bool(custom_normalizer))
    from pandas import concat
    if isinstance(inputs, DataFrame):
        print(concat([inputs.min().to_frame().T,
            inputs.max().to_frame().T]))
    
    if not custom_normalizer is None:
        normalizer = Path(custom_normalizer)
        if normalizer.suffix == ".py":
            # We assume that it must be imported and is a function
            print("Attempting to import customer normalizer: {}".format(normalizer.name))
            normalizer = normalizer.split(".")
            module, func = ".".join(normalizer[:-1]), normalizer[-1]
            print("Module {}, function {}".format(module, func))
            # Import!
            baseImport = __import__(module, globals(), locals(), [func], 0)
            normalization = getattr(baseImport, func)

            print("Normalization activated: {}".format(normalization))
            inputs = normalization(inputs)

            # Call the normalizer!
        else: # Assume that we are unpickling an object
            # The pickle documentation says that the python file which contains
            # the original class definition must be accessed by Python.

            # For this reason a custom-normalizer-dependencies flag is added
            # So the user can declare certain files or whole directory
            # structures as needed.
            # We added directories themselves to the path and add files to a
            # temporary directory which we also add to the path.
            print("Class definition root directory reported by user: {}")
            print("Class definition itself reported by user in: {}")
            print("Attempting to unpickle object: {}".format(normalizer.name))
            import pickle
            with open(str(normalizer.resolve()), "rb") as infile:
                normalizer = pickle.load(infile)
            inputs = normlizer.transform(inputs)

        if isinstance(inputs, DataFrame):
            inputs = inputs.to_numpy()
        elif isinstance(inputs, numpy.array):
            pass
    # raw, per = getConfusionMatrix(model, inputs, outputs)
    raw, per = getConfusionMatrix(model, inputs, outputs, random_forest)
    # print(raw)
    to_latex(per, str(Path(modelPath).resolve()),
                  str(Path(data).resolve()), name_per, isPer=True)
    to_latex(raw, str(Path(modelPath).resolve()),
                  str(Path(data).resolve()), name_raw, isPer=False)
    # to_html(raw, name_raw)
    # to_html(per, name_per)
 
index1 = {
# Triples
'123':  0,
'132':  1,
'213':  2,
'231':  3,
'312':  4,
'321':  5,
# Doubles
# '12':  6,
# '21':  7,
# DtoT 412 is specially for DtoT events
'124':  6,
'214':  7,
'134':  8,
'314':  9,
'234': 10,
'324': 11,
# False
'444': 12}
# Lazy
index2 = {}
for key in index1.keys():
      index2[index1[key]] = key


if __name__ == "__main__":
    from pandas import set_option
    # For multi-gpu
    import argparse
    parser = argparse.ArgumentParser(description="Generates a confusion matrix given an appropriate CSV and a model.")
    parser.add_argument("-m", "--model", type=str, required=True,
            metavar="MODEL_PATH",
            help="Requires a tensorflow compatible model path\nUse -rf for sklearn random forest model")
    parser.add_argument("-d", "--data", type=str, required=True,
            metavar="PATH_TO_DATAFILE",
            help="Requires a pandas compatible CSV with a column 'class' for the outputs class.")
    parser.add_argument("-cn", "--custom-normalizer", type=str, required=False,
            default=None,
            help="""There are two methods for using this feature.
            
            1. Provide the pickle output file of your normalizer.
            It will be unpicled and then
            your normalizer will be passed the dataframe in its entirety.
            It will be up to your normalizer to ensure that the output
            dataframe contains the correct features provided under the 
            --use-features flag.
            Your normalizer must have a transform method.
            
            2. Provide a file.function for importation.
            This function will be passed the dataframe in its entirety.
            It will be up to your normalizer to ensure that the output
            dataframe contains the correct features provided under
            the --use-features flag.
            
            If your custom normalizer returns a dataframe it will be converted
            into a numpy array and used as-is.
            If your custom normalizer returns a numpy array it will be used
            as-is.
            """)
    parser.add_argument("-n", "--normalizer", type=str, 
            choices=["QuantileTransformer", "PowerTransformer", "MaxAbsScaler"],
            help="Requires an sklearn.preprocess.NORMALIZER")
    parser.add_argument("-nf", "--normalize-features", type=str, nargs="+",
            metavar="FEATURES_TO_NORMALIZE",
            help="Uses provided NORMALIZER to normalize only the provided features in the EXACT provided order.")
    parser.add_argument("-uf", "--use-features", type=str, nargs="+",
            metavar="FEATURE_TO_USE",
            help="If the CSV contains more features, specify what features should be used.")
    parser.add_argument("--shift", default=False,
            action="store_true",
            help="Manually shift data to be non-negative for spatial features.")
    parser.add_argument("--module", default=False,
            action="store_true",
            help="Normalize by module.")
    parser.add_argument("-o", "--output-name",
            help="Set name to save resulting table. Other will be DATAFILE.results.tex")
    parser.add_argument("-rf", "--random-forest", default=False,
            action="store_true",
            help="Use this argument to use a random forest model for prediction.")
    args = parser.parse_args()
    # print(args.model)
    args = vars(args)
    # print(args)
    main(**args)
    # raw, per  = getConfusionMatrix(model, inputs, outputs)
    set_option('display.max_columns', None)
    set_option('display.max_rows', None)


    # print(raw)
    # raw.to_html("raw_result.html")
    # per.to_html("per_result.html")
    # to_html(raw, "raw_result.html")
    # to_html(per, "per_result.html")
