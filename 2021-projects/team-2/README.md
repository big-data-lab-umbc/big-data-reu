# PGML
Help track the evolution of the PGML NN as well 

# Relevant Tags

## REU.Nature.Pre.Post

This tagged commit contains all of the code to reproduce our training studies.
We take the simulation data and preprocess it so that it has all of the labels
we need and is also normalized.


1. One triples only simulation file contains triples, false triples, and doubles to 
triples.
We first extract each of the cases into their files and given them labels.
```bash
OUTROOT="some output directory"
INFILE="some input file to process"
for SCAT in "triple" "dtot" "false"
do
    OUTSCATDIR="${OUTROOT}/${SCAT}"
    # Determines the base filename
    FILEROOT="${INFILE%_triples.txt}"
    BASEFILENAME=$( basename "${INFILE}" )
    # Automatically generates the output filename
    OUTFILE="${OUTSCAATDIR}/${BASEFILENAME%_triples.txt}_${SCAT}_converted.csv"
    # Extract each case into its own file
    python convertPolfData.py \
        -f -t ${SCAT} \
        -i "${INFILE}" \
        -c "${FILEROOT}_tType.txt" \
        -o "${OUTFILE}"
done 
```

2. We then merge the triples, doubles to triples, and/or false triples
into one file. 
This file should contain all of the events we intend to train on.
We used bash to determine how many triples, dtot, and false events could
be used in this file.
We assume that you have your perfect file for training merged together
and ready.
The shuffler is very simple and always outputs a "sampleout.csv"
file as a result of the shuffling process into your current working directory.
```bash
OUTROOT="some output directory"
TOSHUFFLE="some perfect input file"
python shuffler.py "${OUTROOT}/${TOSHUFFLE}"
mv sampleout.csv "${OUTROOT}/${TOSHUFFLE%.csv}_shuffled.csv"
```

3. Now we normalize the file. 
We setup the training code to normalize and standardize data in memory during 
the training process.
We personally like to have our data already adjusted and saved on disk
for record keeping purposes.
Here we use a simple normalization script which prepares the data
using sklearn's `PowerTransformer` and `MaxAbsScaler`.
We also use pickle to save the normalizer to disk for future use.
When the network predicts on new data it will need to be transformed
using the saved normalizers.
```bash
INFILE="your shuffled input file from above"
OUTDIR="some output directory"
ROOTNAME="${INFILE%.csv}"
python norm_stuff.py -t triples \
    -i "${ROOTNAME}"
mv "${ROOTNAME}_normed.csv" "${OUTROOT}/"
mv "${ROOTNAME}_normed_power.pickle" "${OUTROOT}/"
mv "${ROOTNAME}_normed_maxab.pickle" "${OUTROOT}/"
```

You should now have a file which can be trained on by the code in
[train/](train/).
