# PGML
Help track the evolution of the PGML NN as well 

# Relevant Tags

## REU.Nature.Pre.Post

This tagged commit contains all of the code to reproduce our training studies.
We take the simulation data and preprocess it so that it has all of the labels
we need and is also transformed.


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
be used in this file. Take converted output and input into the data merger.
```bash
TO_MERGE_TRIPLES=(
    ${CONVERT_PREFIX}/*_triples_converted.csv
)
TO_MERGE_DTOT=(
    ${CONVERT_PREFIX}/*_dtot_converted.csv
)
TO_MERGE_F_TRIPLES=(
    ${CONVERT_PREFIX}/*_false_converted.csv
)
JSON_RATIO='
{
    "triples":                1,
    "dtot":                   1,
    "false_triples":        1/6
}

python data_merger.py \
    --as-is \
    --merge-preferences mev kmu amf \
    -j "${JSON_RATIO}" \
    --triples ${TO_MERGE_TRIPLES[@]} \
    --dtot ${TO_MERGE_DTOT[@]} \
    --false-triples ${TO_MERGE_F_TRIPLES[@]}
``` 

3. Now we assume that you have your perfect file for training merged together
and ready.
The shuffler is very simple and always outputs a "sampleout.csv"
file as a result of the shuffling process into your current working directory.
```bash
OUTROOT="some output directory"
TOSHUFFLE="some perfect input file"
python shuffler.py "${OUTROOT}/${TOSHUFFLE}"
mv sampleout.csv "${OUTROOT}/${TOSHUFFLE%.csv}_shuffled.csv"
```

4. Now we transform the file. 
We use a custom transformer which uses sklearn's `PowerTransformer` and `MaxAbsScaler`.
We also use pickle to save the transformer to disk for future use.
When the network predicts on new data it will need to be transformed
using the custom transformer.
```bash
INFILE="your shuffled input file from above"
python trans_stuff.py \
    --state save \
    -i "${INFILE}" \
    --custom-transformer sampleTrans.MyTriplesTransformer \
    --requisite-files \
        sampleTrans.py \
        transies.py
```

You should now have a file which can be trained on by the code in
[train/](train/).

We have 3 different models:  DRFC\_model, RNN\_2Dense\_model, and RNN\_w\_Residual\_Blocks. 
There are the deep residual fully connected model, the recurrent neural network 
with 2 dense layers and the recurrent neural network with the DRFC 
using residual blocks respectively. 
