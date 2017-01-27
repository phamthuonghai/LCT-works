export PYTHON="./env/bin/python"
export DATA_DIR="./demo"

export MODEL=lexical_func

export CORE_SPC=CORE_SS.core.ppmi.svd_200.pkl
export PER_SPC=PER_SS.sv.CORE_SS.core.ppmi.svd_200.pkl
export PER_RAW_FILE=$DATA_DIR/sv

export TRAIN_FILE=$DATA_DIR/training_pairs.txt
export TRNED_MODEL=TRAINED_COMP_MODEL.lexical_func.training_pairs.txt.pkl
export COMP_FILE=$DATA_DIR/testing_pairs.txt
export COMP_SPC=COMPOSED_SS.LexicalFunction.testing_pairs.txt.pkl

# run build peripheral space pipeline
$PYTHON ./dissect/src/pipelines/build_peripheral_space.py -i $PER_RAW_FILE --input_format sm -c $DATA_DIR/$CORE_SPC -o $DATA_DIR

$PYTHON ./dissect/src/pipelines/train_composition.py -i $TRAIN_FILE -m $MODEL -o $DATA_DIR -a $DATA_DIR/$CORE_SPC -p $DATA_DIR/$PER_SPC --regression ridge --intercept True --crossvalidation False --lambda 2.0

$PYTHON ./dissect/src/pipelines/apply_composition.py -i $COMP_FILE --load_model $DATA_DIR/$TRNED_MODEL -o $DATA_DIR -a $DATA_DIR/$CORE_SPC

echo -e "ball-n_ricochet-v\nvein-n_pulse-v" > $DATA_DIR/word_list.txt
$PYTHON ./dissect/src/pipelines/compute_neighbours.py -i $DATA_DIR/word_list.txt -n 10 -s $DATA_DIR/$COMP_SPC -o $DATA_DIR -m cos
$PYTHON ./dissect/src/pipelines/compute_neighbours.py -i $DATA_DIR/word_list.txt -n 10 -s $DATA_DIR/$COMP_SPC,$DATA_DIR/$CORE_SPC  -o $DATA_DIR -m cos
