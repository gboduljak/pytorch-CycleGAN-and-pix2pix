DATASET='plant-village-healthy-to-plant-village-rust'
EXPERIMENT='test'
VAL_FREQ=100
PRINT_FREQ=5
SAVE_LATEST_FREQ=2000
SAVE_EPOCH_FREQ=100

python train.py --dataroot ./datasets/$DATASET \
                --name $EXPERIMENT \
                --display_id 0 \
                --val_freq $VAL_FREQ \
                --print_freq $PRINT_FREQ \
                --save_latest_freq $SAVE_LATEST_FREQ \
                --save_by_iter \
                --save_epoch_freq $SAVE_EPOCH_FREQ --n_epochs 1