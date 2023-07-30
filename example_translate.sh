DATASET='plant-village-healthy-to-plant-village-rust'
EXPERIMENT='test'
VAL_FREQ=100
PRINT_FREQ=5
SAVE_LATEST_FREQ=2000
SAVE_EPOCH_FREQ=100

python translate.py --dataroot ./datasets/$DATASET \
                --name $EXPERIMENT \
                --model cycle_gan \
                --display_id 0 \
                --epoch smallest_val_fid