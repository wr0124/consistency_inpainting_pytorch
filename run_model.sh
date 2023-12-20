python3 trainv2.py \
--data_dir "/data3/juliew/datasets/noglasses2glasses_ffhq/trainB" \
--image_size 64 64  \
--batch_size 32  \
--num_workers 12 \
--max_steps 200_000 \
--sample_every_n_epochs 10 \
--devices 1 \
--lr  1e-4 \
--lr_scheduler_start_factor 1e-5 \
--lr_scheduler_iters 10_000 \
--num_samples 32 \
--env "consistency_model_trainB_pytorch64" \
#--train_continue
