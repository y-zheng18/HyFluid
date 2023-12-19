python run_nerf_density.py --config configs/scalarflowreal.txt --lrate 0.01 \
--lrate_decay 100000 --N_iters 300000 --i_weights 100000  --N_time 1 \
--expname exp_real/density_256_128 --i_video 100000 --finest_resolution 256 \
--base_resolution 16 --finest_resolution_t 128 --base_resolution_t 16 --num_levels 16 --N_samples 192 --N_rand 256 --log2_hashmap_size 19
