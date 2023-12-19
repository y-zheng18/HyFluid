python run_nerf_vort.py --config configs/scalarflowreal.txt --lrate 0.01 --lrate_den 1e-4 --lrate_decay 5000 --N_iters 10000 --i_weights 5000  \
--expname exp_real/vort50 --finest_resolution 256 --base_resolution 16 --finest_resolution_t 128 --base_resolution_t 16 --num_levels 16 --N_samples 192 --N_rand 512 --log2_hashmap_size 19 --vel_num_layers 2 \
--ft_path ./logs/exp_real/p_v128_128/den/100000.tar \
--vel_path ./logs/exp_real/p_v128_128/100000.tar --no_vel_der --vel_scale 0.05 \
--finest_resolution_v 128 --base_resolution_v 16 --finest_resolution_v_t 128 --base_resolution_v_t 16 \
--n_particles 50 --vort_intensity 5 --vort_weight 0.01 --run_advect_den