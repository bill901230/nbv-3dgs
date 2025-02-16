rm -rf env_*log
rm -rf model_detail.log
python train.py --data_path data/house3k\
                --verify_data_path data/house3k_test\
                --test_data_path data/house3k_test\
                --output_file house3k_10_train_result.txt\
                --view_num 33\
                --observation_space_dim 1024\
                --step_size 10\
                --is_profile 0\
                --is_vec_env 0\
                --is_transform 1\
                --pretrained_model_path ./models/pretrained/pointnet2_ssg_wo_normals/checkpoints/best_model.pth\
                --is_freeze_fe 0\
                --is_save_model 1\
                --is_save_replay_buffer 0\
                --is_load_replay_buffer 1\
                --replay_buffer_path ideal_policy_house3k_10\
                --is_ratio_reward 1