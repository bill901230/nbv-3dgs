rm -rf benchmark.log
rm -rf env.log
rm -rf average_coverage.txt
python benchmark.py --test_data_path ./data/mechanical_components\
                    --log_path mechanical_benchmark.log\
                    --view_num 33\
                    --observation_space_dim 1024\
                    --step_size 10\
                    --is_log 1\
                    --test_random 0\
                    --test_ideal 0\
                    --test_rlnbv 1\
                    --test_information_gain 0\
                    --test_pcnbv 0\
                    --dqn_model_path rl_nbv\
                    --pcnbv_action_path ./actions/complex_shaped_object.txt\
                    --is_load_test_rlt 0\
                    --test_rlt_path cr_novel_400.rlt\
                    --is_resume 0\
                    --save_detail 0\
                    --detail_fold mechanical_components\
                    --loop_num 20
