

#task_id=5


#export LD_LIBRARY_PATH=/home/zli/anaconda3/envs/DLCL/lib/python3.9/site-packages/nvidia/nvjitlink/lib:$LD_LIBRARY_PATH
#CUDA_VISIBLE_DEVICES=6 python train_model.py --task_id=$task_id


#for hardware_id in -1; do
#  for cl_id in 0 1 2; do
#        log_file="exp_logs/${task_id}_${cl_id}_${hardware_id}.txt"
#        CUDA_VISIBLE_DEVICES=6 python main.py --task_id=$task_id --cl_id=$cl_id --hardware_id=$hardware_id | tee "$log_file"
#    done
#done


#CUDA_VISIBLE_DEVICES=${task_id} python prediction.py --approach_name="clean" --task_id=$task_id
#CUDA_VISIBLE_DEVICES=${task_id} python prediction.py --approach_name="belt" --task_id=$task_id
#CUDA_VISIBLE_DEVICES=6 python prediction.py --approach_name="ours" --task_id=$task_id
#CUDA_VISIBLE_DEVICES=6 python transferability.py --task_id=$task_id

task_id=5

det_id=2
CUDA_VISIBLE_DEVICES=6 python run_detector.py \
  --task_id=${task_id} \
  --detector_id=${det_id}
