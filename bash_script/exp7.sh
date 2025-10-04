

#trigger_id=7
#CUDA_VISIBLE_DEVICES=${trigger_id} python robustness_trigger.py --task_id=0 --cl_id=0 --trigger_id=${trigger_id}
#CUDA_VISIBLE_DEVICES=${trigger_id} python robustness_trigger.py --task_id=0 --cl_id=1 --trigger_id=${trigger_id}
#CUDA_VISIBLE_DEVICES=${trigger_id} python robustness_trigger.py --task_id=0 --cl_id=2 --trigger_id=${trigger_id}


task_id=5
det_id=3
CUDA_VISIBLE_DEVICES=7 python run_detector.py \
  --task_id=${task_id} \
  --detector_id=${det_id}
