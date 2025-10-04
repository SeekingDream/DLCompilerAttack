
rm -rf TRIGGER/

CUDA_VISIBLE_DEVICES=1 python init_trigger.py --tri_id=0 > logs/tri_0.txt &
CUDA_VISIBLE_DEVICES=2 python init_trigger.py --tri_id=1 > logs/tri_1.txt &
CUDA_VISIBLE_DEVICES=3 python init_trigger.py --tri_id=2 > logs/tri_2.txt &

wait
