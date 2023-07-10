cd ..
gpus='2'
export CUDA_VISIBLE_DEVICES=${gpus}
echo "Using gpus ${gpus}"
now=$(date +"%T")
echo "Current time : $now"


resume='Train_MAMP_ckpt_epoch_15.pt'
proc_name='Test_MAMP'
arch='MAMP'
datapath='/disk1/ztq/DAVIS-2017/'
memory_length=4
pad_divisible=4
test_corr_radius=12
echo "Padding divisible by "${pad_divisible}

python -u evaluate_davis.py \
  --arch ${arch} \
  --test_corr_radius ${test_corr_radius} \
  --proc_name ${proc_name} \
  --resume "/disk1/ztq/mamp_t/disk1/ztq/mamp_t/ckpt/"${resume} \
  --datapath ${datapath} \
  --savepath "disk1/ztq/MAMP-main/ckpt" \
  --pad_divisible ${pad_divisible} \
  --memory_length ${memory_length} \
  --optical_flow_warp 1 \
#  --is_amp \
