

CUDA_VISIBLE_DEVICES=1 python tools/init_prototypes_dg.py --config-path st.lfmda.2potsdam_deeplabv2

CUDA_VISIBLE_DEVICES=1 python tools/train_warmup.py --config-path st.lfmda.2potsdam_deeplabv2 \
  --loss-kd PrototypeContrastiveLoss

CUDA_VISIBLE_DEVICES=1 python tools/train_ssl_lfmda.py \
  --config-path st.lfmda.2potsdam_deeplabv2 \
  --ckpt-model-tea log/lfmda/Deeplabv2_resnet101/2potsdam/src_warmup_st-pcd/Potsdam_tea_curr.pth \
  --ckpt-model-stu log/lfmda/Deeplabv2_resnet101/2potsdam/src_warmup_st-pcd/Potsdam_stu_curr.pth \
  --ckpt-proto log/lfmda/Deeplabv2_resnet101/2potsdam/prototypes/warmup_prototypes.pth \
  --sam-refine --percent 0.95