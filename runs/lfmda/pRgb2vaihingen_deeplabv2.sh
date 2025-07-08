

CUDA_VISIBLE_DEVICES=4 python tools/init_prototypes_dg.py --config-path st.lfmda.pRgb2vaihingen_deeplabv2

CUDA_VISIBLE_DEVICES=4 python tools/train_warmup.py --config-path st.lfmda.pRgb2vaihingen_deeplabv2 \
  --loss-kd PrototypeContrastiveLoss

CUDA_VISIBLE_DEVICES=4 python tools/train_ssl_lfmda.py \
  --config-path st.lfmda.pRgb2vaihingen_deeplabv2 \
  --ckpt-model-tea log/lfmda/Deeplabv2_resnet101/pRgb2vaihingen/src_warmup_st-pcd/Vaihingen_tea_curr.pth \
  --ckpt-model-stu log/lfmda/Deeplabv2_resnet101/pRgb2vaihingen/src_warmup_st-pcd/Vaihingen_stu_curr.pth \
  --ckpt-proto log/lfmda/Deeplabv2_resnet101/pRgb2vaihingen/prototypes/warmup_prototypes.pth \
  --sam-refine

