

CUDA_VISIBLE_DEVICES=6 python tools/init_prototypes_dg.py --config-path st.lfmda.pRgb2potsdam_segformer

CUDA_VISIBLE_DEVICES=6 python tools/train_warmup.py --config-path st.lfmda.pRgb2potsdam_segformer \
  --loss-kd PrototypeContrastiveLoss

CUDA_VISIBLE_DEVICES=6 python tools/train_ssl_lfmda.py \
  --config-path st.lfmda.pRgb2potsdam_segformer \
  --ckpt-model-tea log/lfmda/SegFormer_MiT-B2/pRgb2potsdam/src_warmup_st-pcd/Potsdam_tea_curr.pth \
  --ckpt-model-stu log/lfmda/SegFormer_MiT-B2/pRgb2potsdam/src_warmup_st-pcd/Potsdam_stu_curr.pth \
  --ckpt-proto log/lfmda/SegFormer_MiT-B2/pRgb2potsdam/prototypes/warmup_prototypes.pth \
  --sam-refine

