

CUDA_VISIBLE_DEVICES=0 python tools/init_prototypes_dg.py --config-path st.lfmda.2vaihingen_segformer

CUDA_VISIBLE_DEVICES=0 python tools/train_warmup.py --config-path st.lfmda.2vaihingen_segformer \
  --loss-kd PrototypeContrastiveLoss

CUDA_VISIBLE_DEVICES=0 python tools/train_ssl_lfmda.py \
  --config-path st.lfmda.2vaihingen_segformer \
  --ckpt-model-tea log/lfmda/SegFormer_MiT-B2/2vaihingen/src_warmup_st-pcd/Vaihingen_tea_curr.pth \
  --ckpt-model-stu log/lfmda/SegFormer_MiT-B2/2vaihingen/src_warmup_st-pcd/Vaihingen_stu_curr.pth \
  --ckpt-proto log/lfmda/SegFormer_MiT-B2/2vaihingen/prototypes/warmup_prototypes.pth \
  --sam-refine --percent 0.95

