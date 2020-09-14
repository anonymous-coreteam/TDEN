CUDA_VISIBLE_DEVICES=3,2,1,0 NCCL_SOCKET_IFNAME=eth0 python3 -m torch.distributed.launch \
    --nproc_per_node=4 \
    --nnodes=4 \
    --node_rank=0 \
    --master_addr="10.207.174.51" \
    --master_port=12346 \
    main_pretrain.py --folder ./experiments/pretrain
