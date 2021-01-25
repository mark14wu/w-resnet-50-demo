ulimit -u unlimited

/usr/local/openmpi-4.0.3/bin/mpirun --allow-run-as-root --mca btl_tcp_if_include ens11f0 -x LD_LIBRARY_PATH=/usr/local/python-3.7.5/lib -n 8 -H gpu1-docker:1,gpu3-docker:1,gpu4-docker:1,gpu5-docker:1,gpu7-docker:1,gpu10-docker:1,gpu11-docker:2 --output-filename log_output --merge-stderr-to-stdout \
  python train.py --net="resnet50" --dataset="cifar10" --run_distribute=True \
    --device_target="GPU" --dataset_path="/data/cifar-10-batches-bin/" > log 2>&1 &
