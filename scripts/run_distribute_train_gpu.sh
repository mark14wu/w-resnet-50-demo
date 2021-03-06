ulimit -u unlimited

# /usr/local/openmpi-4.0.3/bin/mpirun --allow-run-as-root --mca btl_tcp_if_include ens11f0 -x LD_LIBRARY_PATH=/usr/local/python-3.7.5/lib -n 8 -H gpu7-docker:3,gpu10-docker:2,gpu11-docker:3 --output-filename log_output --merge-stderr-to-stdout \
/usr/local/openmpi-4.0.3/bin/mpirun --allow-run-as-root --mca btl_tcp_if_include ens11f0 -x LD_LIBRARY_PATH=/usr/local/python-3.7.5/lib -n 2 -H gpu3-docker:2 --output-filename log_output --merge-stderr-to-stdout \
  python train.py --net="resnet50" --dataset="cifar10" --run_distribute=True \
    --device_target="GPU" --dataset_path="/data/cifar-10-batches-bin/" > log 2>&1 &
