ulimit -u unlimited

rm -rf ./train_parallel
mkdir ./train_parallel
cp ../*.py ./train_parallel
cp *.sh ./train_parallel
cp -r ../src ./train_parallel
cd ./train_parallel || exit

/usr/local/openmpi-4.0.3/bin/mpirun --allow-run-as-root --mca btl_tcp_if_include ens11f0 -x LD_LIBRARY_PATH=/usr/local/python-3.7.5/lib -x NCCL_SOCKET_IFNAME=ens11f0 -n 8 -H gpu3-docker:2,gpu5-docker:2,gpu7-docker:2,gpu11-docker:2 --output-filename log_output --merge-stderr-to-stdout \
  python train.py --net="resnet50" --dataset="cifar10" --run_distribute=True \
    --device_target="GPU" --dataset_path="/data/cifar-10-batches-bin/" &> log &
