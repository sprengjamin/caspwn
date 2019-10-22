#!/bin/bash

mkdir test_data

# plane-sphere

# T=300, PR

python ../plane-sphere/compute.py 50.e-06 1.e-06 300 >> test_data/data01.out
python ../plane-sphere/compute.py 50.e-06 0.25e-06 300 >> test_data/data02.out
python ../plane-sphere/compute.py 150.e-06 3.e-06 300 >> test_data/data03.out
python ../plane-sphere/compute.py 150.e-06 0.75e-06 300 >> test_data/data04.out

# T=300, Gold sphere

python ../plane-sphere/compute.py 50.e-06 1.e-06 300 --sphere Gold >> test_data/data05.out
python ../plane-sphere/compute.py 50.e-06 0.25e-06 300 --sphere Gold >> test_data/data06.out
python ../plane-sphere/compute.py 150.e-06 3.e-06 300 --sphere Gold >> test_data/data07.out
python ../plane-sphere/compute.py 150.e-06 0.75e-06 300 --sphere Gold >> test_data/data08.out

# T=0, PR

python ../plane-sphere/compute.py 50.e-06 1.e-06 0. --fcqs --X 40 >> test_data/data09.out
python ../plane-sphere/compute.py 50.e-06 0.25e-06 0. --fcqs --X 40 >> test_data/data10.out
python ../plane-sphere/compute.py 150.e-06 3.e-06 0. --fcqs --X 40 >> test_data/data11.out
python ../plane-sphere/compute.py 150.e-06 0.75e-06 0. --fcqs --X 40  >> test_data/data12.out

# sphere-sphere

python ../sphere-sphere/compute.py 50.e-06 50.e-06 1e-06 300 >> test_data/data13.out
python ../sphere-sphere/compute.py 100.e-06 50.e-06 1e-06 300  >> test_data/data14.out
python ../sphere-sphere/compute.py 50.e-06 100.e-06 1e-06 300  >> test_data/data15.out

python ../sphere-sphere/compute.py 12.5e-06 2.5e-06 1e-06 293 --sphere1 Silica1 --sphere2 Silica1 --medium Water >> test_data/data16.out
python ../sphere-sphere/compute.py 2.5e-06 12.5e-06 1e-06 293 --sphere1 Silica1 --sphere2 Silica1 --medium Water >> test_data/data17.out
python ../sphere-sphere/compute.py 12.5e-06 2.5e-06 1e-07 293 --sphere1 Silica1 --sphere2 Silica1 --medium Water >> test_data/data18.out
python ../sphere-sphere/compute.py 40.e-06 20.e-06 1e-06 300 --sphere1 Gold --sphere2 PS1 --medium Vacuum >> test_data/data19.out
python ../sphere-sphere/compute.py 40.e-06 50.e-06 1e-06 300 --sphere1 Gold --sphere2 Silica1 --medium Water >> test_data/data20.out


