#!/bin/bash

# plane-sphere

# T=300, PR

python ../../plane-sphere/compute.py 50.e-06 1.e-06 300 --energy >> reference_data/nystrom/data01.out
python ../../plane-sphere/compute.py 50.e-06 0.25e-06 300 --energy >> reference_data/nystrom/data02.out
python ../../plane-sphere/compute.py 150.e-06 3.e-06 300 --energy >> reference_data/nystrom/data03.out
python ../../plane-sphere/compute.py 150.e-06 0.75e-06 300 --energy >> reference_data/nystrom/data04.out

# T=300, Gold sphere

python ../../plane-sphere/compute.py 50.e-06 1.e-06 300 --sphere Gold --energy >> reference_data/nystrom/data05.out
python ../../plane-sphere/compute.py 50.e-06 0.25e-06 300 --sphere Gold --energy >> reference_data/nystrom/data06.out
python ../../plane-sphere/compute.py 150.e-06 3.e-06 300 --sphere Gold --energy >> reference_data/nystrom/data07.out
python ../../plane-sphere/compute.py 150.e-06 0.75e-06 300 --sphere Gold --energy >> reference_data/nystrom/data08.out

# T=0, PR

python ../../plane-sphere/compute.py 50.e-06 1.e-06 0. --fcqs --X 40 --energy >> reference_data/nystrom/data09.out
python ../../plane-sphere/compute.py 50.e-06 0.25e-06 0. --fcqs --X 40 --energy >> reference_data/nystrom/data10.out
python ../../plane-sphere/compute.py 150.e-06 3.e-06 0. --fcqs --X 40 --energy >> reference_data/nystrom/data11.out
python ../../plane-sphere/compute.py 150.e-06 0.75e-06 0. --fcqs --X 40 --energy  >> reference_data/nystrom/data12.out

# sphere-sphere

python ../../sphere-sphere/compute.py 50.e-06 50.e-06 1e-06 300 --energy >> reference_data/nystrom/data13.out
python ../../sphere-sphere/compute.py 100.e-06 50.e-06 1e-06 300 --energy  >> reference_data/nystrom/data14.out
python ../../sphere-sphere/compute.py 50.e-06 100.e-06 1e-06 300 --energy  >> reference_data/nystrom/data15.out

python ../../sphere-sphere/compute.py 12.5e-06 2.5e-06 1e-06 293 --sphere1 Silica1 --sphere2 Silica1 --medium Water --energy >> reference_data/nystrom/data16.out
python ../../sphere-sphere/compute.py 2.5e-06 12.5e-06 1e-06 293 --sphere1 Silica1 --sphere2 Silica1 --medium Water --energy >> reference_data/nystrom/data17.out
python ../../sphere-sphere/compute.py 12.5e-06 2.5e-06 1e-07 293 --sphere1 Silica1 --sphere2 Silica1 --medium Water --energy >> reference_data/nystrom/data18.out
python ../../sphere-sphere/compute.py 40.e-06 20.e-06 1e-06 300 --sphere1 Gold --sphere2 PS1 --medium Vacuum --energy >> reference_data/nystrom/data19.out
python ../../sphere-sphere/compute.py 40.e-06 50.e-06 1e-06 300 --sphere1 Gold --sphere2 Silica1 --medium Water --energy >> reference_data/nystrom/data20.out


