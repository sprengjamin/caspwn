#!/bin/bash

mkdir test_data

# plane-sphere

# T=300, PR
echo -ne Computing test data 01/24\\r
python ../../plane-sphere/compute_plsp.py 50.e-06 1.e-06 300 --energy >> test_data/data01.out
echo -ne Computing test data 02/24\\r
python ../../plane-sphere/compute_plsp.py 50.e-06 0.25e-06 300 --energy >> test_data/data02.out
echo -ne Computing test data 03/24\\r
python ../../plane-sphere/compute_plsp.py 150.e-06 3.e-06 300 --energy >> test_data/data03.out
echo -ne Computing test data 04/24\\r
python ../../plane-sphere/compute_plsp.py 150.e-06 0.75e-06 300 --energy >> test_data/data04.out

# T=300, Gold sphere

echo -ne Computing test data 05/24\\r
python ../../plane-sphere/compute_plsp.py 50.e-06 1.e-06 300 --sphere gold_drude --energy >> test_data/data05.out
echo -ne Computing test data 06/24\\r
python ../../plane-sphere/compute_plsp.py 50.e-06 0.25e-06 300 --sphere gold_drude --energy >> test_data/data06.out
echo -ne Computing test data 07/24\\r
python ../../plane-sphere/compute_plsp.py 150.e-06 3.e-06 300 --sphere gold_drude --energy >> test_data/data07.out
echo -ne Computing test data 08/24\\r
python ../../plane-sphere/compute_plsp.py 150.e-06 0.75e-06 300 --sphere gold_drude --energy >> test_data/data08.out

# T=0, PR

echo -ne Computing test data 09/24\\r
python ../../plane-sphere/compute_plsp.py 50.e-06 1.e-06 0. --fcq --X 40 --energy >> test_data/data09.out
echo -ne Computing test data 10/24\\r
python ../../plane-sphere/compute_plsp.py 50.e-06 0.25e-06 0. --fcq --X 40 --energy >> test_data/data10.out
echo -ne Computing test data 11/24\\r
python ../../plane-sphere/compute_plsp.py 150.e-06 3.e-06 0. --fcq --X 40 --energy >> test_data/data11.out
echo -ne Computing test data 12/24\\r
python ../../plane-sphere/compute_plsp.py 150.e-06 0.75e-06 0. --fcq --X 40  --energy >> test_data/data12.out

# sphere-sphere

echo -ne Computing test data 13/24\\r
python ../../sphere-sphere/compute_spsp.py 50.e-06 50.e-06 1e-06 300 --energy >> test_data/data13.out
echo -ne Computing test data 14/24\\r
python ../../sphere-sphere/compute_spsp.py 100.e-06 50.e-06 1e-06 300  --energy >> test_data/data14.out
echo -ne Computing test data 15/24\\r
python ../../sphere-sphere/compute_spsp.py 50.e-06 100.e-06 1e-06 300  --energy >> test_data/data15.out

echo -ne Computing test data 16/24\\r
python ../../sphere-sphere/compute_spsp.py 12.5e-06 2.5e-06 1e-06 293 --sphere1 silica --sphere2 silica --medium water_zwol --energy >> test_data/data16.out
echo -ne Computing test data 17/24\\r
python ../../sphere-sphere/compute_spsp.py 2.5e-06 12.5e-06 1e-06 293 --sphere1 silica --sphere2 silica --medium water_zwol --energy >> test_data/data17.out
echo -ne Computing test data 18/24\\r
python ../../sphere-sphere/compute_spsp.py 12.5e-06 2.5e-06 1e-07 293 --sphere1 silica --sphere2 silica --medium water_zwol --energy >> test_data/data18.out
echo -ne Computing test data 19/24\\r
python ../../sphere-sphere/compute_spsp.py 40.e-06 20.e-06 1e-06 300 --sphere1 gold_drude --sphere2 polystyrene --medium vacuum --energy >> test_data/data19.out
echo -ne Computing test data 20/24\\r
python ../../sphere-sphere/compute_spsp.py 40.e-06 50.e-06 1e-06 300 --sphere1 gold_drude --sphere2 silica --medium water_zwol --energy >> test_data/data20.out

# compare to caps
# plane-sphere (gold-gold), T=300 in vacuum
echo -ne Computing test data 21/24\\r
python ../../plane-sphere/compute_plsp.py 50.e-06 5.e-06 300 --sphere gold_drude --plane gold_drude --energy >> test_data/data21.out
echo -ne Computing test data 22/24\\r
python ../../plane-sphere/compute_plsp.py 100.e-06 2.e-06 300 --sphere gold_drude --plane gold_drude --energy >> test_data/data22.out
echo -ne Computing test data 23/24\\r
python ../../plane-sphere/compute_plsp.py 150.e-06 6.e-06 300 --sphere gold_drude --plane gold_drude --energy >> test_data/data23.out
echo -ne Computing test data 24/24\\r
python ../../plane-sphere/compute_plsp.py 100.e-06 1.e-06 300 --sphere gold_drude --plane gold_drude --energy >> test_data/data24.out
echo
echo done
