#!/bin/bash

mkdir data
# PLANE AND SPHERE
echo -ne computing test data 1/8\\r
python ../../plane-sphere/compute_plsp.py 50e-6 5e-6 300 >> data/out1.dat
python ../../plane-sphere/compute_plsp.py 50e-6 5.005e-6 300 >> data/out1p.dat
python ../../plane-sphere/compute_plsp.py 50e-6 4.995e-6 300 >> data/out1m.dat
echo -ne computing test data 2/8\\r
python ../../plane-sphere/compute_plsp.py 50e-6 5e-6 300 --plane Silica1 --medium Water --sphere PS1 >> data/out2.dat
python ../../plane-sphere/compute_plsp.py 50e-6 5.005e-6 300 --plane Silica1 --medium Water --sphere PS1 >> data/out2p.dat
python ../../plane-sphere/compute_plsp.py 50e-6 4.995e-6 300 --plane Silica1 --medium Water --sphere PS1 >> data/out2m.dat
echo -ne computing test data 3/8\\r
python ../../plane-sphere/compute_plsp.py 150e-6 1e-6 300 >> data/out3.dat
python ../../plane-sphere/compute_plsp.py 150e-6 1.001e-6 300 >> data/out3p.dat
python ../../plane-sphere/compute_plsp.py 150e-6 0.999e-6 300 >> data/out3m.dat
echo -ne computing test data 4/8\\r
python ../../plane-sphere/compute_plsp.py 150e-6 1e-6 300 --plane Silica1 --medium Water --sphere PS1 >> data/out4.dat
python ../../plane-sphere/compute_plsp.py 150e-6 1.001e-6 300 --plane Silica1 --medium Water --sphere PS1 >> data/out4p.dat
python ../../plane-sphere/compute_plsp.py 150e-6 0.999e-6 300 --plane Silica1 --medium Water --sphere PS1 >> data/out4m.dat

# TWO SPHERES
echo -ne Computing test data 5/8\\r
python ../../sphere-sphere/compute_spsp.py 50e-6 50e-6 5e-6 300 >> data/out5.dat
python ../../sphere-sphere/compute_spsp.py 50e-6 50e-6 5.005e-6 300 >> data/out5p.dat
python ../../sphere-sphere/compute_spsp.py 50e-6 50e-6 4.995e-6 300 >> data/out5m.dat
echo -ne computing test data 6/8\\r
python ../../sphere-sphere/compute_spsp.py 50e-6 50e-6 5e-6 300 --sphere1 PS1 --medium Water --sphere2 PS1 >> data/out6.dat
python ../../sphere-sphere/compute_spsp.py 50e-6 50e-6 5.005e-6 300 --sphere1 PS1 --medium Water --sphere2 PS1 >> data/out6p.dat
python ../../sphere-sphere/compute_spsp.py 50e-6 50e-6 4.995e-6 300 --sphere1 PS1 --medium Water --sphere2 PS1 >> data/out6m.dat
echo -ne computing test data 7/8\\r
python ../../sphere-sphere/compute_spsp.py 150e-6 50e-6 1e-6 300 >> data/out7.dat
python ../../sphere-sphere/compute_spsp.py 150e-6 50e-6 1.001e-6 300 >> data/out7p.dat
python ../../sphere-sphere/compute_spsp.py 150e-6 50e-6 0.999e-6 300 >> data/out7m.dat
echo -ne computing test data 8/8\\r
python ../../sphere-sphere/compute_spsp.py 150e-6 50e-6 1e-6 300 --sphere1 PS1 --medium Water --sphere2 PS1 >> data/out8.dat
python ../../sphere-sphere/compute_spsp.py 150e-6 50e-6 1.001e-6 300 --sphere1 PS1 --medium Water --sphere2 PS1 >> data/out8p.dat
python ../../sphere-sphere/compute_spsp.py 150e-6 50e-6 0.999e-6 300 --sphere1 PS1 --medium Water --sphere2 PS1 >> data/out8m.dat
echo
echo done
