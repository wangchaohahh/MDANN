#!/bin/bash
set -x
params_alpha=(0\.01  0\.05  0\.1 0\.5 )
params_lamda=(0\.01  0\.05  0\.1 0\.5 )
for j in  ${params_lamda[@]}
do

for i in ${params_alpha[@]}
do
#    echo "alpha is $i"

        sed -i "28s/alpha\=*[0-9]*\.*[0-9]*/alpha\=$i/" ./DANN_Alexnet.py
        sed -i "29s/lamda\=*[0-9]*\.*[0-9]*/lamda\=$j/" ./DANN_Alexnet.py
        CUDA_VISIBLE_DEVICES=0 python ./DANN_Alexnet.py

done
done

        
    

