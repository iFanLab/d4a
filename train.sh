ds_names=(cora citeseer pubmed)
atk_rates=(0.05 0.15 0.25)
atk_names=(mettack minmax)

for ds_name in "${ds_names[@]}";
do
    python3 main.py --dataset=${ds_name} --atk_name=clean
    wait
    for atk_name in "${atk_names[@]}";
    do
        for atk_rate in "${atk_rates[@]}";
        do
            python3 main.py --dataset=${ds_name} --atk_name=${atk_name} --atk_rate=${atk_rate}
            wait
        done
    done
done
