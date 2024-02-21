#!/bin/bash
#SBATCH -J p30w2
#SBATCH -o ./%x.%j.%N.out
#SBATCH -D ./
#SBATCH --get-user-env
#SBATCH --mail-type=end
#SBATCH --mem=50000mb
#SBATCH --cpus-per-task=28
#SBATCH --export=NONE
#SBATCH --time=70:00:00


module load python/3.8.11-base
module load slurm_setup

source venv/bin/activate


echo "We start training"

printf -v HOUR_TWO_DIGIT "%02d" $HOUR
printf -v END_HOUR "%02d" $((HOUR+1))

#python training.py --num_vehicles=${VEHICLES} --objective=${OBJECTIVE} --stop_max_time=3000 --extended_horizon=${EXTENDED_HORIZON} --perturbed_optimizer ${NUM_PERTURBATIONS} 1 --heuristic_distance_range=${DISTANCE_LOCATION} --heuristic_time_range=${HEURISTIC_TIME_RANGE} --max_iter=70 --sparsity_factor=${SPARSITY} --standardization=${STANDARDIZATION} --policy=$POLICY --model=${MODEL} --learning_rate=${LEARNING_RATE} --normalization=${NORMALIZATION} --num_layers=${NUMLAYERS} --hidden_units=${LAYER_SIZE}

declare -A array_best_learning_iteration
readarray -t lines < "./data/read_in_best_learning_iteration.txt"
for line in "${lines[@]}"; do
   array_best_learning_iteration[${line%%=*}]=${line#*=}
   #echo ${line%%=*}
   #echo ${line#*=}
done

if [[ ${SANITY} == 1 ]]; then
	echo ${SANITY}
	python sanity_check.py --num_vehicles=${VEHICLES} --objective=${OBJECTIVE} --stop_max_time=3000 --extended_horizon=${EXTENDED_HORIZON} --perturbed_optimizer ${NUM_PERTURBATIONS} 1 --heuristic_distance_range=${DISTANCE_LOCATION} --heuristic_time_range=${HEURISTIC_TIME_RANGE} --max_iter=70 --sparsity_factor=${SPARSITY} --standardization=${STANDARDIZATION} --policy=$POLICY --model=${MODEL} --learning_rate=${LEARNING_RATE} --normalization=${NORMALIZATION} --num_layers=${NUMLAYERS} --hidden_units=${LAYER_SIZE} --read_in_iterations=-1 --fix_capacity=${MAXIMUM_CAPACITY_ARRAY_HYP} --rebalancing_action_sampling=1
elif [[ ${SANITY_PERFORMANCE} == 1 ]]; then
	echo ${SANITY_PERFORMANCE}
	python sanity_check_performance.py --num_vehicles=${VEHICLES} --objective=${OBJECTIVE} --stop_max_time=3000 --extended_horizon=${EXTENDED_HORIZON} --perturbed_optimizer ${NUM_PERTURBATIONS} 1 --heuristic_distance_range=${DISTANCE_LOCATION} --heuristic_time_range=${HEURISTIC_TIME_RANGE} --max_iter=70 --sparsity_factor=${SPARSITY} --standardization=${STANDARDIZATION} --policy=$POLICY --model=${MODEL} --learning_rate=${LEARNING_RATE} --normalization=${NORMALIZATION} --num_layers=${NUMLAYERS} --hidden_units=${LAYER_SIZE} --read_in_iterations=-1 --fix_capacity=${MAXIMUM_CAPACITY_ARRAY_HYP} --rebalancing_action_sampling=1

else
	echo "We read in best learning iteration"

	for DISTANCE_LOCATION in ${DISTANCE_LOCATION_ARRAY_HYP}
	do
		for HEURISTIC_TIME_RANGE in ${HEURISTIC_TIME_RANGE_ARRAY_HYP}
		do
			for maximum_capacity in ${MAXIMUM_CAPACITY_ARRAY_HYP}
			do
				INSTANCES_SETS="test" #"test validation"

				for instances_set in ${INSTANCES_SETS}
				do
					echo "We read in validation or test"
					if [[ ${instances_set} == "validation" ]]; then
					    	DAYS_MONTHS="14,1 15,1 16,1 19,1 20,1"
					    	ITERATIONS="1 2 3 4 5 10 15 20 25 30 35 40 45 50 55 60 65 69"
					elif [[ ${instances_set} == "test" ]]; then
						echo "We read in test"
					    	DAYS_MONTHS="21,1 22,1 23,1 26,1 27,1 28,1 29,1 30,1 2,2 3,2 4,2 5,2 6,2 9,2 10,2 11,2 12,2 13,2 16,2 17,2"
					    	echo ${DAYS_MONTHS}
					    	echo "${RUNNING_TYPE}-${OBJECTIVE}-${POLICY}-${VEHICLES}-${SPARSITY}-${MODEL}"
					    	ITERATIONS=${array_best_learning_iteration["${RUNNING_TYPE}-${OBJECTIVE}-${POLICY}-${VEHICLES}-${SPARSITY}-${MODEL}"]}
					    	echo ${ITERATIONS}
					fi
					echo $ITERATIONS
					for DAY_MONTH in ${DAYS_MONTHS}
					do
						IFS=',' read DAY MONTH <<< "${DAY_MONTH}"
						YEAR=2015
						printf -v DAY_TWO_DIGIT "%02d" $DAY
						printf -v MONTH_TWO_DIGIT "%02d" $MONTH
						printf -v DATA_TWO_DIGIT "%02d" $((DAY-1))
						for ITERATION in ${ITERATIONS}
						do
							echo $ITERATION
							#python evaluation.py --num_vehicles=${VEHICLES} --objective=${OBJECTIVE} --extended_horizon=${EXTENDED_HORIZON} --start_date="${YEAR}-${MONTH_TWO_DIGIT}-${DAY_TWO_DIGIT} ${HOUR_TWO_DIGIT}:00:00" --end_date="${YEAR}-${MONTH_TWO_DIGIT}-${DAY_TWO_DIGIT} ${END_HOUR}:00:00" --result_directory="./results_deeplearning_final/results_${RUNNING_TYPE}_${instances_set}" --read_in_iterations=$ITERATION --heuristic_distance_range=${DISTANCE_LOCATION} --heuristic_time_range=${HEURISTIC_TIME_RANGE} --sparsity_factor=${SPARSITY} --standardization=${STANDARDIZATION} --fix_capacity=${maximum_capacity} --policy=$POLICY --perturbed_optimizer ${NUM_PERTURBATIONS} 1 --rebalancing_action_sampling=1 --model=${MODEL} --normalization=${NORMALIZATION} --learning_rate=${LEARNING_RATE} --hidden_units=${LAYER_SIZE} --num_layers=${NUMLAYERS}
						done
					done
				done
			done
		done
	done
fi
