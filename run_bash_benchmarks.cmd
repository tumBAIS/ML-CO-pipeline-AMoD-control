#!/bin/bash
#SBATCH -J p30w2
#SBATCH -o ./%x.%j.%N.out
#SBATCH -D ./
#SBATCH --get-user-env
#SBATCH --mem=10000mb
#SBATCH --cpus-per-task=1
#SBATCH --export=NONE
#SBATCH --time=08:00:00


module load python/3.8.11-base
module load slurm_setup

source venv/bin/activate

if [[ ${RUNNING_TYPE} != "samplingHyperparameterSelection" ]]; then
        if [[ ${POLICY} == "sampling" ]]; then
		EXTENDED_HORIZON="5"
	fi
fi

if [[ ${RUNNING_TYPE} != "samplingReduceFutureSelect" ]]; then
        if [[ ${POLICY} == "sampling" ]]; then
		REDUCE_WEIGHT_OF_FUTURE_REQUEST_FACTOR=0.2
	else
		REDUCE_WEIGHT_OF_FUTURE_REQUEST_FACTOR=1
	fi
fi

printf -v HOUR_TWO_DIGIT "%02d" $HOUR
printf -v END_HOUR "%02d" $((HOUR+1))

if [[ ${RUNNING_TYPE} == "vehicleTest" || ${RUNNING_TYPE} == "sparsityTest" ]]; then
        INSTANCES_SETS="validation test"
else
	INSTANCES_SETS="validation"
fi



for instances_set in ${INSTANCES_SETS}
do
	if [[ ${instances_set} == "validation" ]]; then
	    	DAYS_MONTHS="14,1 15,1 16,1 19,1 20,1"
	elif [[ ${instances_set} == "test" ]]; then
	    	DAYS_MONTHS="21,1 22,1 23,1 26,1 27,1 28,1 29,1 30,1 2,2 3,2 4,2 5,2 6,2 9,2 10,2 11,2 12,2 13,2 16,2 17,2"
	fi
	
	for DAY_MONTH in ${DAYS_MONTHS}
	do
		IFS=',' read DAY MONTH <<< "${DAY_MONTH}"
		YEAR=2015
		printf -v DAY_TWO_DIGIT "%02d" $DAY
		printf -v MONTH_TWO_DIGIT "%02d" $MONTH
		printf -v DATA_TWO_DIGIT "%02d" $((DAY-1))
		python evaluation.py --num_vehicles=${VEHICLES} --objective=${OBJECTIVE} --extended_horizon=${EXTENDED_HORIZON} --start_date="${YEAR}-${MONTH_TWO_DIGIT}-${DAY_TWO_DIGIT} ${HOUR_TWO_DIGIT}:00:00" --end_date="${YEAR}-${MONTH_TWO_DIGIT}-${DAY_TWO_DIGIT} ${END_HOUR}:00:00" --result_directory="./results/results_${RUNNING_TYPE}_${instances_set}" --heuristic_distance_range=${DISTANCE_LOCATION_HYP} --heuristic_time_range=${HEURISTIC_TIME_RANGE_HYP} --sparsity_factor=${SPARSITY} --reduce_weight_of_future_requests_factor=${REDUCE_WEIGHT_OF_FUTURE_REQUEST_FACTOR} --policy=$POLICY --rebalancing_action_sampling=0
	done
done


