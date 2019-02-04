__kernel void calculate_error (
		__constant float *outputs,
		__constant float *results,
		__global float *error,
		const uint size,
		const uint samples) {
	float err = 0;

	const uint personId = get_global_id(0);
	const uint sampleId = get_global_id(1);

	const uint lastId = (personId * samples + sampleId + 1) * size;

	for (uint id = lastId - size, resultId = sampleId * size; id < lastId; ++id, ++resultId) {
		float diff = outputs[id] - results[resultId];
		err += diff * diff;
	}

	error[personId * samples + sampleId] = err / size;
	return;
}

__kernel void calculate_final_error (
		__constant float *errors,
		__global float *finalErrors,
		const uint samples) {

	const uint personId = get_global_id(0);
	const uint lastId = (personId + 1) * samples;

	float err = 0;

	for (uint id = lastId - samples; id < lastId; ++id) {
		err += errors[id];
	}

	finalErrors[personId] = err / samples;
	return;
}

__kernel void calculate_best_error (
		__constant float *errors,
		__global float *bestErrors,
		__constant float *weights,
		__global float *bestWeights,
		const uint size) {
	const uint personId = get_global_id(0);
	const uind lastId = (personId + 1) * size;


	if (errors[personId] >= bestWeights[personId]) {
		bestWeights[personId] = errors[personId];
		for (uint id = lastId - size; id < lastId; ++id) {
			bestWeights[id] = weights[id];
		}
	}
	return;
}

