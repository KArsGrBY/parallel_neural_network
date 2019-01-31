__kernel void activate (
		__constant float *outputs,
		__constant float *results,
		__global float *error,
		const uint size,
		const uint samples) {
	float err = 0;

	const uint personId = get_global_id(0);
	const uint sampleId = get_global_id(1);

	const uint lastId = (personId * samples + sampleId + 1) * size;

	for (uint id = lastId - size; id < lastId; ++id) {
		err += (outputs[id] - results[id]) * (outputs[id] - results[id]);
	}

	error[personId * samples + sampleId] = err / size;
	return;
}