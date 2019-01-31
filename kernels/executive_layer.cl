//#pragma OPENCL EXTENSION cl_amd_printf : enable

__kernel void execute (
		__constant float *inputs,
		__global float *outputs,
		__constant const float *weights,
		const uint sizeIn,
		const uint sizeOut,
		const uint samples) {

	const uint personId = get_global_id(0);
	const uint sampleId = get_global_id(1);
	const uint outputId = get_global_id(2);

	const uint lastInputId = (personId * samples + sampleId + 1) * sizeIn;

	float outputNeuron = 0;

	for (uint inputId = lastInputId - sizeIn, weightId = sizeIn * sizeOut * personId + outputId;
			inputId < lastInputId;
			++inputId, weightId += sizeOut) {
		outputNeuron += weights[weightId] * inputs[inputId];
	}

	outputs[(personId * samples + sampleId) * sizeOut + outputId] = outputNeuron;
	return;
}