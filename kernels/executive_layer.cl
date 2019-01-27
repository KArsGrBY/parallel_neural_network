__kernel void execute (__global const float *inputs, volatile __global float *outputs, __global const float *weights, const uint sizeIn, const uint sizeOut, const uint samples, const uint inputBlock, const uint outputBlock) {
	uint countOfWeights = sizeIn * sizeOut;
	const uint personId = get_global_id(0);
	const uint sampleId = get_global_id(1);
	const uint blockId = get_global_id(2);
	const uint inId = 0;
	const uint outId = 0;
	const uint inShift = (personId * samples + sampleId) * sizeIn + inId;
	const uint outShift = (personId * samples + sampleId) * sizeOut + outId;


	//NEED to write uploading inputs and outputs blocks for local memory !!!
	for (uint outIdLocal = 0, weightIndex = countOfWeights * personId + sizeOut * inId + outId; outIdLocal < inputBlock; weightIndex += sizeOut, ++outIdLocal) {
		for (uint inIdLocal = 0, weightId = weightIndex; inIdLocal < outputBlock; ++weightId, ++inIdLocal) {
			outputs[outShift + outIdLocal] += inputs[inShift + inIdLocal] * weights[weightId];
		}
	}

	return;
}