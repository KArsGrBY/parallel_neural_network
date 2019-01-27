//#pragma OPENCL EXTENSION cl_amd_printf : enable

__kernel void execute (__global float *inputs, __global float *outputs, __global const float *weights, const uint sizeIn, const uint sizeOut, const uint samples, const uint inputBlock, const uint outputBlock) {
	uint numOfWeights = sizeIn * sizeOut;
	const uint personId = get_global_id(0);
	const uint sampleId = get_global_id(1);
	const uint blockId = get_global_id(2);
	const uint inId = blockId / (sizeOut / outputBlock) * inputBlock;
	const uint outId = blockId % (sizeOut / outputBlock) * outputBlock;
	const uint inShift = (personId * samples + sampleId) * sizeIn + inId;
	const uint outShift = (personId * samples + sampleId) * sizeOut + outId;

	//NEED to write uploading inputs and outputs blocks for local memory !!!
	for (uint weightIndex = numOfWeights * personId + inId * sizeOut + outId, localInId = 0; localInId < inputBlock; ++localInId, weightIndex += sizeOut) {
		for (uint localOutId = 0, wId = weightIndex; localOutId < outputBlock; ++localOutId, ++wId) {
			outputs[outShift + localOutId] += weights[weightIndex] * inputs[inShift + localInId];
		}
	}

	return;
}