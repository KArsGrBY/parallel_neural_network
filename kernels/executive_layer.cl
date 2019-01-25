__kernel void execute (__global float *inputs, __global float *outputs, __global float *weights, uint sizeIn, uint sizeOut, uint population, uint inputBlock, uint outputBlock) {
	uint weights = sizeIn * sizeOut;
	uint personId = get_global_id(0);
	uint sampleId = get_global_id(1);
	uint blockId = get_global_id(2);

	printf("%d %d %d\n", personId, sampleId, blockId);
}
