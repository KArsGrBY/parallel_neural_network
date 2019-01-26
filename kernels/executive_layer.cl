__kernel void execute (__global float *inputs, __global float *outputs, __global float *weights, int sizeIn, int sizeOut, int population, int inputBlock, int outputBlock) {
	int cnt = sizeIn * sizeOut;
	int personId = get_global_id(0);
	int sampleId = get_global_id(1);
	int blockId = get_global_id(2);
	return;
}