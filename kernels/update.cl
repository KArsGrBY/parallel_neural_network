float rand (const ulong seed) {
	return ((float) (((seed + 0x5DEECE66DL + 0xBL) & ((1L << 48) - 1)) >> 16) / 4294967296.f);
}

#define K1 0.1f
#define K2 0.1f
#define K3 0.000f

__kernel void update (
		__constant float * bestWeights,
		__global float * motions,
		__constant float * bestPerson,
		__global float * weights,
		const uint sizeIn,
		const uint sizeOut,
		const ulong seed) {
	const uint personId = get_global_id(0);
	const uint inputsId = get_global_id(1);
	const uint size = sizeIn * sizeOut;
	const lastId = personId * size + (inputsId + 1) * sizeOut;

	for (uint id = lastId - sizeOut, bestId = inputsId * sizeOut; id < lastId; ++id, ++bestId) {
		motions[id] += K1 * rand(id) * (bestWeights[id] - weights[id]) + K2 * rand(id ^ 1) * (bestPerson[bestId] - weights[id]) + K3 * (2 * rand(id ^ 2) - 1.f);
		weights[id] += motions[id];
	}
	return;
}