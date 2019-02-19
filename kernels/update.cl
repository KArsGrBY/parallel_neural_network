float rand (const ulong seed) {
	return (float) (((seed * 0x5DEECE66DL + 0xBL) & ((1L << 48) - 1)) >> 16) / 0x10000000f;
}

ulong next (const ulong seed) {
	return ((seed * 0x5DEECE66DL + 0xBL) & ((1L << 48) - 1)) >> 16;
}

#define K1 		0.1f
#define K2 		0.2f
#define SLOW	0.95f

__kernel void update (
		__constant float * bestWeights,
		__global float * motions,
		__constant float * bestPerson,
		__global float * weights,
		const uint sizeIn,
		const uint sizeOut,
		ulong seed) {
	const uint personId = get_global_id(0);
	const uint inputsId = get_global_id(1);
	const uint size = sizeIn * sizeOut;
	const lastId = personId * size + (inputsId + 1) * sizeOut;

	for (uint id = lastId - sizeOut, bestId = inputsId * sizeOut; id < lastId; ++id, ++bestId, seed = next(seed)) {
		motions[id] = SLOW * motions[id] + K1 * rand(id) * (bestWeights[id] - weights[id]) + K2 * rand(id ^ 1) * (bestPerson[bestId] - weights[id]);
//		motions[id] = K1 * rand(id + seed) * (bestWeights[id] - weights[id]) + K2 * rand(seed + id ^ 1) * (bestPerson[bestId] - weights[id]);
		weights[id] += motions[id] + weights[id] * (2 * rand(seed - id) - 1) * 0.01f;
	}
	return;
}