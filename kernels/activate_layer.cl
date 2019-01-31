float activation_function (float x) {
	return 1.f / (1 + exp(-x));
}

__kernel void activate () {
}