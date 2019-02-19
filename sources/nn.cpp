#include "nn.hpp"
#include "random"

inline float randomFromRange (float x, float y) {
	static std::mt19937 gen(time(nullptr));
	static std::uniform_real_distribution dis(x, y);
	return dis(gen);
}

ml::Nn::Nn (const std::vector <size_t> & _architecture) {
	architecture = _architecture;

	using vecs = std::vector <std::vector <float>>;
	weights = vecs((int) architecture.size() - 1);
	bestWeights = vecs((int) architecture.size() - 1);
	motions = vecs((int) architecture.size() - 1);

	curError = bestError = architecture.back() * 100.f;

	for (size_t layer = 0; layer + 1 < architecture.size(); ++layer) {
		size_t input = architecture[layer], output = architecture[layer + 1];
		weights[layer] = std::vector <float>(input * output);
		bestWeights[layer].reserve(input * output);
		motions[layer].reserve(input * output);

//		 		RANGE INITS HERE. 6.f for sigmoid function
		float range = 6.f;
		for (size_t neuronInput = 0; neuronInput < input; ++neuronInput) {
			for (size_t neuronOutput = 0; neuronOutput < output; ++neuronOutput) {
				weights[layer][neuronInput * output + neuronOutput] = randomFromRange(-range / 2.f, range / 2.f);
				motions[layer][neuronInput * output + neuronOutput] = randomFromRange(-range / 8.f, range / 8.f);
//				motions[layer][neuronInput * output + neuronOutput] = 0;
			}
		}
		std::copy(std::begin(weights[layer]), std::end(weights[layer]), std::back_inserter(bestWeights[layer]));
	}
}