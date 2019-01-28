#include "CL/cl.hpp"
#include "learning.hpp"
#include "bits/stdc++.h"

using namespace std;

class Timer {
private:
	// Псевдонимы типов используются для удобного доступа к вложенным типам
	using clock_t = std::chrono::high_resolution_clock;
	using second_t = std::chrono::duration <double, std::ratio <1> >;

	std::chrono::time_point <clock_t> m_beg;

public:
	Timer () : m_beg(clock_t::now()) {
	}

	void reset () {
		m_beg = clock_t::now();
	}

	double elapsed () const {
		return std::chrono::duration_cast <second_t>(clock_t::now() - m_beg).count();
	}
};

const int SAMPLES = 100;
const int ITER = 100;
const int SIZE_IN = 900, SIZE_OUT = 30;


int main (int argc, char ** argv) {
	using vec = std::vector <float>;
	using sample = std::pair <vec, vec>;
	std::vector <sample> samples;
	for (int samp = 0; samp < SAMPLES; samp++) {
		samples.push_back(std::make_pair(vec(SIZE_IN, rand() * 0.001f), vec(SIZE_OUT, rand() * 0.001f)));
	}


	ml::Learning learning({SIZE_IN, 30, SIZE_OUT}, 32, samples);


	std::cerr << "START\n";
	Timer timer;
	for (int iter = 0; iter < ITER; iter++) {
		std::cerr << iter << std::endl;
		learning.iteration();
	}
	std::cout << std::fixed << std::setprecision(5) << timer.elapsed() / ITER << std::endl;

}