#pragma comment( lib, "opencl.lib" )
#define __CL_ENABLE_EXCEPTIONS
#include "CL/cl.hpp"
#include "learning.hpp"
#include "bits/stdc++.h"
using namespace std;

void test () {

	std::vector <cl::Device> devices;

	std::vector <cl::Platform> platforms;
	cl::Platform::get(&platforms);
	for (auto platform : platforms) {

		std::vector <cl::Device> devs;
		platform.getDevices(CL_DEVICE_TYPE_ALL, &devs);
		std::copy(devs.begin(), devs.end(), std::back_inserter(devices));
	}

	cl::Device device = devices.front();
	cl::Context context;
	cl::Buffer bufArr1, bufArr2, bufArr3;
	cl::CommandQueue commandQueue;
	cl::Program::Sources source;
	cl::Program program;
	cl::Kernel kernel;

	try {
		devices = {device};
		context = cl::Context(devices);

		cl::Buffer bufArr1, bufArr2;

		{
			std::vector <float> buf = {1, 2, 3, 4, 5, 6, 7};

			bufArr1 = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, buf.size() * sizeof(float),
									  buf.data());
			buf = {1, 2, 3, 4};
			bufArr2 = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, buf.size() * sizeof(float),
									  buf.data());
		}


		commandQueue = cl::CommandQueue(context, device);

		std::ifstream fin("../kernels/executive_layer.cl");
		std::string sourceCode(std::istreambuf_iterator <char>(fin), (std::istreambuf_iterator <char>()));

		source = cl::Program::Sources(1, std::make_pair(sourceCode.c_str(), sourceCode.size() + 1));
		program = cl::Program(context, source);
		program.build(devices);

		kernel = cl::Kernel(program, "test");

		kernel.setArg(0, bufArr1);
		kernel.setArg(1, bufArr2);

		commandQueue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(1024),
										  device.getInfo <CL_DEVICE_MAX_WORK_GROUP_SIZE>());
		commandQueue.finish();

		float* arr = new float[100];
		arr[0] = -1;
		commandQueue.enqueueReadBuffer(bufArr1, CL_TRUE, 0, 8 * sizeof(float), arr);
		commandQueue.finish();
		std::cerr << arr[0] << std::endl;

	}
	catch (cl::Error err) {
		std::string buildlog = program.getBuildInfo <CL_PROGRAM_BUILD_LOG>(device);
		std::cerr << buildlog << std::endl;
		std::cerr << err.what() << std::endl;
	}
}

int main (int argc, char ** argv) {

//	test();
//	return 0;

	ml::Learning learning({2, 2, 1}, 128, {
		{{0, 0}, {0}},
		{{1, 1}, {0}},
		{{1, 0}, {1}},
		{{0, 1}, {1}}});
	learning.iteration();
}