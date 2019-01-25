#ifndef NN_GPU_SINGLEKERNEL_HPP
#define NN_GPU_SINGLEKERNEL_HPP

#include "ml.hpp"
#include "fstream"
#include "iostream"
#include "CL/cl.hpp"

namespace ml {
	class SingletonKernel {
	public:

		static SingletonKernel & getInstance () {
			static SingletonKernel instance;
			return instance;
		}

		const cl::Program::Sources & getExeLayerSource () {
			return srcExeLayer;
		}

		const cl::Program::Sources & getActLayerSource () {
			return srcActLayer;
		}

		const cl::Program::Sources & getUpdateSource () {
			return srcUpdate;
		}

	private:
		cl::Program::Sources srcExeLayer;
		cl::Program::Sources srcActLayer;
		cl::Program::Sources srcUpdate;

		SingletonKernel () {
			std::ifstream fin;
			std::string sourceCode;

			//upload executive layer kernel
			fin = std::ifstream("../kernels/executive_layer.cl");
			sourceCode = std::string(std::istreambuf_iterator <char>(fin), (std::istreambuf_iterator <char>()));
			srcExeLayer = cl::Program::Sources(1, std::make_pair(sourceCode.c_str(), sourceCode.size() + 1));
			fin.close();

			//upload activate layer kernel
			fin = std::ifstream("../kernels/activate_layer.cl");
			sourceCode = std::string(std::istreambuf_iterator <char>(fin), (std::istreambuf_iterator <char>()));
			srcActLayer = cl::Program::Sources(1, std::make_pair(sourceCode.c_str(), sourceCode.size() + 1));
			fin.close();

			//upload update kernel
			fin = std::ifstream("../kernels/update.cl");
			sourceCode = std::string(std::istreambuf_iterator <char>(fin), (std::istreambuf_iterator <char>()));
			srcUpdate = cl::Program::Sources(1, std::make_pair(sourceCode.c_str(), sourceCode.size() + 1));
			fin.close();
		}

		SingletonKernel (const SingletonKernel & root) = delete;

		SingletonKernel & operator= (const SingletonKernel &) = delete;
	};
}

#endif //NN_GPU_SINGLEKERNEL_HPP
