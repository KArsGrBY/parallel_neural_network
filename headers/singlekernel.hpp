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

		const std::string & getExeLayerCode () {
			return codeExeLayer;
		}

		const std::string & getErrorCode () {
			return codeError;
		}

		const std::string & getUpdateCode () {
			return codeUpdate;
		}

	private:
		std::string codeExeLayer;
		std::string codeError;
		std::string codeUpdate;

		SingletonKernel () {
			std::ifstream fin;
			std::string sourceCode;

			//upload executive layer kernel
			fin = std::ifstream("../kernels/executive_layer.cl");
			codeExeLayer = std::string(std::istreambuf_iterator <char>(fin), (std::istreambuf_iterator <char>()));
			fin.close();

			//upload activate layer kernel
			fin = std::ifstream("../kernels/arror.cl");
			codeError = std::string(std::istreambuf_iterator <char>(fin), (std::istreambuf_iterator <char>()));
			fin.close();

			//upload update kernel
			fin = std::ifstream("../kernels/update.cl");
			codeUpdate = std::string(std::istreambuf_iterator <char>(fin), (std::istreambuf_iterator <char>()));
			fin.close();
		}

		SingletonKernel (const SingletonKernel & root) = delete;

		SingletonKernel & operator= (const SingletonKernel &) = delete;
	};
}

#endif //NN_GPU_SINGLEKERNEL_HPP
