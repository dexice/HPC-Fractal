#define __CL_ENABLE_EXCEPTIONS

#ifdef __APPLE__
#include <OpenCL/cl.hpp>
#else
#include <CL/cl.hpp>
#endif

#include <cstdlib>
#include <iostream>
#include <fstream>
#include <string>

#include <complex>
#include <cmath>

#include "Display.h"

typedef std::complex<float> cplx;

// show available platforms and devices
void showPlatforms(){
	std::vector<cl::Platform> platforms;
	cl::Platform::get(&platforms);
	for(int i=0 ; i<platforms.size(); i++){
		std::cout<<"Platform: "<<platforms[i].getInfo<CL_PLATFORM_NAME>()<<std::endl;
		std::vector<cl::Device> devices;
		platforms[i].getDevices(CL_DEVICE_TYPE_GPU | CL_DEVICE_TYPE_CPU, &devices);
		for(int j=0; j<devices.size(); j++){
			if(devices[j].getInfo<CL_DEVICE_TYPE>() == CL_DEVICE_TYPE_CPU) std::cout << "Device: " << " CPU " << " : "<< devices[j].getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>() << " compute units "<<"( "<<devices[j].getInfo<CL_DEVICE_NAME>()<<" )"<<std::endl;
			if(devices[j].getInfo<CL_DEVICE_TYPE>() == CL_DEVICE_TYPE_GPU) std::cout << "Device: " << " GPU " << " : "<< devices[j].getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>() << " compute units "<<"( "<<devices[j].getInfo<CL_DEVICE_NAME>()<<" )"<<std::endl;
		}
	}
}

// try to construct a context with the first platform of the requested type
cl::Context getContext(cl_device_type requestedDeviceType, std::vector<cl::Platform>& platforms){
	// try to create a context of the requested platform type
	for(unsigned int i=0;i<platforms.size();i++){
		try{
			cl_context_properties cps[] = {CL_CONTEXT_PLATFORM, (cl_context_properties)(platforms[i])(), 0};
			return cl::Context( requestedDeviceType, cps);
		}
		catch(...){}
	}
	throw CL_DEVICE_NOT_AVAILABLE;
}


int main(int argc, char** argv){

	// fractal parameters
	cl_uint maxIter = 1000;
	cl_uint2 size;
	size.x = 2048;
	size.y = 2048;
	cplx lowerLeft = cplx(-1.5,-1.5);
	cplx upperRight = cplx(1.5,1.5);
	cplx c = cplx(-0.726895347709114071439, 0.188887129043845954792);

	showPlatforms();

	std::vector<cl::Platform> platforms;
	cl::Platform::get(&platforms);

	// get a context on a platform containing at least one GPU
	cl::Context context = getContext(CL_DEVICE_TYPE_GPU, platforms);

	// get a list of devices on this platform
	std::vector<cl::Device> devices = context.getInfo<CL_CONTEXT_DEVICES>();

	// create a command queue and use the first device of the platform
	cl::CommandQueue queue = cl::CommandQueue(context, devices[0]);
	std::string deviceName;
	devices[0].getInfo(CL_DEVICE_NAME, &deviceName);
	std::cout<<"Command queue created successfuly, kernels will be executed on : "<<deviceName<<std::endl;

	// read source file
	std::ifstream sourceFile("fractale.cl");
	std::string sourceCode(std::istreambuf_iterator<char>(sourceFile),(std::istreambuf_iterator<char>()));
	cl::Program::Sources source(1, std::make_pair(sourceCode.c_str(), sourceCode.length()+1));

	// make program of the source code in the context
	cl::Program program = cl::Program(context, source);

	// code surounded by try catch in order to get OpenCL errors (eg. compilation error)
	try{

		// build program for these specific device
		program.build(devices);

		// construct kernel
		cl::Kernel kernel(program, "julia");

		// host side memory allocation
		cl_uint* A = new cl_uint[size.x * size.y];

		// device side memory allocation
		cl::Buffer d_A = cl::Buffer(context, CL_MEM_WRITE_ONLY, size.x * size.y * sizeof(cl_uint));

		// prepare parameters for kernel
		cl_float2 ll;
		cl_float2 ur;
		cl_float2 cl_c;

		ll.x = real(lowerLeft); ll.y = imag(lowerLeft);
		ur.x = real(upperRight); ur.y = imag(upperRight);
		cl_c.x = real(c); cl_c.y = imag(c);

		// set arguments to kernel
		kernel.setArg(0, d_A);
		kernel.setArg(1, size);
		kernel.setArg(2, ll);
		kernel.setArg(3, ur);
		kernel.setArg(4, cl_c);
		kernel.setArg(5, maxIter);

		// execute the kernel
		cl::NDRange local(8,8);
		cl::NDRange global(std::ceil(float(size.x)/float(local[0]))*local[0], std::ceil(float(size.y)/float(local[1]))*local[1]);
		std::cout << "Kernel execution" << std::endl;
		queue.enqueueNDRangeKernel(kernel, cl::NullRange, global, local);

		// getting result
		queue.enqueueReadBuffer(d_A, CL_TRUE, 0, size.x * size.y * sizeof(cl_uint), A);

		// display result
		Display d(A, size.x, size.y, maxIter);
		d.show();
		d.waitForKey();

	}
	catch(cl::Error error){
		std::cerr << error.what() << "(" << error.err() << ")" << std::endl;
		std::string buildLog;
		program.getBuildInfo(devices[0], CL_PROGRAM_BUILD_LOG, &buildLog);
		std::cerr<<buildLog<<std::endl;
	}

}
