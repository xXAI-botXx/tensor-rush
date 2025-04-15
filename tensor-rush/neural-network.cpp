#include "pch.h"

#include "neural-network.h"

#include <memory>
#include <vector>
#include <string>

#include "math.h"






// ------> nn <------


// -------------
// >>> Layer <<<
// -------------
//rush::nn::Layer::Layer() {
//	std::vector<std::shared_ptr<Layer>> layers = {};
//}
//
//rush::nn::Layer::~Layer() {
//
//}

// adding new (sub)layers
void rush::nn::Layer::add(std::shared_ptr<rush::nn::Layer> layer) {
	layers.push_back(layer);
}

void rush::nn::Layer::add(std::vector<std::shared_ptr<rush::nn::Layer>> layers) {
	for (auto& cur_layer : layers) {
		this->layers.push_back(cur_layer);
	}
}

// others
std::string rush::nn::Layer::get_as_str() {
	return "";
}

void rush::nn::Layer::plot() {

}

void rush::nn::Layer::get_parameters() {

}






// ------------------
// >>> Sequential <<<
// ------------------
//rush::nn::Sequential::Sequential() {
//	std::vector<std::shared_ptr<Layer>> layers = {};
//}
//
//rush::nn::Sequential::~Sequential() {
//
//}

rush::math::Tensor rush::nn::Sequential::forward(const rush::math::Tensor& input) {
	rush::math::Tensor out = input;
	for (auto& cur_layer : this->layers) {
		out = cur_layer->forward(out);
	}
	return out;
}


rush::math::Tensor rush::nn::Sequential::backward(const rush::math::Tensor& grad_output) {
	rush::math::Tensor grad = grad_output;
	for (size_t i = 0; i < layers.size(); ++i) {
		grad = layers.at(i)->backward(grad);
	}
	return grad;
}






// ---------> activation <---------


// ------------
// >>> ReLu <<<
// ------------
//rush::nn::activation::ReLU::ReLU() {
//	std::vector<std::shared_ptr<Layer>> layers = {};
//}
//
//rush::nn::activation::ReLU::~ReLU() {
//
//}

rush::math::Tensor rush::nn::activation::ReLU::forward(const rush::math::Tensor& input) {
	return input.apply([](float x) { return (0.0f > x ? 0.0f : x); });
}

rush::math::Tensor rush::nn::activation::ReLU::backward(const rush::math::Tensor& grad_output) {
	// FIXME
	return grad_output;
}

// ---------------
// >>> Sigmoid <<<
// ---------------
//rush::nn::activation::Sigmoid::Sigmoid() {
//	std::vector<std::shared_ptr<Layer>> layers = {};
//}
//
//rush::nn::activation::Sigmoid::~Sigmoid() {
//
//}

rush::math::Tensor rush::nn::activation::Sigmoid::forward(const rush::math::Tensor& input) {
	return input.apply([](float x) {
						return 1.0f / (1.0f + rush::math::exp(-x));
								   }
					  );
}

rush::math::Tensor rush::nn::activation::Sigmoid::backward(const rush::math::Tensor& grad_output) {
	// FIXME
	return grad_output;
}

// ------------
// >>> Tanh <<<
// ------------
//rush::nn::activation::Tanh::Tanh() {
//	std::vector<std::shared_ptr<Layer>> layers = {};
//}
//
//rush::nn::activation::Tanh::~Tanh() {
//
//}

rush::math::Tensor rush::nn::activation::Tanh::forward(const rush::math::Tensor& input) {
	return input.apply([](float x) {
						return rush::math::tanh(x);
								   }
					  );
}

rush::math::Tensor rush::nn::activation::Tanh::backward(const rush::math::Tensor& grad_output) {
	// FIXME
	return grad_output;
}

// ------------
// >>> GELU <<<
// ------------
//rush::nn::activation::GELU::GELU() {
//	std::vector<std::shared_ptr<Layer>> layers = {};
//}
//
//rush::nn::activation::GELU::~GELU() {
//
//}

rush::math::Tensor rush::nn::activation::GELU::forward(const rush::math::Tensor& input) {
	return input.apply([](float x) {
							return rush::math::gelu(x);
								   }
					  );
}

rush::math::Tensor rush::nn::activation::GELU::backward(const rush::math::Tensor& grad_output) {
	// FIXME
	return grad_output;
}



