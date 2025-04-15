#include "pch.h"

#include "math.h"


#include <vector>
#include <functional>
#include <cmath>
#include <numeric>
#include <algorithm>
#include <iostream>
#include <sstream>






// ------> math <------


// -----------------
// >>> Constants <<<
// -----------------

constexpr float PI = 3.14159265359f;
constexpr float E = 2.71828182846f;
constexpr float EPSILON = 1e-6f;




// -----------------
// >>> Functions <<<
// -----------------

// Helper functions (wrappers to std)
float rush::math::exp(float x) {
	return std::exp(x);
}

float rush::math::tanh(float x) {
	return std::tanh(x);
}

float rush::math::sigmoid(float x) {
	return 1.0f / (1.0f + std::exp(-x));
}

float rush::math::sigmoid_derivative(float x) {
	float s = rush::math::sigmoid(x);
	return s * (1 - s);
}

float rush::math::relu(float x) {
	return (((0.0f) > (x)) ? (0.0f) : (x));
}

float rush::math::relu_derivative(float x) {
	return x > 0 ? 1.0f : 0.0f;
}

float rush::math::tanh_derivative(float x) {
	float t = std::tanh(x);
	return 1.0f - t * t;
}

float rush::math::gelu(float x) {
	const float sqrt_2_over_pi = std::sqrt(2.0f / 3.14159265359f);
	return 0.5f * x * (1.0f + std::tanh(sqrt_2_over_pi * (x + 0.044715f * x * x * x)));
}

// Optional derivative if needed later
float rush::math::gelu_derivative(float x) {
	const float sqrt_2_over_pi = std::sqrt(2.0f / 3.14159265359f);
	float x_cubed = x * x * x;
	float tanh_arg = sqrt_2_over_pi * (x + 0.044715f * x_cubed);
	float tanh_val = std::tanh(tanh_arg);
	float sech2 = 1.0f - tanh_val * tanh_val;

	return 0.5f * (1.0f + tanh_val) +
		   0.5f * x * sech2 * sqrt_2_over_pi * (1.0f + 3.0f * 0.044715f * x * x);
}

// FIXME






// --------------
// >>> Tensor <<<
// --------------

rush::math::Tensor::Tensor(const float& data_) {
	data = std::vector<float>{ data_ };
	shape = std::vector<int>{1};
}

rush::math::Tensor::Tensor(const float& data_, const std::vector<int>& shape_) {
	data = std::vector<float>{ };
	shape = shape_;
	int goal_size = 1;
	for (auto& cur_dim : shape) {
		for (int i = 0; i < cur_dim; ++i) {
			data.push_back(data_);
		}
	}
}

rush::math::Tensor::Tensor(const std::vector<float>& data_, 
						   const std::vector<int>& shape_) {
	data = data_;
	shape = shape_;

	/*
	shape = 2, 3, 4 
	=> [], []
	=> 2   *    [], [], []
	=> 2*3 *    [], [], [], []
	Needed Elems: 2*3*4
	*/

	// check and apply -1
	// get shape size without -1 + -1 values
	int minus_one_value = NULL;
	int goal_size = len();
	int found_size = 1;
	int minus_one_counter = 0;
	for (int& cur_dim : shape) {
		if (cur_dim > 0) {
			found_size *= cur_dim;
		}
		else {
			minus_one_counter += 1;
			if (minus_one_counter > 1)
				throw std::invalid_argument("Multiple -1 values in the shape definition detected");
		}
	}

	// check if size without -1 value is too big -> the -1 value can only make it even bigger not smaller
	if ((goal_size < goal_size-found_size) || (minus_one_counter > 0 && goal_size <= goal_size - found_size))
		throw std::invalid_argument("Data size does not match shape (shape too big)");

	// calculat -1 value
	if (minus_one_counter > 0) {
		minus_one_value = goal_size - found_size;
		found_size *= minus_one_value;
	}

	// check if size with -1 value fits -> only that the shapes are too small
	if (data.size() != found_size)
		throw std::invalid_argument("Data size does not match shape (shape too small)");

	// save new shape -> with calculated -1 value
	std::vector<int> corrected_shape = {};
	for (int& cur_dim : shape) {
		if (cur_dim > 0) {
			corrected_shape.push_back(cur_dim);
		}
		else {
			corrected_shape.push_back(minus_one_value);
		}
	}
}

rush::math::Tensor rush::math::Tensor::apply(const std::function<float(float)>& func) const {
	std::vector<float> result;
	result.reserve(data.size());
	for (float val : data) result.push_back(func(val));
	return rush::math::Tensor::Tensor(result, shape);
}

void rush::math::Tensor::reshape(const std::vector<int>& new_shape) {
	// Calculate total size of old and new shapes
	int old_size = 1;
	for (int dim : shape) {
		old_size *= dim;
	}

	int new_size = 1;
	for (int dim : new_shape) {
		new_size *= dim;
	}

	// Validate the total number of elements remains unchanged
	if (old_size != new_size) {
		throw std::invalid_argument("New shape must contain the same number of elements.");
	}

	// Update the shape
	shape = new_shape;
}

// Basic math operators
rush::math::Tensor rush::math::Tensor::operator+(const rush::math::Tensor& other) const {
	std::vector<float> result;
	for (int i = 0; i < data.size(); ++i)
		result.push_back(data[i] + other.data[i]);
	return rush::math::Tensor::Tensor(result, shape);
}

rush::math::Tensor rush::math::Tensor::operator-(const Tensor& other) const {
	std::vector<float> result;
	for (int i = 0; i < data.size(); ++i)
		result.push_back(data[i] - other.data[i]);
	return rush::math::Tensor(result, shape);
}

rush::math::Tensor rush::math::Tensor::operator*(const Tensor& other) const {
	std::vector<float> result;
	for (int i = 0; i < data.size(); ++i)
		result.push_back(data[i] * other.data[i]);
	return rush::math::Tensor(result, shape);
}

rush::math::Tensor rush::math::Tensor::operator*(float scalar) const {
	std::vector<float> result;
	for (float val : data)
		result.push_back(val * scalar);
	return rush::math::Tensor(result, std::vector<int>{len(), 1});
}

float& rush::math::Tensor::operator[](std::vector<int> shape_request) {
	// To calculate the index, we have to go backwards, because the dimensions are applied sequentially and now we have to go the way from the other side.
	// Let's say we have a tensor with shape: {2, 3, 4}
	// Example indexing: tensor[{1, 2, 3}]
	// To calculate the flat index for {1, 2, 3}, we use strides:
	//   - stride[0] = 3 * 4 = 12    
	//   - stride[1] = 4             
	//   - stride[2] = 1             
	//
	// Then the flat index is:
	//   index = 1 * 12 + 2 * 4 + 3 * 1 = 12 + 8 + 3 = 23
	//
	// So tensor[{1, 2, 3}] == data[23]
	//
	// General formula:
	//   index = sum_{i=0}^{n-1} (indices[i] * stride[i])
	//
	// We compute the strides from the shape, typically in reverse:
	//   stride[n-1] = 1
	//   stride[n-2] = shape[n-1]
	//   stride[n-3] = shape[n-2] * shape[n-1]
	//   ...
	//
	// This is how we map n-dimensional indexing to a 1D flat vector.
	
	if (shape_request.size() != shape.size())
		throw std::invalid_argument("The given shape for indexing have the wrong amount elemnts, should be the same as the shapes/dimensions of te tensor");

	// replace -1 values with the latest index
	std::vector<int> corrected_request = {};
	for (int i = 0; i < shape_request.size(); ++i) {
		if (shape_request.at(i) < 0) {
			corrected_request.push_back(shape.at(i)-1);
		}
		else {
			corrected_request.push_back(shape_request.at(i));
		}
	}

	// access flat vector with multi-dimensional index
	int index = 0;
	int stride = 1;
	for (int i = shape_request.size() - 1; i >= 0; --i) {
		if (shape_request.at(i) >= shape.at(i) || shape_request.at(i) < 0)
			throw std::invalid_argument("The given shape for indexing tries to access a value which is out of the box. Check your requested index again.");
		
		index += shape_request.at(i) * stride;
		stride *= shape.at(i);
	}

	return data.at(index);
}

const float& rush::math::Tensor::operator[](std::vector<int> shape_request) const {
	if (shape_request.size() != shape.size())
		throw std::invalid_argument("The given shape for indexing have the wrong amount elemnts, should be the same as the shapes/dimensions of te tensor");

	int index = 1;
	for (int i = 0; i < shape_request.size(); ++i) {
		index *= shape_request.at(i);
		if (shape_request.at(i) > shape.at(i) || shape_request.at(i) < 0)
			throw std::invalid_argument("The given shape for indexing tries to access a value which is out of the box. Check your requested index again.");
	}

	return data.at(index);
}

float& rush::math::Tensor::operator[](int shallow_shape_request) {
	if (shallow_shape_request == -1)
		shallow_shape_request = len() - 1;

	if (shallow_shape_request >= len())
		throw std::invalid_argument("The given shallow number for indexing is too high, the Tensor does not have so much values");

	return data.at(shallow_shape_request);
}

const float& rush::math::Tensor::operator[](int shallow_shape_request) const {
	if (shallow_shape_request == -1)
		shallow_shape_request = len() - 1;

	if (shallow_shape_request >= len())
		throw std::invalid_argument("The given shallow number for indexing is too high, the Tensor does not have so much values");

	return data.at(shallow_shape_request);
}

void rush::math::Tensor::print() const {
	std::cout << "Tensor of shape (";
	for (size_t i = 0; i < shape.size(); ++i) {
		std::cout << shape[i];
		if (i < shape.size() - 1) std::cout << ", ";
	}
	std::cout << "):\n";

	
	std::ostringstream oss;
	print_data(*this, oss, {}, 0, 4);
	std::cout << oss.str() << std::endl;
}

void print_data(const rush::math::Tensor& tensor, std::ostream& os, std::vector<int> indices={}, int depth=0, int max_items=4) {
	std::string spacing(depth * 4, ' '); // 4 spaces per depth

	if (depth == tensor.shape.size()) {
		// Base case: we've reached a data point
		os << tensor[indices];
		return;
	}

	os << spacing << "[";
	size_t dim_size = tensor.shape[depth];
	int print_limit = ((int)dim_size < max_items ? (int)dim_size : max_items);

	for (int i = 0; i < print_limit; ++i) {
		std::vector<int> new_indices = indices;
		new_indices.push_back(i);

		if (i > 0)
			os << ", ";

		if (depth + 1 < tensor.shape.size())
			os << "\n";

		print_data(tensor, os, new_indices, depth + 1, max_items);
	}

	if ((int)dim_size > max_items) {
		os << ", ";
		if (depth + 1 < tensor.shape.size())
			os << "\n" << spacing << "    ";
		os << "...";
	}

	if (depth + 1 < tensor.shape.size())
		os << "\n" << spacing;
	os << "]";
}

int rush::math::Tensor::len() const {
	return (int)data.size();
}









