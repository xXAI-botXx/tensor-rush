#pragma once

#include <vector>
#include <functional>
#include <sstream>

namespace rush {
	namespace math {
		class Tensor {

		// Currently based on float32, which is a trade from size and precision, should be fine...

		public:
			Tensor() = default;
			Tensor(const float& data_);
			Tensor(const std::vector<float>& data_, const std::vector<int>& shape_);
			Tensor(const float& data_, const std::vector<int>& shape_);

			Tensor apply(const std::function<float(float)>& func) const;
			void reshape(const std::vector<int>& new_shape);

			// basic math operators
			Tensor operator+(const Tensor& other) const;
			Tensor operator-(const Tensor& other) const;
			Tensor operator*(const Tensor& other) const;
			Tensor operator*(float scalar) const;

			// indexing (flat)
			float& operator[](std::vector<int> shape_request);
			const float& operator[](std::vector<int> shape_request) const;
			float& operator[](int shallow_shape_request);
			const float& operator[](int shallow_shape_request) const;

			// TODO
			// at -> already available
			// get -> already available
			// fill
			// save
			// load
			// plot

			// other methods
			void print() const;
			int len() const;

			// variables
			std::vector<float> data;
			std::vector<int> shape;
		};

		// Helper functions (wrappers to std)
		float exp(float x);
		float tanh(float x);
		float sigmoid(float x);
		float sigmoid_derivative(float x);
		float relu(float x);
		float relu_derivative(float x);
		float tanh_derivative(float x);
		float gelu(float x);
		float gelu_derivative(float x);

	}
}

void print_data(const rush::math::Tensor& tensor, std::ostream& os, std::vector<int> indices, int depth, int max_items);




