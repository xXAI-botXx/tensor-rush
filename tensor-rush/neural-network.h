#pragma once

#include <memory>
#include <vector>
#include <string>

#include "math.h"

namespace rush {
	namespace nn {

		class Layer {
			// consists of layers
			// a Layer have always a input and a output
			// a Layer can be a softmax, activation function, consist of multiple layers
		public:
			Layer() = default;
			virtual ~Layer() = default;

			virtual rush::math::Tensor forward(const rush::math::Tensor& input) = 0;
			virtual rush::math::Tensor backward(const rush::math::Tensor& grad_output) = 0;

			// adding new (sub)layers
			virtual void add(std::shared_ptr<Layer> layer);/* {
				throw std::logic_error("This layer can not contain sub-layers.");
			}*/
			virtual void add(std::vector<std::shared_ptr<Layer>> layers);

			// others
			virtual std::string get_as_str();
			virtual void plot();
			virtual void get_parameters();

		protected:
			std::vector<std::shared_ptr<Layer>> layers;
		};

		class Sequential : public Layer {
		public:
			Sequential() = default;
			virtual ~Sequential() = default;

			virtual rush::math::Tensor forward(const rush::math::Tensor& input);
			virtual rush::math::Tensor backward(const rush::math::Tensor& grad_output);
		};

		namespace activation {
			class ReLU : public Layer {
			public:
				ReLU() = default;
				virtual ~ReLU() = default;

				virtual rush::math::Tensor forward(const rush::math::Tensor& input);
				virtual rush::math::Tensor backward(const rush::math::Tensor& grad_output);
			};

			class Sigmoid : public Layer {
			public:
				Sigmoid() = default;
				virtual ~Sigmoid() = default;

				virtual rush::math::Tensor forward(const rush::math::Tensor& input);
				virtual rush::math::Tensor backward(const rush::math::Tensor& grad_output);
			};

			class Tanh : public Layer {
			public:
				Tanh() = default;
				virtual ~Tanh() = default;

				virtual rush::math::Tensor forward(const rush::math::Tensor& input);
				virtual rush::math::Tensor backward(const rush::math::Tensor& grad_output);
			};

			class GELU : public Layer {
			public:
				GELU() = default;
				virtual ~GELU() = default;

				virtual rush::math::Tensor forward(const rush::math::Tensor& input);
				virtual rush::math::Tensor backward(const rush::math::Tensor& grad_output);
			};
		}

	}
}


