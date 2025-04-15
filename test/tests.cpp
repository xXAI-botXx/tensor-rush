
#define CATCH_CONFIG_MAIN
#include "catch.hpp"

// import code to test
#include "tensor-rush.h"

// start testing

// ---------
//  tensors
// ---------
TEST_CASE("Tensor creation", "[tensor]") {
    rush::math::Tensor t = rush::math::Tensor(1.0f, std::vector<int>{2, 3});
    auto shape = t.shape[0];
    REQUIRE(shape == 2);
    shape = t.shape[1];
    REQUIRE(shape == 3);
    shape = t[std::vector<int>{0,0}];
    REQUIRE(shape == Approx(1.0f));
}

TEST_CASE("Tensor element modification", "[tensor]") {
    rush::math::Tensor t = rush::math::Tensor(0.0f, std::vector<int>{2, 2});
    t[std::vector<int>{0, 0}] = 3.14f;
    t[std::vector<int>{1, 1}] = 2.71f;
    REQUIRE(t[std::vector<int>{0, 0}] == Approx(3.14f));
    REQUIRE(t[std::vector<int>{1, 1}] == Approx(2.71f));
}

TEST_CASE("Tensor addition", "[tensor][math]") {
    rush::math::Tensor a = rush::math::Tensor(1.0f, { 2, 2 });
    rush::math::Tensor b = rush::math::Tensor(2.0f, { 2, 2 });
    rush::math::Tensor result = a + b;
    REQUIRE(result[std::vector<int>{0, 0}] == Approx(3.0f));
    REQUIRE(result[std::vector<int>{1, 1}] == Approx(3.0f));
}

TEST_CASE("Tensor multiplication", "[tensor][math]") {
    rush::math::Tensor a = rush::math::Tensor(2.0f, { 2, 2 });
    rush::math::Tensor b = rush::math::Tensor(3.0f, { 2, 2 });
    rush::math::Tensor result = a * b;
    REQUIRE(result[std::vector<int>{0, 0}] == Approx(6.0f));
    REQUIRE(result[std::vector<int>{1, 1}] == Approx(6.0f));
}

TEST_CASE("Tensor reshape", "[tensor]") {
    rush::math::Tensor t = rush::math::Tensor(1.0f, { 2, 3 });
    t.reshape({ 3, 2 });
    REQUIRE(t.shape[0] == 3);
    REQUIRE(t.shape[1] == 2);
    REQUIRE(t[std::vector<int>{0, 0}] == Approx(1.0f)); // assuming values stay the same
}

TEST_CASE("Tensor indexing", "[tensor]") {
    rush::math::Tensor t = rush::math::Tensor(0.0f, { 2, 2 });
    t[std::vector<int>{0, 1}] = 5.0f;
    REQUIRE(t[std::vector<int>{0, 1}] == Approx(5.0f));
}

// --------
//  layers
// --------
//TEST_CASE("Add layers to Sequential model", "[nn]") {
//    auto model = std::make_shared<rush::nn::Sequential>();
//
//    model->add(std::make_shared<rush::nn::Linear>(3, 4));
//    model->add(std::make_shared<rush::nn::Activation>(rush::Activation::ReLU));
//    model->add(std::make_shared<rush::nn::Linear>(4, 2));
//
//    rush::math::Tensor input = rush::math::Tensor::random({ 1, 3 });
//    rush::math::Tensor output = model->forward(input);
//
//    REQUIRE(output.shape()[1] == 2);
//}

