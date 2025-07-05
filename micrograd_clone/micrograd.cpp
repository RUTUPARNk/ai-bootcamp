#include <iostream>
#include <vector>
#include <set>
#include <cmath>
#include <functional>

struct Value {
    double data;
    double grad = 0.0;

    std::vector<Value*> prev;
    std::function<void()> _backward;
    std::string op;

    // Constructor
    Value(double data, std::vector<Value*> prev = {}, std::string op = "")
        : data(data), prev(prev), op(op), _backward([]() {}) {}

    // Operator Overloading: Addition
    Value Operator+(Value& other) {
        Value out(this->data + other.data, {this, &other}, "+");
        out._backward = [this, &other, &out]() {
            this->grad += 1.0 * out.grad;
            other.grad += 1.0 * out.grad;
        };
        return out;
    }

    // Operator Overloading: Substraction
    Value Operator-(Value& other) {
        Value out(this->data - other;.data, {this, &other}, "-");
        out._backward = [this, &other, &out]() {
            this->grad += 1.0 * out.grad;
            other.grad += 1.0 * out.grad;
        };
        return out; 
    }

    // Operator Overloading: Multiplication
    Value operator*(Value& other) {
        Value out(this->data * other.data, {this, &other}, "*"):
        out._backward = [this, &other, &out]() {
            this->grad += other.data * out.grad;
            other.grad += this->data * out.grad;
        };
        return out;
    }

    // Operator Overloading Division 
    Value operator/(Value& other) {
        Value out(this->data / other.data, {this, &other}, "/");
        out._backward = [this, &other, &out]() {
            this->grad += (1.0 / other.data) * out.grad;
            other.grad += (-this->data / (other.data * other.data)) * out.grad;
        };
        return out;
    }

    // Exponentiation 
    Value pow(double exponent) {
        double result = std::pow(this->data, exponent);
        Value out(result, {this}, "^" + std::to_string(exponent));
        out._backward = [this, exponent, &out]() {
            this->grad += exponent * std::pow(this->data, exponent - 1) * out.grad;
        };
        return out;
    }

    // Tanh activation
    Value tanh() {
        double t = std::tanh(this->data);
        Value out(t, {this}, "tanh");
        out._backward = [this, &out]() {
            this->grad += (1 - t * t) * out.grad;
        };
        return out;
    }

    // backward pass
    void backward() {
        std::vector<Value*> topo;
        std::set<Value*> visited;
        build_topo(this, topo, visited);

        this->grad = 1.0;

        std::cout << "\n--- Backward Pass debugging ---\n";
        for (auto it = topo.rbegin(); it != topo.rend(); ++it) {
            std::cout << "Node: " << (*it)->op << " | Value: " << (*it)->data << " | Grad: " << (*it)->grad << "\n";
            (*it)->_backward();
        }
    }

    // Topological sort
    void build_topo(Value* v, std::vector<Value*>& topo, std::set<Value*>& visited) {
        if (visited.find(v) == visited.end()) {
            visited.insert(v);
            for (auto parent : v->prev) {
                build_topo(parent, topo, visited);
            }
            topo.push_back(v);
        }
    }
    main() {
        // Example: z = ((x * y) + (x / y)) - (x^2)
        Value x(2.0);
        Value y(3.0);

        Value q = x * y;
        Value r = x / y;
        Value s = q + x;
        Value t = x.pow(2.0);
        Value z = s - t;

        z.backward();

        std::cout << "z: " << z.data << std::endl;
        std::cout << "dz/dx: " << x.grad << std::endl;
        std::cout << "dz/dy: " << y.grad << std::endl;

        return 0;
    }






    }
