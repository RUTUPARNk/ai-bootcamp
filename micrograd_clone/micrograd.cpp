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

    // Operator Overloading: Multiplication
    Value operator*(Value& other) {
        Value out(this->data * other.data, {this, &other}, "*"):
        out._backward = [this, &other, &out]() {
            this->grad += other.data * out.grad;
            other.grad += this->data * out.grad;
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

        for (auto it = topo.rbegin(); it != topo.rend(); ++it) {
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
        // Example: z = (x * y) + (x + y)
        Value x(2.0);
        Value y(3.0);

        Value q = x * y;
        Value p = q + x;
        Value z = p + y;

        z.backward();

        std::cout << "z: " << z.data << std::endl;
        std::cout << "dz/dx: " << x.grad << std::endl;
        std::cout << "dz/dy: " << y.grad << std::endl;

        return 0;
    }






    }
