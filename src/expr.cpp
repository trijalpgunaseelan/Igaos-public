// expr.cpp : the tape, and the three sweeps over it.
//
// The whole of automatic differentiation, for a fixed operation set, is one
// table: for each operation, its value, its first partials, and the derivative
// of those partials along a direction.  Everything below is that table plus the
// bookkeeping to walk the tape in the right order.
#include "igaos/expr.hpp"
#include <algorithm>
#include <cmath>

namespace igaos {

Int ExprTape::push(Op op, Int a, Int b, Real v) {
    node_.push_back(Node{op, a, b, v});
    return (Int)node_.size() - 1;
}

Int ExprTape::constant(Real v)          { return push(Op::Const, kNone, kNone, v); }
Int ExprTape::variable(Int j)           { nvar_ = std::max(nvar_, j + 1);
                                          return push(Op::Var, kNone, kNone, (Real)j); }
Int ExprTape::add(Int a, Int b)         { return push(Op::Add, a, b, 0.0); }
Int ExprTape::sub(Int a, Int b)         { return push(Op::Sub, a, b, 0.0); }
Int ExprTape::mul(Int a, Int b)         { return push(Op::Mul, a, b, 0.0); }
Int ExprTape::div(Int a, Int b)         { return push(Op::Div, a, b, 0.0); }
Int ExprTape::neg(Int a)                { return push(Op::Neg, a, kNone, 0.0); }
Int ExprTape::pow(Int a, Real p)        { return push(Op::Pow, a, kNone, p); }
Int ExprTape::square(Int a)             { return push(Op::Square, a, kNone, 0.0); }
Int ExprTape::exp(Int a)                { return push(Op::Exp, a, kNone, 0.0); }
Int ExprTape::log(Int a)                { return push(Op::Log, a, kNone, 0.0); }
Int ExprTape::sqrt(Int a)               { return push(Op::Sqrt, a, kNone, 0.0); }
Int ExprTape::sin(Int a)                { return push(Op::Sin, a, kNone, 0.0); }
Int ExprTape::cos(Int a)                { return push(Op::Cos, a, kNone, 0.0); }

Int ExprTape::linear(const std::vector<std::pair<Int, Real>>& terms, Real c) {
    Int acc = constant(c);
    for (const auto& t : terms) {
        Int prod = mul(constant(t.second), variable(t.first));
        acc = add(acc, prod);
    }
    return acc;
}

// ---------------------------------------------------------------------------
// Which nodes are needed for `root`.  Marking them first means every sweep
// below touches only the subtree, not the whole shared tape -- which matters a
// great deal when one tape holds a thousand constraints and the gradient of one
// of them is wanted.
// ---------------------------------------------------------------------------
static void markLive(const std::vector<ExprTape::Node>& node, Int root,
                     std::vector<uint8_t>& live) {
    live.assign(node.size(), 0);
    if (root == kNone) return;
    live[(size_t)root] = 1;
    for (Int i = root; i >= 0; --i) {
        if (!live[(size_t)i]) continue;
        const ExprTape::Node& nd = node[(size_t)i];
        if (nd.a != kNone) live[(size_t)nd.a] = 1;
        if (nd.b != kNone) live[(size_t)nd.b] = 1;
    }
}

// ---------------------------------------------------------------------------
// Forward pass: value, and optionally the tangent along p.
// ---------------------------------------------------------------------------
static void forward(const std::vector<ExprTape::Node>& node,
                    const std::vector<uint8_t>& live,
                    const std::vector<Real>& x, const std::vector<Real>* p,
                    std::vector<Real>& v, std::vector<Real>& dv) {
    const size_t n = node.size();
    v.assign(n, 0.0);
    if (p) dv.assign(n, 0.0);
    for (size_t i = 0; i < n; ++i) {
        if (!live[i]) continue;
        const ExprTape::Node& nd = node[i];
        const Real a = nd.a != kNone ? v[(size_t)nd.a] : 0.0;
        const Real b = nd.b != kNone ? v[(size_t)nd.b] : 0.0;
        const Real da = (p && nd.a != kNone) ? dv[(size_t)nd.a] : 0.0;
        const Real db = (p && nd.b != kNone) ? dv[(size_t)nd.b] : 0.0;
        Real val = 0.0, tan = 0.0;
        switch (nd.op) {
            case Op::Const:  val = nd.value; break;
            case Op::Var: {
                const Int j = (Int)nd.value;
                val = x[(size_t)j];
                tan = p ? (*p)[(size_t)j] : 0.0;
                break;
            }
            case Op::Add:    val = a + b;  tan = da + db; break;
            case Op::Sub:    val = a - b;  tan = da - db; break;
            case Op::Mul:    val = a * b;  tan = da * b + a * db; break;
            case Op::Div:    val = a / b;  tan = (da * b - a * db) / (b * b); break;
            case Op::Neg:    val = -a;     tan = -da; break;
            case Op::Square: val = a * a;  tan = 2.0 * a * da; break;
            case Op::Pow:    val = std::pow(a, nd.value);
                             tan = nd.value * std::pow(a, nd.value - 1.0) * da; break;
            case Op::Exp:    val = std::exp(a);  tan = val * da; break;
            case Op::Log:    val = std::log(a);  tan = da / a; break;
            case Op::Sqrt:   val = std::sqrt(a); tan = da / (2.0 * val); break;
            case Op::Sin:    val = std::sin(a);  tan = std::cos(a) * da; break;
            case Op::Cos:    val = std::cos(a);  tan = -std::sin(a) * da; break;
        }
        v[i] = val;
        if (p) dv[i] = tan;
    }
}

Real ExprTape::value(Int root, const std::vector<Real>& x) const {
    if (root == kNone) return 0.0;
    markLive(node_, root, live_);
    forward(node_, live_, x, nullptr, v_, dv_);
    return v_[(size_t)root];
}

// ---------------------------------------------------------------------------
// Reverse pass for the gradient.  adj[i] = d(root)/d(node i), accumulated from
// the root downward; at a Var node it lands in the gradient entry.
// ---------------------------------------------------------------------------
void ExprTape::gradient(Int root, const std::vector<Real>& x,
                        std::vector<Real>& g, Real scale) const {
    if (root == kNone) return;
    markLive(node_, root, live_);
    forward(node_, live_, x, nullptr, v_, dv_);

    adj_.assign(node_.size(), 0.0);
    adj_[(size_t)root] = 1.0;
    for (Int i = root; i >= 0; --i) {
        if (!live_[(size_t)i] || adj_[(size_t)i] == 0.0) continue;
        const Node& nd = node_[(size_t)i];
        const Real w = adj_[(size_t)i];
        const Real a = nd.a != kNone ? v_[(size_t)nd.a] : 0.0;
        const Real b = nd.b != kNone ? v_[(size_t)nd.b] : 0.0;
        switch (nd.op) {
            case Op::Const: break;
            case Op::Var:   g[(size_t)(Int)nd.value] += scale * w; break;
            case Op::Add:   adj_[(size_t)nd.a] += w; adj_[(size_t)nd.b] += w; break;
            case Op::Sub:   adj_[(size_t)nd.a] += w; adj_[(size_t)nd.b] -= w; break;
            case Op::Mul:   adj_[(size_t)nd.a] += w * b; adj_[(size_t)nd.b] += w * a; break;
            case Op::Div:   adj_[(size_t)nd.a] += w / b;
                            adj_[(size_t)nd.b] -= w * a / (b * b); break;
            case Op::Neg:   adj_[(size_t)nd.a] -= w; break;
            case Op::Square:adj_[(size_t)nd.a] += w * 2.0 * a; break;
            case Op::Pow:   adj_[(size_t)nd.a] += w * nd.value * std::pow(a, nd.value - 1.0);
                            break;
            case Op::Exp:   adj_[(size_t)nd.a] += w * v_[(size_t)i]; break;
            case Op::Log:   adj_[(size_t)nd.a] += w / a; break;
            case Op::Sqrt:  adj_[(size_t)nd.a] += w / (2.0 * v_[(size_t)i]); break;
            case Op::Sin:   adj_[(size_t)nd.a] += w * std::cos(a); break;
            case Op::Cos:   adj_[(size_t)nd.a] -= w * std::sin(a); break;
        }
    }
}

// ---------------------------------------------------------------------------
// Forward-over-reverse: H*p, exactly.
//
// The forward pass already carries the tangent dv.  The reverse pass carries a
// second adjoint dadj alongside adj, propagated by differentiating the reverse
// recurrence itself:
//
//      adj[child]  += adj[i] * partial
//      dadj[child] += dadj[i] * partial + adj[i] * d(partial)
//
// where d(partial) is the partial's own derivative along the tangent.  At a Var
// node, dadj is the corresponding entry of H*p.  Nothing here is differenced,
// so the result is the Hessian-vector product to full double precision.
// ---------------------------------------------------------------------------
void ExprTape::hessianVector(Int root, const std::vector<Real>& x,
                             const std::vector<Real>& p, std::vector<Real>& hv,
                             Real scale) const {
    if (root == kNone) return;
    markLive(node_, root, live_);
    forward(node_, live_, x, &p, v_, dv_);

    adj_.assign(node_.size(), 0.0);
    dadj_.assign(node_.size(), 0.0);
    adj_[(size_t)root] = 1.0;

    for (Int i = root; i >= 0; --i) {
        if (!live_[(size_t)i]) continue;
        const Real w = adj_[(size_t)i], dw = dadj_[(size_t)i];
        if (w == 0.0 && dw == 0.0) continue;
        const Node& nd = node_[(size_t)i];
        const Real a  = nd.a != kNone ? v_[(size_t)nd.a] : 0.0;
        const Real b  = nd.b != kNone ? v_[(size_t)nd.b] : 0.0;
        const Real da = nd.a != kNone ? dv_[(size_t)nd.a] : 0.0;
        const Real db = nd.b != kNone ? dv_[(size_t)nd.b] : 0.0;

        auto sendA = [&](Real partial, Real dpartial) {
            adj_[(size_t)nd.a]  += w * partial;
            dadj_[(size_t)nd.a] += dw * partial + w * dpartial;
        };
        auto sendB = [&](Real partial, Real dpartial) {
            adj_[(size_t)nd.b]  += w * partial;
            dadj_[(size_t)nd.b] += dw * partial + w * dpartial;
        };

        switch (nd.op) {
            case Op::Const: break;
            case Op::Var:   hv[(size_t)(Int)nd.value] += scale * dw; break;
            case Op::Add:   sendA(1.0, 0.0); sendB(1.0, 0.0); break;
            case Op::Sub:   sendA(1.0, 0.0); sendB(-1.0, 0.0); break;
            // d(a*b)/da = b, and that partial's own derivative is db.
            case Op::Mul:   sendA(b, db); sendB(a, da); break;
            case Op::Div: {
                const Real b2 = b * b, b3 = b2 * b;
                sendA(1.0 / b, -db / b2);
                sendB(-a / b2, -da / b2 + 2.0 * a * db / b3);
                break;
            }
            case Op::Neg:    sendA(-1.0, 0.0); break;
            case Op::Square: sendA(2.0 * a, 2.0 * da); break;
            case Op::Pow: {
                const Real q = nd.value;
                sendA(q * std::pow(a, q - 1.0), q * (q - 1.0) * std::pow(a, q - 2.0) * da);
                break;
            }
            case Op::Exp: {
                const Real e = v_[(size_t)i];
                sendA(e, e * da);
                break;
            }
            case Op::Log:  sendA(1.0 / a, -da / (a * a)); break;
            case Op::Sqrt: {
                const Real r = v_[(size_t)i];
                sendA(1.0 / (2.0 * r), -da / (4.0 * r * r * r));
                break;
            }
            case Op::Sin:  sendA(std::cos(a), -std::sin(a) * da); break;
            case Op::Cos:  sendA(-std::sin(a), -std::cos(a) * da); break;
        }
    }
}

void ExprTape::dependencies(Int root, std::vector<Int>& out) const {
    out.clear();
    if (root == kNone) return;
    markLive(node_, root, live_);
    for (size_t i = 0; i < node_.size(); ++i)
        if (live_[i] && node_[i].op == Op::Var) out.push_back((Int)node_[i].value);
    std::sort(out.begin(), out.end());
    out.erase(std::unique(out.begin(), out.end()), out.end());
}

bool ExprTape::isLinear(Int root) const {
    if (root == kNone) return true;
    markLive(node_, root, live_);
    // "Constant" here means the subtree contains no Var node at all.  A product
    // of a constant subtree and a linear one is linear; a product of two
    // non-constant ones is not.
    std::vector<uint8_t> hasVar(node_.size(), 0), lin(node_.size(), 1);
    for (size_t i = 0; i < node_.size(); ++i) {
        if (!live_[i]) continue;
        const Node& nd = node_[i];
        const bool va = nd.a != kNone && hasVar[(size_t)nd.a];
        const bool vb = nd.b != kNone && hasVar[(size_t)nd.b];
        const bool la = nd.a == kNone || lin[(size_t)nd.a];
        const bool lb = nd.b == kNone || lin[(size_t)nd.b];
        switch (nd.op) {
            case Op::Const:  hasVar[i] = 0; lin[i] = 1; break;
            case Op::Var:    hasVar[i] = 1; lin[i] = 1; break;
            case Op::Add: case Op::Sub:
                             hasVar[i] = va || vb; lin[i] = la && lb; break;
            case Op::Neg:    hasVar[i] = va; lin[i] = la; break;
            case Op::Mul:    hasVar[i] = va || vb;
                             lin[i] = la && lb && !(va && vb); break;
            case Op::Div:    hasVar[i] = va || vb;
                             lin[i] = la && !vb; break;      // linear / constant
            default:         hasVar[i] = va; lin[i] = !va; break;
        }
    }
    return lin[(size_t)root] != 0;
}

// ---------------------------------------------------------------------------
Int NlpProblem::addVariable(Real lo, Real up, const std::string& nm) {
    Int j = (Int)lower.size();
    lower.push_back(lo);
    upper.push_back(up);
    varName.push_back(nm.empty() ? ("v" + std::to_string(j)) : nm);
    tape.setNumVar(j + 1);
    return j;
}

Int NlpProblem::addConstraint(Int node, Real lo, Real up, const std::string& nm) {
    Int i = (Int)constraint.size();
    constraint.push_back(node);
    conLower.push_back(lo);
    conUpper.push_back(up);
    conName.push_back(nm.empty() ? ("c" + std::to_string(i)) : nm);
    return i;
}

} // namespace igaos
