"""Small explicit MLIR generators, with an independent scalar reference."""
import math
import random
import json


def matmul(m: int, n: int, k: int) -> str:
    if any(type(d) is not int or d < 0 for d in (m, n, k)):
        raise ValueError("dimensions must be nonnegative integers")
    return f'''module {{
  func.func @kernel(%a: memref<{m}x{k}xf64>, %b: memref<{k}x{n}xf64>,
                    %c: memref<{m}x{n}xf64>) {{
    %zero = arith.constant 0.0 : f64
    linalg.fill ins(%zero : f64) outs(%c : memref<{m}x{n}xf64>)
    linalg.generic {{indexing_maps = [
      affine_map<(i,j,k)->(i,k)>, affine_map<(i,j,k)->(k,j)>,
      affine_map<(i,j,k)->(i,j)>],
      iterator_types = ["parallel", "parallel", "reduction"]}}
      ins(%a, %b : memref<{m}x{k}xf64>, memref<{k}x{n}xf64>)
      outs(%c : memref<{m}x{n}xf64>) {{
      ^bb0(%x: f64, %y: f64, %sum: f64):
        %p = arith.mulf %x, %y : f64
        %s = arith.addf %sum, %p : f64
        linalg.yield %s : f64
    }}
    return
  }}
}}'''


def matmul_case(m: int, n: int, k: int, seed: int = 7):
    rng = random.Random(seed)
    a = [rng.uniform(-1, 1) for _ in range(m * k)]
    b = [rng.uniform(-1, 1) for _ in range(k * n)]
    # Nonzero initial output detects missing reset, including repeated calls.
    c = [17.0] * (m * n)
    reference = [math.fsum(a[i * k + q] * b[q * n + j] for q in range(k))
                 for i in range(m) for j in range(n)]
    return [a, b, c], reference


def matvec(m: int, k: int) -> str:
    if any(type(d) is not int or d < 0 for d in (m, k)):
        raise ValueError("dimensions must be nonnegative integers")
    return f'''module {{
  func.func @kernel(%a: memref<{m}x{k}xf64>, %b: memref<{k}xf64>, %c: memref<{m}xf64>) {{
    %zero = arith.constant 0.0 : f64
    linalg.fill ins(%zero : f64) outs(%c : memref<{m}xf64>)
    linalg.generic {{indexing_maps = [affine_map<(i,k)->(i,k)>,
      affine_map<(i,k)->(k)>, affine_map<(i,k)->(i)>],
      iterator_types = ["parallel", "reduction"]}}
      ins(%a, %b : memref<{m}x{k}xf64>, memref<{k}xf64>)
      outs(%c : memref<{m}xf64>) {{
      ^bb0(%x: f64, %y: f64, %sum: f64):
        %p = arith.mulf %x, %y : f64
        %s = arith.addf %sum, %p : f64
        linalg.yield %s : f64
    }}
    return
  }}
}}'''


def matvec_case(m: int, k: int, seed: int = 7):
    return matmul_case(m, 1, k, seed)


def _generic(shapes, maps, iterators, body):
    if any(type(d) is not int or d < 0 for shape in shapes for d in shape):
        raise ValueError("dimensions must be nonnegative integers")
    types = ["memref<" + "x".join(map(str, shape)) + "xf64>" for shape in shapes]
    return f'''module {{
  func.func @kernel(%a: {types[0]}, %b: {types[1]}, %c: {types[2]}) {{
    %zero = arith.constant 0.0 : f64
    linalg.fill ins(%zero : f64) outs(%c : {types[2]})
    linalg.generic {{indexing_maps = [{', '.join(maps)}],
      iterator_types = {json.dumps(iterators)}}}
      ins(%a, %b : {types[0]}, {types[1]}) outs(%c : {types[2]}) {{
      ^bb0(%x: f64, %y: f64, %sum: f64):
        {body}
        linalg.yield %s : f64
    }}
    return
  }}
}}'''


def row_reduce(m, k):
    return _generic([(m, k), (k,), (m,)],
                    ['affine_map<(i,k)->(i,k)>', 'affine_map<(i,k)->(k)>', 'affine_map<(i,k)->(i)>'],
                    ['parallel', 'reduction'], '%s = arith.addf %sum, %x : f64')


def row_reduce_case(m, k, seed=7):
    rng = random.Random(seed)
    a = [rng.uniform(-1, 1) for _ in range(m*k)]
    return [a, [0.] * k, [17.] * m], [math.fsum(a[i*k:(i+1)*k]) for i in range(m)]


def elementwise(m, n):
    return _generic([(m,n)] * 3, ['affine_map<(i,j)->(i,j)>'] * 3,
                    ['parallel', 'parallel'], '%p = arith.mulf %x, %y : f64\n        %s = arith.addf %p, %x : f64')


def elementwise_case(m, n, seed=7):
    rng = random.Random(seed)
    a, b = [[rng.uniform(-1,1) for _ in range(m*n)] for _ in range(2)]
    return [a, b, [17.] * (m*n)], [x*y+x for x,y in zip(a,b)]


def stencil(h, w):
    """Nine-point 2D averaging stencil, valid interior with an explicit input halo."""
    return _generic([(h+2,w+2), (3,3), (h,w)],
                    ['affine_map<(i,j,r,s)->(i+r,j+s)>', 'affine_map<(i,j,r,s)->(r,s)>', 'affine_map<(i,j,r,s)->(i,j)>'],
                    ['parallel','parallel','reduction','reduction'],
                    '%p = arith.mulf %x, %y : f64\n        %s = arith.addf %sum, %p : f64')


def stencil_case(h, w, seed=7):
    rng = random.Random(seed)
    a = [rng.uniform(-1,1) for _ in range((h+2)*(w+2))]
    weights = [1./9.] * 9
    answer = [math.fsum(a[(i+r)*(w+2)+j+s] * weights[r*3+s] for r in range(3) for s in range(3))
              for i in range(h) for j in range(w)]
    return [a, weights, [17.] * (h*w)], answer


def conv1d(length, width):
    """Direct valid cross-correlation, the kernel convention used in ML convolutions."""
    if width < 1:
        raise ValueError("filter width must be positive")
    return _generic([(length+width-1,), (width,), (length,)],
                    ['affine_map<(i,k)->(i+k)>', 'affine_map<(i,k)->(k)>', 'affine_map<(i,k)->(i)>'],
                    ['parallel','reduction'], '%p = arith.mulf %x, %y : f64\n        %s = arith.addf %sum, %p : f64')


def conv1d_case(length, width, seed=7):
    rng = random.Random(seed)
    a = [rng.uniform(-1,1) for _ in range(length+width-1)]
    b = [rng.uniform(-1,1) for _ in range(width)]
    return [a,b,[17.] * length], [math.fsum(a[i+k] * b[k] for k in range(width)) for i in range(length)]


GENERATORS = {"matmul": matmul, "matvec": matvec, "row_reduce": row_reduce,
              "elementwise": elementwise, "stencil": stencil, "conv1d": conv1d}
REFERENCES = {"matmul": matmul_case, "matvec": matvec_case, "row_reduce": row_reduce_case,
              "elementwise": elementwise_case, "stencil": stencil_case, "conv1d": conv1d_case}
