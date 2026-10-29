"""Small explicit MLIR generators, with an independent scalar reference."""
import math
import random


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
